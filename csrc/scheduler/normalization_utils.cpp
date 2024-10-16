// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_evaluator.h>
#include <grouped_reduction.h>
#include <id_model/id_model.h>
#include <instrumentation.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <utils.h>
#include <val_graph_visitor.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {
namespace normalization_scheduler_utils {

using scheduler_debug_utils::log;

PreferredLaunchConfig::PreferredLaunchConfig() : valid_(true) {
  initValidGdims();
  resetBdim();
  resetGdim();
}

bool PreferredLaunchConfig::isNextSmallerBdimx() const {
  return grid_dims_pos_ + 1 == (int)valid_grid_dims_.size();
}

bool PreferredLaunchConfig::canLowerBdimx() const {
  return bdimx() > kMinBdimx;
}

bool PreferredLaunchConfig::setBdimx(int bdimx, bool dry_run) {
  constexpr int block_size = 256;

  if (bdimx < kMinBdimx || bdimx > kMaxBdimx) {
    return false;
  }

  NVF_ERROR(block_size % bdimx == 0, "Invalid bdimx: ", bdimx);
  int bdimy = block_size / bdimx;

  if (!dry_run) {
    bdimy_ = bdimy;
    bdimx_ = bdimx;
  }

  return true;
}

// Populate the list of valid gridDim configs for persistent grid
// normalization kernels in the order of increasing gridDim.y.
// Start
// with gridDim.y == 2. For example, on A100, the list would be: [(54,
// 2), (36, 3), (27, 4), (21, 5), (18, 6), (15, 7), (13, 8), (12, 9),
// (10, 10), (9, 12), (8, 13), (7, 15), (6, 18), (5, 21), (4, 27), (3,
// 36), (2, 54)].
void PreferredLaunchConfig::initValidGdims() {
  std::vector<std::pair<int, int>> grid_dims;
  const int num_sms =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int max_first_half =
      static_cast<int>(std::sqrt(static_cast<float>(num_sms)));
  for (int gdimy = 2; gdimy <= max_first_half; ++gdimy) {
    int gdimx = num_sms / gdimy;
    grid_dims.emplace_back(gdimx, gdimy);
  }
  // Reverse the first half and swap gridDim.x and gridDim.y. That
  // list becomes the latter half
  auto latter_half = grid_dims;
  std::reverse(latter_half.begin(), latter_half.end());
  for (const auto& gdimx_gdimy : latter_half) {
    if (gdimx_gdimy.second == gdimx_gdimy.first) {
      // This is already in the first half
      continue;
    }
    grid_dims.emplace_back(gdimx_gdimy.second, gdimx_gdimy.first);
  }
  valid_grid_dims_ = grid_dims;
}

bool PreferredLaunchConfig::moveToNextConfig() {
  if (moveToNextGdim()) {
    return true;
  }

  // Can't increase gdimy. Try bdimy next.
  if (moveToNextBdim()) {
    return true;
  }

  // No more valid config
  invalidate();
  return false;
}

bool PreferredLaunchConfig::moveToNextBdim() {
  const int new_bdimx = bdimx() / 2;
  if (setBdimx(new_bdimx)) {
    resetGdim();
    return true;
  } else {
    invalidate();
    return false;
  }
}

bool PreferredLaunchConfig::moveToNextGdim() {
  auto grid_dims_next_pos = getNextGdimsPos();
  if (grid_dims_next_pos >= 0) {
    grid_dims_pos_ = grid_dims_next_pos;
    return true;
  } else {
    return false;
  }
}

int PreferredLaunchConfig::peekNextGdimx() const {
  auto grid_dims_next_pos = getNextGdimsPos();
  if (grid_dims_next_pos >= 0) {
    return gdimxAt(grid_dims_next_pos);
  } else {
    return -1;
  }
}

int PreferredLaunchConfig::peekNextGdimy() const {
  auto grid_dims_next_pos = getNextGdimsPos();
  if (grid_dims_next_pos >= 0) {
    return gdimyAt(grid_dims_next_pos);
  } else {
    return -1;
  }
}

int PreferredLaunchConfig::getNextGdimsPos() const {
  auto grid_dims_next_pos = grid_dims_pos_ + 1;
  if (grid_dims_next_pos < (int)valid_grid_dims_.size()) {
    return grid_dims_next_pos;
  } else {
    return -1;
  }
}

namespace {

// Estimated register count available for persistent buffer. The
// available space is considered to depend on the size of the
// persistent buffer itself due to the predicate caching
int64_t getAvailableRegisterCount(int64_t persistent_buffer_factor) {
  // The thread block size is (currently) always 256, so each thread
  // can use up to 255 registers
  int64_t register_count = scheduler_utils::max_registers_per_thread;

  // Offset a constant overhead
  register_count -= scheduler_utils::register_overhead;

  // Allow small number of spills
  register_count += 5;

  // account for index caching, assuming each cache entry
  //  consumes one register
  // TODO: Consider relaxing this reduction. It seems likes
  //  overestimation.
  register_count -= persistent_buffer_factor;

  return register_count;
}

int64_t getMinPersistentBufferSize(
    const int64_t total_reduction_numel,
    const int64_t bdimy,
    const int64_t gdimy) {
  return ceilDiv(ceilDiv(total_reduction_numel, bdimy), gdimy);
}

// Return true if a given combination of parameters is likely to
// result in no (or little) register spilling
bool checkIfWithinRegisterSpace(
    int64_t total_reduction_numel,
    int64_t persistent_buffer_size,
    int64_t vectorize_factor,
    int64_t bdimy,
    int64_t gdimy) {
  // The extent of the persistent buffer domain
  auto pb_factor =
      getMinPersistentBufferSize(total_reduction_numel, bdimy, gdimy);

  NVF_ERROR(pb_factor > 0);

  const auto available_reg_count = getAvailableRegisterCount(pb_factor);

  auto per_thread_persistent_buffer_size =
      ceilDiv(ceilDiv(persistent_buffer_size, bdimy), gdimy) * vectorize_factor;

  auto persistent_buffer_reg_count =
      ceilDiv(per_thread_persistent_buffer_size, sizeof(int));

  log("persistent_buffer_reg_count: ",
      persistent_buffer_reg_count,
      ", available_reg_count: ",
      available_reg_count);

  return persistent_buffer_reg_count <= available_reg_count;
}

// Calculate the factor of work of the last thread block in each of
// reductions. More specifically, use the number of serial
// iterations for the persistent buffer loop as a proxy of the
// amount of work. The rest of the blocks should execute the loop
// buffer_size times, whereas the last block only processes the
// remaining iterations.
double getLastBlockWorkRatio(
    const int64_t total_reduction_numel,
    const int64_t bdimy,
    const int64_t persistent_buffer_size) {
  auto last_block_pb =
      total_reduction_numel % (persistent_buffer_size * bdimy) / bdimy;
  return ((double)last_block_pb) / (double)persistent_buffer_size;
};

// In the current outer normalization scheduling, only the last thread
// block of each reduction group hits the fallback path of the
// unswitched loops, so it can be significantly slower than the
// rest. This is particularly problematic with grid persistence as all
// thread blocks need to synchronize, so the slowest block determines
// the performance. This could be to some extent mitigated by
// adjusting the buffer size such that the work assigned to the last
// block is relatively smaller than the work assigned to the
// rest.
//
// Here, given a valid launch config, we try to slightly adjust it so
// that the ratio of the last work becomes the smallest. We do this by
// increasing buffer sizes and in turn decreasing gdimy and picking the
// configuration that has the smallest work ratio. All of this is done
// with some bounds, e.g., the buffer size should still be within the
// register space, the decrease of gdimy should be less than 10%,
// etc. These threshold values are experimentally picked on A100 with
// the current benchmarks, but more tuning would likely lead to better
// performance.
//
// The function returns the adjusted gdimy and persistent buffer size
// as well as a bool indicating whether the work size is
// sufficiently reduced. Nullopt is returned if no adjustment is
// successfully done and the search should continue.
std::optional<std::tuple<int64_t, int64_t, bool>> reduceWorkOfLastBlock(
    const PreferredLaunchConfig& launch_cfg,
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t persistent_buffer_size,
    const int64_t vectorize_factor) {
  const auto bdimy = launch_cfg.bdimy();

  // Aim to reduce the work size of the last block to be smaller than
  // some factor of the rest of the blocks.
  const double target_last_block_work_ratio = 0.25;

  // Start with the current gdimy and buffer size. Gradually increase
  // the buffer size and in turn decrease gdimy with the bounds set as
  // below.
  int64_t current_gdimy = launch_cfg.gdimy();
  auto current_buffer_size =
      getMinPersistentBufferSize(total_reduction_numel, bdimy, current_gdimy);

  log("reduceWorkOfLastBlock: ", current_gdimy, ", ", current_buffer_size);

  // Threshold to stop decreasing gdimy
  const auto min_gdimy = static_cast<int64_t>((double)current_gdimy * 0.9);

  // Keep track of the best gdimy and buffer size configuration
  auto optimal_size = current_buffer_size;
  auto optimal_gdimy = current_gdimy;
  double optimal_work_ratio =
      getLastBlockWorkRatio(total_reduction_numel, bdimy, current_buffer_size);

  // Find the best gdimy and buffer size configuration by lowering
  // gdimy. Stop if the minimum gdimy is hit or the register limit is
  // reached.
  while (current_gdimy >= min_gdimy &&
         checkIfWithinRegisterSpace(
             total_reduction_numel,
             persistent_buffer_size,
             vectorize_factor,
             bdimy,
             current_gdimy)) {
    auto ratio_of_last_block_work = getLastBlockWorkRatio(
        total_reduction_numel, bdimy, current_buffer_size);
    log("Ratio of last block work: ",
        ratio_of_last_block_work,
        ", persistent_buffer: ",
        current_buffer_size,
        ", gdimy: ",
        current_gdimy);

    if (ratio_of_last_block_work < optimal_work_ratio) {
      optimal_work_ratio = ratio_of_last_block_work;
      optimal_size = current_buffer_size;
      optimal_gdimy = current_gdimy;
    }

    if (ratio_of_last_block_work < target_last_block_work_ratio) {
      // Good enough config found; stop searching
      break;
    }

    // not good enough; increase persistent buffer
    ++current_buffer_size;
    // adjust gdimy (decreased as persistent_buffer is increased)
    current_gdimy =
        ceilDiv(ceilDiv(total_reduction_numel, bdimy), current_buffer_size);

    log("Next buffer size: ",
        current_buffer_size,
        ", Next gdimy: ",
        current_gdimy);
  }

  // Use the optimal ratio if it's within the threshold
  if (optimal_work_ratio < target_last_block_work_ratio) {
    log("Successfully reduced to ", optimal_work_ratio);
    return std::make_tuple(optimal_gdimy, optimal_size, true);
  }

  // Acceptable config not found. Continue searching a better config
  // by moving to the next candidate. However, if the next candidate
  // incurs a larger number of grid syncs, i.e., the serial factor of
  // the iteration domain is larger, the additional overhead would
  // likely to outweigh the benefit of potentially better block
  // specialization, so pick the best among found so far.
  auto next_gdimx = launch_cfg.peekNextGdimx();

  // If the next gdimx is negative, that means there's no more config
  // candidate or the next would decrease the bdimx, which could be a
  // large perf degradation, so stop the search then.
  if (next_gdimx < 0) {
    log("Stop as there's no more search space left for gdimx");
    return std::make_tuple(optimal_gdimy, optimal_size, false);
  }

  if (next_gdimx > 0) {
    auto remaining_iteration_factor = ceilDiv(
        ceilDiv(total_iteration_numel, vectorize_factor), launch_cfg.bdimx());
    auto current_iterration_count =
        ceilDiv(remaining_iteration_factor, launch_cfg.gdimx());
    auto next_iteration_count = ceilDiv(remaining_iteration_factor, next_gdimx);
    log("Next iteration count: ",
        next_iteration_count,
        ", next gdimx: ",
        next_gdimx,
        ", current iteration: ",
        current_iterration_count,
        ", curreng gdimx: ",
        launch_cfg.gdimx());
    if (next_iteration_count > current_iterration_count) {
      log("Still not good but stop here to avoid increase of iteration count");
      return std::make_tuple(optimal_gdimy, optimal_size, false);
    }
  }

  log("Acceptable config not found. Continue search");
  return std::nullopt;
}

} // namespace

// Iterate configurations from largest blockDim.x and smallest
// gridDim.y until the per-thread size of the persistent buffer
// becomes sufficiently small enough not to cause (significant)
// register spill.
std::optional<GridOuterNormalizationParams> getGridOuterNormalizationParams(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t persistent_buffer_size) {
  PreferredLaunchConfig launch_cfg;

  // The launch config starts with the largest blockDim.x, which may
  // be larger than the iteration size. Decrease it until it doesn't
  // exceed the iteration size.
  const auto max_bdimx = ceilDiv(total_iteration_numel, vectorize_factor);
  while (launch_cfg.bdimx() > max_bdimx) {
    if (!launch_cfg.moveToNextBdim()) {
      // The iteration size is too small. It might still be worthwhile
      // to be persistent, but it's unlikely to be performant anyway
      return std::nullopt;
    }
  }

  // Iterate candidates of launch configurations
  while (!launch_cfg.isInvalid()) {
    log("Current config: ", launch_cfg);

    // Skip if iterations are not evenly distributed among thread
    // blocks unless the remaining factor is smaller than
    // gridDim.x. However, don't skip if this is the last valid config
    // within the same blockDim config.
    auto remaining_gdimx_factor =
        ceilDiv(total_iteration_numel / vectorize_factor, launch_cfg.bdimx());
    // TODO: Needs better tuning. Probably want to allow
    // configurations that are slightly uneven
    if (remaining_gdimx_factor > launch_cfg.gdimx() &&
        remaining_gdimx_factor % launch_cfg.gdimx() != 0 &&
        !launch_cfg.isNextSmallerBdimx()) {
      log("Rejected due to uneven iteration domain");
      launch_cfg.moveToNextConfig();
      continue;
    }

    if (!checkIfWithinRegisterSpace(
            total_reduction_numel,
            persistent_buffer_size,
            vectorize_factor,
            launch_cfg.bdimy(),
            launch_cfg.gdimy())) {
      log("Rejected due to register spill");
      launch_cfg.moveToNextConfig();
      continue;
    }

    // At this point, gdimy is large enough to keep the register
    // pressure low enough.

    // In case the iteration domain is small, the gdimx and bdimx pair
    // may be too large and some threads/blocks may be idle.

    if (remaining_gdimx_factor < launch_cfg.gdimx()) {
      log("gdimx too large: ",
          remaining_gdimx_factor,
          ", vec: ",
          vectorize_factor);
      launch_cfg.moveToNextConfig();
      continue;
    }

    // If there's idle tidx threads, don't accept if there's further
    // config candidates with smaller bdimx
    if (vectorize_factor * launch_cfg.bdimx() * launch_cfg.gdimx() >
            total_iteration_numel &&
        launch_cfg.canLowerBdimx()) {
      log("Skip due to too large bdimx: ", launch_cfg.bdimx());
      launch_cfg.moveToNextBdim();
      continue;
    }

    // Adjust gdimy and buffer size for processing predicates more
    // efficiently through the block specialization, so that the last
    // block is assigned with a relatively small chunk of work.
    // For some reason, this doesn't work well on Titan RTX. It seems
    // it's just better unswitching by a small factor.
    // TODO: Test other generations of GPUs
    int64_t adjusted_gdimy = -1;
    int64_t adjusted_buffer_size = -1;
    bool last_block_work_reduced = false;
    const auto major_ver = at::cuda::getCurrentDeviceProperties()->major;
    const auto minor_ver = at::cuda::getCurrentDeviceProperties()->minor;
    if (major_ver == 7 && minor_ver == 5) {
      adjusted_gdimy = launch_cfg.gdimy();
      adjusted_buffer_size = getMinPersistentBufferSize(
          total_reduction_numel, launch_cfg.bdimy(), launch_cfg.gdimy());
      last_block_work_reduced = false;
    } else {
      auto gdimy_pb_size = reduceWorkOfLastBlock(
          launch_cfg,
          total_reduction_numel,
          total_iteration_numel,
          persistent_buffer_size,
          vectorize_factor);
      if (!gdimy_pb_size.has_value()) {
        launch_cfg.moveToNextConfig();
        continue;
      }
      std::tie(adjusted_gdimy, adjusted_buffer_size, last_block_work_reduced) =
          *gdimy_pb_size;
    }

    // Acceptable configuration found
    auto launch_params = LaunchParams(
        launch_cfg.gdimx(),
        adjusted_gdimy,
        LaunchParams::UNINITIALIZED_VAL,
        launch_cfg.bdimx(),
        launch_cfg.bdimy(),
        LaunchParams::UNINITIALIZED_VAL);

    // If the last block is sufficiently reduced, unswitch the whole
    // persistent buffer. Otherwise, unswitch by a factor of 4.
    int64_t unswitch_factor = last_block_work_reduced
        ? adjusted_buffer_size
        : std::min(4l, adjusted_buffer_size);

    GridOuterNormalizationParams params = {
        .launch_params = launch_params,
        .persistent_buffer_factor = adjusted_buffer_size,
        .unswitch_factor = unswitch_factor};
    return params;
  }

  // No valid config found. Return launch_cfg, which should be marked
  // as invalid
  NVF_ERROR(launch_cfg.isInvalid());
  return std::nullopt;
}

bool checkIfReductionsAreInnerOuter(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs) {
  bool pass_combined_heck = true;
  // inner reduction must be [I,I,...R,R]
  auto innerReductionCheck = [](TensorView* tv) {
    int ndim = static_cast<int>(tv->nDims());
    int lastIter = -1;
    while (lastIter < ndim - 1 && tv->axis(lastIter + 1)->isIteration()) {
      lastIter++;
    }
    int firstRedu = ndim;
    while (firstRedu > 0 && tv->axis(firstRedu - 1)->isReduction()) {
      firstRedu--;
    }
    return lastIter >= 0 && firstRedu < ndim && lastIter == firstRedu - 1;
  };
  // outer reduction must be [R,R,..I,I]
  auto outerReductionCheck = [](TensorView* tv) {
    int ndim = static_cast<int>(tv->nDims());
    int lastRedu = -1;
    while (lastRedu < ndim - 1 && tv->axis(lastRedu + 1)->isReduction()) {
      lastRedu++;
    }
    int firstIter = ndim;
    while (firstIter > 0 && tv->axis(firstIter - 1)->isIteration()) {
      firstIter--;
    }
    return lastRedu >= 0 && firstIter < ndim && lastRedu == firstIter - 1;
  };
  for (auto itv : inner_reduction_tvs) {
    if (!innerReductionCheck(itv)) {
      pass_combined_heck = false;
      break;
    }
  }
  for (auto otv : outer_reduction_tvs) {
    if (!outerReductionCheck(otv)) {
      pass_combined_heck = false;
      break;
    }
  }
  return pass_combined_heck;
}

bool hasSharedInput(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs) {
  bool has_shared_input = false;
  std::unordered_set<TensorView*> input_inner_reduction_tvs;
  for (auto tv : inner_reduction_tvs) {
    for (auto input_tv : ir_utils::inputTvsOf(tv)) {
      input_inner_reduction_tvs.emplace(input_tv);
    }
  }
  for (auto tv : outer_reduction_tvs) {
    for (auto input_tv : ir_utils::inputTvsOf(tv)) {
      if (input_inner_reduction_tvs.find(input_tv) !=
          input_inner_reduction_tvs.end()) {
        has_shared_input = true;
        break;
      }
    }
    if (has_shared_input) {
      break;
    }
  }
  return has_shared_input;
}

bool isConnectedOnlyThroughReductionProducer(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs) {
  const std::unordered_set<TensorView*> outer_tv_set{
      outer_reduction_tvs.begin(), outer_reduction_tvs.end()};
  // initialize disjoint sets with tvs connected to inner reduction tvs
  std::unordered_set<TensorView*> disjoint_tvs =
      scheduler_utils::getAllTvsFrom(inner_reduction_tvs, outer_tv_set);
  // get disjoint sets with tvs connected to outer reduction tvs
  // check if there is any intersection
  for (auto otv : outer_reduction_tvs) {
    const auto& producers = ir_utils::producerTvsOf(otv);
    // cutoff at producers of outer reduction tvs as they are computed with
    // inner reducitons
    const auto& connected_tv_set = scheduler_utils::getAllTvsFrom(
        {otv}, {producers.begin(), producers.end()});
    for (auto tv : connected_tv_set) {
      if (!disjoint_tvs.emplace(tv).second) {
        return false;
      }
    }
  }
  return true;
}

bool isReductionIterationAxisMatched(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs) {
  // set up reference, checkIfReductionsAreInnerOuter already ensures all the
  // tensor domains are either iteration or reduction, so we can just use a
  // vector of bool.
  auto reference_tv = inner_reduction_tvs[0];
  std::vector<bool> is_reduction(reference_tv->nDims(), false);
  for (const auto i : c10::irange(reference_tv->nDims())) {
    auto id = reference_tv->axis(i);
    NVF_CHECK(
        id->getIterType() == IterType::Iteration ||
            id->getIterType() == IterType::Reduction,
        "Invalid iteration type: ",
        id->getIterType());
    if (id->isReduction()) {
      is_reduction[i] = true;
    }
  }
  // check other inner reduction tvs, the corresponding axis should be
  // reduction.
  for (auto i : c10::irange(1, inner_reduction_tvs.size())) {
    auto tv = inner_reduction_tvs[i];
    for (const auto i : c10::irange(tv->nDims())) {
      auto id = tv->axis(i);
      NVF_CHECK(
          id->getIterType() == IterType::Iteration ||
              id->getIterType() == IterType::Reduction,
          "Invalid iteration type: ",
          id->getIterType());

      if (id->isReduction() != is_reduction.at(i)) {
        return false;
      }
    }
  }
  // check outer reduction tvs, the corresponding axis should be iteration.
  for (auto tv : outer_reduction_tvs) {
    for (const auto i : c10::irange(tv->nDims())) {
      auto id = tv->axis(i);
      NVF_CHECK(
          id->getIterType() == IterType::Iteration ||
              id->getIterType() == IterType::Reduction,
          "Invalid iteration type: ",
          id->getIterType());

      if (id->isIteration() != is_reduction.at(i)) {
        return false;
      }
    }
  }

  return true;
}

int64_t partialReductionBufferSize(
    const std::vector<TensorView*>& outer_reduction_tvs,
    SchedulerRuntimeInfo& runtime_info) {
  int64_t partial_reduction_buffer_size = 0;
  for (auto buffer : outer_reduction_tvs) {
    int64_t buffer_size = -1;
    for (auto id : buffer->getLogicalDomain()) {
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }
      auto id_size = runtime_info.expressionEvaluator().evaluate(id->extent());
      NVF_ERROR(id_size.hasValue(), "Could not infer persistent buffer size.");
      if (buffer_size == -1) {
        buffer_size = id_size.as<int64_t>();
      } else {
        buffer_size *= id_size.as<int64_t>();
      }
    }
    buffer_size = (buffer_size == -1) ? 0
                                      : buffer_size *
            (int64_t)dataTypeSize(buffer->getDataType().value(),
                                  runtime_info.getIndexType());
    partial_reduction_buffer_size += buffer_size;
  }
  return partial_reduction_buffer_size;
}

// Get the appropriate scheduler based on reduction type
SchedulerType getPersistentHeuristicFor(ReductionType reduction_type) {
  switch (reduction_type) {
    case ReductionType::Inner:
      return SchedulerType::InnerPersistent;
    case ReductionType::Outer:
      return SchedulerType::OuterPersistent;
    case ReductionType::InnerOuter:
      return SchedulerType::InnerOuterPersistent;
    default:
      NVF_THROW(
          "Reduction type not supported! reduction_type: ", reduction_type);
  }
}

void checkReductionTvForScheduling(Fusion* fusion, TensorView* ref_red_tv) {
  NVF_ERROR(ref_red_tv != nullptr, "Reduction TensorView wasn't found.");
  NVF_ERROR(ref_red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  NVF_ERROR(
      ir_utils::isReductionOp(ref_red_tv->definition()),
      "TensorView doesn't have a reduction.");
  NVF_ERROR(
      std::any_of(
          fusion->inputs().begin(),
          fusion->inputs().end(),
          [](Val* inp) { return inp->isA<TensorView>(); }),
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");
}

int64_t getMaxRegOrSharedMemorySizeForPersistentBuffer(
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*>& persistent_buffers,
    const bool can_use_smem_persistent) {
  // Init to register file size, which is half of the full register file size
  int64_t available_persistent_buffer_size =
      scheduler_utils::register_file_size;
  // shared memory persistent is not implemented for 3D inner reduction
  if (!can_use_smem_persistent) {
    return available_persistent_buffer_size;
  }
  // Check available shared memory
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t max_shared_memory_size =
      (int64_t)dev_prop->sharedMemPerBlockOptin;
  // Some shared memories are reserved for kernel launch overhead and
  // reduction_broadcast_workspace. Estimation is conservative, but should
  // be good enough. The actual threads per block is set in the heuristics
  // and it may be smaller than maxThreadsPerBlock.
  // TODO: More accurate estimation of available shared memory size.
  const int64_t kernel_overhead = (int64_t)dev_prop->reservedSharedMemPerBlock;
  int64_t max_buffer_dtype_size = 1;
  for (auto tv : persistent_buffers) {
    max_buffer_dtype_size = std::max(
        max_buffer_dtype_size,
        dataTypeSize(tv->getDataType().value(), runtime_info.getIndexType()));
  }
  const int64_t reduction_broadcast_workspace =
      (int64_t)(dev_prop->maxThreadsPerBlock) * max_buffer_dtype_size;
  const int64_t available_shared_memory_size =
      max_shared_memory_size - kernel_overhead - reduction_broadcast_workspace;
  available_persistent_buffer_size =
      std::max(available_persistent_buffer_size, available_shared_memory_size);
  return available_persistent_buffer_size;
}

// Returns true if persistent buffers are projected to inputs, meaning the
// inputs are cached instead of the persistent buffers.
bool isProjectBufferToInputs(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const scheduler_utils::PersistentBufferInfo& persistent_buffer_info,
    const scheduler_utils::PersistentBufferSizeReturn&
        persistent_buffer_size_info,
    const SchedulerType scheduler_type,
    const bool can_use_smem_persistent,
    const bool check_projected_buffer_size) {
  // don't project if there are view ops and no buffer can be projected
  bool can_project = !persistent_buffer_info.has_view_ops &&
      persistent_buffer_size_info.projected_persistent_buffer_size > 0;
  if (!can_project) {
    return false;
  }

  // Enusre project to inputs can save persistent buffer size,
  // unless it's innerOuter with outer broadcast where project to inputs reduces
  // gmem access.
  if (check_projected_buffer_size &&
      persistent_buffer_size_info.projected_persistent_buffer_size >=
          persistent_buffer_size_info.persistent_buffer_size) {
    return false;
  }

  // must project to inputs otherwise don't have enough register or shared
  // memory to store the buffers. Even after projecting, may still not have
  // enough register or shared memory, then canScheduleRunTime will return
  // false. For InnerOuterPersistent, both register and shared memory are used
  // and will be handled in getPersistentBufferStorageParams.
  if (scheduler_type != SchedulerType::InnerOuterPersistent) {
    int64_t max_available_buffer =
        getMaxRegOrSharedMemorySizeForPersistentBuffer(
            runtime_info,
            persistent_buffer_info.persistent_buffers,
            can_use_smem_persistent);
    if (max_available_buffer <
        persistent_buffer_size_info.persistent_buffer_size) {
      return true;
    }
  }

  // don't project if recompute requires rng op
  if (persistent_buffer_info.projection_with_rng_op) {
    return false;
  }

  // free to project if no exp op
  if (!persistent_buffer_info.projection_with_exp_op) {
    return true;
  }

  // consider buffer size when exp op exists
  if (scheduler_type == SchedulerType::InnerPersistent) {
    // check if the non-projected persistent buffer is small enough,
    // i.e., not affecting the occupancy, projecting back to the inputs
    // isn't buying us anything.
    // This check only works for inner persistent as outer persistent and
    // inner outer persistent usually have large register pressure and always
    // want to project back to the inputs.
    // Assumptions:
    // (1) 50% occupancy, which is 1024 active threads per SM.
    // (2) 128 threads per block.
    // (3) 24 registers per thread for overhead.
    // (4) 8 extra register per thread allowing register spills.
    // The derived [buffer_per_block] is 48*128*4 = 24KB.
    constexpr int64_t active_threads_per_sm = 1024l;
    constexpr int64_t threads_per_block = 128l;
    constexpr int64_t overhead_register_per_thread = 24l;
    constexpr int64_t extra_register_allowing_spills = 8l;
    constexpr int64_t total_register_per_thread =
        scheduler_utils::register_file_size_full /
        scheduler_utils::bytes_per_register / active_threads_per_sm;
    constexpr int64_t buffer_register_per_thread = total_register_per_thread -
        overhead_register_per_thread + extra_register_allowing_spills;
    constexpr int64_t buffer_per_block = threads_per_block *
        buffer_register_per_thread * scheduler_utils::bytes_per_register;
    if (persistent_buffer_size_info.persistent_buffer_size <=
        buffer_per_block) {
      return false;
    }
  }

  return true;
}

PersistentKernelProperties getPersistentKernelProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE(
      "normalization_scheduler_utils::getPersistentKernelProperties");

  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();
  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");
  auto ref_red_tv = reduction_tvs[0];

  checkReductionTvForScheduling(fusion, ref_red_tv);

  scheduler_utils::ReductionTvProperties properties;
  TensorView* reduced_tv = nullptr;
  int64_t vectorize_factor = -1;

  properties =
      scheduler_utils::getReductionProperties(fusion, runtime_info, ref_red_tv);
  reduced_tv = ir_utils::getSoleProducerTv(ref_red_tv);

  // Although properties contains runtime information
  // "inner_most_dimension_ndims" is a compile time value
  auto vec_break_point = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizationBreakPointOfReductionProducer>(
      data_cache, [&ref_red_tv, &reduced_tv, &properties]() {
        return std::make_unique<int64_t>(
            vectorize_helper::getVectorizationBreakPointOfReductionProducer(
                ref_red_tv, reduced_tv, properties.inner_most_dimension_ndims));
      });

  vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info, reduced_tv, data_cache, vec_break_point.get());

  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });
  auto& persistent_buffer_info = persistent_buffer_info_entry.get();
  NVF_ERROR(
      !persistent_buffer_info.persistent_buffers.empty(),
      "Persistent scheduler requires persistent buffers.");
  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Can project to input?
  // Figure out if we want to projet persistent buffers to the inputs for
  // exmaple if we have an input tensor t0 that's fp16:
  //
  // t0 = makeSymbolicTensor(2, DataType::Half)
  // t1 = castOp(DataType::Float, t0)
  // t2 = sum(t1, 1)
  // t3 = broadcast(t2, {false, true})
  // t4 = set(t1)
  // t5 = add(t4, t3)
  // t6 = castOp(DataType::Half, t5)
  //
  // The persistent buffer is detected as being t1, which would save the
  // persistent buffer as a float, however we could obviously just save t0 which
  // is half and would take half the memory. A more complex scenario of this
  // which requires more advanced analysis is batch norm backwards.
  // TODO: Fix projected persistent buffers with view
  // https://github.com/csarofeen/pytorch/issues/2054

  // Project to input when it can reduce buffer size and the gains of
  // reducing buffer size is larger than the pains of recalculations.
  bool can_use_smem_persistent =
      properties.inner_most_dimension_numel == properties.total_reduction_numel;
  bool project_persistent_buffers = isProjectBufferToInputs(
      fusion,
      runtime_info,
      persistent_buffer_info,
      persistent_buffer_size_info,
      scheduler_type,
      can_use_smem_persistent);
  int64_t max_persistent_buffer_size = project_persistent_buffers
      ? persistent_buffer_size_info.projected_persistent_buffer_size
      : persistent_buffer_size_info.persistent_buffer_size;

  // Info about input and output tensors
  // Base max dtype and n_tensor_inputs on tensors that are vectorizable (i.e.
  // share inner dimension with data pattern we're looking at).
  // TODO: This might be better if it was the larger of input or outputs. Would
  // be even better if we had better analysis as not all unrolled elements have
  // to be alive at the same time.
  auto unrollable_inputs_outputs_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::UnrollableInputsAndOutputs>(
          data_cache, [&reduced_tv]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    reduced_tv, false, false));
          });

  // Info about ops in the fusion, used to set model specific parameters
  int64_t max_dtype_size = 1;
  int64_t n_tensor_inputs = 0;

  auto& unrollable_inputs_outputs = unrollable_inputs_outputs_entry.get();
  for (auto tv : unrollable_inputs_outputs) {
    if (!tv->isFusionInput()) {
      continue;
    }
    max_dtype_size = std::max(
        max_dtype_size,
        dataTypeSize(tv->getDataType().value(), runtime_info.getIndexType()));
    n_tensor_inputs++;
  }
  // To prevent division by zero, ensure that n_tensor_inputs is not equal to
  // zero.
  n_tensor_inputs = std::max(n_tensor_inputs, (int64_t)1);

  // Exp op typically used in softmax is expensive and needs more registers.
  bool has_exp_op = false;

  // Could save fusion->exprs() instead of doing this, but allTvs is already
  // cached in fusion so using that for now.
  for (auto tv : fusion->allTvs()) {
    if (tv->definition() == nullptr) {
      continue;
    }
    if (tv->definition()->isA<UnaryOp>() &&
        tv->definition()->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Exp) {
      has_exp_op = true;
      break;
    }
  }

  // Return collected properties to get heuristics.
  return PersistentKernelProperties{
      .inner_most_dimension_numel = properties.inner_most_dimension_numel,
      .total_reduction_numel = properties.total_reduction_numel,
      .total_iteration_numel = properties.total_iteration_numel,
      .max_persistent_buffer_size = max_persistent_buffer_size,
      .n_tensor_inputs = n_tensor_inputs,
      .max_dtype_size = max_dtype_size,
      .vectorize_factor = vectorize_factor,
      .project_persistent_buffers = project_persistent_buffers,
      .index_type = runtime_info.getIndexType(),
      .has_exp_op = has_exp_op,
      .persistent_buffers = persistent_buffer_info.persistent_buffers};
}

bool checkOpsAndInputs(Fusion* fusion, SchedulerType scheduler_type) {
  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "Fusion is resharding.");
    return false;
  }

  // Needs at least one reduction to consider.
  if (!ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "needs a reduction op");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, scheduler_type)) {
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type,
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  return true;
}

bool checkReductionPattern(
    Fusion* fusion,
    SchedulerType scheduler_type,
    const std::vector<TensorView*>& reduction_tvs1,
    const std::vector<TensorView*>& reduction_tvs2) {
  // Ensure that the reduction operations share the same axes in their root
  // domains
  FusionGuard fg(fusion);
  ComputeAtLogicalDomainMap logical_map;
  logical_map.build(true);

  // Helper function to check the pattern equivalence for a list of
  // TensorViews
  auto checkPattern = [&](const std::vector<TensorView*>& rtvs) -> bool {
    for (const auto it : c10::irange(1, rtvs.size())) {
      if (!registry_utils::checkPatternEquivalence(
              rtvs[it - 1], rtvs[it], logical_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            scheduler_type,
            "Un-mapped multi-reduction: ",
            rtvs[it - 1]->toString(),
            " and ",
            rtvs[it]->toString());
        return false;
      }
    }
    return true;
  };

  // Check the pattern equivalence for the first set of reduction TensorViews
  if (!checkPattern(reduction_tvs1)) {
    return false;
  }

  // Return if there is no second set of reduction TensorViews
  if (reduction_tvs2.empty()) {
    return true;
  }

  // Check the pattern equivalence for the second set of reduction TensorViews
  // if provided.
  if (!checkPattern(reduction_tvs2)) {
    return false;
  }

  return true;
}

// The identical compile time check of InnerPersistentKernelScheduler and
// OuterPersistentKernelScheduler.
bool compileTimeCheck(Fusion* fusion, SchedulerType scheduler_type) {
  // common checks for all persistent heuristics
  if (!normalization_scheduler_utils::checkOpsAndInputs(
          fusion, scheduler_type)) {
    return false;
  }

  // check reduction types and pattern
  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "no reduction tv");
    return false;
  }
  auto reduction_type =
      reduction_scheduler_utils::getReductionType(reduction_tvs);
  const SchedulerType persistent_heuristic =
      getPersistentHeuristicFor(reduction_type);
  if (persistent_heuristic != scheduler_type) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type,
        "scheduler_type doesn't match with reduction type `",
        persistent_heuristic,
        "`.");
    return false;
  }

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          scheduler_type, "Fusion requires view being reversible.");
      return false;
    }

    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    auto reference_tv = reduction_tvs[0];
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          scheduler_type, "View may interfere with normalization scheduling.");
      return false;
    }
  }

  // Before examining the reduction axes want to quickly
  //   check the reductions have the same axis width
  //   to avoid building root domain map in easier cases
  bool valid_axis_count = false;
  size_t axis_count = 0;
  auto reduction_root_size = [](TensorView* red_tv) {
    size_t count = 0;
    for (auto id : red_tv->getMaybeRootDomain()) {
      if (!id->isBroadcast()) {
        count++;
      }
    }
    return count;
  };

  for (auto red : reduction_tvs) {
    if (!valid_axis_count) {
      valid_axis_count = true;
      axis_count = reduction_root_size(red);
    } else {
      if (reduction_root_size(red) != axis_count) {
        scheduler_debug_utils::canScheduleRejectReason(
            scheduler_type,
            "Inconsistent reduction root size: ",
            red->toString(),
            ", expected: ",
            axis_count);
        return false;
      }
    }
  }

  if (!checkReductionPattern(fusion, scheduler_type, reduction_tvs)) {
    return false;
  }

  // Only accept persistent kernels
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "has unsupported gather-like ops before normalization");
    return false;
  }

  return true;
}

std::vector<TensorView*> movePersistentBufferToSmem(
    Fusion* fusion,
    const ReductionParams* rparams,
    const std::vector<TensorView*>& cached_inputs) {
  std::vector<TensorView*> smem_consumers;
  // Transfer the persistent buffer tensors to shared memory. These tensors are
  // housed in smem_persistent_buffers. If a candidate tensor is input, move its
  // associated cached tensors.
  if (rparams->smem_persistent_buffers.empty()) {
    return {};
  }
  const auto& persistent_buffers =
      scheduler_utils::persistentBuffers(fusion).persistent_buffers;
  auto isSharedMemoryPersistent = [&rparams](const TensorView* lookup_tv) {
    return std::any_of(
        rparams->smem_persistent_buffers.begin(),
        rparams->smem_persistent_buffers.end(),
        [lookup_tv](const auto* tv) {
          // can't use `tv->sameAs(lookup_tv)` since the saved tvs in
          // smem_persistent_buffers are from a cloned fusion.
          return tv->name() == lookup_tv->name();
        });
  };
  for (auto tv : persistent_buffers) {
    // Persistent buffers are categorized into two types:
    // (1) Cached input tensors.
    //     For these, [smem_persistent_buffers] holds the original input
    //     tensors, not the cached.
    // (2) Intermediate tensors: Other tensors used throughout computation.

    // If a buffer is absent from [smem_persistent_buffers], it may be a
    // cached input. In such cases, verify if the original input tensor is
    // stored in [smem_persistent_buffers]. So, we may need to call
    // isSharedMemoryPersistent() twice, one for the buffer iteself and the
    // other for the buffer's input tensor if the buffer is a cached input
    // and it is not in [smem_persistent_buffers].
    bool is_cached_input = false;
    bool use_smem = isSharedMemoryPersistent(tv);
    if (!use_smem &&
        std::find(cached_inputs.begin(), cached_inputs.end(), tv) !=
            cached_inputs.end()) {
      auto input_tv = ir_utils::producerTvsOf(tv).at(0);
      use_smem = isSharedMemoryPersistent(input_tv);
      is_cached_input = true;
    }
    if (use_smem) {
      tv->setMemoryType(MemoryType::Shared);
      // When loading from global memory (gmem), use CpAsync with a short data
      // path of gmem -> smem to reduce temporary register usage. Otherwise, the
      // data path from gmem to shared memory (smem) follows this sequence: gmem
      // -> L1 cache -> register -> smem.
      // Only supported after device 8.0 and requires vectorized load.
      int hw_major = at::cuda::getCurrentDeviceProperties()->major;
      if (rparams->vectorize_inner_reduction && is_cached_input &&
          hw_major >= 8) {
        tv->definition()->as<LoadStoreOp>()->setOpType(
            LoadStoreOpType::CpAsync);
        tv->definition()->as<LoadStoreOp>()->setCacheOp(CacheOp::Unspecified);
      }
      // T4_s_float = CpAsync(T0_g_float)
      // do a register cache for all the uses of this smem tv
      auto cached_tv = tv->cacheAfter();
      const auto& consumers = ir_utils::consumerTvsOf(cached_tv);
      smem_consumers.push_back(cached_tv);
      for (auto i = 1; i < (int)consumers.size(); i++) {
        auto consumer = consumers.at(i);
        // recompute cached_tv for each consumer, so it is no longer persistent
        // similar to project to inputs, here we are projecting to the shared
        // memory buffer.
        auto cached_tv_replicate = RecomputeTv::recompute(cached_tv, {tv});
        ir_utils::replaceValInExprInputs(
            consumer->definition(), cached_tv, cached_tv_replicate);
        smem_consumers.push_back(cached_tv_replicate);
      }
    }
  }
  return smem_consumers;
}

// common prepare for all persistent schedulers
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams* rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<TensorView*>& smem_consumers,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs) {
  // Project the persistent buffers to the inputs. Inputs will be cached in a
  // later step, this will move them to be in a register buffer as expected.
  // dummy outputs are helper tensors to make sure persistent buffer projection
  // does not create trouble for transform propagation.
  dummy_outputs = reduction_scheduler_utils::projectPersistentBuffers(
      fusion, rparams->project_persistent_buffers);

  // Cache tensors before grabbing any references to reductions as cache_before
  // can invalidate the references since when applied to a reduction tensor view
  // the new tensor view contains the reduction and original doesn't.
  bool unroll = rparams->isUnrolled();
  // Cache inputs even if not unrolled, as otherwise we may not create a
  // persistent buffer if that persistent buffer would be the input.
  cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, unroll);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);
  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  // move persistent buffer marked in [smem_persistent_buffers] from register to
  // smem
  smem_consumers = movePersistentBufferToSmem(fusion, rparams, cached_inputs);

  reduction_tvs = scheduler_utils::getReductionTvs(fusion);
}

TensorView* scheduleReductionGeneral(
    Fusion* fusion,
    const ReductionParams* rparams,
    std::vector<TensorView*>& reduction_tvs,
    SchedulerType scheduler_type) {
  NVF_ERROR(!reduction_tvs.empty());
  // Registry assumes the reference tv is the first reduction_tv, if this
  // changes registry needs to change.
  auto reduction_tv = reduction_tvs[0];

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion, ca_map);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    reduction_tv->reorder(
        scheduler_utils::domainReorderAsLogicalMap(reduction_tv));
  }

  if (scheduler_type == SchedulerType::OuterPersistent &&
      rparams->cross_grid_inner_reduction && reduction_tvs.size() > 1) {
    groupReductions(reduction_tvs, false);
  }

  auto dim_analysis = scheduler_utils::canonicalDimReduction(
      fusion, reduction_tv, rparams->fastest_dim && rparams->schedule_3D);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  NVF_ERROR(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    NVF_ERROR(
        rparams->fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  return reduction_scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);
}

// fusion is the input IR that will be modified by this function
void schedulePersistentKernel(
    Fusion* fusion,
    const ReductionParams* rparams,
    SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE("schedulePersistentKernel");

  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs,
      smem_consumers;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
      smem_consumers,
      cached_outputs);

  TensorView* reference_tv =
      scheduleReductionGeneral(fusion, rparams, reduction_tvs, scheduler_type);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  NVF_ERROR(
      reference_tv != nullptr && reduction_tvs[0] != nullptr,
      "Need these two tensor views to finish the scheduling.");

  scheduler_utils::moveNonConcretizedBroadcastInnermost(fusion, {reference_tv});

  for (auto output : dummy_outputs) {
    fusion->addOutput(output);
  }

  const bool unroll = rparams->isUnrolled();
  const bool vectorize =
      rparams->vectorize_inner_reduction || rparams->vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams->persistent_kernel &&
      rparams->cross_grid_inner_reduction && !rparams->fastest_dim;
  reduction_scheduler_utils::multiReductionInliner(
      fusion,
      reduction_tvs[0],
      reference_tv,
      unroll,
      vectorize,
      is_outer_grid_persistence,
      reduction_tvs,
      cached_inputs,
      cached_outputs,
      smem_consumers,
      dummy_outputs);

  if (rparams->compute_persistent_buffer_with_first_consumer) {
    NVF_ERROR(
        rparams->persistent_kernel,
        "computeWith should be only used with persistent kernels");
    for (const auto persistent_buffer : cached_inputs) {
      persistent_buffer->computeWith(-1, true);
    }
  }

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  refineCachePolicy(fusion);
}

namespace {

class PersistentBufferResolution : public IterVisitor {
 public:
  static std::vector<TensorView*> getResolutionPointsOf(
      TensorView* persistent_buffer) {
    PersistentBufferResolution resolution(persistent_buffer);

    NVF_ERROR(
        !resolution.resolution_points_.empty(),
        "Could not resolve persistent buffer: ",
        persistent_buffer->toString());

    return resolution.resolution_points_;
  }

  PersistentBufferResolution() = delete;

 private:
  PersistentBufferResolution(TensorView* persistent_buffer)
      : persistent_buffer_(persistent_buffer),
        exact_graph_(
            IdModel(persistent_buffer->fusion(), /*build_graphs=*/false)
                .buildExactGraph()) {
    traverse(persistent_buffer->fusion());
  }

 private:
  void dispatch(Val* val) final {
    if (!val->isA<TensorView>()) {
      return;
    }
    auto tv = val->as<TensorView>();
    if (tv == persistent_buffer_) {
      persistent_buffer_hit_ = true;
      return;
    }

    if (!persistent_buffer_hit_) {
      return;
    }

    if (!tv->hasReduction()) {
      return;
    }

    if (std::any_of(
            resolution_points_.begin(),
            resolution_points_.end(),
            [&tv](TensorView* resolution_point) {
              return DependencyCheck::isDependencyOf(resolution_point, tv);
            })) {
      // If already resolved, don't start a new reduction path.
      return;
    }

    if (!DependencyCheck::isDependencyOf(persistent_buffer_, tv)) {
      // Not a dependent reduction
      return;
    }

    auto resolution_tvs = getResolutionTvs(tv);
    resolution_points_.insert(
        resolution_points_.end(), resolution_tvs.begin(), resolution_tvs.end());
  }

  // Traverse from consumers of the persistent tensor and find if it
  // reaches at a dependent tensor of the reduction tensor.
  std::vector<TensorView*> getResolutionTvs(TensorView* reduction_tv) {
    // When traversing from the persistent tensor, it should not be
    // necessary to traverse any of the tensors between the persistent
    // tensor and reduction tensor since the resolution point must be on
    // the other paths.
    const auto reduction_producers = DependencyCheck::getAllValsBetween(
        {persistent_buffer_}, {reduction_tv});
    const std::unordered_set<Val*> reduction_producer_set(
        reduction_producers.begin(), reduction_producers.end());

    // Resolution points must be a dependent tensor of the reduction tensor
    const std::unordered_set<Val*> reduction_dep_tvs =
        DependencyCheck::getAllDependentVals({reduction_tv});

    // Not all reduction is for normalization. There can be no val
    // after this TV, e.g., a Welford output that is also a segment
    // output (can happen due to segmentation)
    if (reduction_dep_tvs.empty()) {
      return {};
    }

    // The resolution tensor must have iter domains that are reachable
    // from the persistent iter domains
    ValGroups persistent_ids;
    for (auto id : reduction_tv->getLogicalDomain()) {
      if (id->isReduction()) {
        persistent_ids.pushBack(exact_graph_.toGroup(id));
      }
    }

    std::deque<TensorView*> tvs_to_visit;
    std::unordered_set<TensorView*> visited_tvs;
    std::vector<TensorView*> resolution_tvs;

    // Traversing from consumers of persistent tensor
    for (auto tv : ir_utils::consumerTvsOf(persistent_buffer_)) {
      if (!reduction_producer_set.count(tv)) {
        tvs_to_visit.push_back(tv);
      }
    }

    // Check if a tensor should be visited. It should not be visited
    // if any of the following conditions is true:
    //
    // - It is the persistent buffer. The traversal starts from the
    //   consumers of the persistent buffer. Since it should not need
    //   to visit the producers of the persistent buffer, it should
    //   not need to visit the persistent buffer itself.
    // - It's already visited
    // - It's between the persistent buffer and reduction tensor. The
    //   persistent buffer should have multiple consumers, and one of
    //   them leads to the reduction tensor. The resolution point
    //   should be reachable by traversing the other consumers.
    // - It has no logical ID that is reachable from the
    //   persistent IDs. That means the tensor has nothing to do with
    //   the persistent IDs.
    auto should_visit = [&](TensorView* tv) -> bool {
      if (tv == persistent_buffer_ || visited_tvs.count(tv) != 0 ||
          reduction_producer_set.count(tv) != 0) {
        return false;
      }

      // Check if any of the logical IDs are reachable from the
      // persistent IDs. If not, the tensor should have nothing to do
      // with the persistence of the persistent tensor
      const auto& producer_logical_ids =
          exact_graph_.toGroups(tv->getLogicalDomain());
      auto reachable_ids = ValGraphBFS::getReachableValsFrom(
          exact_graph_, persistent_ids, producer_logical_ids);

      return !reachable_ids.empty();
    };

    while (!tvs_to_visit.empty()) {
      auto tv = tvs_to_visit.front();
      tvs_to_visit.pop_front();

      if (reduction_dep_tvs.count(tv)) {
        resolution_tvs.emplace_back(tv);
        // Do not further traverse beyond this tv
        continue;
      }

      // Further traversal to producers
      for (auto producer : ir_utils::producerTvsOf(tv)) {
        if (!should_visit(producer)) {
          continue;
        }
        tvs_to_visit.emplace_back(producer);
      }

      // Further traversal to consumers
      for (auto consumer : ir_utils::consumerTvsOf(tv)) {
        if (!should_visit(consumer)) {
          continue;
        }
        tvs_to_visit.emplace_back(consumer);
      }

      visited_tvs.emplace(tv);
    }

    return resolution_tvs;
  }

 private:
  TensorView* persistent_buffer_ = nullptr;
  ValGraph exact_graph_;

  // Don't do processing until we see the buffer we're looking for
  bool persistent_buffer_hit_ = false;

  // Tracks where the persistent buffer (key) is resolved (values)
  std::vector<TensorView*> resolution_points_;
};

} // namespace

std::vector<TensorView*> getResolutionPointsOf(TensorView* persistent_buffer) {
  return PersistentBufferResolution::getResolutionPointsOf(persistent_buffer);
}

} // namespace normalization_scheduler_utils
} // namespace nvfuser
