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
#include <iter_visitor.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/domain_map.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
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
    int64_t persistent_buffer_size_bit,
    int64_t vectorize_factor,
    int64_t bdimy,
    int64_t gdimy) {
  // The extent of the persistent buffer domain
  auto pb_factor =
      getMinPersistentBufferSize(total_reduction_numel, bdimy, gdimy);

  NVF_ERROR(pb_factor > 0);

  const auto available_reg_count = getAvailableRegisterCount(pb_factor);

  auto per_thread_persistent_buffer_size_bit =
      ceilDiv(ceilDiv(persistent_buffer_size_bit, bdimy), gdimy) *
      vectorize_factor;

  auto persistent_buffer_reg_count =
      ceilDiv(per_thread_persistent_buffer_size_bit, sizeof(int) * 8);

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
    const int64_t persistent_buffer_size_bit) {
  auto last_block_pb =
      total_reduction_numel % (persistent_buffer_size_bit * bdimy) / bdimy;
  return ((double)last_block_pb) / (double)persistent_buffer_size_bit;
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
    const int64_t persistent_buffer_size_bit,
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
             persistent_buffer_size_bit,
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
    int64_t persistent_buffer_size_bit) {
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
            persistent_buffer_size_bit,
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
          persistent_buffer_size_bit,
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
  for (const auto i : arange(reference_tv->nDims())) {
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
  for (auto i : arange(1, inner_reduction_tvs.size())) {
    auto tv = inner_reduction_tvs[i];
    for (const auto i : arange(tv->nDims())) {
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
    for (const auto i : arange(tv->nDims())) {
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
            dataTypeSizeByte(buffer->getDataType().value(),
                             runtime_info.getIndexType());
    partial_reduction_buffer_size += buffer_size;
  }
  return partial_reduction_buffer_size;
}

std::vector<TensorView*> getOuterBroadcastTvs(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs) {
  // set reference broadcast mask using the first inner reduction tv
  std::vector<bool> ref_broadcast_mask;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      const auto& logical = tv->getLogicalDomain();
      ref_broadcast_mask.reserve(logical.size());
      for (const auto i : arange(logical.size())) {
        ref_broadcast_mask.push_back(!logical.at(i)->isReduction());
      }
      break;
    }
  }
  NVF_ERROR(!ref_broadcast_mask.empty(), "ref_broadcast_mask is empty!");

  // find the broadcast tensor whose broadcast mask is same to the reference
  std::vector<TensorView*> outer_broadcast_tvs;
  for (auto tv : fusion->allTvs()) {
    if (std::any_of(
            tv->getLoopDomain().begin(),
            tv->getLoopDomain().end(),
            [](IterDomain* id) { return id->isBroadcast(); })) {
      if (auto bcast = dynamic_cast<BroadcastOp*>(tv->definition())) {
        if (bcast->getBroadcastDimFlags() == ref_broadcast_mask) {
          outer_broadcast_tvs.emplace_back(tv);
        }
      }
    }
  }
  return outer_broadcast_tvs;
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
      "Tried to schedule a fusion with no tensor inputs, currently not "
      "supported.");
}

namespace {
// For inner persistent kernel, shared memory is allocated as:
// ceilDiv(N/vect, batch) * vect * batch. The required shared memory size is
// larger than buffer size when split is not divisible. The difference is
// counted as roundup overhead. This function estimates the maximum possible
// shared memory size due to this round up.
int64_t roundUpSharedMemory(
    int64_t tv_buffer_size_bit,
    int64_t data_type_size_bit) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t max_threads_per_block = (int64_t)dev_prop->maxThreadsPerBlock;
  int64_t max_smem_bit = 0;
  int64_t max_vectorize_factor =
      getMaxVectorizationSizeInBit() / data_type_size_bit;
  int64_t dim_size = tv_buffer_size_bit / data_type_size_bit;
  // Check all possible combinations of vectorization factor, batch size and
  // threads per block
  for (int64_t vectorize_factor = 1; vectorize_factor <= max_vectorize_factor;
       vectorize_factor *= 2) {
    // heuristic only uses divisible vectorization factor
    if (dim_size % vectorize_factor != 0) {
      continue;
    }
    int64_t after_vect = dim_size / vectorize_factor;
    // For shared memory persistence, heuristic always uses maximum threads
    // per block
    int64_t threads_per_block = max_threads_per_block;
    int64_t persistent_batch = ceilDiv(after_vect, threads_per_block);
    max_smem_bit = std::max(
        max_smem_bit,
        persistent_batch * vectorize_factor * threads_per_block *
            data_type_size_bit);
  }
  return max_smem_bit;
}
int64_t sharedMemoryRoundUpOverheadBit(
    SchedulerRuntimeInfo& runtime_info,
    const scheduler_utils::PersistentBufferInfo& persistent_buffer_info,
    const bool project_to_inputs) {
  auto buffers = project_to_inputs
      ? persistent_buffer_info.projectable_buffer_inputs
      : persistent_buffer_info.persistent_buffers;
  int64_t total_smem_overhead_bit = 0;
  for (auto buffer : buffers) {
    // Buffer size derived from shape and dtype of the persistent tensor
    int64_t logical_buffer_size_bit =
        scheduler_utils::getPersistentBufferSizeBitOfTensor(
            buffer, runtime_info, persistent_buffer_info);
    // Required shared memory size if store that tensor in shared memory
    int64_t buffer_size_smem = roundUpSharedMemory(
        logical_buffer_size_bit,
        dataTypeSizeBit(buffer->getDataType().value()));
    // The difference is counted as roundup overhead
    total_smem_overhead_bit += (buffer_size_smem - logical_buffer_size_bit);
  }
  return total_smem_overhead_bit;
}
} // namespace

int64_t getMaxRegOrSharedMemorySizeBitForPersistentBuffer(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*>& reduction_tvs,
    const scheduler_utils::PersistentBufferInfo& persistent_buffer_info,
    const bool can_use_smem_persistent,
    const bool project_to_inputs) {
  // Init to register file size, which is half of the full register file size
  int64_t available_persistent_buffer_size_bit =
      scheduler_utils::register_file_size_bit;
  // shared memory persistent is not implemented for 3D inner reduction
  if (!can_use_smem_persistent) {
    return available_persistent_buffer_size_bit;
  }
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t smem_overhead_bit =
      scheduler_utils::getReductionSmemWorkspaceBit(fusion, reduction_tvs);

  smem_overhead_bit += sharedMemoryRoundUpOverheadBit(
      runtime_info, persistent_buffer_info, project_to_inputs);

  int64_t available_shared_memory_size_bit =
      (int64_t)dev_prop->sharedMemPerBlockOptin * 8 - smem_overhead_bit;

  available_persistent_buffer_size_bit = std::max(
      available_persistent_buffer_size_bit, available_shared_memory_size_bit);
  return available_persistent_buffer_size_bit;
}

// Returns BufferProjectionStrategy based on buffer size, hardware, and fusion
// ops. ProjectToInputs: recompute buffer from cached inputs to save
// register/shared memory usage. NoProjectToAvoidRecompute: don't recompute to
// reduce computation cost. NoProjectOtherReasons: don't recompute due to other
// reasons.
BufferProjectionStrategy isProjectBufferToInputs(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*>& reduction_tvs,
    const scheduler_utils::PersistentBufferInfo& persistent_buffer_info,
    const scheduler_utils::PersistentBufferSizeReturn&
        persistent_buffer_size_info,
    const SchedulerType scheduler_type,
    const bool can_use_smem_persistent,
    const bool check_projected_buffer_size) {
  // don't project if there are view ops and no buffer can be projected
  bool can_project = !persistent_buffer_info.has_view_ops &&
      persistent_buffer_size_info.projected_persistent_buffer_size_bit > 0;
  if (!can_project) {
    return BufferProjectionStrategy::NoProjectOtherReasons;
  }

  // Enusre project to inputs can save persistent buffer size,
  // unless it's innerOuter with outer broadcast where project to inputs reduces
  // gmem access.
  if (check_projected_buffer_size &&
      persistent_buffer_size_info.projected_persistent_buffer_size_bit >=
          persistent_buffer_size_info.persistent_buffer_size_bit) {
    return BufferProjectionStrategy::NoProjectOtherReasons;
  }

  // must project to inputs otherwise don't have enough register or shared
  // memory to store the buffers. Even after projecting, may still not have
  // enough register or shared memory, then canScheduleRunTime will return
  // false. For InnerOuterPersistent, both register and shared memory are used
  // and will be handled in getPersistentBufferStorageParams.
  if (scheduler_type != SchedulerType::InnerOuterPersistent) {
    int64_t max_available_buffer_bit =
        getMaxRegOrSharedMemorySizeBitForPersistentBuffer(
            fusion,
            runtime_info,
            reduction_tvs,
            persistent_buffer_info,
            can_use_smem_persistent,
            false);
    if (max_available_buffer_bit <
        persistent_buffer_size_info.persistent_buffer_size_bit) {
      return BufferProjectionStrategy::ProjectToInputs;
    }
  }

  // don't project if recompute requires rng op
  if (persistent_buffer_info.projection_with_rng_op) {
    return BufferProjectionStrategy::NoProjectOtherReasons;
  }

  // free to project if no exp op
  if (!persistent_buffer_info.projection_with_exp_op) {
    return BufferProjectionStrategy::ProjectToInputs;
  }

  // Recompute from inputs reduces regisger usage which may lead to higher
  // occupancy and better performance. However, it also increases computation
  // cost which may lead to lower performance, especially when the device has a
  // high bandwidth to flops ratio where the kernel may change from memory bound
  // to compute bound. Here, we use two empirical values derived from tests of
  // softmax on H100 and B100/200. B100/200 are considered as devices with high
  // bandwidth to flops ratio.
  if (scheduler_type == SchedulerType::InnerPersistent) {
    bool is_high_bandwidth_flops_ratio =
        scheduler_utils::isHighBandwidthFlopsRatio();
    int64_t buffer_per_block_bit =
        is_high_bandwidth_flops_ratio ? 24 * 4 * 1024 * 8 : 6 * 4 * 1024 * 8;
    if (persistent_buffer_size_info.persistent_buffer_size_bit <=
        buffer_per_block_bit) {
      return BufferProjectionStrategy::NoProjectToAvoidRecompute;
    }
  }

  return BufferProjectionStrategy::ProjectToInputs;
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

  vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      vec_break_point.get(),
      std::getenv("USE_MAIN") || !properties.fastest_dim_reduction
          ? 128
          : getMaxVectorizationSizeInBit());

  // Used by inner persistent and outer persistent kernels.
  // These two heurisics were tuned to work with a maximum vectorization factor
  // of 8. This change is to allow the use of vectorization of 8 when there are
  // both bfloat16 and float tensors in the fusion inputs, e.g. rms norm may
  // have input tensor as bfloat16 and weight as float.
  vectorize_factor = std::min(vectorize_factor, (int64_t)8);

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
  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSizeBit(
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
  auto project_strategy = isProjectBufferToInputs(
      fusion,
      runtime_info,
      reduction_tvs,
      persistent_buffer_info,
      persistent_buffer_size_info,
      scheduler_type,
      can_use_smem_persistent);
  bool project_persistent_buffers =
      (project_strategy == BufferProjectionStrategy::ProjectToInputs);
  bool disable_project_to_avoid_recompute =
      (project_strategy == BufferProjectionStrategy::NoProjectToAvoidRecompute);
  int64_t max_persistent_buffer_size_bit = project_persistent_buffers
      ? persistent_buffer_size_info.projected_persistent_buffer_size_bit
      : persistent_buffer_size_info.persistent_buffer_size_bit;

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
  int64_t max_dtype_size_bit = 1;
  int64_t n_tensor_inputs = 0;

  auto& unrollable_inputs_outputs = unrollable_inputs_outputs_entry.get();
  for (auto tv : unrollable_inputs_outputs) {
    if (!tv->isFusionInput()) {
      continue;
    }
    max_dtype_size_bit = std::max(
        max_dtype_size_bit,
        dataTypeSizeBit(
            tv->getDataType().value(), runtime_info.getIndexType()));
    n_tensor_inputs++;
  }
  // To prevent division by zero, ensure that n_tensor_inputs is not equal to
  // zero.
  n_tensor_inputs = std::max(n_tensor_inputs, (int64_t)1);

  // Exp op typically used in softmax is expensive and needs more registers.
  bool has_exp_op = false;
  bool has_rng_op = false;

  // Could save fusion->exprs() instead of doing this, but allTvs is already
  // cached in fusion so using that for now.
  for (auto tv : fusion->allTvs()) {
    if (tv->definition() == nullptr) {
      continue;
    }
    if (tv->definition()->isA<UnaryOp>() &&
        tv->definition()->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Exp) {
      has_exp_op = true;
    }
    if (tv->definition()->isA<RNGOp>()) {
      has_rng_op = true;
    }
  }
  auto buffers = project_persistent_buffers
      ? persistent_buffer_info.projectable_buffer_inputs
      : persistent_buffer_info.persistent_buffers;

  // Add buffers that are not projectable.
  if (project_persistent_buffers) {
    std::unordered_set<TensorView*> projectable_set(
        persistent_buffer_info.projectable_persistent_buffers.begin(),
        persistent_buffer_info.projectable_persistent_buffers.end());
    for (auto tv : persistent_buffer_info.persistent_buffers) {
      if (projectable_set.find(tv) == projectable_set.end()) {
        buffers.push_back(tv);
      }
    }
  }

  // Return collected properties to get heuristics.
  return PersistentKernelProperties{
      .inner_most_dimension_numel = properties.inner_most_dimension_numel,
      .total_reduction_numel = properties.total_reduction_numel,
      .total_iteration_numel = properties.total_iteration_numel,
      .max_persistent_buffer_size_bit = max_persistent_buffer_size_bit,
      .n_tensor_inputs = n_tensor_inputs,
      .max_dtype_size_bit = max_dtype_size_bit,
      .vectorize_factor = vectorize_factor,
      .project_persistent_buffers = project_persistent_buffers,
      .index_type = runtime_info.getIndexType(),
      .has_exp_op = has_exp_op,
      .has_rng_op = has_rng_op,
      .disable_project_to_avoid_recompute = disable_project_to_avoid_recompute,
      .persistent_buffers = buffers};
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
    for (const auto it : arange(1, rtvs.size())) {
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

  // Reject when output IDs are not covered by reference tv. Assuming reduction
  // scheduler simply uses reduction_tvs[0] as the reference, if that changes,
  // this needs to be changed. see issue
  // https://github.com/NVIDIA/Fuser/issues/3811
  scheduler_tools::DomainMap domain_map(fusion);
  if (!domain_map.isValidReference(reduction_tvs[0], /*check_inputs=*/false)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type,
        "Output contains ID that's not scheduled by reference tv.");
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
  auto supportCpAsync = [rparams](const TensorView* smem_tv) {
    // Only supported after device 8.0
    int hw_major = at::cuda::getCurrentDeviceProperties()->major;
    if (hw_major < 8) {
      return false;
    }
    // requires 4, 8, or 16 loading bytes.
    int vect_factor = rparams->vectorize_inner_reduction
        ? (int)rparams->unroll_factor_inner_reduction
        : 1;
    size_t loading_size =
        dataTypeSizeByte(smem_tv->getDataType().value()) * vect_factor;
    bool is_supported_bytes =
        (loading_size == 4 || loading_size == 8 || loading_size == 16);
    return is_supported_bytes;
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
    // For warp specialized, may direct load non-circular buffered tv to regs
    // Non-circular buffered tvs are those have broadcast dimensions that mapped
    // with reduction dimensions.
    if (rparams->tma_warp_specialized &&
        rparams->is_non_circular_buffer_gmem_to_regs) {
      const auto& outer_broadcast_tvs = getOuterBroadcastTvs(
          fusion, scheduler_utils::getReductionTvs(fusion));
      if (std::any_of(
              outer_broadcast_tvs.begin(),
              outer_broadcast_tvs.end(),
              [&tv](TensorView* bcast_tv) {
                return DependencyCheck::isDependencyOf(tv, bcast_tv);
              })) {
        use_smem = false;
      }
    }
    if (use_smem) {
      tv->setMemoryType(MemoryType::Shared);
      // Use 1D TMA, CpAsyncBulk
      if (rparams->tma_warp_specialized && is_cached_input) {
        tv->definition()->as<LoadStoreOp>()->setOpType(
            LoadStoreOpType::CpAsyncBulk);
      } else if (supportCpAsync(tv) && is_cached_input) {
        // When loading from global memory (gmem), use CpAsync with a short data
        // path of gmem -> smem to reduce temporary register usage. Otherwise,
        // the data path from gmem to shared memory (smem) follows this
        // sequence: gmem -> L1 cache -> register -> smem.
        tv->definition()->as<LoadStoreOp>()->setOpType(
            LoadStoreOpType::CpAsync);
        tv->definition()->as<LoadStoreOp>()->setCacheOp(CacheOp::Unspecified);
      }
      // do a register cache for all the uses of this smem tv.
      // The load from smem to register cache will then be vectorized to avoid
      // bank conflicts. The determination of bank conflicts is made per
      // transaction, with 16 bytes vectorized load, each warp needs 4
      // transactions (32 threads * 16 bytes per thread / 128 bytes per
      // transaction). In each transaction, different banks are visited, e.g.
      // transaction-1, threads 0-7 visit banks 0-31
      auto cached_tv = tv->cacheAfter();
      smem_consumers.push_back(cached_tv);
      // At this point, if cached_tv has multiple uses,  it becomes the
      // persistent buffer instead of smem tv, due to the way the persistent
      // buffer selector works. To make smem tv remain as the persistent buffer,
      // all of the uses must be privatized. However, for tma warp specialized
      // case, we don't need to privatize the cached_tv, so the smem tv is only
      // consumed by its register cache. It can be used to issue the next TMA
      // load right after the copy from shared memory to register cache.
      // Otherwise, it needs to wait all the computations to finish before
      // issuing the next TMA.
      if (!rparams->tma_warp_specialized ||
          !rparams->is_circular_buffer_regs_cached) {
        const auto& consumers = ir_utils::consumerTvsOf(cached_tv);
        for (auto i = 1; i < (int)consumers.size(); i++) {
          auto consumer = consumers.at(i);
          // recompute cached_tv for each consumer, so it is no longer
          // persistent similar to project to inputs, here we are projecting to
          // the shared memory buffer.
          auto cached_tv_replicate = RecomputeTv::recompute(cached_tv, {tv});
          ir_utils::replaceValInExprInputs(
              consumer->definition(), cached_tv, cached_tv_replicate);
          smem_consumers.push_back(cached_tv_replicate);
        }
      }
    }
  }
  return smem_consumers;
}

namespace {
void recomputeNonPersistentUnmappbleTvs(
    const scheduler_utils::PersistentBufferInfo& persistent_info) {
  for (auto non_persistent_buffer : persistent_info.non_persistent_buffers) {
    // If there's only one use, it must be cached
    if (non_persistent_buffer->uses().size() == 1) {
      auto caching_load = non_persistent_buffer->uses().at(0);
      NVF_ERROR(caching_load->isA<LoadStoreOp>());
      non_persistent_buffer =
          caching_load->as<LoadStoreOp>()->out()->as<TensorView>();
    }
    NVF_ERROR(non_persistent_buffer->uses().size() > 1);
    bool is_first = true;
    for (const auto& use : non_persistent_buffer->uses()) {
      // No need to clone the tv for the first use
      if (is_first) {
        is_first = false;
        continue;
      } else {
        auto recomputed_tv = RecomputeTv::recompute(non_persistent_buffer);
        ir_utils::replaceValInExprInputs(
            use, non_persistent_buffer, recomputed_tv);
      }
    }
  }
}
} // namespace

// common prepare for all persistent schedulers
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams* rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<TensorView*>& smem_consumers,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs) {
  const scheduler_utils::PersistentBufferInfo persistent_info =
      scheduler_utils::persistentBuffers(fusion);

  // Project the persistent buffers to the inputs. Inputs will be cached in a
  // later step, this will move them to be in a register buffer as expected.
  // dummy outputs are helper tensors to make sure persistent buffer projection
  // does not create trouble for transform propagation.
  dummy_outputs = reduction_scheduler_utils::projectPersistentBuffers(
      fusion, persistent_info, rparams->project_persistent_buffers);

  // Cache tensors before grabbing any references to reductions as cache_before
  // can invalidate the references since when applied to a reduction tensor view
  // the new tensor view contains the reduction and original doesn't.
  bool unroll = rparams->isUnrolled();
  // Cache inputs even if not unrolled, as otherwise we may not create a
  // persistent buffer if that persistent buffer would be the input.
  cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  recomputeNonPersistentUnmappbleTvs(persistent_info);

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
        "If all dims are reduction, should be sending it to fastest dim "
        "scheduler.");
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

  const bool is_unroll_or_vectorization = rparams->isUnrolled();
  const bool is_vectorize =
      rparams->vectorize_inner_reduction || rparams->vectorize_iter_dom;
  const bool use_grouped_reduction = rparams->persistent_kernel &&
      rparams->cross_grid_inner_reduction && !rparams->fastest_dim;

  // Propagate transformations before we rfactor the other reductions
  auto reduction_tv = reduction_tvs.at(0);
  reduction_scheduler_utils::propagateTransformation(reference_tv);
  // If reduction_tv is rfactored, rfactor all reductions.
  if (reference_tv != reduction_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, reduction_tv, reduction_tvs);
  }

  const auto& unroll_vectorizable_cached_tvs =
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          reference_tv, is_vectorize, cached_inputs, cached_outputs);
  reduction_scheduler_utils::propagateParallelization(
      reduction_tv,
      reference_tv,
      is_unroll_or_vectorization,
      use_grouped_reduction,
      reduction_tvs,
      unroll_vectorizable_cached_tvs);

  // Needs special handling of vectorized loading from shared memory due to
  // potential different data types of inputs and shared memory tensor.
  if (is_vectorize) {
    int64_t vectorization_factor = rparams->unroll_factor_inner_reduction;
    reduction_scheduler_utils::sharedMemoryConsumerVectorization(
        smem_consumers, vectorization_factor);
  }

  // Remove dummy outputs as they can inadvertently affect CA positions
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }

  // Inline the schedule
  inlineMost();

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
      TensorView* persistent_buffer,
      IdModel& id_model) {
    PersistentBufferResolution resolution(persistent_buffer, id_model);
    return resolution.resolution_points_;
  }

  PersistentBufferResolution() = delete;

 private:
  PersistentBufferResolution(TensorView* persistent_buffer, IdModel& id_model)
      : persistent_buffer_(persistent_buffer),
        exact_graph_(id_model.maybeBuildGraph(IdMappingMode::EXACT)) {
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
      auto reachable_ids = getReachableValsFrom<ValGraphBFS>(
          persistent_ids.vector(),
          producer_logical_ids.vector(),
          Direction::Undefined,
          exact_graph_);

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

std::vector<TensorView*> getResolutionPointsOf(
    TensorView* persistent_buffer,
    IdModel& id_model) {
  return PersistentBufferResolution::getResolutionPointsOf(
      persistent_buffer, id_model);
}

int64_t getInnerPersistentMaxBatchSize(bool is_high_bandwidth_flops_ratio) {
  return is_high_bandwidth_flops_ratio ? 12l : 10l;
}

bool isCacheableUnmappableTv(
    TensorView* unmappable_tv,
    const std::vector<TensorView*>& reduction_tvs,
    const ValGraph& almost_exact_graph) {
  // To make an unmmapble tensor persistent, we need to make sure it
  // can be parallelized in the same way as the following reduction
  // and residual paths. While the unmmapble tensor is transformed in
  // the same way, since it is not inlineable, the effect of loop
  // promotion by broadcast inling is not propagated to the unmappable
  // tensor. For example, in the following fusion, both t2 and t3 are the
  // unmappable tensors but t2 is problematic.
  //
  // t0: [i0]
  // t1: [i1, i2]
  //
  // t2 = t0 // caching
  // t3 = t1 // caching
  // t4 = broadcast(t2, {false, true})
  // t5 = t4 + t3
  // t6 = sum(t5, {0, 1})
  // t7 = broadcast(t6, {true, true})
  // t8 = broadcast(t2, {false, true})
  // t9 = t8 + t3
  // t10 = t7 + t9
  //
  // The immediate consumer of t4 has a broadcast ID, which will be
  // inlined into t5, making it effectively have the same extent as
  // the corresponding non-broadcast ID of t5. Moreover, our
  // schedulers are likely to merge all reductions IDs before applying
  // parallelization. What this means is that the parallelized loop
  // IDs of t4 will not be mapped with any of the loop IDs of t2, and
  // thus a synchronization will be required. This may not a problem
  // when t2 is cached on the shared memory, however, otherwise, it
  // will result in a sync error when the fusion is lowered.
  //
  // It may be possible to avoid the issue by changing the scheduling
  // and parallelization of these tensors, but a much simpler
  // workaround here is to just give up caching such tensors. While
  // it may cause some performance regressions, it is expected the
  // impact would be limited since this pattern itself is not common.
  //
  // To find if a given unmappable tensor may result in cases
  // like the above, we need to make sure that it is parallelized in
  // the same way in its all use paths. Since this is an unmappable
  // tensor in a normalization fusion, all we need to check is if it
  // can be parallelized in the same way as the following reduction
  // tensors.

  // reduction_tvs are all reduction tensors in the fusion. Those
  // tensors that do not appear immediately after unmappable_tv can be
  // ignored since they
  std::vector<TensorView*> immediate_reduction_tvs;
  for (const auto& reduction_tv : reduction_tvs) {
    auto all_vals = DependencyCheck::getAllValsBetween(
        std::unordered_set<Val*>{unmappable_tv},
        std::vector<Val*>{reduction_tv});
    // If the reduction tv doesn't depend on unmappable tv,
    // all_vals will be empty.
    if (all_vals.empty() ||
        std::any_of(
            reduction_tvs.begin(),
            reduction_tvs.end(),
            [&](const auto& reduction_tv_j) {
              return reduction_tv_j != reduction_tv &&
                  std::find(all_vals.begin(), all_vals.end(), reduction_tv_j) !=
                  all_vals.end();
            })) {
      continue;
    }
    immediate_reduction_tvs.push_back(reduction_tv);
  }

  NVF_ERROR(!immediate_reduction_tvs.empty());

  // For each (indirect) consumer reduction tensor, make sure the
  // unmappble tensor is consistent with the reduction tensor with
  // respect to the reduction IDs. The reduction IDs are those that
  // are not inlineable, so they won't get the effect of loop
  // promotion if that happens inside the group of inlined tensors.
  for (const auto& reduction_tv : immediate_reduction_tvs) {
    bool missing_reduction_id_found = false;
    bool mapped_reduction_id_found = false;
    for (const auto& reduction_id : reduction_tv->getLogicalDomain()) {
      if (!reduction_id->isReduction()) {
        continue;
      }

      // Here, we only look for a logical ID of the unmappble tensor
      // that is mapped with the reduction ID. If found,
      // parallelization of this reduction ID should be consistently
      // applied to the unmappble tensor as well.
      //
      // TODO: Even if they are not mapped, is it possible that they
      // are still mapped through reshape ops?
      auto it = std::find_if(
          unmappable_tv->getLogicalDomain().begin(),
          unmappable_tv->getLogicalDomain().end(),
          [&](const auto& unmappable_tv_logical_id) {
            return almost_exact_graph.disjointValSets().strictAreMapped(
                reduction_id, unmappable_tv_logical_id);
          });
      if (it == unmappable_tv->getLogicalDomain().end()) {
        missing_reduction_id_found = true;
      } else {
        mapped_reduction_id_found = true;
      }
    }
    if (missing_reduction_id_found && mapped_reduction_id_found) {
      return false;
    }
  }
  return true;
}

} // namespace normalization_scheduler_utils
} // namespace nvfuser
