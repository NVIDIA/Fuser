// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_evaluator.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/registry.h>
#include <utils.h>
#include <array>

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

  TORCH_INTERNAL_ASSERT(block_size % bdimx == 0, "Invalid bdimx: ", bdimx);
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

  TORCH_INTERNAL_ASSERT(pb_factor > 0);

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
    // adjust gdimy (decreased as persitent_buffer is increased)
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
  // the iteration domain is larger, the additional overheaad would
  // likely to outweight the benefit of potentially better block
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
  TORCH_INTERNAL_ASSERT(launch_cfg.isInvalid());
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

int64_t partialReductionBufferSize(
    const std::vector<TensorView*>& outer_reduction_tvs,
    SchedulerRuntimeInfo& runtime_info) {
  int64_t partial_reduction_buffer_size = 0;
  for (auto buffer : outer_reduction_tvs) {
    int64_t buffer_size = -1;
    for (auto id : buffer->getMaybeRFactorDomain()) {
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }
      auto id_size = runtime_info.expressionEvaluator().evaluate(id->extent());
      TORCH_INTERNAL_ASSERT(
          id_size.hasValue(), "Could not infer persistent buffer size.");
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

std::pair<int64_t, int64_t> getInnerOuterPersistentBufferBatches(
    const int64_t inner_dim_numel,
    const int64_t outer_dim_numel,
    const int64_t regs_buffer_size,
    const int64_t smem_buffer_size,
    const int64_t vectorize_factor,
    const int64_t warp_size) {
  // if inner_dim_numel <= 1024, we are doing multiple reductions per block
  // with a constant batch size of 1 if vectorized. See Step 5 of
  // innerOuterPersistentHeuristic. Although batch size is 1, each thread also
  // needs to do serial reduction of [vectorize_factor] elements. However, if
  // vectorize_factor is 1, we can increase batch size to set a minimum serial
  // reduction workload for each thread to take advantage of zero intra-threads
  // communication cost. Here a middle value of 4 is selected without spending
  // time to tune as these un-vectorized small cases should be rare in real
  // world.
  if (inner_dim_numel <= 1024l) {
    const int64_t batch = (vectorize_factor == 1) ? 4l : 1l;
    return std::make_pair(
        batch, ceilDiv(inner_dim_numel, batch * vectorize_factor));
  }
  // Set a minimum workload for each thread to take advantage of low
  // intra-threads communication cost. Tuned for layer_norm backward on A100.
  auto getMinimumBatch = [&]() -> int64_t {
    if (inner_dim_numel >= 3072l) {
      if (outer_dim_numel <= 2048l && inner_dim_numel == 3072l) {
        return 3l;
      } else {
        return 4l;
      }
    } else if (inner_dim_numel >= 2048l) {
      return 2l;
    }
    return 1l;
  };
  //! Each thread can use a maximum of 255 registers, and assume 40 of them are
  //! reserved for indexing and other purposes. So, each thread can use up to
  //! 215 registers for persistent buffer. Calculate number of buffer batches
  //! using these 215 registers. total_buffer_bytes is the total size of
  //! persistent buffers in bytes. reduction_elements is the number of elements
  //! in the reduction domain. vectorization_factor is the vectorization factor
  //! of inputs and outputs.
  auto getMaximumInnerOuterPersistentBufferBatch = [&]() -> int64_t {
    int64_t register_per_batch = ceilDiv(
        regs_buffer_size / inner_dim_numel * vectorize_factor,
        scheduler_utils::bytes_per_register);
    return scheduler_utils::safeDiv(
        scheduler_utils::max_registers_per_thread -
            scheduler_utils::register_overhead,
        register_per_batch);
  };

  const int64_t after_vectorization = inner_dim_numel / vectorize_factor;
  const int64_t threads_per_block_min = std::min(after_vectorization, 128l);
  const int64_t threads_per_block_max = std::min(
      after_vectorization,
      vectorize_factor > 1
          ? scheduler_utils::max_threads_per_block_combined
          : scheduler_utils::max_threads_per_block_combined_unvectorized);
  const int64_t batch_min = getMinimumBatch();
  const int64_t batch_max = getMaximumInnerOuterPersistentBufferBatch();
  int64_t threads_per_block = threads_per_block_min;
  int64_t inner_batch = ceilDiv(after_vectorization, threads_per_block);

  // When shared memory is used to store persistent buffers, hidden size is
  // large. Set threads_per_block to maximum to avoid large batch sizes.
  if (smem_buffer_size > 0) {
    threads_per_block = threads_per_block_max;
    inner_batch = ceilDiv(after_vectorization, threads_per_block);
  } else {
    // Start from the smallest threads_per_block. If the corresponding batch
    // size is larger than batch_max, try double threads per block
    // until the threads_per_block reaches threads_per_block_max or the batch
    // size reaches batch_min.
    while (inner_batch > batch_max &&
           threads_per_block * 2l <= threads_per_block_max &&
           ceilDiv(after_vectorization, threads_per_block * 2l) >= batch_min) {
      threads_per_block *= 2;
      inner_batch = ceilDiv(after_vectorization, threads_per_block);
    }
  }
  return std::make_pair(inner_batch, threads_per_block);
}

int64_t getSharedMemoryOverheadPerBlock(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t max_threads_per_block) {
  const auto& dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t dtype_size = 1;
  for (auto tv : reduction_tvs) {
    dtype_size = std::max(dtype_size, dataTypeSize(tv->getDataType().value()));
  }
  auto hasWelford = [&fusion]() -> bool {
    for (auto expr : fusion->exprs()) {
      if (expr->isA<WelfordOp>()) {
        return true;
      }
    }
    return false;
  };
  int64_t welford_factor = hasWelford() ? 3l : 1l;
  int64_t reduction_broadcast_workspace =
      max_threads_per_block * dtype_size * welford_factor;
  int64_t smem_overhead_per_block =
      (int64_t)dev_prop->reservedSharedMemPerBlock +
      reduction_broadcast_workspace;
  return smem_overhead_per_block;
}

TensorView* getReferenceReductionTv(
    const std::vector<TensorView*>& reduction_tvs) {
  TensorView* first_inner_tv = nullptr;
  TensorView* first_outer_tv = nullptr;
  for (auto tv : reduction_tvs) {
    bool is_inner = scheduler_utils::isFastestDimReduction(tv);

    if (is_inner && !first_inner_tv) {
      first_inner_tv = tv;
    } else if (!is_inner && !first_outer_tv) {
      first_outer_tv = tv;
    }

    if (first_inner_tv && first_outer_tv) {
      return first_inner_tv;
    }
  }

  return reduction_tvs[0];
}

namespace {

// shared memory is configured to specific sizes, e.g. 8, 16, 32, 64, 100,
// 132, 164, 196, 228 KB per SM on H100. Here, smem_config_options is set to
// H100's shared memory configuration. The returned value is the smallest shared
// memory size required to launch the kernel. The driver may use a larger value
// if there is enough shared memory to launch more than one block per SM. It's
// the caller's responsibility to check if the returned value is smaller than
// the device's shared memory size.
static const std::array<int64_t, 9> smem_config_options = {
    8l * 1024l,
    16l * 1024l,
    32l * 1024l,
    64l * 1024l,
    100l * 1024l,
    132l * 1024l,
    164l * 1024l,
    196l * 1024l,
    228l * 1024l};

int64_t getSharedMemoryConfigSize(int64_t request_size) {
  auto it = std::upper_bound(
      smem_config_options.begin(), smem_config_options.end(), request_size);
  return (it != smem_config_options.end()) ? *it : smem_config_options.back();
}

// The roundup is due to the fact that the shared memory buffer is allocated
// as: ceilDiv(ceilDiv(dim_size, vect), threadsPerBlock)
int64_t roundUpSharedMemory(
    TensorView* tv,
    int64_t tv_buffer_size,
    int64_t vectorize_factor,
    int64_t threads_per_block) {
  const int64_t data_type_size = dataTypeSize(tv->getDataType().value());
  const int64_t n_elements = tv_buffer_size / data_type_size;
  const int64_t n_batch =
      ceilDiv(ceilDiv(n_elements, vectorize_factor), threads_per_block);
  return n_batch * vectorize_factor * threads_per_block * data_type_size;
}

bool isDirectlyUsedByBroadcast(TensorView* tv) {
  for (auto consumer : ir_utils::consumerTvsOf(tv)) {
    if (consumer->hasBroadcast()) {
      return true;
    } else if (auto op = dynamic_cast<UnaryOp*>(consumer->definition())) {
      return op->getUnaryOpType() == UnaryOpType::Cast
          ? isDirectlyUsedByBroadcast(consumer)
          : false;
    }
  }
  return false;
}

} // namespace

PersistentBufferStorageParams getPersistentBufferStorageParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t vectorize_factor) {
  PersistentBufferStorageParams buffer_params;

  // Check if the reduction is inner, outer, or combined inner-outer
  bool inner_reduction = false;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction = true;
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  buffer_params.combined_reduction =
      inner_reduction && !outer_reduction_tvs.empty();

  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

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
  // Note that projected buffer size can be zero
  // for layer_norm backward, enable project to input can reuse weight shared
  // among different rows. Although it increased register usage and may lead
  // to register spills, the overall performance is increased. This is a
  // temporary solution, the issue is tracked by
  // https://github.com/csarofeen/pytorch/issues/2525
  buffer_params.project_to_input = ir_utils::getViewOps(fusion).empty() &&
      persistent_buffer_size_info.projected_persistent_buffer_size > 0 &&
      (persistent_buffer_size_info.projected_persistent_buffer_size <
           persistent_buffer_size_info.persistent_buffer_size ||
       buffer_params.combined_reduction);
  auto total_buffer_size = buffer_params.project_to_input
      ? persistent_buffer_size_info.projected_persistent_buffer_size
      : persistent_buffer_size_info.persistent_buffer_size;
  const auto& persistent_buffers = buffer_params.project_to_input
      ? persistent_buffer_info.projectable_buffer_inputs
      : persistent_buffer_info.persistent_buffers;
  if (buffer_params.combined_reduction) {
    // In the case of combined_reduction, the scheduler will create
    // additional tensors in the schedule process to hold the intermediate
    // results of the outer reduction. These tensors are persistent but are not
    // captured in the persistent_buffer_info, since they are not exist at this
    // point.
    const auto intermediate_buffer_size =
        normalization_scheduler_utils::partialReductionBufferSize(
            outer_reduction_tvs, runtime_info);
    total_buffer_size += intermediate_buffer_size;
  }

  // At this point, we use a much larger register file size for the combined
  // case as it is rarely fused with other ops.
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t available_regs = scheduler_utils::register_file_size;
  if (buffer_params.combined_reduction) {
    available_regs = vectorize_factor > 1
        ? scheduler_utils::register_file_size_combined
        : scheduler_utils::register_file_size_combined_unvectorized;
  }
  const int64_t max_threads_per_block = vectorize_factor > 1
      ? scheduler_utils::max_threads_per_block_combined
      : scheduler_utils::max_threads_per_block_combined_unvectorized;
  // Shared memory persistent is only implemented for the inner and combined
  // case.
  buffer_params.smem_overhead = getSharedMemoryOverheadPerBlock(
      fusion, reduction_tvs, max_threads_per_block);
  int64_t available_smem = inner_reduction
      ? (int64_t)dev_prop->sharedMemPerBlockOptin - buffer_params.smem_overhead
      : 0l;

  // Put all the persistent tensors in registers
  buffer_params.regs_buffer_size = total_buffer_size;
  buffer_params.smem_buffer_size = 0;

  //! Move buffers to shared memory following the 3 rules:
  //! (1) Prioritize moving buffers that are directly used by broadcast ops.
  //! (2) Move N buffers until the register buffer size is below the available
  //! limit. (3) Move one more if leads to higher shared memory utilization
  //! ratio.
  if (buffer_params.regs_buffer_size > available_regs) {
    const int64_t n_buffers = (int64_t)persistent_buffers.size();
    int64_t n_broadcast_buffers = 0;
    std::vector<TensorView*> sorted_candidate_tvs;
    sorted_candidate_tvs.reserve(n_buffers);
    for (auto tv : persistent_buffers) {
      if (isDirectlyUsedByBroadcast(tv)) {
        sorted_candidate_tvs.insert(sorted_candidate_tvs.begin(), tv);
        n_broadcast_buffers++;
      } else {
        sorted_candidate_tvs.push_back(tv);
      }
    }
    std::cout << "n_broadcast_buffers= " << n_broadcast_buffers
              << ", n_buffers= " << n_buffers << std::endl;
    // calculate the accumulated buffer size of the first N buffers
    std::vector<int64_t> acc_regs_buffer_sizes(n_buffers + 1, 0);
    std::vector<int64_t> acc_smem_buffer_sizes(n_buffers + 1, 0);
    for (int i = 1; i <= n_buffers; i++) {
      int64_t tv_buffer_size_regs = scheduler_utils::getOnePersistentBufferSize(
          sorted_candidate_tvs[i - 1], runtime_info, persistent_buffer_info);
      int64_t tv_buffer_size_smem = roundUpSharedMemory(
          sorted_candidate_tvs[i - 1],
          tv_buffer_size_regs,
          vectorize_factor,
          max_threads_per_block);

      acc_regs_buffer_sizes[i] =
          acc_regs_buffer_sizes[i - 1] + tv_buffer_size_regs;
      acc_smem_buffer_sizes[i] =
          acc_smem_buffer_sizes[i - 1] + tv_buffer_size_smem;
    }

    // Determine the least number of buffers to transfer to shared memory
    // to ensure the register buffer size doesn't exceed the available limit.
    int64_t n_smem_buffer = -1;
    for (int i = 1; i <= n_buffers; i++) {
      if (buffer_params.regs_buffer_size - acc_regs_buffer_sizes[i] <=
          available_regs) {
        n_smem_buffer = i;
        break;
      }
    }

    // Can't be scheduled if n_smem_buffer is not set or requested shared memory
    // is larger than available.
    if (n_smem_buffer == -1 ||
        acc_smem_buffer_sizes[n_smem_buffer] > available_smem) {
      buffer_params.has_enough_regs_and_smem = false;
      return buffer_params;
    }

    // Evaluate the possibility of shifting an additional one, but don't move
    // all of them except for combined case where additional register buffer
    // will be created to store intermediate outer reduction results.. Doing so
    // might enhance the efficiency of shared memory utilization due to preset
    // shared configuration sizes. For instance, a 65K shared memory requirement
    // might default to a 100K configuration. But transferring an extra buffer,
    // raising the need to 130K, could lead to a selection of the 132K
    // configuration, optimizing usage.
    const int64_t min_resg_buffer = buffer_params.combined_reduction ? 0l : 1l;
    if (n_smem_buffer > n_broadcast_buffers &&
        n_buffers - n_smem_buffer > min_resg_buffer) {
      int64_t smem_buffer_size = acc_smem_buffer_sizes[n_smem_buffer];
      int64_t smem_config_size = getSharedMemoryConfigSize(
          smem_buffer_size + buffer_params.smem_overhead);
      double buffer_config_ratio = static_cast<double>(smem_buffer_size) /
          static_cast<double>(smem_config_size);
      if (buffer_config_ratio < 0.8 &&
          smem_config_size < smem_config_options.back() &&
          smem_config_size < available_smem) {
        int64_t smem_buffer_size_tmp = acc_smem_buffer_sizes[n_smem_buffer + 1];
        int64_t smem_config_size_tmp = getSharedMemoryConfigSize(
            smem_buffer_size_tmp + buffer_params.smem_overhead);
        double buffer_config_ratio_tmp =
            static_cast<double>(smem_buffer_size_tmp) /
            static_cast<double>(smem_config_size_tmp);
        if (buffer_config_ratio_tmp > buffer_config_ratio &&
            smem_config_size_tmp < available_smem) {
          std::cout << "New n_smem_buffer detected! new n_smem_buffer= "
                    << n_smem_buffer + 1
                    << ", smem_config_size_tmp= " << smem_config_size_tmp
                    << ", buffer_config_ratio_tmp= " << buffer_config_ratio_tmp
                    << ", buffer_config_ratio_old= " << buffer_config_ratio
                    << ", n_broadcast_buffers= " << n_broadcast_buffers
                    << std::endl;
          n_smem_buffer++;
        }
      }
    }
    // move n_smem_buffer buffers to shared memory
    for (int i = 0; i < n_smem_buffer; i++) {
      buffer_params.smem_tvs.emplace_back(sorted_candidate_tvs[i]);
    }
    buffer_params.regs_buffer_size -= acc_regs_buffer_sizes[n_smem_buffer];
    buffer_params.smem_buffer_size = acc_smem_buffer_sizes[n_smem_buffer];
  }

  buffer_params.has_enough_regs_and_smem =
      (buffer_params.smem_buffer_size <= available_smem) &&
      (buffer_params.regs_buffer_size <= available_regs);

  std::cout << "regs_buffer_size: " << buffer_params.regs_buffer_size
            << ", smem_buffer_size: " << buffer_params.smem_buffer_size
            << ", available_regs: " << available_regs
            << ", available_smem: " << available_smem
            << ", has_enough_regs_and_smem: "
            << buffer_params.has_enough_regs_and_smem << std::endl;
  TORCH_INTERNAL_ASSERT(
      buffer_params.has_enough_regs_and_smem,
      "Not enough registers and shared memory for persistence! Should return early.");
  return buffer_params;
}

} // namespace normalization_scheduler_utils
} // namespace nvfuser
