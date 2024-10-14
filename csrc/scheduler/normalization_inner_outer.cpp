// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_inner_outer.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {
namespace {

// The roundup is due to the fact that the shared memory buffer is allocated
// as: ceilDiv(dim_size / vectorize_factor, threads_per_block).
// Let after_vect = dim_size / vectorize_factor;
// n_batch = ceilDiv(after_vect, threads_per_block);
// Then the shared memory buffer size is n_batch * vectorize_factor *
// threads_per_block * data_type_size. This function returns the maximum
// possible shared memory buffer size considering all possible block sizes.
int64_t roundUpSharedMemory(
    int64_t tv_buffer_size,
    int64_t data_type_size,
    int64_t vectorize_factor,
    int64_t threads_per_block_min,
    int64_t threads_per_block_max,
    int64_t threads_per_block_step) {
  int64_t dim_size = tv_buffer_size / data_type_size;
  int64_t after_vect = dim_size / vectorize_factor;
  int64_t max_smem = 0;
  for (int64_t threads_per_block = threads_per_block_min;
       threads_per_block <= threads_per_block_max;
       threads_per_block += threads_per_block_step) {
    int64_t n_batch = ceilDiv(after_vect, threads_per_block);
    max_smem = std::max(
        max_smem,
        n_batch * vectorize_factor * threads_per_block * data_type_size);
  }
  return max_smem;
}

// Return the broadcast tvs that are broadcast to the iteration dimensions of
// the inner reduction tv. These tvs are reused in the loop over the iteration
// dimension. This reuse reduced the number loads from gmem and this tensor
// is likely the first candidate to be moved to shared memory when the register
// space runs low.
std::vector<TensorView*> getOuterBroadcastTvs(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs) {
  // set reference broadcast mask using the first inner reduction tv
  std::vector<bool> ref_broadcast_mask;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      const auto& logical = tv->getLogicalDomain();
      ref_broadcast_mask.reserve(logical.size());
      for (const auto i : c10::irange(logical.size())) {
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

// Size of buffers storing intermediate outer reduction results
// TODO: check if we can directly start with [buffer_size = 1]
int64_t partialOuterReductionBufferSize(
    const std::vector<TensorView*>& reduction_tvs,
    SchedulerRuntimeInfo& runtime_info) {
  int64_t partial_reduction_buffer_size = 0;
  for (auto buffer : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(buffer)) {
      continue;
    }
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

// Decide where to store persistent buffers.
// By default, they reside in registers.
// If register space runs low but there's ample shared memory,
// move one or more buffers to shared memory until the register space is
// sufficient.
struct PersistentBufferStorageParams {
  // representing buffers that are stored in shared memory, other buffers are
  // stored in registers.
  std::vector<TensorView*> smem_persistent_buffers;

  // Total number of bytes occupied by all persistent buffers stored in shared
  // memory.
  int64_t smem_buffer_size = -1;

  // Total number of bytes occupied by all persistent buffers stored in
  // registers.
  int64_t regs_buffer_size = -1;

  // Additional shared memory usage per block that is not associated with
  // persistent buffers. This includes memory for driver overhead and workspace
  // for reductions.
  int64_t smem_overhead = -1;

  // Flag indicating whether there are sufficient registers and shared memory
  // available to accommodate all persistent buffers as required for efficient
  // execution.
  bool has_enough_regs_and_smem = false;

  // Flag indicating whether the persistent buffers are recomputed using inputs.
  bool project_to_input = false;
};

// Prioritize moving buffers used by outer broadcast tensors to shared memory
// because:
// (1) They are reused in every iteration of the outer loop, has lower IO.
// (2) Load occurs before the outer loop. Temporary register usage won't
//     increase register pressure since the loop is the high-pressure region.
std::vector<TensorView*> sortProjectableBufferInputs(
    const std::vector<TensorView*>& projectable_buffer_inputs,
    const std::vector<TensorView*>& outer_broadcast_tvs) {
  // mark whether the buffer is used by outer broadcast tensors
  std::unordered_map<TensorView*, bool> is_used_by_outer_bcast;
  for (auto buffer : projectable_buffer_inputs) {
    is_used_by_outer_bcast[buffer] = std::any_of(
        outer_broadcast_tvs.begin(),
        outer_broadcast_tvs.end(),
        [&buffer](TensorView* tv) {
          return DependencyCheck::isDependencyOf(buffer, tv);
        });
  }

  // sort based on [is_used_by_outer_bcast]
  std::vector<TensorView*> sorted_buffer = projectable_buffer_inputs;
  std::sort(
      sorted_buffer.begin(),
      sorted_buffer.end(),
      [&](TensorView* a, TensorView* b) {
        return is_used_by_outer_bcast[a] && !is_used_by_outer_bcast[b];
      });
  return sorted_buffer;
}

PersistentBufferStorageParams getPersistentBufferStorageParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t vectorize_factor) {
  FUSER_PERF_SCOPE(
      "normalization_inner_outer::getPersistentBufferStorageParams");

  PersistentBufferStorageParams buffer_params;

  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Project to inputs when there is at least one outer broadcast tensor or
  // projected persistent buffer size is smaller. When projecting to inputs, the
  // outer broadcast tensor is reused in the loop over the iteration dimension,
  // test shows it is faster than the non-projected version which requires
  // reload from gmem for each iteration.
  // Note: in current use cases (layer norm bwd and RMS norm bwd), there are
  // outer broadcast tvs and always project to inputs.
  const auto& outer_broadcast_tvs = getOuterBroadcastTvs(fusion, reduction_tvs);
  buffer_params.project_to_input =
      normalization_scheduler_utils::isProjectBufferToInputs(
          fusion,
          runtime_info,
          persistent_buffer_info,
          persistent_buffer_size_info,
          InnerOuterPersistentKernelScheduler::schedulerType(),
          /*can_use_smem_persistent=*/true,
          outer_broadcast_tvs.empty());

  auto total_buffer_size = buffer_params.project_to_input
      ? persistent_buffer_size_info.projected_persistent_buffer_size
      : persistent_buffer_size_info.persistent_buffer_size;
  total_buffer_size +=
      partialOuterReductionBufferSize(reduction_tvs, runtime_info);

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t smem_overhead = scheduler_utils::getSharedMemoryOverheadPerBlock(
      fusion,
      reduction_tvs,
      InnerOuterPersistentKernelScheduler::threads_per_block_max);
  int64_t available_smem =
      (int64_t)dev_prop->sharedMemPerMultiprocessor - smem_overhead;
  int64_t available_regs = scheduler_utils::register_file_size_56k;
  buffer_params.smem_overhead = smem_overhead;

  // (1) init the buffer_params by putting all the persistent tensors in
  // registers
  buffer_params.regs_buffer_size = total_buffer_size;
  buffer_params.smem_buffer_size = 0;

  // (2) If the available register is larger than current register buffer size,
  // no need to move buffers to shared memory, return early.
  bool use_register_only = std::getenv("USE_OLD") != nullptr;
  if (use_register_only && buffer_params.regs_buffer_size <= available_regs) {
    buffer_params.has_enough_regs_and_smem = true;
    return buffer_params;
  }

  // (3) Relocate buffers to shared memory until the buffer size in registers is
  // within the allowable limit.
  // (3.1) Sort the candidate persistent buffers
  const auto buffers = buffer_params.project_to_input
      ? sortProjectableBufferInputs(
            persistent_buffer_info.projectable_buffer_inputs,
            outer_broadcast_tvs)
      : persistent_buffer_info.persistent_buffers;

  // (3.2) Before this loop, all buffers are in registers.
  // Try to move buffer from register to shared memroy.
  // After one buffer is moved to shared memory, the buffer size in registers
  // and shared memory are updated accordingly. Break if required register and
  // shared memory are lower than limit or shared memory exceeds the limit.
  int64_t n_smem_buffer = -1;
  int64_t regs_buffer_size = buffer_params.regs_buffer_size;
  int64_t smem_buffer_size = buffer_params.smem_buffer_size;
  int64_t register_smem_diff = regs_buffer_size - smem_buffer_size;
  const int n_buffers = (int)buffers.size();
  for (int i = 0; i < n_buffers; i++) {
    auto current_tv = buffers[i];

    // calculate the size of this buffer & reduce the register buffer size
    int64_t tv_buffer_size_regs =
        scheduler_utils::getPersistentBufferSizeOfTensor(
            current_tv, runtime_info, persistent_buffer_info);
    regs_buffer_size -= tv_buffer_size_regs;

    // round up the buffer size to shared memory & increase the shared memory
    // buffer size
    int64_t tv_buffer_size_smem = roundUpSharedMemory(
        tv_buffer_size_regs,
        dataTypeSize(current_tv->getDataType().value()),
        vectorize_factor,
        InnerOuterPersistentKernelScheduler::threads_per_block_min,
        InnerOuterPersistentKernelScheduler::threads_per_block_max,
        dev_prop->warpSize);
    smem_buffer_size += tv_buffer_size_smem;

    // The first-i buffers are moved from register to shared memory
    // If both the register buffer size and shared memory buffer size are within
    // the allowable limit, we found a good configuration. Record the number of
    // buffers to be moved to shared memory. Instead of break from the loop, we
    // keep looping to find a better configuration where the difference between
    // register buffer size and shared memory buffer size is minimized with the
    // constraint that register buffer size is still larger than shared memory
    // buffer size.
    if (regs_buffer_size <= available_regs &&
        smem_buffer_size <= available_smem) {
      int64_t diff = regs_buffer_size - smem_buffer_size;
      // if we don't have a valid configuration yet or a better configuration
      // is found, then use it
      if (true || n_smem_buffer == -1 ||
          ((diff > 0 && diff < register_smem_diff))) {
        n_smem_buffer = i + 1;
        register_smem_diff = diff;
        buffer_params.regs_buffer_size = regs_buffer_size;
        buffer_params.smem_buffer_size = smem_buffer_size;
      } else {
        break;
      }
    }
    // shared memory buffer size exceeds the limit, not a good configuration.
    // break the loop, n_smem_buffer remains [-1] indicating a bad
    // configuration.
    if (smem_buffer_size > available_smem) {
      break;
    }
  }

  // n_smem_buffer > 0, has_enough_regs_and_smem = true, move the
  // first n_smem_buffer buffers to shared memory. otherwise, we
  // don't have enough shared memory and registers to accommodate all persistent
  // buffers, has_enough_regs_and_smem = false.
  if (n_smem_buffer > 0) {
    buffer_params.has_enough_regs_and_smem = true;
    buffer_params.smem_persistent_buffers.reserve(n_smem_buffer);
    for (int i = 0; i < n_smem_buffer; i++) {
      buffer_params.smem_persistent_buffers.emplace_back(buffers[i]);
    }
  } else {
    buffer_params.has_enough_regs_and_smem = false;
  }
  return buffer_params;
}

// Calculate the persistent buffer batches and threads per block.
// Start from a large value of inner_dim_numel / (inner_vect * warpSize/4),
// gradually reduce to small values but not smaller than a threshold determined
// by inner_dim_numel and outer_dim_numel. If the persistent buffer batch is
// smaller than the maximum allowed batch which is determined by the avilable
// registers, this function will return that batch value. Otherwise, it will
// return nullopt except when ignore_register_size_limit is true where it will
// return whatever the batch value is.
std::pair<int64_t, int64_t> getBufferBatchSizeAndThreadsPerBlock(
    const int64_t inner_dim_numel,
    const int64_t outer_dim_numel,
    const int64_t register_buffer_size,
    const int64_t shared_memory_buffer_size,
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
    int64_t batch = (vectorize_factor == 1) ? 4l : 1l;
    batch = std::min(batch, inner_dim_numel);
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
  // Each thread can use a maximum of 255 registers, and assume 40 of them are
  // reserved for indexing and other purposes. So, each thread can use up to
  // 215 registers for persistent buffer. Calculate number of buffer batches
  // using these 215 registers. total_buffer_bytes is the total size of
  // persistent buffers in bytes. reduction_elements is the number of elements
  // in the reduction domain. vectorization_factor is the vectorization factor
  // of inputs and outputs.
  auto getMaximumInnerOuterPersistentBufferBatch = [&]() -> int64_t {
    int64_t register_per_batch = ceilDiv(
        register_buffer_size / inner_dim_numel * vectorize_factor,
        scheduler_utils::bytes_per_register);
    int64_t max_persistent_batch = scheduler_utils::safeDiv(
        scheduler_utils::max_registers_per_thread -
            scheduler_utils::register_overhead,
        register_per_batch);
    return std::min(14L, max_persistent_batch);
  };

  const int64_t after_vectorization = inner_dim_numel / vectorize_factor;
  const int64_t threads_per_block_min = std::min(
      after_vectorization,
      InnerOuterPersistentKernelScheduler::threads_per_block_min);
  const int64_t threads_per_block_max =
      InnerOuterPersistentKernelScheduler::threads_per_block_max;
  const int64_t batch_min = getMinimumBatch();
  const int64_t batch_max = getMaximumInnerOuterPersistentBufferBatch();

  // Start from the smallest threads_per_block. If the corresponding batch size
  // is larger than batch_max, try increase threads per block by a warp until
  // the threads_per_block reaches threads_per_block_max or the batch size
  // reaches batch_min.
  int64_t threads_per_block = threads_per_block_min;
  int64_t inner_batch = ceilDiv(after_vectorization, threads_per_block);
  while (inner_batch > batch_max &&
         threads_per_block * 2 <= threads_per_block_max &&
         ceilDiv(after_vectorization, threads_per_block * 2) >= batch_min) {
    threads_per_block *= 2;
    inner_batch = ceilDiv(after_vectorization, threads_per_block);
  }
  return std::make_pair(inner_batch, threads_per_block);
}

// The innerOuterPersistentHeuristic is tuned for layer_norm backward on A100
// ======= Method if hidden_size > 1024 =======
// (1) Inner reduction is one reduction per block. Reduction domain is
// parallelized by TIDx and TIDy, Iteration domain is parallelized by BIDy.
// (2) Outer reduction is done in two-steps. The first step is partial
// reduction, reduction domain is parallelized by BIDy, iteration domain is
// parallelized by TIDx and TIDy. The partial results are written to gmem
// followed by a grid sync. The second step is block reduction, the reduction
// domain is parallelized by TIDy, the iteration domain is parallelized by TIDx
// and BIDy.
// ======= Method if hidden_size <= 1024 =======
// (1) Inner reduction is multi-reductions per blocks. Reduction domain is
// parallelized by TIDx, Iteration domain is parallelized by BIDy and TIDy.
// (2) Outer reduction is same to cases where hidden_size > 1024 except the
// second step where in this case, the reduction domain is parallelized by TIDx
// and the iteration domain is parallelized by TIDy and BIDy. This switch
// between TIDx and TIDy is because:
// (a) We can do warp reduction with TIDx
// (b) TIDx*BIDy is usually much larger than hidden_size, e.g. 128*216 = 1024*27
// this means without switch only 1/27 of the threads is used.
std::unique_ptr<ReductionParams> innerOuterPersistentHeuristic(
    const int64_t outer_dim_numel,
    const int64_t inner_dim_numel,
    const int64_t regs_buffer_size,
    const int64_t smem_buffer_size,
    const int64_t smem_overhead,
    const size_t tmp_gmem_dtype_size,
    const size_t vectorize_factor,
    const bool project_to_input,
    const PrimDataType index_type) {
  // Initialize reduction parameters
  auto rparams = std::make_unique<ReductionParams>(
      InnerOuterPersistentKernelScheduler::schedulerType());
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t sm_count = (int64_t)dev_prop->multiProcessorCount;
  const int64_t warp_size = (int64_t)dev_prop->warpSize;

  // Estimate registers per thread based on buffer size
  auto getEstimatedRegisterUsage = [&](int64_t batch_mul_vect) -> int64_t {
    constexpr int64_t bytes_per_register = 4;
    return (regs_buffer_size / inner_dim_numel * batch_mul_vect) /
        bytes_per_register +
        scheduler_utils::register_overhead;
  };

  // Calculate the number of blocks per SM based on threads and warps
  auto getBlocksPerSM = [&](int64_t threads_per_sm,
                            int64_t threads_per_block) -> int64_t {
    constexpr int64_t warp_allocation_granularity = 4;
    const int64_t allocated_warps_per_block =
        ceilDiv(
            ceilDiv(threads_per_block, warp_size),
            warp_allocation_granularity) *
        warp_allocation_granularity;
    return scheduler_utils::safeDiv(
        threads_per_sm / warp_size, allocated_warps_per_block);
  };

  // Calculate the inner batch size
  auto getInnerBatch = [&](int64_t vectorize_factor,
                           int64_t threads_per_block) -> int64_t {
    return ceilDiv(inner_dim_numel / vectorize_factor, threads_per_block);
  };
  auto getUnusedThreads = [&](int64_t inner_vect,
                              int64_t threads_per_block,
                              int64_t inner_batch) -> int64_t {
    return threads_per_block * inner_batch - inner_dim_numel / inner_vect;
  };

  // Compute the number of grid dimensions for the inner reduction
  auto getGdimy = [&](int64_t inner_vect,
                      int64_t inner_batch,
                      int64_t threads_per_block) -> int64_t {
    int64_t reg_per_thread =
        getEstimatedRegisterUsage(inner_vect * inner_batch);
    reg_per_thread =
        std::min(reg_per_thread, scheduler_utils::max_registers_per_thread);
    int64_t threads_per_sm = getThreadsPerSMGivenRegPerThread(reg_per_thread);
    int64_t blocks_per_sm_regs =
        getBlocksPerSM(threads_per_sm, threads_per_block);
    int64_t blocks_per_sm_smem = dev_prop->sharedMemPerMultiprocessor /
        (smem_overhead + smem_buffer_size);
    int64_t blocks_per_sm = std::min(blocks_per_sm_regs, blocks_per_sm_smem);
    int64_t gdimy = blocks_per_sm * sm_count;

    const int64_t outer_iter_min = 8;
    const int64_t gdimy_max = scheduler_utils::roundUpToN(
        ceilDiv(outer_dim_numel, outer_iter_min), sm_count);

    while (gdimy > gdimy_max && blocks_per_sm > 1) {
      blocks_per_sm--;
      gdimy = blocks_per_sm * sm_count;
    }

    return gdimy;
  };

  auto getOuterReductionVectorizationFactor =
      [&](int64_t tmp_gmem_dtype_size,
          int64_t inner_vect) -> std::pair<int64_t, int64_t> {
    // Set vectorization factor for global memory writes
    constexpr int64_t max_gmem_vect_access_bytes = 16;
    int64_t tmp_gmem_write_vect =
        std::min(max_gmem_vect_access_bytes / tmp_gmem_dtype_size, inner_vect);

    // Set outer reduction parameters
    int64_t vectorization_factor_outer =
        std::min(inner_dim_numel >= 4096 ? 4l : 2l, tmp_gmem_write_vect);

    return std::make_pair(tmp_gmem_write_vect, vectorization_factor_outer);
  };

  auto getBdimxBdimy =
      [&](int64_t gdimy,
          int64_t threads_per_block,
          int64_t vectorization_factor_outer) -> std::pair<int64_t, int64_t> {
    int64_t bdimx = scheduler_utils::roundUpPow2Or8(
        ceilDiv(inner_dim_numel / vectorization_factor_outer, gdimy));

    while (threads_per_block % bdimx) {
      bdimx = std::min(bdimx + 8, threads_per_block);
    }

    int64_t bdimy = threads_per_block / bdimx;
    NVF_ERROR(
        bdimy * bdimx == threads_per_block,
        "Threads per block must be divisible by bdimx and bdimy.");

    return std::make_pair(bdimx, bdimy);
  };

  // Define inner and outer parameters struct
  struct InnerOuterParams {
    int64_t inner_vect = -1;
    int64_t inner_batch = -1;
    int64_t bdimx = -1;
    int64_t bdimy = -1;
    int64_t bdimz = -1;
    int64_t gdimy = -1;
    int64_t tmp_gmem_write_vect = -1;
    int64_t vectorization_factor_outer = -1;

    // If not divisible, last batch has unused threads
    int64_t unused_threads = -1;
    int64_t required_regs = -1;
    int64_t avilable_regs = -1;
    int64_t warps_per_sm = -1;

    void verify() {
      NVF_ERROR(inner_vect != -1, "inner_vect is not set.");
      NVF_ERROR(inner_batch != -1, "inner_batch is not set.");
      NVF_ERROR(bdimx != -1, "bdimx is not set.");
      NVF_ERROR(bdimy != -1, "bdimy is not set.");
      NVF_ERROR(bdimz != -1, "bdimz is not set.");
      NVF_ERROR(gdimy != -1, "gdimy is not set.");
      NVF_ERROR(tmp_gmem_write_vect != -1, "tmp_gmem_write_vect is not set.");
      NVF_ERROR(
          vectorization_factor_outer != -1,
          "vectorization_factor_outer is not set.");
    }
    std::string toString() const {
      std::stringstream ss;
      ss << "inner_vect: " << inner_vect << ", inner_batch: " << inner_batch
         << ", bdimx: " << bdimx << ", bdimy: " << bdimy
         << ", unused_threads: " << unused_threads << ", gdimy: " << gdimy
         << ", tmp_gmem_write_vect: " << tmp_gmem_write_vect
         << ", required_regs: " << required_regs
         << ", avilable_regs: " << avilable_regs
         << ", warps_per_sm: " << warps_per_sm
         << ", vectorization_factor_outer: " << vectorization_factor_outer;
      return ss.str();
    }
  };

  auto getInnerOuterParams =
      [&](int64_t inner_vect, int64_t threads_per_block) -> InnerOuterParams {
    InnerOuterParams iop;
    iop.inner_vect = inner_vect;
    // inner dim
    iop.inner_batch = getInnerBatch(iop.inner_vect, threads_per_block);
    // outer dim
    iop.gdimy = getGdimy(iop.inner_vect, iop.inner_batch, threads_per_block);
    // dump partial outer reduction results to gmem
    std::tie(iop.tmp_gmem_write_vect, iop.vectorization_factor_outer) =
        getOuterReductionVectorizationFactor(
            tmp_gmem_dtype_size, iop.inner_vect);
    // outer reduction
    std::tie(iop.bdimx, iop.bdimy) = getBdimxBdimy(
        iop.gdimy, threads_per_block, iop.vectorization_factor_outer);

    iop.unused_threads =
        getUnusedThreads(iop.inner_vect, threads_per_block, iop.inner_batch);
    iop.required_regs =
        getEstimatedRegisterUsage(iop.inner_batch * iop.inner_vect);
    iop.avilable_regs = getRegPerThreadGivenThreadsPerSM(
        threads_per_block * iop.gdimy / sm_count);
    iop.warps_per_sm = iop.gdimy * threads_per_block / warp_size;
    return iop;
  };

  const int64_t max_vect_factor = (int64_t)vectorize_factor;
  std::vector<int64_t> vect_candidates = {max_vect_factor};
  const int64_t after_vect = inner_dim_numel / max_vect_factor;
  std::vector<int64_t> threads_candidates;
  if (after_vect > 256) {
    threads_candidates = std::vector<int64_t>{128, 256, 512};
  } else if (after_vect > 128) {
    threads_candidates = std::vector<int64_t>{128, 256};
  } else {
    threads_candidates = std::vector<int64_t>{after_vect};
  }

  std::vector<InnerOuterParams> iop_candidates;
  for (auto vect : vect_candidates) {
    for (auto threads : threads_candidates) {
      iop_candidates.emplace_back(getInnerOuterParams(vect, threads));
    }
  }
  if (iop_candidates.size() > 1) {
    std::stable_sort(
        iop_candidates.begin(),
        iop_candidates.end(),
        [](const InnerOuterParams& a, const InnerOuterParams& b) {
          // register based sorting
          if (a.required_regs <= a.avilable_regs &&
              b.required_regs > b.avilable_regs) {
            return true;
          } else if (
              b.required_regs <= b.avilable_regs &&
              a.required_regs > a.avilable_regs) {
            return false;
          } else if (
              a.required_regs > a.avilable_regs &&
              b.required_regs > b.avilable_regs) {
            return a.required_regs - a.avilable_regs <
                b.required_regs - b.avilable_regs;
          }
          // occupancy
          if (a.warps_per_sm != b.warps_per_sm) {
            return a.warps_per_sm > b.warps_per_sm;
          }
          // // prefer divisible split, may be slower, e.g. at 17K uses batch
          // size of 17. if ((a.unused_threads == 0 && b.unused_threads != 0) ||
          //     (a.unused_threads != 0 && b.unused_threads == 0)) {
          //   return a.unused_threads < b.unused_threads;
          // }

          // persistent batch size
          if (a.inner_batch != b.inner_batch) {
            return a.inner_batch < b.inner_batch;
          }
          NVF_ERROR(false, "sort idp_candidates failed.");
          return false;
        });
  }

  InnerOuterParams iop = iop_candidates.at(0);

  if (std::getenv("THREADS") != nullptr && std::getenv("VECT") != nullptr) {
    auto threads = std::stoi(std::getenv("THREADS"));
    auto vect = std::stoi(std::getenv("VECT"));
    iop = getInnerOuterParams(vect, threads);
  } else {
    for (auto iop : iop_candidates) {
      std::cout << iop.toString() << std::endl;
    }
  }

  // std::min(8L, ceilDiv(iop.gdimy, iop.bdimy));
  // Step-5, special case, when inner_dim_numel <= 1024, bdimx is usually small
  // after divide by inner_vect and inner_batch. In this case, bdimy is used to
  // parallelize outer_dim instead of inner_dim. This pattern is named multi
  // reductions per block (mrpb).
  if (inner_dim_numel <= 1024) {
    rparams->multiple_reds_per_blk = true;
    rparams->tidx_for_outer_reduction = true;
    constexpr int64_t threads_per_block_mrpb = 512;

    // Step-1, InnerParams, Reduction dim: inner_vect(reuse),
    // inner_batch(reuse), bdimx
    iop.bdimx = ceilDiv(inner_dim_numel, iop.inner_vect * iop.inner_batch);
    iop.bdimz = ceilDiv(
        ceilDiv(inner_dim_numel / iop.inner_vect, iop.bdimx), iop.inner_batch);
    // Step-2, InnerParams, Iteration dim: gdimy, bdimy (in next step)
    int64_t reg_per_thread =
        getEstimatedRegisterUsage(iop.inner_vect * iop.inner_batch);
    reg_per_thread =
        std::min(reg_per_thread, scheduler_utils::max_registers_per_thread);
    int64_t threads_per_sm = getThreadsPerSMGivenRegPerThread(reg_per_thread);
    int64_t blocks_per_sm =
        getBlocksPerSM(threads_per_sm, threads_per_block_mrpb);
    iop.gdimy = blocks_per_sm * sm_count;

    // Step-3, OuterParams, Iteration dim: vectorization_factor_outer(reuse),
    // bdimy, gdimy (in previous step). We prefer bdimy to be larger enough to
    // cover what is left in both the outer_dim and inner_dim. However, it
    // should not exceed the limitation set by threads_per_block_mrpb.
    int64_t bdimy_tmp = std::max(
        ceilDiv(outer_dim_numel, iop.gdimy),
        ceilDiv(inner_dim_numel, iop.vectorization_factor_outer * iop.gdimy));
    iop.bdimy = std::min(threads_per_block_mrpb / iop.bdimx, bdimy_tmp);

    iop.avilable_regs = getRegPerThreadGivenThreadsPerSM(
        iop.bdimx * iop.bdimy * iop.gdimy / sm_count);
    iop.warps_per_sm = iop.gdimy * iop.bdimx * iop.bdimy / warp_size;
    // Step-4, OuterParams, Reduction dim: bdimx (already done)

    if (iop.bdimx % dev_prop->warpSize == 0) {
      rparams->pad_inner_reduction_to_warp = true;
      rparams->pad_outer_reduction_to_warp = true;
    }
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  } else {
    rparams->block_dim_inner_reduction_extra = ParallelType::TIDy;
    rparams->static_bdimx = true;
    rparams->static_bdimy = true;
    iop.bdimz = ceilDiv(
        ceilDiv(
            ceilDiv(inner_dim_numel / iop.inner_vect, iop.bdimx), iop.bdimy),
        iop.inner_batch);
  }
  NVF_ERROR(iop.bdimz == 1, "bdimz must be 1.");

  // check all the parameters in InnerOuterParams are set.
  iop.verify();
  rparams->combined_outer_reduction_static_bdimy = true;
  rparams->unroll_factor_outer_reduction = 1;
  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;
  // tmp_gmem is the intermediate result of outer reduction, its dtype is float,
  // so the maximum vectorization factor is 4.
  rparams->vectorization_factor_outer = iop.vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = iop.tmp_gmem_write_vect;
  rparams->cparams.maxrregcount = iop.avilable_regs;
  rparams->unroll_factor_inner_reduction = iop.inner_vect;
  rparams->batches_per_block_inner_reduction = iop.inner_batch;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->vectorize_inner_reduction = iop.inner_vect > 1;
  rparams->split_grid_dim_iter_dom_outer = true;
  rparams->grid_dim_iter_dom = ParallelType::BIDy;

  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      iop.gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      iop.bdimx,
      iop.bdimy,
      LaunchParams::UNINITIALIZED_VAL);
  if (!rparams->smem_persistent_buffers.empty()) {
    rparams->tag =
        "InnerOuter Register and Shared Memory Persistent Heuristic.\n";
  } else {
    rparams->tag = "InnerOuter Register Persistent Heuristic.\n";
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Combined InnerOuter Reduction Stats ========\n"
            << "outer_dim_numel: " << outer_dim_numel << "\n"
            << "inner_dim_numel: " << inner_dim_numel << "\n"
            << "regs_buffer_size: " << regs_buffer_size << "\n"
            << "smem_buffer_size: " << smem_buffer_size << "\n"
            << "smem_overhead: " << smem_overhead << "\n"
            << "vectorize_factor_input: " << iop.inner_vect << "\n"
            << "vectorization_factor_tmp_gmem_write: "
            << iop.tmp_gmem_write_vect << "\n"
            << "vectorization_factor_outer: " << iop.vectorization_factor_outer
            << "\n"
            << "multiple_reds_per_blk: " << rparams->multiple_reds_per_blk
            << "\n"
            << "warps_per_sm: " << iop.warps_per_sm << "\n"
            << "gdimy: " << iop.gdimy << "\n"
            << "block(" << (iop.bdimx) << ", " << iop.bdimy << ", " << 1 << ")";
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

std::unique_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  // Get dtype used to store partial outer reduction
  // Get the first inner reduction tv and use it as the reference tv
  int64_t max_outer_reduction_dtype_size = 1;
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
    } else {
      max_outer_reduction_dtype_size = std::max(
          max_outer_reduction_dtype_size,
          dataTypeSize(tv->getDataType().value()));
    }
  }
  auto ref_red_tv = first_inner_reduction_tv;

  // Verify the presence of a reduction TensorView connected to a Fusion input
  normalization_scheduler_utils::checkReductionTvForScheduling(
      fusion, ref_red_tv);

  auto properties =
      scheduler_utils::getReductionProperties(fusion, runtime_info, ref_red_tv);
  auto reduced_tv = ir_utils::getSoleProducerTv(ref_red_tv);

  // Although properties contains runtime information
  // "inner_most_dimension_ndims" is a compile time value
  auto vec_break_point = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizationBreakPointOfReductionProducer>(
      data_cache, [&ref_red_tv, &reduced_tv, &properties]() {
        return std::make_unique<int64_t>(
            vectorize_helper::getVectorizationBreakPointOfReductionProducer(
                ref_red_tv, reduced_tv, properties.inner_most_dimension_ndims));
      });

  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
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
  auto buffer_params = getPersistentBufferStorageParams(
      fusion, runtime_info, data_cache, reduction_tvs, vectorize_factor);
  std::unique_ptr<ReductionParams> rparams = innerOuterPersistentHeuristic(
      properties.total_iteration_numel,
      properties.total_reduction_numel,
      buffer_params.regs_buffer_size,
      buffer_params.smem_buffer_size,
      buffer_params.smem_overhead,
      max_outer_reduction_dtype_size,
      vectorize_factor,
      buffer_params.project_to_input,
      runtime_info.getIndexType());

  // save persistent tvs should use shared memory, to avoid calling
  // getPersistentBufferStorageParams again during the scheduling.
  rparams->smem_persistent_buffers = buffer_params.smem_persistent_buffers;

  return rparams;
}

void scheduleReductionCombinedOuter(
    Fusion* fusion,
    const ReductionParams* rparams,
    const std::vector<TensorView*>& outer_reduction_tvs,
    std::vector<TensorView*>& cached_gmem,
    std::vector<TensorView*>& cached_gmem_reload,
    std::vector<TensorView*>& outer_reference_tvs,
    std::unordered_set<TensorView*>& boundaryNodesSet) {
  auto mergeReductionOrIterDomains = [](TensorView* tv, bool mergeReduction) {
    int prev_i = -1;
    for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
      if (mergeReduction == tv->axis(i)->isReduction()) {
        if (prev_i == -1) {
          prev_i = i;
        } else {
          tv->merge(i, prev_i);
          prev_i = i;
        }
      }
    }
  };
  for (auto& outer_reduction_tv : outer_reduction_tvs) {
    // merge tensorview to [reduction, iteraiton] domains
    mergeReductionOrIterDomains(outer_reduction_tv, true);
    mergeReductionOrIterDomains(outer_reduction_tv, false);
    if (rparams->multiple_reds_per_blk) {
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(rparams->block_dim_iter_dom));
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(rparams->grid_dim_iter_dom), false);
    } else {
      outer_reduction_tv->split(0, rparams->lparams.gdimy());
    }

    if (rparams->multiple_reds_per_blk) {
      outer_reduction_tv->rFactor({1});
    }
    TensorView* partialResult = rparams->multiple_reds_per_blk
        ? outer_reduction_tv->rFactor({1})
        : outer_reduction_tv->rFactor({0});
    partialResult->cacheBefore();
    partialResult->setMemoryType(MemoryType::Global);
    TensorView* partialResultReload = partialResult->cacheAfter();

    boundaryNodesSet.insert(partialResultReload);
    cached_gmem.emplace_back(partialResult);
    cached_gmem_reload.emplace_back(partialResultReload);

    if (rparams->multiple_reds_per_blk) {
      if (rparams->tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
        // to use warp reduction
        if (rparams->pad_outer_reduction_to_warp) {
          outer_reduction_tv->axis(1)->padToMultipleOfWarp();
        }
      } else {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
      }
      // iteration domain
      int axisID = -1;
      if (rparams->vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams->vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams->tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDy);
      } else {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }
      if (rparams->combined_split_grid_inner_dim) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::BIDy));
      }
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);

    } else {
      // reduction domain
      if (rparams->combined_outer_reduction_static_bdimy) {
        outer_reduction_tv->split(0, rparams->lparams.bdimy());
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
      } else {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);

        // [I/TIDy, TIDy]
        if (rparams->unroll_factor_outer_reduction > 1) {
          outer_reduction_tv->split(0, rparams->unroll_factor_outer_reduction);
          outer_reduction_tv->axis(1)->parallelize(ParallelType::Unroll);
          // [I/TIDy/Unroll, Unroll, TIDy]
          outer_reduction_tv->split(0, 1);
          outer_reduction_tv->axis(1)->parallelize(ParallelType::Unswitch);
        }
      }

      // iteration domain
      int axisID = -1;
      if (rparams->vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams->vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams->lparams.bdimx() > 1) {
        outer_reduction_tv->split(axisID, rparams->lparams.bdimx());
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }

      if (rparams->combined_split_grid_inner_dim) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::BIDy));
      }

      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);
    }
    auto outer_reference_tv =
        reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
    outer_reference_tvs.emplace_back(outer_reference_tv);
  }
}

// fusion is the input IR that will be modified by this function
void scheduleInnerOuterPersistentKernel(
    Fusion* fusion,
    const ReductionParams* rparams) {
  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs,
      smem_consumers;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  normalization_scheduler_utils::beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
      smem_consumers,
      cached_outputs);

  // split reduction_tvs into inner and outer reduction_tvs
  std::vector<TensorView*> inner_reduction_tvs, outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  NVF_ERROR(
      !inner_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no inner reduction is found.");
  NVF_ERROR(
      !outer_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no outer reduction is found.");

  // schedule inner reduction, only schedule the first inner reduction tv,
  // then will be propagated to other inner reduction tvs.
  TensorView* inner_reference_tv =
      normalization_scheduler_utils::scheduleReductionGeneral(
          fusion,
          rparams,
          inner_reduction_tvs,
          InnerOuterPersistentKernelScheduler::schedulerType());

  // schedule outer reduction, schedule all the outer reduction tvs since we
  // need to store the intermediate results.
  std::vector<TensorView*> cached_gmem;
  std::vector<TensorView*> cached_gmem_reload;
  std::vector<TensorView*> outer_reference_tvs;
  std::unordered_set<TensorView*> boundaryNodesSet;
  scheduleReductionCombinedOuter(
      fusion,
      rparams,
      outer_reduction_tvs,
      cached_gmem,
      cached_gmem_reload,
      outer_reference_tvs,
      boundaryNodesSet);

  // Propagate inner reduction and outer reductions
  for (auto output : dummy_outputs) {
    fusion->addOutput(output);
  }

  const bool unroll = rparams->isUnrolled();
  const bool vectorize =
      rparams->vectorize_inner_reduction || rparams->vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams->persistent_kernel &&
      rparams->cross_grid_inner_reduction && !rparams->fastest_dim;

  // Propagate inner reduction. There is a cutoff at boundaryNodesSet, so this
  // propagation will not propagate to the final outer reduction.
  reduction_scheduler_utils::propagateTransformation(
      inner_reference_tv, boundaryNodesSet);
  reduction_scheduler_utils::propagateRFactor(
      inner_reference_tv, inner_reduction_tvs[0], inner_reduction_tvs);

  // Don't allow parallelization propagation goes through boundaryNodesSet
  const auto& selected_tvs_inner =
      scheduler_utils::getAllTvsFrom(inner_reduction_tvs, boundaryNodesSet);
  reduction_scheduler_utils::propagateParallelization(
      fusion,
      inner_reduction_tvs[0],
      inner_reference_tv,
      unroll,
      vectorize,
      is_outer_grid_persistence,
      inner_reduction_tvs,
      cached_inputs,
      cached_outputs,
      smem_consumers,
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  // Propagate outer reduction. Each outer reduction is connected with its
  // cached_gmem and output, since we added all the cached_gmem to the
  // boundaryNodesSet, the transformation from one outer reduction can't
  // propagate to other outer reductions due to the cutoff at
  // boundaryNodesSet. Thus, we need a loop to initiate the propagation from
  // each outer reduction. Don't allow parallelization propagation goes
  // through cached_gmem, see issue 246.
  // don't do grouped reduction for now since it uses more registers.
  bool is_grouped_reduction = false; //! rparams->tidx_for_outer_reduction;
  for (long unsigned int i = 0; i < outer_reference_tvs.size(); i++) {
    const auto& selected_tvs_outer = scheduler_utils::getAllTvsFrom(
        {outer_reduction_tvs[i]}, {cached_gmem[i]});
    reduction_scheduler_utils::propagateTransformation(
        outer_reference_tvs[i], boundaryNodesSet);
    reduction_scheduler_utils::propagateParallelization(
        fusion,
        outer_reduction_tvs[i],
        outer_reference_tvs[i],
        unroll,
        vectorize,
        is_grouped_reduction,
        outer_reduction_tvs,
        cached_inputs,
        cached_outputs,
        smem_consumers,
        {selected_tvs_outer.begin(), selected_tvs_outer.end()});
  }

  // special vectorization of temp gmem, vectorization_factor_tmp_gmem_write
  // is guaranteed to be smaller or equal to input vectorization factor.
  if (rparams->vectorization_factor_tmp_gmem_write > 1) {
    for (auto tv : cached_gmem) {
      NVF_ERROR(
          rparams->vectorization_factor_tmp_gmem_write <=
              rparams->unroll_factor_inner_reduction,
          "vectorization factor of temp gmem write should be smaller than that of inner reduction.")
      if (rparams->vectorization_factor_tmp_gmem_write <
          rparams->unroll_factor_inner_reduction) {
        tv->split(-1, rparams->vectorization_factor_tmp_gmem_write);
      }
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }
  // vectorization propagate through propagateParallelization only works for
  // input and output tensors. propagate vectorization to cached_gmem_reload
  // directly from output tv using parallelizeAllLike. must propagate
  // seperaely for different tvs as outer reductions are transformed
  // seperately.
  if (rparams->vectorization_factor_outer > 1) {
    for (auto tv : cached_gmem_reload) {
      auto output_tvs = ir_utils::outputTvsOf(tv);
      NVF_ERROR(
          !output_tvs.empty(),
          "cached_gmem_reload should have at least one output tensor.")
      scheduler_utils::parallelizeAllLike(
          output_tvs[0],
          -1,
          {cached_gmem_reload.begin(), cached_gmem_reload.end()},
          {ParallelType::Vectorize});
    }
  }

  // Remove dummy outputs as they can inadvertently affect CA positions
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }
  inlineMost();
}

} // namespace

bool InnerOuterPersistentKernelScheduler::canScheduleCompileTime(
    Fusion* fusion) {
  FUSER_PERF_SCOPE(
      "InnerOuterPersistentKernelScheduler::canScheduleCompileTime");
  // common checks for all persistent heuristics
  if (!normalization_scheduler_utils::checkOpsAndInputs(
          fusion, schedulerType())) {
    return false;
  }

  // check reduction type
  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no reduction tv");
    return false;
  }
  auto reduction_type =
      reduction_scheduler_utils::getReductionType(reduction_tvs);
  const SchedulerType persistent_heuristic =
      normalization_scheduler_utils::getPersistentHeuristicFor(reduction_type);
  if (persistent_heuristic != schedulerType()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "schedulerType() doesn't match with reduction type `",
        persistent_heuristic,
        "`.");
    return false;
  }
  std::vector<TensorView*> inner_reduction_tvs;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }

  // check connections between inner reduction and outer reduction tvs.
  if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction tensor should be [I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
    return false;
  }

  if (!normalization_scheduler_utils::hasSharedInput(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should have shared input.");
    return false;
  }

  if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should not have shared consumer, their consumers should not have shared non-outer-reduction producer.");
    return false;
  }

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }
    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    auto reference_tv = inner_reduction_tvs[0];
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "View may interfere with normalization scheduling.");
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
            schedulerType(),
            "inconsistent reduction root size: ",
            red->toString(),
            ", expected: ",
            axis_count);
        return false;
      }
    }
  }

  // the reduction axis of outer reduction tv should match to the iteration axis
  // of the inner reduction tv.
  if (!normalization_scheduler_utils::isReductionIterationAxisMatched(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, every iteration axis in inner reduction tv should match to a reduction domain in outer reduction tv.");
    return false;
  }

  if (!normalization_scheduler_utils::checkReductionPattern(
          fusion, schedulerType(), inner_reduction_tvs, outer_reduction_tvs)) {
    return false;
  }

  // Only accept persistent kernels
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "has unsupported gather-like ops before normalization");
    return false;
  }

  return true;
}

bool InnerOuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::canScheduleRunTime");
  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      break;
    }
  }
  auto reference_tv = first_inner_reduction_tv;

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);

  const int64_t warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;

  auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      (int)(reduced_tv->nDims() - properties.inner_most_dimension_ndims));

  // check if there is enough register and shared memory for persistence
  const auto buffer_params = getPersistentBufferStorageParams(
      fusion, runtime_info, data_cache, reduction_tvs, vectorize_factor);

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (!buffer_params.has_enough_regs_and_smem) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "not enough registers or shared memory for persistence");
    return false;
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t required_sm_per_norm = ceilDiv(
      buffer_params.regs_buffer_size, scheduler_utils::register_file_size);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "requires over half GPU persistence.");
    return false;
  }

  // Don't go persistent if we can't use a small fraction of the
  // available SMs yet have a large reduction size.
  if ( // Large reduction dim
      properties.total_reduction_numel >=
          device_max_threads_per_multiprocessor * 4 &&
      properties.total_iteration_numel <
          (properties.fastest_dim_reduction
               ? scheduler_utils::safeDiv(device_multiprocessor_count, 8)
               // Make sure we at least use a quarter of the device * a
               // half warp
               : (warp_size / 8) * device_multiprocessor_count)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "not enough blocks");
    return false;
  }

  return true;
}

std::unique_ptr<HeuristicParams> InnerOuterPersistentKernelScheduler::
    computeHeuristics(
        Fusion* fusion,
        SchedulerRuntimeInfo& runtime_info,
        HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::computeHeuristics");
  auto rparams =
      getInnerOuterPersistentHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void InnerOuterPersistentKernelScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::schedule");
  auto rparams = dynamic_cast<const ReductionParams*>(params);
  NVF_ERROR(
      rparams != nullptr && rparams->scheduler_type == schedulerType(),
      "Incorrect parameters sent to InnerOuterPersistentKernelScheduler::schedule",
      params);
  scheduleInnerOuterPersistentKernel(fusion, rparams);
}
} // namespace nvfuser
