// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <ops/arith.h>
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

// Prioritize keeping buffers used by outer broadcast tensors to shared memory
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
        return !is_used_by_outer_bcast[a] && is_used_by_outer_bcast[b];
      });
  return sorted_buffer;
}

PersistentBufferStorageParams getPersistentBufferStorageParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t vectorize_factor,
    const int64_t threads_per_block_min,
    const int64_t threads_per_block_max) {
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
  // Warp specialized persistent kernel always cache inputs in shared memory,
  // should project to inputs.
  const auto& outer_broadcast_tvs = getOuterBroadcastTvs(fusion, reduction_tvs);
  bool skip_check_buffer_size = !outer_broadcast_tvs.empty() ||
      isOptionEnabled(EnableOption::WarpSpecializedNormalization);
  normalization_scheduler_utils::BufferProjectionStrategy project_strategy =
      normalization_scheduler_utils::isProjectBufferToInputs(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          persistent_buffer_size_info,
          InnerOuterPersistentKernelScheduler::schedulerType(),
          /*can_use_smem_persistent=*/true,
          !skip_check_buffer_size);

  buffer_params.project_to_input =
      (project_strategy ==
       normalization_scheduler_utils::BufferProjectionStrategy::
           ProjectToInputs);

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t smem_overhead = scheduler_utils::getSharedMemoryOverheadPerBlock(
      fusion, reduction_tvs, threads_per_block_max);
  int64_t available_smem =
      (int64_t)dev_prop->sharedMemPerMultiprocessor - smem_overhead;
  int64_t available_regs = scheduler_utils::register_file_size_56k;
  buffer_params.smem_overhead = smem_overhead;

  // (1) Use both register and shared memory.
  // Start with all the cached input buffers in shared memory, they are loaded
  // from global memory uses async copy which bypasses L1 cache. Outer reduction
  // buffers are used to accumulate partial results of the outer reduction. They
  // are not loaded from global memory and requires frequent read/write. So,
  // they are always stored in registers.
  // TODO: We may also move outer reduction buffers to shared
  // memory to avoid segmentation when there are many outer reductions and
  // hardware has larger shared memory, but these applications are rare, so this
  // is not considered here.
  auto buffers = buffer_params.project_to_input
      ? persistent_buffer_info.projectable_buffer_inputs
      : persistent_buffer_info.persistent_buffers;

  // Add buffers that are inputs to the fusion. They are not included in
  // projectable_buffer_inputs since they are not projectable.
  if (buffer_params.project_to_input) {
    for (auto tv : persistent_buffer_info.persistent_buffers) {
      if (tv->isFusionInput()) {
        buffers.push_back(tv);
      }
    }
  }

  // Needs to use rounded shared memory size to avoid over usage.
  // key : buffer tv.
  // val : register size and rounded shared memory size
  std::unordered_map<TensorView*, std::pair<int64_t, int64_t>>
      required_size_regs_smem_map;
  int64_t total_smem_buffer_size = 0;
  for (auto buffer : buffers) {
    int64_t buffer_size_regs = scheduler_utils::getPersistentBufferSizeOfTensor(
        buffer, runtime_info, persistent_buffer_info);
    int64_t buffer_size_smem = roundUpSharedMemory(
        buffer_size_regs,
        dataTypeSize(buffer->getDataType().value()),
        vectorize_factor,
        threads_per_block_min,
        threads_per_block_max,
        dev_prop->warpSize);
    required_size_regs_smem_map[buffer] =
        std::make_pair(buffer_size_regs, buffer_size_smem);
    total_smem_buffer_size += buffer_size_smem;
  }
  buffer_params.smem_buffer_size = total_smem_buffer_size;
  buffer_params.regs_buffer_size =
      partialOuterReductionBufferSize(reduction_tvs, runtime_info);
  if (buffer_params.regs_buffer_size <= available_regs &&
      buffer_params.smem_buffer_size <= available_smem) {
    buffer_params.smem_persistent_buffers = buffers;
    buffer_params.has_enough_regs_and_smem = true;
    return buffer_params;
  }

  // Moving outer reduction buffer to shared memory is not considered yet,
  // set to false if the outer reduction buffer size exceeds the register size.
  if (buffer_params.regs_buffer_size > available_regs) {
    buffer_params.has_enough_regs_and_smem = false;
    return buffer_params;
  }

  // (2) Now, shared memory is overused, move some buffers to registers.
  // (2.1) Sort the candidate persistent buffers. No need to sort since the
  // sorting is based on whether the buffer is used by outer broadcast tensors.
  if (!outer_broadcast_tvs.empty()) {
    buffers = sortProjectableBufferInputs(buffers, outer_broadcast_tvs);
  }
  // (2.2) Before this loop, all cached input buffers are in shared memory. Move
  // buffer from shared memory to register.
  int64_t n_regs_buffer = -1;
  const int n_buffers = (int)buffers.size();
  for (int i = 0; i < n_buffers; i++) {
    auto current_tv = buffers[i];
    auto [buffer_size_regs, buffer_size_smem] =
        required_size_regs_smem_map.at(current_tv);
    buffer_params.regs_buffer_size += buffer_size_regs;
    buffer_params.smem_buffer_size -= buffer_size_smem;

    // The first-i buffers to are moved from shared memory to register
    // If both the register buffer size and shared memory buffer size are within
    // the allowable limit, we found a good configuration.
    if (buffer_params.regs_buffer_size <= available_regs &&
        buffer_params.smem_buffer_size <= available_smem) {
      n_regs_buffer = i + 1;
      break;
    }
    // Register buffer size exceeds the limit, can't move more to registers.
    // Break the loop.
    if (buffer_params.regs_buffer_size > available_regs) {
      break;
    }
  }

  // n_regs_buffer > 0 indicats a good configuration is found.
  // The first n_regs_buffer buffers are stored in registers and last [n_buffers
  // - n_regs_buffer] are stored in shared memory.
  if (n_regs_buffer > 0) {
    buffer_params.has_enough_regs_and_smem = true;
    auto n_smem_buffer = n_buffers - n_regs_buffer;
    buffer_params.smem_persistent_buffers.reserve(n_smem_buffer);
    for (int i = 0; i < n_smem_buffer; i++) {
      buffer_params.smem_persistent_buffers.emplace_back(
          buffers[n_buffers - 1 - i]);
    }
  } else {
    buffer_params.has_enough_regs_and_smem = false;
  }
  return buffer_params;
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
    const int64_t hp_threads_per_block_min,
    const int64_t hp_threads_per_block_max,
    const bool project_to_input,
    const PrimDataType index_type) {
  auto rparams = std::make_unique<ReductionParams>(
      InnerOuterPersistentKernelScheduler::schedulerType());
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;
  // Parameters for inner reduction:
  // Reduction dim: inner_vect, inner_batch, bdimx and bdimy
  // Iteration dim: gdimy

  // Parameters for outer reduction:
  // Reduction dim: bdimy
  // Iteration dim: vectorization_factor_outer, bdimx, gdimy
  struct InnerOuterParams {
    int64_t inner_vect = -1;
    int64_t inner_batch = -1;
    int64_t bdimx = -1;
    int64_t bdimy = -1;
    int64_t bdimz = -1;
    int64_t gdimy = -1;
    int64_t tmp_gmem_write_vect = -1;
    int64_t vectorization_factor_outer = -1;
    int64_t threads_per_block = -1;
    // derived metrics for sorting
    int64_t warps_per_sm = -1;
    int64_t required_register_per_thread = -1;
    int64_t available_register_per_thread = -1;

    void verify() {
      NVF_ERROR(inner_vect != -1, "inner_vect is not set.");
      NVF_ERROR(inner_batch != -1, "inner_batch is not set.");
      NVF_ERROR(bdimx != -1, "bdimx is not set.");
      NVF_ERROR(bdimy != -1, "bdimy is not set.");
      NVF_ERROR(gdimy != -1, "gdimy is not set.");
      NVF_ERROR(tmp_gmem_write_vect != -1, "tmp_gmem_write_vect is not set.");
      NVF_ERROR(
          vectorization_factor_outer != -1,
          "vectorization_factor_outer is not set.");
    }
    std::string toString() const {
      std::stringstream ss;
      ss << "inner_vect: " << inner_vect << ", inner_batch: " << inner_batch
         << ", bdimx: " << bdimx << ", bdimy: " << bdimy << ", bdimz: " << bdimz
         << ", gdimy: " << gdimy
         << ", tmp_gmem_write_vect: " << tmp_gmem_write_vect
         << ", vectorization_factor_outer: " << vectorization_factor_outer
         << ", threads_per_block: " << threads_per_block
         << ", warps_per_sm: " << warps_per_sm
         << ", required_register_per_thread: " << required_register_per_thread
         << ", available_register_per_thread: "
         << available_register_per_thread;
      return ss.str();
    }
  };

  // Set a minimum workload for each thread to take advantage of low
  // intra-threads communication cost.
  // Tuned for layer_norm backward on A100, still works fine on H100.
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

  // Estimate register usage per thread based on buffer size.
  // Assuming a constant register overhead for non-buffer related usage,
  // and all the register buffers are stored in registers.
  auto getEstimatedRegisterUsage = [&](int64_t batch_mul_vect) {
    int64_t persistent_buffer_size =
        regs_buffer_size / inner_dim_numel * batch_mul_vect;
    int64_t estimated_register_count =
        persistent_buffer_size / scheduler_utils::bytes_per_register +
        scheduler_utils::register_overhead;
    return std::min(
        estimated_register_count, scheduler_utils::max_registers_per_thread);
  };

  // Estimate max blocks per sm based on register and shared memory usage.
  auto getBlocksPerSM = [&](const int64_t threads_per_sm,
                            const int64_t threads_per_block,
                            const int64_t warp_size) {
    // check register limitation on blocks per sm
    constexpr int64_t warp_allocation_granularity = 4;
    const int64_t allocated_warps_per_block =
        ceilDiv(
            ceilDiv(threads_per_block, warp_size),
            warp_allocation_granularity) *
        warp_allocation_granularity;
    int64_t max_blocks_per_sm_regs = scheduler_utils::safeDiv(
        threads_per_sm / warp_size, allocated_warps_per_block);
    // check shared memory limitation on blocks per sm
    int64_t max_blocks_per_sm_smem =
        (int64_t)dev_prop->sharedMemPerMultiprocessor /
        (smem_overhead + smem_buffer_size);
    return std::min(max_blocks_per_sm_regs, max_blocks_per_sm_smem);
  };

  // In the inner reduction part of the kernel, gdimy is used to parallelize the
  // outer dimension. The kernel is a cooperative kernel, so the number of
  // blocks should be as large as possible to achieve a high occupancy unless
  // outer dim is too small which may lead large workload for the final outer
  // reduction. So, gdimy is drvied from the number of blocks per sm and limited
  // to ensure at least 8 rows per block.
  // TODO: re-evaluate this 8 rows per block requirement.
  auto getGdimy = [&](int64_t inner_vect,
                      int64_t threads_per_block,
                      int64_t inner_batch) {
    int64_t reg_per_thread =
        getEstimatedRegisterUsage(inner_vect * inner_batch);
    int64_t threads_per_sm = getThreadsPerSMGivenRegPerThread(reg_per_thread);
    int64_t blocks_per_sm =
        getBlocksPerSM(threads_per_sm, threads_per_block, dev_prop->warpSize);
    int64_t gdimy = blocks_per_sm * device_multiprocessor_count;
    const int64_t outer_iter_min = 8;
    const int64_t gdimy_max = scheduler_utils::roundUpToN(
        ceilDiv(outer_dim_numel, outer_iter_min), device_multiprocessor_count);
    while (gdimy > gdimy_max && blocks_per_sm > 1) {
      blocks_per_sm -= 1;
      gdimy = blocks_per_sm * device_multiprocessor_count;
    }
    return gdimy;
  };

  // The inner reduction part of the kernel also does a partial outer reduction
  // and stores the partial results in tmp gmem and then reloaded to finish the
  // outer reduciton. This function set the vectorization factor for write and
  // and read of the partial outer reduction result.
  // For write to tmp gmem, follows vectorization factor of inner reduction
  //                        but don't exceed 16 bytes.
  // For read from tmp gmem, since the paralelization is changed, a different
  //                         vectorization factor is used to optimize the
  //                         number of reaductions per thread.
  auto getOuterReductionBufferVectFactor = [&](int64_t inner_vect) {
    constexpr int64_t max_gmem_vect_access_bytes = 16;
    const int64_t max_tmp_gmem_vect_factor = std::min(
        max_gmem_vect_access_bytes / (int64_t)tmp_gmem_dtype_size, inner_vect);
    int64_t tmp_gmem_write_vect = max_tmp_gmem_vect_factor;
    const int64_t workload_per_thread = inner_dim_numel >= 4096 ? 4l : 2l;
    int64_t vectorization_factor_outer =
        std::min(workload_per_thread, max_tmp_gmem_vect_factor);
    return std::make_pair(tmp_gmem_write_vect, vectorization_factor_outer);
  };

  // In the outer reduction part of the kernel, inner and outer dims are
  // parallelized as:
  // --- inner dim: vect, bdimx, gdimy ----
  // --- outer dim: bdimy -----------------
  // This function splits the threads_per_block into bdimx and bdimy using:
  // bdimx = ceilDiv(inner_dim_numel / vect, gdimy)
  // bdimy = threads_per_block / bdimx
  auto getBdimxBdimy = [&](int64_t threads_per_block,
                           int64_t vectorization_factor_outer,
                           int64_t gdimy) {
    // For widely used hidden sizes, threads_per_block has factor of 8, roundup
    // to increase the probability of bdimx * bdimy == threads_per_block.
    int64_t bdimx = scheduler_utils::roundUpPow2Or8(
        ceilDiv(inner_dim_numel / vectorization_factor_outer, gdimy));
    // if still not divisible, e.g. threads_per_block = 256, bdimx = 40.
    // increase bdimx to make it divisible. Under worst case, bdimx equals to
    // threads_per_block.
    while (threads_per_block % bdimx) {
      bdimx = std::min(bdimx + 8, threads_per_block);
    }
    // Set OuterParams Reduction dim: bdimy.
    int64_t bdimy = threads_per_block / bdimx;
    NVF_ERROR(
        bdimy * bdimx == threads_per_block,
        " threads_per_block must be divisible by bdimx and bdimy.");
    return std::make_pair(bdimx, bdimy);
  };

  // Get the heuristics given vectorization factor and threads per block
  auto getHeuristicsGivenVectThreads = [&](int64_t vect_factor,
                                           int64_t threads_per_block) {
    InnerOuterParams iop;
    // (1) inner reduction
    // Reduction dim: inner_batch, threads_per_block, vect_factor
    // Iteration dim: gdimy
    iop.inner_vect = vect_factor;
    iop.threads_per_block = threads_per_block;
    iop.inner_batch =
        ceilDiv(inner_dim_numel / iop.inner_vect, iop.threads_per_block);
    iop.gdimy =
        getGdimy(iop.inner_vect, iop.threads_per_block, iop.inner_batch);
    // (2) outer reduction
    // Iteration dim: gdimy, bdimx, vectorization_factor_outer
    // Reduction dim: bdimy
    std::tie(iop.tmp_gmem_write_vect, iop.vectorization_factor_outer) =
        getOuterReductionBufferVectFactor(iop.inner_vect);
    auto [bdimx, bdimy] = getBdimxBdimy(
        threads_per_block, iop.vectorization_factor_outer, iop.gdimy);
    iop.bdimx = bdimx;
    iop.bdimy = bdimy;
    // (3) Derived metrics warps_per_sm and register usage for sorting
    iop.warps_per_sm = ceilDiv(iop.threads_per_block, dev_prop->warpSize) *
        iop.gdimy / device_multiprocessor_count;
    iop.available_register_per_thread =
        getRegPerThreadGivenThreadsPerSM(dev_prop->warpSize * iop.warps_per_sm);
    iop.required_register_per_thread =
        getEstimatedRegisterUsage(iop.inner_vect * iop.inner_batch);
    return iop;
  };

  // Use the maximum vectorization factor
  const int64_t vect_factor = (int64_t)vectorize_factor;

  // Set a reasonable range for threads per block based on the number of
  // elements in the inner dimension after vectorization.
  // Start from 128 or a smaller number if inner dim is small.
  const int64_t after_vect = inner_dim_numel / vect_factor;
  const int64_t batch_min = getMinimumBatch();
  int64_t threads_per_block_min = hp_threads_per_block_min;
  threads_per_block_min = std::min(threads_per_block_min, after_vect);
  threads_per_block_min = scheduler_utils::roundUpPow2(threads_per_block_min);

  // star max threads per block from min threads per block
  int64_t threads_per_block_max = threads_per_block_min;
  // increase to cover the whole inner dim
  threads_per_block_max =
      std::max(threads_per_block_max, ceilDiv(after_vect, batch_min));
  // round up to power of 2
  threads_per_block_max = scheduler_utils::roundUpPow2(threads_per_block_max);
  // don't go beyond the maximum threads per block
  threads_per_block_max =
      std::min(threads_per_block_max, hp_threads_per_block_max);

  // Store all the possible heuristics based on different threads per block.
  // Vectorizaton is fixed at the maximum value.
  std::vector<InnerOuterParams> iop_candidates;
  for (auto threads_per_block = threads_per_block_max;
       threads_per_block >= threads_per_block_min;
       threads_per_block /= 2) {
    iop_candidates.emplace_back(
        getHeuristicsGivenVectThreads(vect_factor, threads_per_block));
  }

  // Sort the heuristics based on the register usage and occupancy.
  std::stable_sort(
      iop_candidates.begin(),
      iop_candidates.end(),
      [](const InnerOuterParams& a, const InnerOuterParams& b) {
        // If a thread can use more registers than required, there is a high
        // chance that it can avoid register spilling and compiler can optimize
        // for better instruction level parallelism.
        int64_t extra_regs_a =
            a.available_register_per_thread - a.required_register_per_thread;
        int64_t extra_regs_b =
            b.available_register_per_thread - b.required_register_per_thread;
        if (extra_regs_a > 0 && extra_regs_b < 0) {
          return true;
        } else if (extra_regs_a < 0 && extra_regs_b > 0) {
          return false;
        }
        // High occupancy provides better threads level parallelism.
        // 25% is sufficient since ILP is high due to persistent batch sizes
        // which is equivalent to unrolling inner dim.
        if (a.warps_per_sm != b.warps_per_sm &&
            (a.warps_per_sm < 16 || b.warps_per_sm < 16)) {
          return a.warps_per_sm > b.warps_per_sm;
        }
        // Tie breaker, smaller threads_per_block to reduce communication
        // overhead
        return a.threads_per_block < b.threads_per_block;
      });

  // Pick the best heuristic
  auto iop = iop_candidates.front();

  // Special case, when inner_dim_numel <= 1024, bdimx is usually small
  // after divide by inner_vect and inner_batch. In this case, bdimy is used to
  // parallelize outer_dim instead of inner_dim. This pattern is named multi
  // reductions per block (mrpb).
  if (inner_dim_numel <= 1024) {
    rparams->multiple_reds_per_blk = true;
    rparams->tidx_for_outer_reduction = true;

    // Step-1, InnerParams, Reduction dim: inner_vect(reuse),
    // inner_batch(reuse), bdimx
    iop.bdimx = ceilDiv(inner_dim_numel, iop.inner_vect * iop.inner_batch);

    // Step-2, InnerParams, Iteration dim: gdimy, bdimy (in next step)
    iop.gdimy = getGdimy(iop.inner_vect, iop.bdimx, iop.inner_batch);

    // Step-3, OuterParams, Iteration dim: vectorization_factor_outer(reuse),
    // bdimy, gdimy (in previous step).
    // WAR for https://github.com/NVIDIA/Fuser/issues/3428
    iop.bdimy = 1;

    // Step-4, OuterParams, Reduction dim: bdimx (already done)
    iop.warps_per_sm = ceilDiv(iop.bdimx * iop.bdimy, dev_prop->warpSize) *
        iop.gdimy / device_multiprocessor_count;
    iop.available_register_per_thread =
        getRegPerThreadGivenThreadsPerSM(dev_prop->warpSize * iop.warps_per_sm);

    if (iop.bdimx % dev_prop->warpSize == 0) {
      rparams->pad_inner_reduction_to_warp = true;
      rparams->pad_outer_reduction_to_warp = true;
    }
    rparams->block_dim_iter_dom = ParallelType::TIDy;
    rparams->combined_split_grid_inner_dim =
        iop.vectorization_factor_outer * iop.bdimy * iop.gdimy <
        inner_dim_numel;
  } else {
    rparams->block_dim_inner_reduction_extra = ParallelType::TIDy;
    rparams->combined_split_grid_inner_dim =
        iop.vectorization_factor_outer * iop.bdimx * iop.gdimy <
        inner_dim_numel;
    rparams->static_bdimx = true;
    rparams->static_bdimy = true;
    iop.bdimz = ceilDiv(
        ceilDiv(
            ceilDiv(inner_dim_numel / iop.inner_vect, iop.bdimx), iop.bdimy),
        iop.inner_batch);
    NVF_ERROR(iop.bdimz == 1, "bdimz must be 1.");
  }

  // check all the parameters in InnerOuterParams are set.
  iop.verify();

  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;
  // tmp_gmem is the intermediate result of outer reduction, its dtype is float,
  // so the maximum vectorization factor is 4.
  rparams->vectorization_factor_outer = iop.vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = iop.tmp_gmem_write_vect;
  rparams->cparams.maxrregcount = iop.available_register_per_thread;
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

std::unique_ptr<ReductionParams> innerOuterWarpSpecializedTmaHeuristic(
    const int64_t outer_dim_numel,
    const int64_t inner_dim_numel,
    const int64_t regs_buffer_size,
    const int64_t smem_buffer_size,
    const int64_t smem_overhead,
    const size_t tmp_gmem_dtype_size,
    const size_t vectorize_factor,
    const int64_t hp_threads_per_block_min,
    const int64_t hp_threads_per_block_max,
    const bool project_to_input,
    const PrimDataType index_type) {
  auto rparams = std::make_unique<ReductionParams>(
      InnerOuterPersistentKernelScheduler::schedulerType());
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;
  // Parameters for inner reduction:
  // Reduction dim: inner_vect, inner_batch, bdimx and bdimy
  // Iteration dim: gdimy

  // Parameters for outer reduction:
  // Reduction dim: bdimy
  // Iteration dim: vectorization_factor_outer, bdimx, gdimy
  struct InnerOuterParams {
    int64_t inner_vect = -1;
    int64_t inner_batch = -1;
    int64_t bdimx = -1;
    int64_t bdimy = -1;
    int64_t bdimz = -1;
    int64_t gdimy = -1;
    int64_t tmp_gmem_write_vect = -1;
    int64_t vectorization_factor_outer = -1;
    int64_t threads_per_block = -1;
    // derived metrics for sorting
    int64_t warps_per_sm = -1;
    int64_t required_register_per_thread = -1;
    int64_t available_register_per_thread = -1;

    void verify() {
      NVF_ERROR(inner_vect != -1, "inner_vect is not set.");
      NVF_ERROR(inner_batch != -1, "inner_batch is not set.");
      NVF_ERROR(bdimx != -1, "bdimx is not set.");
      NVF_ERROR(bdimy != -1, "bdimy is not set.");
      NVF_ERROR(gdimy != -1, "gdimy is not set.");
      NVF_ERROR(tmp_gmem_write_vect != -1, "tmp_gmem_write_vect is not set.");
      NVF_ERROR(
          vectorization_factor_outer != -1,
          "vectorization_factor_outer is not set.");
    }
    std::string toString() const {
      std::stringstream ss;
      ss << "inner_vect: " << inner_vect << ", inner_batch: " << inner_batch
         << ", bdimx: " << bdimx << ", bdimy: " << bdimy << ", bdimz: " << bdimz
         << ", gdimy: " << gdimy
         << ", tmp_gmem_write_vect: " << tmp_gmem_write_vect
         << ", vectorization_factor_outer: " << vectorization_factor_outer
         << ", threads_per_block: " << threads_per_block
         << ", warps_per_sm: " << warps_per_sm
         << ", required_register_per_thread: " << required_register_per_thread
         << ", available_register_per_thread: "
         << available_register_per_thread;
      return ss.str();
    }
  };

  // Set a minimum workload for each thread to take advantage of low
  // intra-threads communication cost.
  // Tuned for layer_norm backward on A100, still works fine on H100.
  auto get_minimum_batch = [&]() -> int64_t {
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

  // Estimate register usage per thread based on buffer size.
  // Assuming a constant register overhead for non-buffer related usage,
  // and all the register buffers are stored in registers.
  auto get_estimated_register_usage = [&](int64_t batch_mul_vect) {
    int64_t persistent_buffer_size =
        regs_buffer_size / inner_dim_numel * batch_mul_vect;
    int64_t estimated_register_count =
        persistent_buffer_size / scheduler_utils::bytes_per_register +
        scheduler_utils::register_overhead;
    return std::min(
        estimated_register_count, scheduler_utils::max_registers_per_thread);
  };

  // The inner reduction part of the kernel also does a partial outer reduction
  // and stores the partial results in tmp gmem and then reloaded to finish the
  // outer reduciton. This function set the vectorization factor for write and
  // and read of the partial outer reduction result.
  // For write to tmp gmem, follows vectorization factor of inner reduction
  //                        but don't exceed 16 bytes.
  // For read from tmp gmem, since the paralelization is changed, a different
  //                         vectorization factor is used to optimize the
  //                         number of reaductions per thread.
  auto get_outer_reduction_buffer_vect_factor = [&](int64_t inner_vect) {
    constexpr int64_t max_gmem_vect_access_bytes = 16;
    const int64_t max_tmp_gmem_vect_factor = std::min(
        max_gmem_vect_access_bytes / (int64_t)tmp_gmem_dtype_size, inner_vect);
    int64_t tmp_gmem_write_vect = max_tmp_gmem_vect_factor;
    const int64_t workload_per_thread = inner_dim_numel >= 4096 ? 4l : 2l;
    int64_t vectorization_factor_outer =
        std::min(workload_per_thread, max_tmp_gmem_vect_factor);
    return std::make_pair(tmp_gmem_write_vect, vectorization_factor_outer);
  };

  // In the outer reduction part of the kernel, inner and outer dims are
  // parallelized as:
  // --- inner dim: vect, bdimx, gdimy ----
  // --- outer dim: bdimy -----------------
  // This function splits the threads_per_block into bdimx and bdimy using:
  // bdimx = ceilDiv(inner_dim_numel / vect, gdimy)
  // bdimy = threads_per_block / bdimx
  auto get_bdimx_bdimy = [&](int64_t threads_per_block,
                             int64_t vectorization_factor_outer,
                             int64_t gdimy) {
    // For widely used hidden sizes, threads_per_block has factor of 8, roundup
    // to increase the probability of bdimx * bdimy == threads_per_block.
    int64_t bdimx = scheduler_utils::roundUpPow2Or8(
        ceilDiv(inner_dim_numel / vectorization_factor_outer, gdimy));
    // if still not divisible, e.g. threads_per_block = 256, bdimx = 40.
    // increase bdimx to make it divisible. Under worst case, bdimx equals to
    // threads_per_block.
    while (threads_per_block % bdimx) {
      bdimx = std::min(bdimx + 8, threads_per_block);
    }
    // Set OuterParams Reduction dim: bdimy.
    int64_t bdimy = threads_per_block / bdimx;
    NVF_ERROR(
        bdimy * bdimx == threads_per_block,
        " threads_per_block must be divisible by bdimx and bdimy.");
    return std::make_pair(bdimx, bdimy);
  };

  // Get the heuristics given vectorization factor and threads per block
  auto get_heuristics_given_vect_threads = [&](int64_t vect_factor,
                                               int64_t threads_per_block) {
    InnerOuterParams iop;
    // (1) inner reduction
    // Reduction dim: inner_batch, threads_per_block, vect_factor
    // Iteration dim: gdimy
    iop.inner_vect = vect_factor;
    iop.threads_per_block = threads_per_block;
    iop.inner_batch =
        ceilDiv(inner_dim_numel / iop.inner_vect, iop.threads_per_block);
    iop.gdimy = device_multiprocessor_count;

    // (2) outer reduction
    // Iteration dim: gdimy, bdimx, vectorization_factor_outer
    // Reduction dim: bdimy
    std::tie(iop.tmp_gmem_write_vect, iop.vectorization_factor_outer) =
        get_outer_reduction_buffer_vect_factor(iop.inner_vect);
    auto [bdimx, bdimy] = get_bdimx_bdimy(
        threads_per_block, iop.vectorization_factor_outer, iop.gdimy);
    iop.bdimx = bdimx;
    iop.bdimy = bdimy;
    // (3) Derived metrics warps_per_sm and register usage for sorting
    iop.warps_per_sm = ceilDiv(iop.threads_per_block, dev_prop->warpSize) *
        iop.gdimy / device_multiprocessor_count;
    iop.available_register_per_thread =
        getRegPerThreadGivenThreadsPerSM(dev_prop->warpSize * iop.warps_per_sm);
    iop.required_register_per_thread =
        get_estimated_register_usage(iop.inner_vect * iop.inner_batch);
    return iop;
  };

  // Use the maximum vectorization factor
  const int64_t vect_factor = (int64_t)vectorize_factor;

  // Set a reasonable range for threads per block based on the number of
  // elements in the inner dimension after vectorization.
  // Start from 128 or a smaller number if inner dim is small.
  const int64_t after_vect = inner_dim_numel / vect_factor;
  const int64_t batch_min = get_minimum_batch();
  int64_t threads_per_block_min = hp_threads_per_block_min;
  threads_per_block_min = std::min(threads_per_block_min, after_vect);
  threads_per_block_min = scheduler_utils::roundUpPow2(threads_per_block_min);

  // star max threads per block from min threads per block
  int64_t threads_per_block_max = threads_per_block_min;
  // increase to cover the whole inner dim
  threads_per_block_max =
      std::max(threads_per_block_max, ceilDiv(after_vect, batch_min));
  // round up to power of 2
  threads_per_block_max = scheduler_utils::roundUpPow2(threads_per_block_max);
  // don't go beyond the maximum threads per block
  threads_per_block_max =
      std::min(threads_per_block_max, hp_threads_per_block_max);

  // Store all the possible heuristics based on different threads per block.
  // Vectorizaton is fixed at the maximum value.
  std::vector<InnerOuterParams> iop_candidates;
  for (auto threads_per_block = threads_per_block_max;
       threads_per_block >= threads_per_block_min;
       threads_per_block /= 2) {
    iop_candidates.emplace_back(
        get_heuristics_given_vect_threads(vect_factor, threads_per_block));
  }

  // Sort the heuristics based on the register usage and occupancy.
  std::stable_sort(
      iop_candidates.begin(),
      iop_candidates.end(),
      [](const InnerOuterParams& a, const InnerOuterParams& b) {
        // If a thread can use more registers than required, there is a high
        // chance that it can avoid register spilling and compiler can optimize
        // for better instruction level parallelism.
        int64_t extra_regs_a =
            a.available_register_per_thread - a.required_register_per_thread;
        int64_t extra_regs_b =
            b.available_register_per_thread - b.required_register_per_thread;
        if (extra_regs_a > 0 && extra_regs_b < 0) {
          return true;
        } else if (extra_regs_a < 0 && extra_regs_b > 0) {
          return false;
        }
        // High occupancy provides better threads level parallelism.
        // 25% is sufficient since ILP is high due to persistent batch sizes
        // which is equivalent to unrolling inner dim.
        if (a.warps_per_sm != b.warps_per_sm &&
            (a.warps_per_sm < 16 || b.warps_per_sm < 16)) {
          return a.warps_per_sm > b.warps_per_sm;
        }
        // Tie breaker, smaller threads_per_block to reduce communication
        // overhead
        return a.threads_per_block < b.threads_per_block;
      });

  // Pick the best heuristic
  auto iop = iop_candidates.front();
  rparams->block_dim_inner_reduction_extra = ParallelType::TIDy;
  rparams->combined_split_grid_inner_dim =
      iop.vectorization_factor_outer * iop.bdimx * iop.gdimy < inner_dim_numel;
  rparams->static_bdimx = true;
  rparams->static_bdimy = true;
  iop.bdimz = ceilDiv(
      ceilDiv(ceilDiv(inner_dim_numel / iop.inner_vect, iop.bdimx), iop.bdimy),
      iop.inner_batch);
  NVF_ERROR(iop.bdimz == 1, "bdimz must be 1.");

  // check all the parameters in InnerOuterParams are set.
  iop.verify();

  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;
  // tmp_gmem is the intermediate result of outer reduction, its dtype is float,
  // so the maximum vectorization factor is 4.
  rparams->vectorization_factor_outer = iop.vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = iop.tmp_gmem_write_vect;
  rparams->cparams.maxrregcount = iop.available_register_per_thread;
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

  rparams->tag = "TMA Warp Specialized Persistent Heuristic.\n";

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

  auto scheduler_hyperparameters_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::SchedulerHyperParameters>(
          data_cache, [&]() {
            return std::make_unique<scheduler_utils::SchedulerHyperParameters>(
                /*vectorize_factor=*/vectorize_factor,
                /*unroll_factor=*/1,
                /*threads_per_block_min=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_min,
                /*threads_per_block_max=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_max);
          });
  scheduler_utils::SchedulerHyperParameters& hp =
      scheduler_hyperparameters_entry.get();

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();
  NVF_ERROR(
      !persistent_buffer_info.persistent_buffers.empty(),
      "Persistent scheduler requires persistent buffers.");
  auto buffer_params = getPersistentBufferStorageParams(
      fusion,
      runtime_info,
      data_cache,
      reduction_tvs,
      hp.vectorize_factor,
      hp.threads_per_block_min,
      hp.threads_per_block_max);

  std::unique_ptr<ReductionParams> rparams;

  // Ultimately, we want the heuristic to decide between using the
  // warp-specialized version or the multi-wave version. The enable option is a
  // temporary configuration to facilitate testing during development without
  // disrupting existing behavior.
  if (isOptionEnabled(EnableOption::WarpSpecializedNormalization)) {
    rparams = innerOuterWarpSpecializedTmaHeuristic(
        properties.total_iteration_numel,
        properties.total_reduction_numel,
        buffer_params.regs_buffer_size,
        buffer_params.smem_buffer_size,
        buffer_params.smem_overhead,
        max_outer_reduction_dtype_size,
        hp.vectorize_factor,
        hp.threads_per_block_min,
        hp.threads_per_block_max,
        buffer_params.project_to_input,
        runtime_info.getIndexType());
    rparams->tma_warp_specialized = true;
  } else {
    rparams = innerOuterPersistentHeuristic(
        properties.total_iteration_numel,
        properties.total_reduction_numel,
        buffer_params.regs_buffer_size,
        buffer_params.smem_buffer_size,
        buffer_params.smem_overhead,
        max_outer_reduction_dtype_size,
        hp.vectorize_factor,
        hp.threads_per_block_min,
        hp.threads_per_block_max,
        buffer_params.project_to_input,
        runtime_info.getIndexType());
  }

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
    // Similar to the inner reduction, we need to reorder the outer reduction tv
    // when there are view operations.
    if (!ir_utils::getViewOps(fusion).empty()) {
      // Reorder reference_tv after propagating the view operation. This will
      // reorder for better merging.
      outer_reduction_tv->reorder(
          scheduler_utils::domainReorderAsLogicalMap(outer_reduction_tv));
    }

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
      outer_reduction_tv->split(0, rparams->lparams.bdimy());
      outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);

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

  const bool is_unroll_or_vectorization = rparams->isUnrolled();
  const bool is_vectorize =
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
  const auto& unroll_vectorizable_cached_tvs =
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          inner_reference_tv, is_vectorize, cached_inputs, cached_outputs);
  reduction_scheduler_utils::propagateParallelization(
      inner_reduction_tvs[0],
      inner_reference_tv,
      is_unroll_or_vectorization,
      is_outer_grid_persistence,
      inner_reduction_tvs,
      unroll_vectorizable_cached_tvs,
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  // Propagate outer reduction. Each outer reduction is connected with its
  // cached_gmem and output, since we added all the cached_gmem to the
  // boundaryNodesSet, the transformation from one outer reduction can't
  // propagate to other outer reductions due to the cutoff at
  // boundaryNodesSet. Thus, we need a loop to initiate the propagation from
  // each outer reduction. Don't allow parallelization propagation goes
  // through cached_gmem, see issue 246.
  for (long unsigned int i = 0; i < outer_reference_tvs.size(); i++) {
    const auto& selected_tvs_outer = scheduler_utils::getAllTvsFrom(
        {outer_reduction_tvs[i]}, {cached_gmem[i]});
    reduction_scheduler_utils::propagateTransformation(
        outer_reference_tvs[i], boundaryNodesSet);
    const auto& unroll_vectorizable_cached_tvs =
        reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
            outer_reference_tvs[i],
            is_vectorize,
            cached_inputs,
            cached_outputs);
    reduction_scheduler_utils::propagateParallelization(
        outer_reduction_tvs[i],
        outer_reference_tvs[i],
        is_unroll_or_vectorization,
        is_outer_grid_persistence,
        outer_reduction_tvs,
        unroll_vectorizable_cached_tvs,
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

  // Needs special handling of vectorized loading from shared memory due to
  // potential different data types of inputs and shared memory tensor.
  if (is_vectorize) {
    reduction_scheduler_utils::sharedMemoryConsumerVectorization(
        smem_consumers, rparams->unroll_factor_inner_reduction);
  }

  // Remove dummy outputs as they can inadvertently affect CA positions
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }
  inlineMost();
}

void scheduleTmaWarpSpecializedOuter(
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
    // Similar to the inner reduction, we need to reorder the outer reduction tv
    // when there are view operations.
    if (!ir_utils::getViewOps(fusion).empty()) {
      // Reorder reference_tv after propagating the view operation. This will
      // reorder for better merging.
      outer_reduction_tv->reorder(
          scheduler_utils::domainReorderAsLogicalMap(outer_reduction_tv));
    }

    // merge tensorview to [reduction, iteraiton] domains
    mergeReductionOrIterDomains(outer_reduction_tv, true);
    mergeReductionOrIterDomains(outer_reduction_tv, false);

    // First-stage of outer reduction
    outer_reduction_tv->split(0, rparams->lparams.gdimy());

    TensorView* partialResult = outer_reduction_tv->rFactor({0});
    partialResult->cacheBefore();
    partialResult->setMemoryType(MemoryType::Global);
    TensorView* partialResultReload = partialResult->cacheAfter();

    boundaryNodesSet.insert(partialResultReload);
    cached_gmem.emplace_back(partialResult);
    cached_gmem_reload.emplace_back(partialResultReload);

    // Second-stage of outer reduction
    // reduction domain, [I1/TIDy, TIDy]
    outer_reduction_tv->split(0, rparams->lparams.bdimy());
    outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
    // iteration domain, [BIDy, TIDx, Vect]
    int axisID = -1;
    if (rparams->vectorization_factor_outer > 1) {
      outer_reduction_tv->split(axisID, rparams->vectorization_factor_outer);
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::Vectorize);
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

    auto outer_reference_tv =
        reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
    outer_reference_tvs.emplace_back(outer_reference_tv);
  }
}

void scheduleTmaWarpSpecializedInnerOuter(
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
  scheduleTmaWarpSpecializedOuter(
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

  // Collect tvs loaded with TMA, they require special scheduling.
  std::vector<TensorView*> tma_load_tvs;
  if (rparams->tma_warp_specialized) {
    for (auto tv : smem_consumers) {
      auto smem_tv = ir_utils::getSoleProducerTv(tv);
      if (std::find(tma_load_tvs.begin(), tma_load_tvs.end(), smem_tv) ==
          tma_load_tvs.end()) {
        tma_load_tvs.emplace_back(smem_tv);
      }
    }
  }

  const bool is_unroll_or_vectorization = rparams->isUnrolled();
  const bool is_vectorize =
      rparams->vectorize_inner_reduction || rparams->vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams->persistent_kernel &&
      rparams->cross_grid_inner_reduction && !rparams->fastest_dim;

  // Propagate transformations for inner reduction.
  // Two steps are used since tma tvs are scheduled differently.
  // Step-1, propagate iteration domain in inner reduction.
  // Step-2, propagate reduction domain in inner reduction.
  if (rparams->tma_warp_specialized) {
    // Find the axis that splits the reduction domain and iteration domain.
    int first_redu_axis = -1;
    int n_dims = (int)inner_reference_tv->nDims();
    for (auto i = 0; i < n_dims; i++) {
      if (inner_reference_tv->axis(i)->isReduction() ||
          inner_reference_tv->axis(i)->isRFactorProduct()) {
        first_redu_axis = i;
        break;
      }
    }

    // Step-1, propagate iteration domain in inner reduction.
    // outer_reference_tvs are excluded since they are already scheduled
    // with a different pattern for the final step of outer reduciton.
    if (first_redu_axis > 0) {
      TransformPropagator propagator(inner_reference_tv, first_redu_axis - 1);
      std::vector<TensorView*> all_tvs_except = ir_utils::allTvsExcept(
          fusion, {outer_reference_tvs.begin(), outer_reference_tvs.end()});
      SetSelector selector({all_tvs_except.begin(), all_tvs_except.end()});
      MaxLogicalDomainInfoSpanningTree(inner_reference_tv, &selector)
          .traverse(&propagator);
    }

    // Step-2, propagate reduction domain in inner reduction.
    // (a) Tvs in boundaryNodesSet are excluded since they should follow outer
    // reduction pattern.
    // (b) TMA tvs are excluded since they require special scheduling.
    // (3) Excluding tma tvs breaks the propagation path from inner reduction tv
    // to cached_gmem which stores the results of the first-stage of outer
    // reduction. The solution is adding a dummy output to link them. The same
    // trick is used when projecting persistent buffers to inputs.
    auto inner_reduction_input =
        ir_utils::getSoleProducerTv(inner_reference_tv);
    for (auto tv : cached_gmem) {
      // T1(smem) --> T2 (l) --> T3 = OuterRedu(T2) --> T4(cached_gmem)
      // outer_reduction_input: T2
      // partial_outer_redu_tv: T3
      auto partial_outer_redu_tv = ir_utils::getSoleProducerTv(tv);
      auto outer_reduction_input =
          ir_utils::getSoleProducerTv(partial_outer_redu_tv);
      auto dummy_output = add(inner_reduction_input, outer_reduction_input);
      fusion->addOutput(dummy_output);
      dummy_outputs.emplace_back(dummy_output);
    }

    // Tvs requiring special scheduling
    std::unordered_set<TensorView*> special_tvs{
        tma_load_tvs.begin(), tma_load_tvs.end()};
    for (auto tv : boundaryNodesSet) {
      if (special_tvs.count(tv) == 0) {
        special_tvs.emplace(tv);
      }
    }
    TransformPropagator propagator(inner_reference_tv);
    std::vector<TensorView*> all_tvs_except_cache = ir_utils::allTvsExcept(
        fusion, {special_tvs.begin(), special_tvs.end()});
    SetSelector selector(
        {all_tvs_except_cache.begin(), all_tvs_except_cache.end()});
    MaxLogicalDomainInfoSpanningTree(inner_reference_tv, &selector)
        .traverse(&propagator);
  } else {
    reduction_scheduler_utils::propagateTransformation(
        inner_reference_tv, boundaryNodesSet);
  }
  reduction_scheduler_utils::propagateRFactor(
      inner_reference_tv, inner_reduction_tvs[0], inner_reduction_tvs);

  // parallelization propagation
  const auto& selected_tvs_inner =
      scheduler_utils::getAllTvsFrom(inner_reduction_tvs, boundaryNodesSet);
  const auto& unroll_vectorizable_cached_tvs =
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          inner_reference_tv, is_vectorize, cached_inputs, cached_outputs);
  reduction_scheduler_utils::propagateParallelization(
      inner_reduction_tvs[0],
      inner_reference_tv,
      is_unroll_or_vectorization,
      is_outer_grid_persistence,
      inner_reduction_tvs,
      unroll_vectorizable_cached_tvs,
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  // Propagate outer reduction. Each outer reduction is connected with its
  // cached_gmem and output, since we added all the cached_gmem to the
  // boundaryNodesSet, the transformation from one outer reduction can't
  // propagate to other outer reductions due to the cutoff at
  // boundaryNodesSet. Thus, we need a loop to initiate the propagation from
  // each outer reduction. Don't allow parallelization propagation goes
  // through cached_gmem, see issue 246.
  for (long unsigned int i = 0; i < outer_reference_tvs.size(); i++) {
    const auto& selected_tvs_outer = scheduler_utils::getAllTvsFrom(
        {outer_reduction_tvs[i]}, {cached_gmem[i]});
    reduction_scheduler_utils::propagateTransformation(
        outer_reference_tvs[i], boundaryNodesSet);
    const auto& unroll_vectorizable_cached_tvs =
        reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
            outer_reference_tvs[i],
            is_vectorize,
            cached_inputs,
            cached_outputs);
    reduction_scheduler_utils::propagateParallelization(
        outer_reduction_tvs[i],
        outer_reference_tvs[i],
        is_unroll_or_vectorization,
        is_outer_grid_persistence,
        outer_reduction_tvs,
        unroll_vectorizable_cached_tvs,
        {selected_tvs_outer.begin(), selected_tvs_outer.end()});
  }

  // Up to this point, the outer dimension of the TMA tv is scheduled
  // the same way as the inner reduction tv. However, the inner dimension
  // has not been scheduled yet. Since 1D TMA allows unrestricted load size,
  // we can simply parallelize the entire inner dimension using bulk.
  // Example: 2D tensor, [BIDy, S, | Bulk]
  // Example: 1D tensor, [Bulk]
  if (rparams->tma_warp_specialized) {
    for (auto tv : tma_load_tvs) {
      tv->axis(-1)->parallelize(ParallelType::Bulk);
    }
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

  // Needs special handling of vectorized loading from shared memory due to
  // potential different data types of inputs and shared memory tensor.
  if (is_vectorize) {
    reduction_scheduler_utils::sharedMemoryConsumerVectorization(
        smem_consumers, rparams->unroll_factor_inner_reduction);
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

  auto scheduler_hyperparameters_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::SchedulerHyperParameters>(
          data_cache, [&]() {
            return std::make_unique<scheduler_utils::SchedulerHyperParameters>(
                /*vectorize_factor=*/vectorize_factor,
                /*unroll_factor=*/1,
                /*threads_per_block_min=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_min,
                /*threads_per_block_max=*/
                InnerOuterPersistentKernelScheduler::threads_per_block_max);
          });
  scheduler_utils::SchedulerHyperParameters& hp =
      scheduler_hyperparameters_entry.get();

  // check if there is enough register and shared memory for persistence
  const auto buffer_params = getPersistentBufferStorageParams(
      fusion,
      runtime_info,
      data_cache,
      reduction_tvs,
      hp.vectorize_factor,
      hp.threads_per_block_min,
      hp.threads_per_block_max);

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
  if (rparams->tma_warp_specialized) {
    scheduleTmaWarpSpecializedInnerOuter(fusion, rparams);
  } else {
    scheduleInnerOuterPersistentKernel(fusion, rparams);
  }
}
} // namespace nvfuser
