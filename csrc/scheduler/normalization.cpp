// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/reduction.h>

#include <debug.h>
#include <executor_utils.h>
#include <grouped_reduction.h>
#include <inlining.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <options.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include <transform_replay.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>

namespace nvfuser {

namespace {
// In normalization heuristics, we usually have several free parameters, e.g.
// persistent batch size, unroll factors, thread block size, etc. This wrapper
// class is used to make sure the parameters are set before they are used and
// they will not be changed after they are finalized.
class HeuristicParameterWrapper {
 private:
  int64_t value_;
  bool mutable_;

 public:
  HeuristicParameterWrapper() : value_(-1), mutable_(true) {}
  void set(int64_t val) {
    if (mutable_) {
      value_ = val;
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Trying to set a non-mutable heuristic parameter!");
    }
  }

  int64_t get() const {
    TORCH_INTERNAL_ASSERT(value_ != -1, "Heuristic parameter is not set!");
    return value_;
  }

  void finalize() {
    TORCH_INTERNAL_ASSERT(value_ != -1, "Heuristic parameter is not set!");
    mutable_ = false;
  }

  bool isMutable() const {
    return mutable_;
  }
};

} // namespace

namespace {

// The innerOuterPersistentHeuristic is tuned for layer_norm backward on A100
// ======= Method if hidden_size > 1024 =======
// (1) Inner reduction is one reduction per block. Reduction domain is
// parallelized by TIDx and TIDy, Iteration domain is parallelized by BIDy. (2)
// Outer reduction is done in two-steps. The first step is partial reduction,
// reduction domain is parallelized by BIDy, iteration domain is parallelized by
// TIDx and TIDy. The partial results are written to gmem followed by a grid
// sync. The second step is block reduction, the reduction domain is
// parallelized by TIDy, the iteration domain is parallelized by TIDx and BIDy.
// ======= Method if hidden_size <= 1024 =======
// (1) Inner reduction is multi-reductions per blocks. Reduction domain is
// parallelized by TIDx, Iteration domain is parallelized by BIDy and TIDy
// (2) Outer reduction is same to cases where hidden_size > 1024 except the
// second step where in this case, the reduction domain is parallelized by TIDx
// and the iteration domain is parallelized by TIDy and BIDy. This switch
// between TIDx and TIDy is because (a) We can do warp reduction with TIDx and
// (b) TIDx*BIDy is usually much larger than hidden_size, e.g. 128*216 = 1024*27
// this means without switch only 1/27 of the threads is used.
std::shared_ptr<ReductionParams> innerOuterPersistentHeuristic(
    const int64_t outer_dim_numel,
    const int64_t inner_dim_numel,
    const int64_t register_persistent_buffer_size,
    const int64_t shared_memory_persistent_buffer_size,
    const int64_t shared_memory_overhead_per_block,
    const size_t tmp_gmem_dtype_size,
    const size_t vectorize_factor) {
  auto rparams = std::make_shared<ReductionParams>();
  rparams->shared_mem_persistent_buffer =
      shared_memory_persistent_buffer_size > 0;
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
    int64_t gdimy = -1;
    int64_t tmp_gmem_write_vect = -1;
    int64_t vectorization_factor_outer = -1;

    void verify() {
      TORCH_INTERNAL_ASSERT(inner_vect != -1, "inner_vect is not set.");
      TORCH_INTERNAL_ASSERT(inner_batch != -1, "inner_batch is not set.");
      TORCH_INTERNAL_ASSERT(bdimx != -1, "bdimx is not set.");
      TORCH_INTERNAL_ASSERT(bdimy != -1, "bdimy is not set.");
      TORCH_INTERNAL_ASSERT(gdimy != -1, "gdimy is not set.");
      TORCH_INTERNAL_ASSERT(
          tmp_gmem_write_vect != -1, "tmp_gmem_write_vect is not set.");
      TORCH_INTERNAL_ASSERT(
          vectorization_factor_outer != -1,
          "vectorization_factor_outer is not set.");
    }
  };

  InnerOuterParams iop;

  // Estimate register per thread based on buffer size, since inner reduction
  // dim is fully parallelized, the buffer size of each thread equals the total
  // buffer size divide by inner_dim_numel.
  auto getEstimatedRegisterUsage = [&](int64_t batch_mul_vect) {
    constexpr int64_t bytes_per_register = 4;
    const int64_t persistent_buffer_size =
        register_persistent_buffer_size / inner_dim_numel * batch_mul_vect;
    const int64_t estimated_register_count =
        persistent_buffer_size / bytes_per_register +
        scheduler_utils::register_overhead;
    return std::min(
        estimated_register_count, scheduler_utils::max_registers_per_thread);
  };

  auto getBlocksPerSM = [&](const int64_t threads_per_sm,
                            const int64_t threads_per_block,
                            const int64_t warp_size) {
    constexpr int64_t warp_allocation_granularity = 4;
    const int64_t allocated_warps_per_block =
        ceilDiv(
            ceilDiv(threads_per_block, warp_size),
            warp_allocation_granularity) *
        warp_allocation_granularity;
    return scheduler_utils::safeDiv(
        threads_per_sm / warp_size, allocated_warps_per_block);
  };

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // Step-1, set InnerParams reduction dim: inner_vect, inner_batch,
  // threads_per_block (bdimx * bdimy). Start threads_per_block from a quarter
  // warp, gradually increase it. Runtime checkCombinedReductionShape ensures
  // inner_dim_numel is dividable by the multiplication of a quarter warp and
  // vectorize_factor.
  iop.inner_vect = (int64_t)vectorize_factor;

  // ignore_register_size_limit will return a valid batch size.
  // This is needed because we enforced projection for fp32 if the feature size
  // is less or equal 14K. It leads to register spills but still faster than the
  // unprojected version due to the reuse of a input para in this grid
  // persistent kernel. However, when we do register usage check in
  // canScheduleRuntime, the enforced projection is not considered. Thus,
  // register_persistent_buffer_size used here is larger than the value used in
  // canScheduleRuntime.
  // This is a tmp solution before we have a new persistent heuristics, where
  // the projection is not solely based on size of buffers. The enforced buffer
  // projection is not considered in canScheduleRuntime Thus,
  constexpr bool ignore_register_size_limit = true;
  const auto& batch_and_block_size = normalization_scheduler_utils::
      getOptionalInnerOuterPersistentBufferBatches(
          inner_dim_numel,
          outer_dim_numel,
          register_persistent_buffer_size,
          shared_memory_persistent_buffer_size,
          iop.inner_vect,
          dev_prop->warpSize,
          ignore_register_size_limit);
  auto opt_inner_batch = batch_and_block_size.first;
  TORCH_INTERNAL_ASSERT(opt_inner_batch.has_value());
  iop.inner_batch = opt_inner_batch.value();
  int64_t threads_per_block = batch_and_block_size.second;

  TORCH_INTERNAL_ASSERT(
      iop.inner_vect * iop.inner_batch * threads_per_block >= inner_dim_numel,
      " iop.inner_vect * iop.inner_batch * threads_per_block should >= inner_dim_numel.");

  // Step-2, set InnerParams Iteration dim: gdimy. reg_per_thread is estimated
  // from buffer size, then it is used to calculate threads_per_sm and gdimy.
  // gdimy_max ensures each block processes at least 8 rows to
  // reduce the workload of the final outer reduction.
  int64_t reg_per_thread =
      getEstimatedRegisterUsage(iop.inner_vect * iop.inner_batch);
  int64_t threads_per_sm = getThreadsPerSMGivenRegPerThread(reg_per_thread);
  int64_t blocks_per_sm_regs =
      getBlocksPerSM(threads_per_sm, threads_per_block, dev_prop->warpSize);
  // check shared memory limitation on blocks per sm
  int64_t blocks_per_sm_smem = dev_prop->sharedMemPerMultiprocessor /
      (shared_memory_overhead_per_block + shared_memory_persistent_buffer_size);
  int64_t blocks_per_sm = std::min(blocks_per_sm_regs, blocks_per_sm_smem);
  std::cout << "sharedMemPerMultiprocessor= "
            << dev_prop->sharedMemPerMultiprocessor / 1024
            << ", shared_memory_persistent_buffer_size= "
            << shared_memory_persistent_buffer_size / 1024
            << ", blocks_per_sm_smem= " << blocks_per_sm_smem
            << ", blocks_per_sm_regs= " << blocks_per_sm_regs << std::endl;
  iop.gdimy = blocks_per_sm * device_multiprocessor_count;
  const int64_t outer_iter_min = 8;
  const int64_t gdimy_max = scheduler_utils::roundUpToN(
      ceilDiv(outer_dim_numel, outer_iter_min), device_multiprocessor_count);
  while (iop.gdimy > gdimy_max && blocks_per_sm > 1) {
    blocks_per_sm -= 1;
    iop.gdimy = blocks_per_sm * device_multiprocessor_count;
  }

  // set the vectorization factor for the write to tmp gmem, may be different
  // from inner_vect due to different data types, e.g. input is half and
  // tmp_gmem is float
  constexpr int64_t max_gmem_vect_access_bytes = 16;
  const int64_t max_tmp_gmem_vect_factor = std::min(
      max_gmem_vect_access_bytes / (int64_t)tmp_gmem_dtype_size,
      iop.inner_vect);
  iop.tmp_gmem_write_vect = max_tmp_gmem_vect_factor;

  // Step-3, set OuterParams Iteration dim: vectorization_factor_outer, bdimx,
  // gdimy (already done) The partial outer reduction result is stored in tmp
  // gmem, set the vectorization factor for write and read
  const int64_t workload_per_thread = inner_dim_numel >= 4096 ? 4l : 2l;
  iop.vectorization_factor_outer =
      std::min(workload_per_thread, max_tmp_gmem_vect_factor);
  // For widely used hidden sizes, threads_per_block has factor of 8, roundup to
  // increase the probability of bdimx * bdimy == threads_per_block.
  iop.bdimx = scheduler_utils::roundUpPow2Or8(
      ceilDiv(inner_dim_numel / iop.vectorization_factor_outer, iop.gdimy));
  // if still not divisible, e.g. threads_per_block = 256, bdimx = 40.
  // increase bdimx to make it divisible. Under worst case, bdimx equals to
  // threads_per_block.
  while (threads_per_block % iop.bdimx) {
    iop.bdimx = std::min(iop.bdimx + 8, threads_per_block);
  }
  // Step-4, set OuterParams Reduction dim: bdimy.
  iop.bdimy = threads_per_block / iop.bdimx;
  TORCH_INTERNAL_ASSERT(
      iop.bdimy * iop.bdimx == threads_per_block,
      " threads_per_block must be divisible by bdimx and bdimy.");
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

    // Step-2, InnerParams, Iteration dim: gdimy, bdimy (in next step)
    reg_per_thread =
        getEstimatedRegisterUsage(iop.inner_vect * iop.inner_batch);
    threads_per_sm = getThreadsPerSMGivenRegPerThread(reg_per_thread);
    blocks_per_sm = getBlocksPerSM(
        threads_per_sm, threads_per_block_mrpb, dev_prop->warpSize);
    iop.gdimy = blocks_per_sm * device_multiprocessor_count;

    // Step-3, OuterParams, Iteration dim: vectorization_factor_outer(reuse),
    // bdimy, gdimy (in previous step). We prefer bdimy to be larger enough to
    // cover what is left in both the outer_dim and inner_dim. However, it
    // should not exceed the limitation set by threads_per_block_mrpb.
    int64_t bdimy_tmp = std::max(
        ceilDiv(outer_dim_numel, iop.gdimy),
        ceilDiv(inner_dim_numel, iop.vectorization_factor_outer * iop.gdimy));
    iop.bdimy = std::min(threads_per_block_mrpb / iop.bdimx, bdimy_tmp);

    // Step-4, OuterParams, Reduction dim: bdimx (already done)

    if (iop.bdimx % dev_prop->warpSize == 0) {
      rparams->pad_inner_reduction_to_warp = true;
      rparams->pad_outer_reduction_to_warp = true;
    }
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  } else {
    rparams->block_dim_inner_reduction_extra = ParallelType::TIDy;
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
  rparams->cparams.maxrregcount = (int)getRegPerThreadGivenThreadsPerSM(
      iop.bdimx * iop.bdimy * blocks_per_sm);
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

  if (rparams->shared_mem_persistent_buffer) {
    rparams->tag =
        "InnerOuter Register and Shared Memory Persistent Heuristic.\n";
  } else {
    rparams->tag = "InnerOuter Register Persistent Heuristic.\n";
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Combined InnerOuter Reduction Stats ========\n"
            << "outer_dim_numel: " << outer_dim_numel << "\n"
            << "inner_dim_numel: " << inner_dim_numel << "\n"
            << "register_persistent_buffer_size: "
            << register_persistent_buffer_size << "\n"
            << "shared_memory_persistent_buffer_size: "
            << shared_memory_persistent_buffer_size << "\n"
            << "shared_memory_overhead_per_block: "
            << shared_memory_overhead_per_block << "\n"
            << "vectorize_factor_input: " << iop.inner_vect << "\n"
            << "vectorization_factor_tmp_gmem_write: "
            << iop.tmp_gmem_write_vect << "\n"
            << "vectorization_factor_outer: " << iop.vectorization_factor_outer
            << "\n"
            << "multiple_reds_per_blk: " << rparams->multiple_reds_per_blk
            << "\n"
            << "threads_per_sm: " << threads_per_sm << "\n"
            << "gdimy: " << iop.gdimy << "\n"
            << "block(" << (iop.bdimx) << ", " << iop.bdimy << ", " << 1 << ")";
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

std::shared_ptr<ReductionParams> innerPersistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t inner_most_dimension_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    const int64_t register_persistent_buffer_size,
    const int64_t shared_memory_persistent_buffer_size,
    const size_t vectorize_factor) {
  auto rparams = std::make_shared<ReductionParams>();
  rparams->shared_mem_persistent_buffer =
      shared_memory_persistent_buffer_size > 0;

  // Set some targets for parallelization
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;

  const int64_t outer_reduction_numel =
      total_reduction_numel / inner_most_dimension_numel;

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // WARNING: At some point we may want to generate heuristics for another
  // device that is not the current device.
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      16l / max_input_dtype_size,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      scheduler_utils::lastPow2(std::max(n_tensor_inputs >> 2, 1l)));

  // Conservative value, could be set to larger based on arch if necessary.
  constexpr int64_t l1_cache = 32l * 1024l;
  // Could change per generation, but for l1 we want to consider active threads,
  // not resident
  constexpr int64_t active_threads = 1024;

  // if data fits in l2 and we need more parallelization in the reduction dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // reduction dim is really small, we can use <32 threads per warp.
  const bool fits_in_l2 =
      n_elems * max_input_dtype_size * n_tensor_inputs < dev_prop->l2CacheSize;

  // If it fits in l2, we just want to make sure each warp uses 32Bytes. Set
  // minimum warp as 16 threads instead of 32 as if we have a small reduction
  // dim going a bit smaller than 32 usually helps.
  const int64_t warp_size_based_on_l2 =
      fits_in_l2 ? 32l / max_input_dtype_size : 16l;

  // Check how many elements it would take per thread to start thrashing l1
  // set that to minimum number we want to reduce per thread.
  const int64_t warp_size_based_on_l1 = std::min(
      ceilDiv(
          total_reduction_numel,
          scheduler_utils::safeDiv(
              l1_cache,
              n_tensor_inputs * max_input_dtype_size * active_threads)),
      16l);

  // Take the smaller, warp_size may be a odd number, e.g. 15
  // Tracked at https://github.com/NVIDIA/Fuser/issues/107
  const int64_t warp_size =
      std::min(warp_size_based_on_l1, warp_size_based_on_l2);

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_iterations = 1;

  // Try to set a minmum amount of work for each thread, as cross thread
  // communication is slow so it shouldn't be done for every element in the
  // reduction.
  int64_t min_target_iterations =
      scheduler_utils::safeDiv(32, max_input_dtype_size);

  // Start trying to break parallelization up across threads,
  // unrolling/iterations, and blocks.

  // max_threads_in_block is the cap on a thread block, the minimum is based on
  // warp_size
  int64_t max_threads_in_block = std::max(
      warp_size, ceilDiv(total_reduction_numel, min_target_iterations));

  // If we have one warp per block, check if that's enough to saturate the SMs
  target_blocks = ceilDiv(n_elems, warp_size);

  // If we have more than a wave of blocks, put parallelism into unrolling and
  // target iterations
  if (target_blocks > device_multiprocessor_count) {
    auto available_unroll = scheduler_utils::safeDiv(
        n_elems, warp_size * device_multiprocessor_count);

    // Spread across unrolling and iterations, want a balance of the two so flip
    // back and forth to alternate adding to them.
    bool flip = true;

    while (available_unroll > 1 &&
           (target_unroll < max_unroll ||
            // Prefer unrolling
            target_iterations < max_unroll)) {
      if (target_unroll * 2 <= max_unroll && flip) {
        target_unroll *= 2;
      }

      if (target_iterations * 2 <= max_unroll && !flip) {
        target_iterations *= 2;
      }

      available_unroll = scheduler_utils::safeDiv(
          n_elems,
          warp_size * device_multiprocessor_count * target_unroll *
              target_iterations);
      flip = !flip;
    }

    // Recompute target blocks
    target_blocks =
        ceilDiv(n_elems, warp_size * target_unroll * target_iterations);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * target_iterations < n_elems) {
    if (outer_reduction_numel == 1) {
      // set to hardware limit to use small persistent buffer for large
      // reductions
      max_threads_in_block = std::min(
          ceilDiv(n_elems, target_blocks * target_unroll),
          (int64_t)dev_prop->maxThreadsPerBlock);
    } else {
      // targetting 4 waves, so try to use a quarter of available threads
      max_threads_in_block = std::min(
          ceilDiv(n_elems, target_blocks * target_unroll),
          ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
    }
  }

  // Round up to nearest warp.
  if (max_threads_in_block % warp_size != 0) {
    max_threads_in_block += warp_size - max_threads_in_block % warp_size;
    max_threads_in_block =
        std::min(max_threads_in_block, (int64_t)dev_prop->maxThreadsPerBlock);
  }
  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size. Bounded by the wave count for utilization of
  // SMs.
  const int64_t max_multi_reduction_factor = std::min(
      scheduler_utils::safeDiv(
          scheduler_utils::register_file_size, register_persistent_buffer_size),
      ceilDiv(total_iteration_numel, device_multiprocessor_count));
  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

  // Blocks for outputs
  int64_t godim = 1;

  // Threads for reduction
  int64_t bdimx = 1;
  // Threads for outputs
  int64_t bdimy = 1;
  // Threads for outer reduction dimension
  int64_t bdimz = 1;

  // Unroll amount
  int64_t inner_reduction_unroll_factor = 1;
  int64_t outer_reduction_unroll_factor = 1;
  int64_t iter_unroll_factor = 1;

  inner_reduction_unroll_factor =
      vectorize_factor > 1 ? (int64_t)vectorize_factor : 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(
      std::max(
          ceilDiv(inner_most_dimension_numel, inner_reduction_unroll_factor),
          (int64_t)warp_size),
      max_threads_in_block);

  // If we're not just barely covering the dimension, round to a more friendly
  // number
  if (bdimx * inner_reduction_unroll_factor != inner_most_dimension_numel) {
    bdimx = bdimx > warp_size ? bdimx - bdimx % warp_size
                              : scheduler_utils::lastPow2(bdimx);

    // Round bdimx down to multiple of warp size or power 2
    if (bdimx < warp_size) {
      bdimx = scheduler_utils::lastPow2(bdimx);
    } else {
      bdimx = bdimx - bdimx % warp_size;
    }
  }

  // Put everything else in bdimy for now
  bdimy = std::min(
      scheduler_utils::safeDiv(warp_size, bdimx), max_multi_reduction_factor);
  // If 3D fill the rest of the threads into bdimz
  bdimz = std::min(
      std::min(
          scheduler_utils::safeDiv(max_threads_in_block, bdimx * bdimy),
          outer_reduction_numel),
      scheduler_utils::z_block_limit);

  bool vectorize = false;

  // Move unrolling factor into vectorization upto vectorization limit.
  if (vectorize_factor > 1 && inner_reduction_unroll_factor > 1) {
    vectorize = true;
    inner_reduction_unroll_factor = std::min(
        scheduler_utils::lastPow2(inner_reduction_unroll_factor),
        (int64_t)vectorize_factor);
  }

  // start from small block size to minimize expensive inter-threads reduction
  const int64_t threads_after_vectorize =
      inner_most_dimension_numel / inner_reduction_unroll_factor;

  // Test min_threads_per_block using 3 values:
  // (1) One warp, so we can use single warp reduction and sync.
  // (2) Two warps, so we can achieve 100% occupancy since most GPUs allow 32
  //     blocks per SM.
  // (3) Four warps, number recommended by the cuda-c-best-practices-guide.
  const int64_t min_threads_per_block = 4l * dev_prop->warpSize;

  // start bdimx with min_threads_per_block then increase if we have too many
  // persistent buffer batches per block
  if (outer_reduction_numel == 1 && vectorize) {
    bdimx = std::min(min_threads_per_block, threads_after_vectorize);
  }

  // If we don't have enough threads, let's do multiple reductions per block.
  // Multiple reductions per block shows better performance than unroll
  // iterations. Still keep vectorization as it is important for performance
  // since V100.
  if (bdimx * bdimy * bdimz < min_threads_per_block) {
    bdimy = std::min(
        scheduler_utils::safeDiv(min_threads_per_block, bdimx * bdimz),
        max_multi_reduction_factor);
  }

  // Set size of persistent per thread buffer on inner reduction buffer
  // if too large, will be reduced later to reduce register usage
  int64_t batches_per_block_inner_reduction = ceilDiv(
      inner_most_dimension_numel, bdimx * inner_reduction_unroll_factor);

  // Attempt to put some unrolling into the outer reduction if inner hasn't
  // taken the max unrolling
  if (inner_reduction_unroll_factor < max_unroll) {
    outer_reduction_unroll_factor = std::min(
        ceilDiv(max_unroll, inner_reduction_unroll_factor),
        ceilDiv(outer_reduction_numel, bdimz));
  }

  godim = ceilDiv(total_iteration_numel, bdimy);

  // Prefer putting iterations into unrolling over having a very large
  // persistent buffer.
  while (!vectorize && inner_reduction_unroll_factor < max_unroll &&
         batches_per_block_inner_reduction >= 2) {
    inner_reduction_unroll_factor *= 2;
    batches_per_block_inner_reduction = scheduler_utils::roundUpPow2Or8(ceilDiv(
        inner_most_dimension_numel, bdimx * inner_reduction_unroll_factor));
  }

  // Set size of persistent per thread buffer on outer reduction buffer
  int64_t batches_per_block_outer_reduction =
      scheduler_utils::roundUpPow2Or8(ceilDiv(
          ceilDiv(total_reduction_numel, inner_most_dimension_numel),
          bdimz * outer_reduction_unroll_factor));

  // Prefer putting iterations into unrolling over having a very large
  // persistent buffer.
  while (outer_reduction_unroll_factor < max_unroll &&
         batches_per_block_outer_reduction >= 2) {
    outer_reduction_unroll_factor *= 2;
    batches_per_block_outer_reduction = scheduler_utils::roundUpPow2Or8(
        ceilDiv(outer_reduction_numel, bdimz * outer_reduction_unroll_factor));
  }

  // Adjust bdimx based on batches_per_block and unroll factor set as they could
  // have moved a bit since they're the free variables, not the buffers
  bdimx = ceilDiv(
      inner_most_dimension_numel,
      inner_reduction_unroll_factor * batches_per_block_inner_reduction);
  bdimz = ceilDiv(
      outer_reduction_numel,
      outer_reduction_unroll_factor * batches_per_block_outer_reduction);

  // Try moving persistent buffer factors into threads until we have too many
  // threads.
  constexpr int batches_per_block_inner_reduction_max = 10;
  while (
      // If block size can be doubled
      bdimx * bdimy * bdimz * 2 <= max_threads_in_block &&
      // And batches_per_block_inner_reduction can be divided by two
      (batches_per_block_inner_reduction >
           batches_per_block_inner_reduction_max ||
       batches_per_block_outer_reduction >= 2)) {
    // Try to decrease per thread register allocation persistence size on inner
    // reduction by double bdimx.
    if (batches_per_block_inner_reduction >
        batches_per_block_inner_reduction_max) {
      bdimx *= 2;
      batches_per_block_inner_reduction = ceilDiv(
          inner_most_dimension_numel, inner_reduction_unroll_factor * bdimx);
      continue;
    }

    // Try to decrease per thread register allocation persistence size on outer
    // reduction
    if (batches_per_block_outer_reduction >= 2 &&
        batches_per_block_outer_reduction !=
            scheduler_utils::roundUpPow2Or8(
                batches_per_block_outer_reduction / 2) &&
        bdimz * 2 <= scheduler_utils::z_block_limit) {
      batches_per_block_outer_reduction = scheduler_utils::roundUpPow2Or8(
          batches_per_block_outer_reduction / 2);
      bdimz = ceilDiv(
          outer_reduction_numel,
          batches_per_block_outer_reduction * outer_reduction_unroll_factor);
      continue;
    }
    break;
  }

  // Register pressure is really high per thread, which could lead to local
  // memory leaks, if using less than maximum threads, decrease batches per
  // block by a factor of 2
  if (batches_per_block_outer_reduction * batches_per_block_inner_reduction *
              inner_reduction_unroll_factor * outer_reduction_unroll_factor *
              4l >
          scheduler_utils::max_registers_per_thread * 3l &&
      bdimx * bdimy * bdimz * 2l <= max_threads_in_block &&
      batches_per_block_inner_reduction >
          batches_per_block_inner_reduction_max) {
    batches_per_block_inner_reduction = batches_per_block_inner_reduction / 2;
  }

  // Do the same on the outer reduction dimension
  if (batches_per_block_outer_reduction * batches_per_block_inner_reduction *
              inner_reduction_unroll_factor * outer_reduction_unroll_factor *
              4l >
          scheduler_utils::max_registers_per_thread * 3l &&
      bdimx * bdimy * bdimz * 2l <= device_max_threads_per_multiprocessor &&
      batches_per_block_outer_reduction >= 2l) {
    batches_per_block_outer_reduction /= 2l;
  }

  auto device_warp_size = (int64_t)at::cuda::warp_size();
  auto padded_bdimx = bdimx % device_warp_size == 0
      ? bdimx
      : bdimx + (device_warp_size - bdimx % device_warp_size);

  bool pad_bdimx = bdimx > 16 &&
      padded_bdimx * bdimy * bdimz < (int64_t)dev_prop->maxThreadsPerBlock;

  // estimate register usage and occupancy raito.
  // If occupancy raito is less than a preset occupancy_ratio, reduce register
  // usage register per thread is estimated as overhead + buffer_size /
  // bytes_per_register
  int64_t nvrtc_register_per_thread = scheduler_utils::max_registers_per_thread;
  const int64_t blocksPerKernel = godim;
  // register estimation is only valid for vectorized gmem access
  // we've seen unexpectedly high register counts with vectorization factor less
  // than 4, which would make the below estimate inaccurate.
  // TODO: support the non vectorized case. consider shmem.
  // only need to balance register and occupancy ratio if there are enough
  // blocks and buffers
  if (vectorize && blocksPerKernel > device_multiprocessor_count &&
      batches_per_block_inner_reduction > 1) {
    // Estimate register per thread based on buffer size, since inner reduction
    // dim is fully parallelized, the buffer size of each element equals the
    // total buffer size divide by inner_most_dimension_numel. Each thread will
    // hold batches_per_block_inner_reduction * inner_reduction_unroll_factor
    // elements.
    const int64_t persistent_buffer_size = register_persistent_buffer_size /
        inner_most_dimension_numel * batches_per_block_inner_reduction *
        inner_reduction_unroll_factor;

    // persistent_buffer_size = 4*2, 8*2, 32*2, 64*2, 128*2
    // register_used_on_a100  = 27,  40,  62,   73,   105
    // register_used_on_v100  = xx,  xx,  45,   62,   93
    // estimated_register_num = 42,  44,  56,   72,   104
    // safe for both v100 & a100
    constexpr int64_t bytes_per_register = 4;
    int64_t estimated_register_count =
        persistent_buffer_size / bytes_per_register +
        scheduler_utils::register_overhead;

    // check occupancy using blocks per sm
    const int64_t threads_per_block =
        pad_bdimx ? padded_bdimx * bdimy * bdimz : bdimx * bdimy * bdimz;
    const int64_t blocks_per_sm_estimated =
        getThreadsPerSMGivenRegPerThread(estimated_register_count) /
        threads_per_block;
    // only allow adjust to 90% of estimated_register_count to avoid too much
    // spills. initially we used 80%, however, the drop from 160 to 128 leads to
    // too much spills in Layer Norm with fused ops, see
    // https://github.com/NVIDIA/Fuser/issues/335
    // 90% allows edge cases, e.g. 72 to 64 which is important for 32K fp16
    // where batch = 8. With this change, however, we lost 10 % performance on
    // Softmax_Inner_fp16/16384/4096, where the perf is best when using 64
    // registers with 232 bytes spill stores and 276 bytes spill loads. The
    // estimated register for this case is 104 adjusting it to 64 is too
    // aggressive.
    constexpr double max_adjust_fraction = 0.9;
    int64_t register_count_minimum = static_cast<int64_t>(
        max_adjust_fraction * static_cast<double>(estimated_register_count));
    const int64_t blocks_per_sm_maximum = scheduler_utils::safeDiv(
        getThreadsPerSMGivenRegPerThread(register_count_minimum),
        threads_per_block);
    register_count_minimum = getRegPerThreadGivenThreadsPerSM(
        blocks_per_sm_maximum * threads_per_block);

    // minimum occupancy we want to achieve
    constexpr double occupancy_ratio = 0.4;
    const int64_t blocks_per_sm_wanted = ceilDiv(
        static_cast<int64_t>(
            dev_prop->maxThreadsPerMultiProcessor * occupancy_ratio),
        threads_per_block);

    // if estimated blocks is smaller than wanted and decrease register usage
    // can increase blocks per sm, try to decrease register usage to increase
    // occupancy but don't go below register_count_minimum
    if (blocks_per_sm_estimated < blocks_per_sm_wanted &&
        blocks_per_sm_maximum > blocks_per_sm_estimated) {
      const int64_t register_count_occupancy = getRegPerThreadGivenThreadsPerSM(
          blocks_per_sm_wanted * threads_per_block);

      nvrtc_register_per_thread =
          std::max(register_count_minimum, register_count_occupancy);
    } else {
      // recalculate estimated_register_count using blocks_per_sm_estimated
      // this may increase estimated_register_count due to allocation
      // granularity e.g. 104 -> 128
      nvrtc_register_per_thread = getRegPerThreadGivenThreadsPerSM(
          blocks_per_sm_estimated * threads_per_block);
    }
  }

  // Will be used once supporting inter-block persistence
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimz = LaunchParams::UNINITIALIZED_VAL;

  rparams->cparams.maxrregcount = (int)nvrtc_register_per_thread;
  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;

  // Inner reduction domain
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->pad_inner_reduction_to_warp = pad_bdimx;
  rparams->batches_per_block_inner_reduction =
      batches_per_block_inner_reduction;

  // For persistent schedules always have to mark the reduction unrolled
  // otherwise rfactor can fail
  rparams->unroll_factor_inner_reduction = inner_reduction_unroll_factor;
  rparams->vectorize_inner_reduction = vectorize;

  // Iter domain
  rparams->multiple_reds_per_blk = bdimy > 1;
  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  if (godim > 1) {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  if (iter_unroll_factor > 1) {
    rparams->unroll_factor_iter_dom = iter_unroll_factor;
  }

  // Outer reduction domain
  rparams->schedule_3D = total_reduction_numel != inner_most_dimension_numel;
  if (rparams->schedule_3D) {
    rparams->batches_per_block_outer_reduction =
        batches_per_block_outer_reduction;
    rparams->block_dim_outer_reduction = ParallelType::TIDz;
    rparams->cross_block_outer_reduction = true;
    rparams->unroll_factor_outer_reduction = outer_reduction_unroll_factor;
  }

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      gdimz,
      LaunchParams::UNINITIALIZED_VAL,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  rparams->tag = "Inner Persistent Heuristic.\n";

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "inner_most_dimension_numel: " << inner_most_dimension_numel
            << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_input_dtype_size: " << max_input_dtype_size << "\n"
            << "register_persistent_buffer_size: "
            << register_persistent_buffer_size << "\n"
            << "max_multi_reduction_factor: " << max_multi_reduction_factor
            << "\n"
            << "block(" << (pad_bdimx ? padded_bdimx : bdimx) << ", " << bdimy
            << ", " << bdimz << ")";
    debug() << rparams->toString() << std::endl;
  }

  return rparams;
}

// Heuristics for grid outer normalizations
std::shared_ptr<ReductionParams> gridOuterPersistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    const int64_t register_persistent_buffer_size,
    const size_t vectorize_factor) {
  auto outer_params =
      normalization_scheduler_utils::getGridOuterNormalizationParams(
          total_reduction_numel,
          total_iteration_numel,
          (int64_t)vectorize_factor,
          register_persistent_buffer_size);

  TORCH_INTERNAL_ASSERT(outer_params.has_value(), "No valid config found");

  const auto pb_size = outer_params->persistent_buffer_factor;
  const auto unswitch_factor = outer_params->unswitch_factor;

  auto rparams = std::make_shared<ReductionParams>();

  rparams->persistent_kernel = true;
  rparams->cross_block_inner_reduction = true;
  rparams->cross_grid_inner_reduction = true;
  rparams->grid_dim_iter_dom = ParallelType::BIDx;
  rparams->grid_dim_inner_reduction = ParallelType::BIDy;
  rparams->block_dim_inner_reduction = ParallelType::TIDy;
  rparams->batches_per_block_inner_reduction = pb_size;
  rparams->multiple_reds_per_blk = true;
  rparams->vectorize_iter_dom = true;
  rparams->unroll_factor_iter_dom = (int64_t)vectorize_factor;
  rparams->block_dim_iter_dom = ParallelType::TIDx;
  rparams->unroll_factor_inner_reduction = unswitch_factor;
  rparams->split_grid_dim_iter_dom_inner =
      ceilDiv(
          total_iteration_numel / (int64_t)vectorize_factor,
          outer_params->launch_params.bdimx()) >
      outer_params->launch_params.gdimx();
  rparams->compute_persistent_buffer_with_first_consumer = true;
  rparams->static_bdimx = true;
  rparams->static_bdimy = true;

  rparams->lparams = LaunchParams(
      rparams->split_grid_dim_iter_dom_inner
          ? outer_params->launch_params.gdimx()
          : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      outer_params->launch_params.bdimx(),
      outer_params->launch_params.bdimy(),
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_input_dtype_size: " << max_input_dtype_size << "\n"
            << "register_persistent_buffer_size: "
            << register_persistent_buffer_size << "\n"
            << "persistent_buffer_factor: " << pb_size << "\n"
            << "block(" << outer_params->launch_params.bdimx() << ", "
            << outer_params->launch_params.bdimy() << ", 1)" << std::endl;
    debug() << rparams->toString() << std::endl;
  }

  return rparams;
}

// Copied from reduction scheduler, should generalize. Simply needed to take out
// grid reductions.
// TODO: Check adding iteration domain unrolling
std::shared_ptr<ReductionParams> outerPersistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    const int64_t register_persistent_buffer_size,
    const int64_t shared_memory_persistent_buffer_size,
    const size_t vectorize_factor) {
  TORCH_INTERNAL_ASSERT(
      shared_memory_persistent_buffer_size == 0,
      "shared memory persistent is not supported in outer normalizations!\n");

  // Set some targets for parallelization
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();

  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // If it fits in l2, we just want to make sure each warp uses 32Bytes. Set
  // minimum warp as 16 threads instead of 32 as if we have a small reduction
  // dim going a bit smaller than 32 usually helps.
  const int64_t warp_size =
      n_elems * max_input_dtype_size * n_tensor_inputs < dev_prop->l2CacheSize
      ? (int64_t)32 / max_input_dtype_size
      : 16;

  const auto register_file_size =
      dev_prop->regsPerBlock * scheduler_utils::bytes_per_register;
  const int64_t device_warp_size = (int64_t)dev_prop->warpSize;

  // Each block runs N reductions, where N is defined as:
  // vectorize_factor * blockDim.x. The minimum number of SMs to run
  // this as a persistent kernel is thus defined as:
  const int64_t min_required_sm_per_norm = ceilDiv(
      register_persistent_buffer_size * (int64_t)vectorize_factor *
          normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx,
      (int64_t)register_file_size);

  if (min_required_sm_per_norm > 1) {
    return gridOuterPersistentHeuristic(
        total_reduction_numel,
        total_iteration_numel,
        n_tensor_inputs,
        max_input_dtype_size,
        register_persistent_buffer_size,
        vectorize_factor);
  }

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size
  const int64_t max_multi_reduction_factor = scheduler_utils::safeDiv(
      scheduler_utils::register_file_size, register_persistent_buffer_size);

  struct HeuristicParams {
    // Iteration dim, each CTA covers [bdimx] * [iter_unroll_factor] reductions.
    // Needs total_iteration_numel / (bdimx * iter_unroll_factor) CTAs.
    HeuristicParameterWrapper iter_unroll_factor;
    HeuristicParameterWrapper bdimx;
    // Reduction dim, each thread do [batches_per_block * redu_unroll_factor]
    // serial reductions, then do block reductions along [bdimy].
    // Total_reduction_numel <= bdimy [dynamic] * batches_per_block *
    // redu_unroll_factor
    HeuristicParameterWrapper redu_unroll_factor;
    HeuristicParameterWrapper batches_per_block;
    HeuristicParameterWrapper bdimy;
    void verify() {
      TORCH_INTERNAL_ASSERT(
          !iter_unroll_factor.isMutable(),
          "iter_unroll_factor is not finalized.");
      TORCH_INTERNAL_ASSERT(!bdimx.isMutable(), "bdimx is not finalized.");
      TORCH_INTERNAL_ASSERT(
          !redu_unroll_factor.isMutable(),
          "redu_unroll_factor is not finalized.");
      TORCH_INTERNAL_ASSERT(
          !batches_per_block.isMutable(),
          "batches_per_block is not finalized.");
      TORCH_INTERNAL_ASSERT(!bdimy.isMutable(), "bdimy is not finalized.");
    }
  };
  HeuristicParams hp;

  // set iter_unroll_factor
  // This controls vectorized load/store along the iteration dimension.
  // The kernel calls block reduction [iter_unroll_factor] times.
  // Test shows performance regression when iter_unroll_factor > 1 due to
  // the high cost of calling block reduction multiple times per block.
  hp.iter_unroll_factor.set(1l);
  hp.iter_unroll_factor.finalize();

  // set redu_unroll_factor
  // This controls unroll along the reduction dimension.
  // For case InstanceNormFP32 of [256, 28, 28, 128], if unroll 2, register
  // usage increased from 89 to 118 but the occupancy is not changed. However,
  // the bandwidth is dropped from 1029 GB/s to 840 GB/s due to more stalled
  // warps. Unroll by 4 increased performance for some cases but has regression
  // in many others. So we set redu_unroll_factor to 1.
  hp.redu_unroll_factor.set(1l);
  hp.redu_unroll_factor.finalize();

  // set bdimx
  // Start from warp_size, and decrease it until we can make more than 4 waves
  const int64_t bdimx_max =
      max_multi_reduction_factor / hp.iter_unroll_factor.get();
  int64_t tmp_bdimx = std::min(bdimx_max, warp_size);
  if (tmp_bdimx < warp_size) {
    tmp_bdimx = scheduler_utils::lastPow2(tmp_bdimx);
  }
  // check if we can make more than 4 waves to hide memory access latency.
  // InstanceNormFP32 of [32, 32, 32, 128] increased from 618 to 770 GB/s
  int64_t num_CTAs =
      ceilDiv(total_iteration_numel, tmp_bdimx * hp.iter_unroll_factor.get());
  while (
      num_CTAs < 4l * device_multiprocessor_count &&
      tmp_bdimx >= 2l *
              normalization_scheduler_utils::PreferredLaunchConfig::kMinBdimx) {
    tmp_bdimx /= 2l;
    num_CTAs =
        ceilDiv(total_iteration_numel, tmp_bdimx * hp.iter_unroll_factor.get());
  }
  // we are not finalizing bdimx here, because we may need to change it later if
  // bdimy is very small
  hp.bdimx.set(tmp_bdimx);

  // set bdimy and batches_per_block
  // These two parameters controls the reduction. Each reduction is split into
  // [batches_per_block] serial reductions and a block reduction along [bdimy].
  // Test shows setting a serial workload larger than 8 improves performance
  // since it reduces inter-threads communication.
  const int64_t batches_per_block_min = std::min(8l, total_reduction_numel);

  // A minimum of 128 threads in a block ensures the four warp schedulers are
  // fully utilized even in cases where only one CTA is active per SM.
  const int64_t min_threads_in_block = 128l;

  // A maximum of 256 threads in a block ensures each thread can use up to 255
  // registers.
  const int64_t max_threads_in_block = 256l;

  // Split reduction domain into redu_unroll_factor, bdimy, and
  // batches_per_block. redu_unroll_factor is already finalized, so the problem
  // changes to split after_unroll into bdimy and batches_per_block. The
  // strategy is: prioritize divisible splits and search for bdimy in a fixed
  // range under the constraint of batches_per_block_min.
  const int64_t after_unroll =
      total_reduction_numel / hp.redu_unroll_factor.get();
  const int64_t bdimy_max = std::min(
      ceilDiv(after_unroll, batches_per_block_min),
      max_threads_in_block / hp.bdimx.get());
  const int64_t bdimy_min =
      std::min(bdimy_max, min_threads_in_block / hp.bdimx.get());
  const int64_t bdimy_step = std::max(1l, device_warp_size / hp.bdimx.get());
  TORCH_INTERNAL_ASSERT(
      device_warp_size % hp.bdimx.get() == 0,
      "bdimx is no divisible by warp_size. bdimx= ",
      hp.bdimx.get());

  auto maybeNextDivisibleFactor =
      [&after_unroll, &bdimy_step, &bdimy_max](int64_t cur) {
        auto next = cur + bdimy_step;
        while (next <= bdimy_max && after_unroll % next) {
          next += bdimy_step;
        }
        return std::min(next, bdimy_max);
      };
  int64_t tmp_bdimy = bdimy_min;
  int64_t tmp_batch = ceilDiv(after_unroll, tmp_bdimy);
  while (tmp_bdimy < bdimy_max) {
    int64_t next_bdimy = maybeNextDivisibleFactor(tmp_bdimy);
    int64_t next_batch = ceilDiv(after_unroll, next_bdimy);
    if (next_batch >= batches_per_block_min) {
      tmp_bdimy = next_bdimy;
      tmp_batch = next_batch;
    } else {
      break;
    }
  }
  hp.bdimy.set(tmp_bdimy);
  hp.bdimy.finalize();
  hp.batches_per_block.set(tmp_batch);
  hp.batches_per_block.finalize();

  // final check on bdimx to avoid small threads_in_block
  if (hp.bdimx.get() * hp.bdimy.get() < min_threads_in_block) {
    hp.bdimx.set(min_threads_in_block / hp.bdimy.get());
  }
  hp.bdimx.finalize();

  // make sure all paras are set
  hp.verify();

  // Final check of the requested registers
  int64_t sm_required_per_norm_set = ceilDiv(
      register_persistent_buffer_size * hp.bdimx.get() *
          hp.iter_unroll_factor.get(),
      scheduler_utils::register_file_size);
  TORCH_INTERNAL_ASSERT(
      sm_required_per_norm_set == 1,
      "Tried to use multiple SMs on an outer persistent kernel ",
      "yet this kernel should have been within block persistent.",
      "\nbdimx= ",
      hp.bdimx.get(),
      ", iter_unroll_factor= ",
      hp.iter_unroll_factor.get());

  // copy to ReductionParams
  auto rparams = std::make_shared<ReductionParams>();
  auto gdimx = ceilDiv(total_iteration_numel, hp.bdimx.get());
  rparams->batches_per_block_inner_reduction = hp.batches_per_block.get();
  rparams->persistent_kernel = true;

  rparams->fastest_dim = false;
  rparams->cross_block_inner_reduction = true;
  rparams->cross_grid_inner_reduction = false;
  rparams->multiple_reds_per_blk = hp.bdimx.get() > 1;

  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDx;
  }

  rparams->grid_dim_iter_dom = ParallelType::BIDx;
  rparams->split_grid_dim_iter_dom_outer =
      gdimx > scheduler_utils::x_grid_limit;

  if (rparams->block_dim_iter_dom == ParallelType::TIDx) {
    rparams->block_dim_inner_reduction = ParallelType::TIDy;
  } else {
    rparams->block_dim_inner_reduction = ParallelType::TIDx;
  }

  // Always need to mark inner reduction unroll for rfactor in outer persitent
  // kernels
  rparams->unroll_factor_inner_reduction = hp.redu_unroll_factor.get();

  rparams->unroll_factor_iter_dom = hp.iter_unroll_factor.get();

  rparams->vectorize_iter_dom =
      vectorize_factor > 1 && hp.iter_unroll_factor.get() > 1;

  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      rparams->multiple_reds_per_blk ? hp.bdimx.get() : hp.bdimy.get(),
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  rparams->tag = "Outer persistent kernel heuristic.\n";

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_input_dtype_size: " << max_input_dtype_size << "\n"
            << "register_persistent_buffer_size: "
            << register_persistent_buffer_size << "\n"
            << "max_multi_reduction_factor: " << max_multi_reduction_factor
            << "\n"
            << "block(" << hp.bdimx.get() << ", " << hp.bdimy.get() << ", 1)"
            << std::endl;
    debug() << rparams->toString() << std::endl;
  }

  return rparams;
}

} // namespace

std::shared_ptr<ReductionParams> persistentHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t inner_most_dimension_numel,
    const bool fastest_dim_reduction,
    const size_t n_tensor_inputs,
    const size_t max_input_dtype_size,
    const size_t tmp_gmem_dtype_size,
    const int64_t register_persistent_buffer_size,
    const int64_t shared_memory_persistent_buffer_size,
    const int64_t shared_memory_overhead_per_block,
    std::vector<TensorView*> shared_memory_persistent_tensors,
    size_t vectorize_factor,
    bool project_persistent_buffers,
    const bool combined_inner_outer_reduction) {
  std::shared_ptr<ReductionParams> rparams;
  if (combined_inner_outer_reduction) {
    const int64_t outer_dim_numel = total_iteration_numel;
    const int64_t inner_dim_numel = inner_most_dimension_numel;
    rparams = innerOuterPersistentHeuristic(
        outer_dim_numel,
        inner_dim_numel,
        register_persistent_buffer_size,
        shared_memory_persistent_buffer_size,
        shared_memory_overhead_per_block,
        tmp_gmem_dtype_size,
        vectorize_factor);
  } else if (fastest_dim_reduction) {
    rparams = innerPersistentHeuristic(
        total_reduction_numel,
        total_iteration_numel,
        inner_most_dimension_numel,
        (int64_t)n_tensor_inputs,
        (int64_t)max_input_dtype_size,
        register_persistent_buffer_size,
        shared_memory_persistent_buffer_size,
        vectorize_factor);
  } else {
    rparams = outerPersistentHeuristic(
        total_reduction_numel,
        total_iteration_numel,
        (int64_t)n_tensor_inputs,
        (int64_t)max_input_dtype_size,
        register_persistent_buffer_size,
        shared_memory_persistent_buffer_size,
        vectorize_factor);
  }
  rparams->project_persistent_buffers = project_persistent_buffers;
  rparams->shared_memory_persistent_tensors =
      std::move(shared_memory_persistent_tensors);
  return rparams;
}

std::shared_ptr<ReductionParams> getPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getPersistentHeuristics");
  FusionGuard fg(fusion);

  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  TORCH_INTERNAL_ASSERT(
      !reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  int64_t n_tensor_inner_reduction = 0;
  int64_t n_tensor_outer_reduction = 0;
  std::vector<TensorView*> inner_reduction_tvs;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      n_tensor_inner_reduction++;
      inner_reduction_tvs.emplace_back(tv);
    } else {
      n_tensor_outer_reduction++;
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  const bool combined_inner_outer_reduction =
      n_tensor_inner_reduction && n_tensor_outer_reduction;
  auto first_red_tv = combined_inner_outer_reduction ? inner_reduction_tvs[0]
                                                     : reduction_tvs[0];

  TORCH_INTERNAL_ASSERT(
      first_red_tv != nullptr, "Reduction TensorView wasn't found.");

  TORCH_INTERNAL_ASSERT(
      first_red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = first_red_tv->definition();

  TORCH_INTERNAL_ASSERT(
      ir_utils::isReductionOp(red_expr),
      "TensorView doesn't have a reduction.");

  auto tv_inps = ir_utils::filterByType<TensorView>(fusion->inputs());
  TORCH_INTERNAL_ASSERT(
      std::distance(tv_inps.begin(), tv_inps.end()) > 0,
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, first_red_tv);

  auto reduced_tv = ir_utils::getSoleProducerTv(first_red_tv);

  auto unrollable_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::UnrollableInputsAndOutputs>(
          data_cache, [&reduced_tv]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    reduced_tv, false, false));
          });

  auto& unrollable_inputs_outputs = unrollable_inputs_outputs_entry.get();

  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      vectorize_helper::getVectorizationBreakPointOfReductionProducer(
          first_red_tv, reduced_tv, properties.inner_most_dimension_ndims));

  // dtype used to store partial outer reduction in combined reduction
  const int64_t tmp_gmem_dtype_size = combined_inner_outer_reduction
      ? dataTypeSize(outer_reduction_tvs[0]->getDataType().value())
      : dataTypeSize(first_red_tv->getDataType().value());

  const auto buffer_params =
      normalization_scheduler_utils::getPersistentBufferStorageParams(
          fusion,
          runtime_info,
          data_cache,
          reduction_tvs,
          properties,
          vectorize_factor);

  // Base max dtype and n_tensor_inputs on tensors that are vectorizable (i.e.
  // share inner dimension with data pattern we're looking at).
  int64_t max_dtype_size = 1;

  // TODO: This might be better if it was the larger of input or outputs. Would
  // be even better if we had better analysis as not all unrolled elements have
  // to be alive at the same time.
  int64_t n_tensor_inputs = 0;
  for (auto tv : unrollable_inputs_outputs) {
    if (!tv->isFusionInput()) {
      continue;
    }

    max_dtype_size = std::max(
        max_dtype_size,
        dataTypeSize(tv->getDataType().value(), runtime_info.getIndexType()));
    n_tensor_inputs++;
  }

  // Protect heuristics div by 0:
  n_tensor_inputs = std::max(n_tensor_inputs, (int64_t)1);

  auto heuristic = persistentHeuristic(
      properties.total_reduction_numel,
      properties.total_iteration_numel,
      properties.inner_most_dimension_numel,
      properties.fastest_dim_reduction,
      n_tensor_inputs,
      max_dtype_size,
      tmp_gmem_dtype_size,
      buffer_params.register_persistent_buffer_size,
      buffer_params.shared_memory_persistent_buffer_size,
      buffer_params.shared_memory_overhead_per_block,
      buffer_params.shared_memory_persistent_tensors,
      vectorize_factor,
      buffer_params.project_to_input,
      combined_inner_outer_reduction);
  heuristic->cparams.index_type = runtime_info.getIndexType();
  return heuristic;
}

std::shared_ptr<ReductionParams> getPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getPersistentHeuristicsFromIValue");
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs);
  return getPersistentHeuristics(fusion, runtime_info, data_cache);
}

// common prepare for both inner outer combined and seperated reductions
void beforeSchedule(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& dummy_outputs,
    std::vector<TensorView*>& cached_inputs,
    std::vector<TensorView*>& reduction_tvs,
    std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs) {
  // Project the persistent buffers to the inputs. Inputs will be cached in a
  // later step, this will move them to be in a register buffer as expected.
  // dummy outputs are helper tensors to make sure persistent buffer projection
  // does not create trouble for transform propagation.
  // TODO: Fix projected persistent buffers with view
  // https://github.com/csarofeen/pytorch/issues/2054
  const bool project_to_inputs = rparams.project_persistent_buffers &&
      ir_utils::getViewOps(fusion).empty();
  dummy_outputs = reduction_scheduler_utils::projectPersistentBuffers(
      fusion, project_to_inputs);
  // Cache tensors before grabbing any references to reductions as cache_before
  // can invalidate the references since when applied to a reduction tensor view
  // the new tensor view contains the reduction and original doesn't.
  bool unroll = rparams.isUnrolled();
  // Cache inputs even if not unrolled, as otherwise we may not create a
  // persistent buffer if that persistent buffer would be the input.
  cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, unroll);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);
  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  // Change storage type to shared memory.
  // Check persistent tensors first, if they are not in the container, check
  // their input if it is a cached input. When the
  // shared_memory_persistent_tensors container was created, cached inputs were
  // not created yet.
  if (rparams.shared_mem_persistent_buffer) {
    const auto& persistent_buffers =
        scheduler_utils::persistentBuffers(fusion).persistent_buffers;
    auto isSharedMemoryPersistent = [&rparams](const TensorView* lookup_tv) {
      return std::any_of(
          rparams.shared_memory_persistent_tensors.begin(),
          rparams.shared_memory_persistent_tensors.end(),
          [lookup_tv](const auto* tv) {
            return tv->name() == lookup_tv->name();
          });
    };
    for (auto tv : persistent_buffers) {
      bool use_smem = isSharedMemoryPersistent(tv);
      if (!use_smem &&
          std::find(cached_inputs.begin(), cached_inputs.end(), tv) !=
              cached_inputs.end()) {
        auto input_tv = ir_utils::producerTvsOf(tv)[0];
        use_smem = isSharedMemoryPersistent(input_tv);
      }
      if (use_smem) {
        tv->setMemoryType(MemoryType::Shared);
      }
      std::cout << "may_be_smem_tv: " << tv->toString() << std::endl;
    }
  }

  reduction_tvs = scheduler_utils::getReductionTvs(fusion);
}

// If called from schedulePersistentKernel, reduction_tvs are either inner
// reductions or outer reductions. If called from
// schedulePersistentKernelInnerOuter, reduction_tvs are inner reductions, outer
// reductions are handled by scheduleCombinedOuter.
TensorView* scheduleReductionGeneral(
    Fusion* fusion,
    const ReductionParams& rparams,
    std::vector<TensorView*>& reduction_tvs) {
  TORCH_INTERNAL_ASSERT(!reduction_tvs.empty());
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
        scheduler_utils::domainReorderAsRfactorMap(reduction_tv));
  }

  if (rparams.persistent_kernel && rparams.cross_grid_inner_reduction &&
      !rparams.fastest_dim && reduction_tvs.size() > 1 &&
      !rparams.combined_inner_outer) {
    groupReductions(reduction_tvs, false);
  }

  auto dim_analysis = scheduler_utils::canonicalDimReduction(
      fusion, reduction_tv, rparams.fastest_dim && rparams.schedule_3D);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  TORCH_INTERNAL_ASSERT(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  return reduction_scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);
}

// fusion is the input IR that will be modified by this function
void schedulePersistentKernel(Fusion* fusion, const ReductionParams& rparams) {
  std::cout << "trace: schedulePersistentKernel\n";

  FUSER_PERF_SCOPE("schedulePersistentKernel");
  if (rparams.combined_inner_outer) {
    return schedulePersistentKernelInnerOuter(fusion, rparams);
  }
  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
      cached_outputs);

  TensorView* reference_tv =
      scheduleReductionGeneral(fusion, rparams, reduction_tvs);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  TORCH_INTERNAL_ASSERT(
      reference_tv != nullptr && reduction_tvs[0] != nullptr,
      "Need these two tensor views to finish the scheduling.");

  for (auto output : dummy_outputs) {
    fusion->addOutput(output);
  }

  const bool unroll = rparams.isUnrolled();
  const bool vectorize =
      rparams.vectorize_inner_reduction || rparams.vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams.persistent_kernel &&
      rparams.cross_grid_inner_reduction && !rparams.fastest_dim;
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
      dummy_outputs);

  if (rparams.compute_persistent_buffer_with_first_consumer) {
    TORCH_INTERNAL_ASSERT(
        rparams.persistent_kernel,
        "computeWith should be only used with persistent kernels");
    for (const auto persistent_buffer : cached_inputs) {
      persistent_buffer->computeWith(-1, true);
    }
  }

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);
}

void scheduleReductionCombinedOuter(
    Fusion* fusion,
    const ReductionParams& rparams,
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
    if (rparams.multiple_reds_per_blk) {
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(rparams.block_dim_iter_dom));
    }
    outer_reduction_tv->split(
        0, NamedScalar::getParallelDim(rparams.grid_dim_iter_dom), false);

    if (rparams.multiple_reds_per_blk) {
      outer_reduction_tv->rFactor({1});
    }
    TensorView* partialResult = outer_reduction_tv->rFactor({1});
    partialResult->cacheBefore();
    partialResult->setMemoryType(MemoryType::Global);
    TensorView* partialResultReload = partialResult->cacheAfter();

    boundaryNodesSet.insert(partialResultReload);
    cached_gmem.emplace_back(partialResult);
    cached_gmem_reload.emplace_back(partialResultReload);

    if (rparams.multiple_reds_per_blk) {
      if (rparams.tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
        // to use warp reduction
        if (rparams.pad_outer_reduction_to_warp) {
          outer_reduction_tv->axis(1)->padToMultipleOfWarp();
        }
      } else {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
      }
      // iteration domain
      int axisID = -1;
      if (rparams.vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams.vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams.tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDy);
      } else {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }

      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);

    } else {
      // reduction domain
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(ParallelType::TIDy));
      outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);

      // iteration domain
      int axisID = -1;
      if (rparams.vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams.vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams.lparams.bdimx() > 1) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }

      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));

      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);
    }
    auto outer_reference_tv =
        reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
    outer_reference_tvs.emplace_back(outer_reference_tv);
  }
}

void schedulePersistentKernelInnerOuter(
    Fusion* fusion,
    const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("schedulePersistentKernelInnerOuter");

  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
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
  TORCH_INTERNAL_ASSERT(
      !inner_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no inner reduction is found.");
  TORCH_INTERNAL_ASSERT(
      !outer_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no outer reduction is found.");

  // schedule inner reduction, only schedule the first inner reduction tv, then
  // will be propagated to other inner reduction tvs.
  TensorView* inner_reference_tv =
      scheduleReductionGeneral(fusion, rparams, inner_reduction_tvs);

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

  const bool unroll = rparams.isUnrolled();
  const bool vectorize =
      rparams.vectorize_inner_reduction || rparams.vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams.persistent_kernel &&
      rparams.cross_grid_inner_reduction && !rparams.fastest_dim;

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
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  // Propagate outer reduction. Each outer reduction is connected with its
  // cached_gmem and output, since we added all the cached_gmem to the
  // boundaryNodesSet, the transformation from one outer reduction can't
  // propagate to other outer reductions due to the cutoff at boundaryNodesSet.
  // Thus, we need a loop to initiate the propagation from each outer reduction.
  // Don't allow parallelization propagation goes through cached_gmem, see issue
  // 246.
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
        is_outer_grid_persistence,
        outer_reduction_tvs,
        cached_inputs,
        cached_outputs,
        {selected_tvs_outer.begin(), selected_tvs_outer.end()});
  }

  // special vectorization of temp gmem, vectorization_factor_tmp_gmem_write is
  // guaranteed to be smaller or equal to input vectorization factor.
  if (rparams.vectorization_factor_tmp_gmem_write > 1) {
    for (auto tv : cached_gmem) {
      TORCH_INTERNAL_ASSERT(
          rparams.vectorization_factor_tmp_gmem_write <=
              rparams.unroll_factor_inner_reduction,
          "vectorization factor of temp gmem write should be smaller than that of inner reduction.")
      if (rparams.vectorization_factor_tmp_gmem_write <
          rparams.unroll_factor_inner_reduction) {
        tv->split(-1, rparams.vectorization_factor_tmp_gmem_write);
      }
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }
  // vectorization propagate through propagateParallelization only works for
  // input and output tensors. propagate vectorization to cached_gmem_reload
  // directly from output tv using parallelizeAllLike. must propagate seperaely
  // for different tvs as outer reductions are transformed seperately.
  if (rparams.vectorization_factor_outer > 1) {
    for (auto tv : cached_gmem_reload) {
      auto output_tvs = ir_utils::outputTvsOf(tv);
      TORCH_INTERNAL_ASSERT(
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
} // namespace nvfuser
