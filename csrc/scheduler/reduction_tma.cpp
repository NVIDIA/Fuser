// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/reduction_tma.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

namespace nvfuser {
namespace reduction {
namespace tma {

std::unique_ptr<TmaInnerReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props) {
  FusionGuard fg(fusion);

  // Get device properties
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
  const int64_t target_threads_per_sm = max_threads_per_sm / 2;
  const int64_t threads_per_warp = dev_prop->warpSize;

  const int64_t smem_elems = dev_prop->sharedMemPerBlockOptin /
      props.max_dtype_size_bit_for_vectorization;

  if (props.inner_most_dimension_numel > smem_elems) {
    return nullptr;
  }

  // Only support 2D case for now (single reduction dimension)
  if (props.total_reduction_numel != props.inner_most_dimension_numel) {
    return nullptr;
  }

  const int64_t total_reduction_numel = props.total_reduction_numel;

  // Get target vectorization/unroll factor
  auto target_vect_unroll = reduction_scheduler_utils::getVectUnroll(
      props.max_dtype_size_bit_for_vectorization,
      props.vectorize_factor,
      props.n_tensor_inputs,
      target_threads_per_sm,
      props.has_mufu_computation);

  // Initialize split factors
  int64_t vectorization_factor = 1;
  int64_t threads_per_block = 1;
  int64_t unroll_factor = 1;

  // Helper to compute remaining elements after all splits
  auto getSerialRemainder = [&]() {
    return ceilDiv(
        ceilDiv(
            total_reduction_numel / vectorization_factor, threads_per_block),
        unroll_factor);
  };

  // Step 1: Set vectorization factor (inner serial split for TMA)
  // Start with max possible, but don't exceed total reduction numel
  vectorization_factor = std::min(
      scheduler_utils::lastPow2(target_vect_unroll),
      (int64_t)props.vectorize_factor);
  vectorization_factor = std::min(vectorization_factor, total_reduction_numel);

  // Step 2: Set threads per block
  // Empirical CTA size based on computation intensity
  int64_t target_threads_per_block = props.has_mufu_computation ? 128 : 256;

  int64_t after_vect = total_reduction_numel / vectorization_factor;

  // Ensure we have enough elements for at least target threads
  if (after_vect < target_threads_per_block) {
    // Reduce vectorization to allow more threads
    while (vectorization_factor > 1 && after_vect < target_threads_per_block) {
      vectorization_factor /= 2;
      after_vect = total_reduction_numel / vectorization_factor;
    }
  }

  // If still not enough elements, reduce target threads
  if (after_vect < target_threads_per_block) {
    target_threads_per_block = scheduler_utils::lastPow2(after_vect);
    // Ensure at least one warp
    target_threads_per_block =
        std::max(target_threads_per_block, threads_per_warp);
  }

  threads_per_block = target_threads_per_block;

  // Step 3: Set inner unroll factor (outside TIDx)
  int64_t target_inner_unroll = target_vect_unroll / vectorization_factor;
  int64_t after_vect_tidx = after_vect / threads_per_block;

  if (after_vect_tidx > 1 && target_inner_unroll > 1) {
    unroll_factor = std::min(
        scheduler_utils::lastPow2(target_inner_unroll),
        scheduler_utils::lastPow2(after_vect_tidx));
    unroll_factor = std::max(unroll_factor, (int64_t)1);
  }

  // Final validation: ensure we have at least 1 serial iteration
  if (getSerialRemainder() < 1) {
    return nullptr;
  }

  // Ensure minimum threads for efficient reduction
  if (threads_per_block < threads_per_warp) {
    return nullptr;
  }

  auto params = std::make_unique<TmaInnerReductionParams>();
  params->vectorization_factor = vectorization_factor;
  params->threads_per_block = threads_per_block;
  params->unroll_factor = unroll_factor;

  params->tag = "Reduction TMA heuristics";
  params->cparams.index_type = runtime_info.getIndexType();

  return params;
}

void scheduleReduction(Fusion* fusion, const TmaInnerReductionParams* rparams) {
  FusionGuard fg(fusion);

  // Always cache inputs for TMA
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  std::vector<TensorView*> tma_tvs;
  for (auto [tv, input_idx] : cached_inputs) {
    if (auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition())) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulk);
      tv->setMemoryType(MemoryType::Shared);
      tma_tvs.push_back(tv);
    }
  }

  NVF_ERROR(!tma_tvs.empty());

  int64_t iter_axis = 0;
  int64_t inner_reduce_axis = 1;

  // Schedule TMA tvs as [BIDx, Bulk]
  tma_tvs[0]->axis(iter_axis)->parallelize(ParallelType::BIDx);
  tma_tvs[0]->axis(inner_reduce_axis)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(tma_tvs[0], tma_tvs);

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  NVF_ERROR(!reduction_tvs.empty());
  TensorView* reduction_tv = reduction_tvs.at(0);

  // Merge all iteration and reduction dimensions into canonical form [I, R]
  auto dim_analysis =
      scheduler_utils::canonicalDimReduction(fusion, reduction_tv, false);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;
  NVF_ERROR(has_iter_axis && has_red_axis);

  // Propagate the merges to all TMA TVs
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator tma_propagator(reduction_tv);
  SetSelector tma_selector({tma_tvs.begin(), tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reduction_tv, &tma_selector)
      .traverse(&tma_propagator);

  // Non-TMA scheduling
  //
  // Apply splits following the pattern:
  // [I, R] -> [I, R/vect, vect] -> [I, R/vect/tidx, tidx, vect]
  //        -> [I, R/vect/tidx/unroll, unroll, tidx, vect]
  //        -> [I, serial, unswitch, unroll, tidx, vect]

  // Split 1: Vectorization factor (innermost serial split for TMA)
  if (rparams->vectorization_factor > 1) {
    reduction_tv->split(inner_reduce_axis, rparams->vectorization_factor);
  }

  // Split 2: TIDx (always applied)
  reduction_tv->split(inner_reduce_axis, rparams->threads_per_block);

  // Split 3: Inner unroll (outside of TIDx)
  if (rparams->unroll_factor > 1) {
    reduction_tv->split(inner_reduce_axis, rparams->unroll_factor);
  }

  // Split 4: Unswitch (always applied)
  reduction_tv->split(inner_reduce_axis, 1);

  // Calculate axis positions after all splits
  // Starting from inner_reduce_axis = 1, after N splits we have:
  // [I, serial, unswitch?, unroll?, tidx, vect?]
  int64_t current_axis = inner_reduce_axis;

  // Serial outer loop (remainder after all splits)
  reduction_tv->axis(current_axis)->parallelize(ParallelType::Serial);
  current_axis++;

  // Unswitch axis (always present)
  reduction_tv->axis(current_axis)->parallelize(ParallelType::Unswitch);
  current_axis++;

  // Unroll axis (if unroll_factor > 1)
  if (rparams->unroll_factor > 1) {
    reduction_tv->axis(current_axis)->parallelize(ParallelType::Unroll);
    current_axis++;
  }

  // TIDx axis (always present)
  reduction_tv->axis(current_axis)->parallelize(ParallelType::TIDx);
  current_axis++;

  // Vectorization axis (inner serial for TMA, if > 1)
  if (rparams->vectorization_factor > 1) {
    reduction_tv->axis(current_axis)->parallelize(ParallelType::Serial);
    current_axis++;
  }

  // Parallelize iteration axis
  reduction_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);

  // Collect rFactor axes: all reduction axes that are not parallelized with
  // threads (i.e., not TIDx). This includes serial, unswitch, unroll, and vect
  // axes.
  std::vector<int64_t> rfactor_axes;
  for (int64_t i = 0; i < reduction_tv->nDims(); i++) {
    if (reduction_tv->axis(i)->isReduction() &&
        !reduction_tv->axis(i)->isThread()) {
      rfactor_axes.push_back(i);
    }
  }

  TensorView* reference_tv = reduction_tv;
  if (!rfactor_axes.empty()) {
    reference_tv = reduction_tv->rFactor(rfactor_axes);
  }

  // Schedule non-TMA tvs based on reference tv
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector non_tma_selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &non_tma_selector)
      .traverse(&non_tma_propagator);

  if (reference_tv != reduction_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, reduction_tv, reduction_tvs);
    non_tma_tvs =
        ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  }

  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  inlineMost();

  refineCachePolicy(fusion);
}
} // namespace tma
} // namespace reduction
} // namespace nvfuser
