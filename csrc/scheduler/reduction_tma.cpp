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

  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
  const int64_t target_threads_per_sm = max_threads_per_sm / 2;

  auto target_vect_unroll = reduction_scheduler_utils::getVectUnroll(
      props.max_dtype_size_bit_for_vectorization,
      props.vectorize_factor,
      props.n_tensor_inputs,
      target_threads_per_sm,
      props.has_mufu_computation);

  // Initialize split factors
  int64_t vectorization_factor =
      std::min(target_vect_unroll, props.vectorize_factor);
  int64_t threads_per_block = 128;
  int64_t unroll_factor = target_vect_unroll / vectorization_factor;

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
  TransformPropagator tma_propagator(reduction_tv);
  SetSelector tma_selector({tma_tvs.begin(), tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reduction_tv, &tma_selector)
      .traverse(&tma_propagator);

  int64_t iter_axis = 0;
  int64_t inner_reduce_axis = 1;

  // Schedule TMA tvs as [BIDx, Bulk]
  tma_tvs[0]->axis(iter_axis)->parallelize(ParallelType::BIDx);
  tma_tvs[0]->axis(inner_reduce_axis)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(tma_tvs[0], tma_tvs);

  // Non-TMA scheduling
  //
  // Apply splits following the pattern:
  // [I, R] -> [I, R/vect, vect]
  //        -> [I, R/vect/tidx, tidx, vect]
  //        -> [I, R/vect/tidx/unroll, unroll, tidx, vect]

  // Split 1: Vectorization factor (innermost serial split for TMA)
  reduction_tv->split(inner_reduce_axis, rparams->vectorization_factor);
  reduction_tv->axis(inner_reduce_axis + 1)->parallelize(ParallelType::Serial);

  // Split 2: TIDx (always applied)
  reduction_tv->split(inner_reduce_axis, rparams->threads_per_block);
  reduction_tv->axis(inner_reduce_axis + 1)->parallelize(ParallelType::TIDx);

  // Split 3: Inner unroll (outside of TIDx)
  if (rparams->unroll_factor > 1) {
    reduction_tv->split(inner_reduce_axis, rparams->unroll_factor);
    reduction_tv->axis(inner_reduce_axis + 1)
        ->parallelize(ParallelType::Unroll);
  }

  // Split 4: Unswitch (always applied)
  reduction_tv->split(inner_reduce_axis, 1);
  reduction_tv->axis(inner_reduce_axis + 1)
      ->parallelize(ParallelType::Unswitch);

  // Serial outer loop (remainder after all splits)
  reduction_tv->axis(inner_reduce_axis)->parallelize(ParallelType::Serial);

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
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
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
