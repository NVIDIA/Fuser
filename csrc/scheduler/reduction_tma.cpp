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
  [[maybe_unused]] auto dev_prop = at::cuda::getCurrentDeviceProperties();
  [[maybe_unused]] const int64_t max_threads_per_sm =
      dev_prop->maxThreadsPerMultiProcessor;
  [[maybe_unused]] const int64_t target_threads_per_sm = max_threads_per_sm / 2;

  const int64_t smem_elems = dev_prop->sharedMemPerBlockOptin /
      props.max_dtype_size_bit_for_vectorization;

  if (props.inner_most_dimension_numel > smem_elems) {
    return nullptr;
  }

  // Set target_vect_unroll
  [[maybe_unused]] auto target_inner_unroll =
      reduction_scheduler_utils::getVectUnroll(
          props.max_dtype_size_bit_for_vectorization,
          props.vectorize_factor,
          props.n_tensor_inputs,
          target_threads_per_sm,
          props.has_mufu_computation);

  int64_t inner_unroll = std::min(
      scheduler_utils::lastPow2(target_inner_unroll),
      (int64_t)props.vectorize_factor);

  auto params = std::make_unique<TmaInnerReductionParams>();

  params->inner_unroll = inner_unroll;

  int64_t threads_per_block = props.has_mufu_computation ? 128 : 256;

  params->threads_per_block = threads_per_block;

  int64_t after_unroll = props.total_reduction_numel / inner_unroll;

  if (after_unroll < threads_per_block) {
    return nullptr;
  }

  // TODO: Support merging for contiguous inner dimensions
  if (props.total_reduction_numel != props.inner_most_dimension_numel) {
    return nullptr;
  }

  params->tag = "Reduction TMA heuristics";
  params->cparams.index_type = runtime_info.getIndexType();

  return params;
}

void scheduleReduction(Fusion* fusion, const TmaInnerReductionParams* rparams) {
  FusionGuard fg(fusion);

  //   return;

  // bool isUnrolled = rparams->isUnrolled();
  bool isUnrolled = true;

  // ndim_3_inner_size_256 = [64, 8, 128] ->

  // Cache inputs if unrolled
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, isUnrolled);

  // Cache and fork outputs
  auto cached_outputs =
      scheduler_utils::cacheAndForkOutputs(fusion, isUnrolled);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  [[maybe_unused]] std::vector<TensorView*> ldg_tvs, tma_tvs;
  for (auto [tv, input_idx] : cached_inputs) {
    if (auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition())) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulk);
      tv->setMemoryType(MemoryType::Shared);
      tma_tvs.push_back(tv);
      // TODO cache to registers
    } else {
      ldg_tvs.push_back(tv);
    }
  }

  NVF_ERROR(!tma_tvs.empty());

  int64_t iter_axis = 0;
  int64_t inner_reduce_axis = 1;

  // Schedule TMA tvs as [BIDx, Bulk]
  tma_tvs[0]->axis(iter_axis)->parallelize(ParallelType::BIDx);
  tma_tvs[0]->axis(inner_reduce_axis)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(tma_tvs[0], tma_tvs);

  // Schedule reduction tv
  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  NVF_ERROR(!reduction_tvs.empty());
  TensorView* reduction_tv = reduction_tvs.at(0);

  // Merge contiguous reduction dimensions
  // [I, R1, R2] -> [I, R1*R2]
  for (int64_t i = reduction_tv->nDims() - 2; i >= 0; --i) {
    if (reduction_tv->axis(i)->isReduction() &&
        reduction_tv->axis(i + 1)->isReduction()) {
      reduction_tv->merge(i, i + 1);
    }
  }

  //   auto dim_analysis = scheduler_utils::canonicalDimReduction(
  //       fusion, reduction_tv, rparams->fastest_dim && rparams->schedule_3D);

  if (rparams->inner_unroll > 1) {
    reduction_tv->split(inner_reduce_axis, rparams->inner_unroll);
    reduction_tv->axis(inner_reduce_axis + 1)
        ->parallelize(ParallelType::Serial);
  }

  reduction_tv->split(inner_reduce_axis, rparams->threads_per_block);
  reduction_tv->axis(inner_reduce_axis + 1)->parallelize(ParallelType::TIDx);

  reduction_tv->split(inner_reduce_axis, 1);
  reduction_tv->axis(inner_reduce_axis + 1)
      ->parallelize(ParallelType::Unswitch);

  reduction_tv->axis(inner_reduce_axis)->parallelize(ParallelType::Serial);
  reduction_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);

  int64_t vectorize_pos = inner_reduce_axis + 2;
  auto reference_tv = reduction_tv->rFactor({inner_reduce_axis, vectorize_pos});

  reduction_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
  reduction_tv->axis(inner_reduce_axis)->parallelize(ParallelType::Serial);

  // Schedule non-TMA tvs based on reduction tv
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reduction_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reduction_tv, &selector)
      .traverse(&non_tma_propagator);

  if (reference_tv != reduction_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, reduction_tv, reduction_tvs);
    non_tma_tvs =
        ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  }

  scheduler_utils::parallelizeAllLike(reduction_tv, non_tma_tvs);

  // TODO: Vectorize load from smem
  // TODO: TMA or vectorize store

  inlineMost();

  refineCachePolicy(fusion);
}
} // namespace tma
} // namespace reduction
} // namespace nvfuser
