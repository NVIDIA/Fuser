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
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  [[maybe_unused]] auto k_params =
      reduction_scheduler_utils::getReductionKernelParams(
          fusion, runtime_info, data_cache);

  NVF_ERROR(k_params.fastest_dim_reduction);
  NVF_ERROR(
      k_params.total_reduction_numel == k_params.inner_most_dimension_numel);

  // Get device properties
  [[maybe_unused]] auto dev_prop = at::cuda::getCurrentDeviceProperties();
  [[maybe_unused]] const int64_t threads_per_warp = dev_prop->warpSize;
  [[maybe_unused]] const int64_t max_threads_per_block =
      dev_prop->maxThreadsPerBlock;
  [[maybe_unused]] const int64_t max_threads_per_sm =
      dev_prop->maxThreadsPerMultiProcessor;
  [[maybe_unused]] const int64_t sm_count = dev_prop->multiProcessorCount;
  [[maybe_unused]] const int64_t target_threads_per_sm = max_threads_per_sm / 2;

  [[maybe_unused]] const int64_t min_warp_size =
      reduction_scheduler_utils::getL1L2WarpSize(
          k_params.total_reduction_numel,
          k_params.total_iteration_numel,
          k_params.n_tensor_inputs,
          k_params.max_dtype_size_bit_for_vectorization);

  // Set target_vect_unroll
  [[maybe_unused]] auto target_vect_unroll =
      reduction_scheduler_utils::getVectUnroll(
          k_params.max_dtype_size_bit_for_vectorization,
          k_params.vectorize_factor,
          k_params.n_tensor_inputs,
          target_threads_per_sm,
          k_params.has_mufu_computation);

  [[maybe_unused]] int64_t vect_factor = 1;

  // Max vectorization factor
  vect_factor = std::min(
      scheduler_utils::lastPow2(target_vect_unroll),
      (int64_t)k_params.vectorize_factor);
  [[maybe_unused]] const int64_t four_warps = 4 * threads_per_warp;
  [[maybe_unused]] int64_t target_inner_unroll =
      target_vect_unroll / vect_factor;
  [[maybe_unused]] int64_t after_vect =
      k_params.total_reduction_numel / vect_factor;

  auto params = std::make_unique<TmaInnerReductionParams>();

  params->unroll_factor = vect_factor;

  params->target_threads_per_block = k_params.has_mufu_computation ? 128 : 256;

  params->tag = "Reduction TMA heuristics";
  params->cparams.index_type = runtime_info.getIndexType();
  return params;
}

void scheduleReduction(Fusion* fusion, const TmaInnerReductionParams* pparams) {
  FusionGuard fg(fusion);

  // bool isUnrolled = rparams->isUnrolled();
  bool isUnrolled = true;

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

  // TODO: Compute as heuristic on TmaInnerReductionParams
  const int64_t unroll = 4;

  if (pparams->unroll_factor > 1) {
    reduction_tv->split(inner_reduce_axis, pparams->unroll_factor);
    reduction_tv->axis(inner_reduce_axis + 1)
        ->parallelize(ParallelType::Serial);
  }

  reduction_tv->split(inner_reduce_axis, pparams->target_threads_per_block);
  reduction_tv->axis(inner_reduce_axis + 1)->parallelize(ParallelType::TIDx);

  reduction_tv->split(inner_reduce_axis, unroll);
  reduction_tv->axis(inner_reduce_axis + 1)->parallelize(ParallelType::Unroll);

  reduction_tv->split(inner_reduce_axis, 1);
  reduction_tv->axis(inner_reduce_axis + 1)
      ->parallelize(ParallelType::Unswitch);

  reduction_tv->axis(inner_reduce_axis)->parallelize(ParallelType::Serial);
  reduction_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);

  int64_t vectorize_pos = inner_reduce_axis + 3;
  auto reference_tv = reduction_tv->rFactor({inner_reduce_axis, vectorize_pos});

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
