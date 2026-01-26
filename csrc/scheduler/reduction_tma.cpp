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
namespace {

// Find the smallest split factor that:
// 1. Evenly divides reduction_numel
// 2. Results in an element count that is divisible by alignment
// 3. Results in an element count that is inside [lower_elem_bound,
//    upper_elem_bound]
int64_t getTmaSplit(
    int64_t numel,
    int64_t alignment,
    int64_t lower_elem_bound,
    int64_t upper_elem_bound) {
  // Lower & upper bounds for the split factor
  const int64_t split_lower = ceilDiv(numel, upper_elem_bound);
  const int64_t split_upper = std::max(int64_t(1), numel / lower_elem_bound);

  // Rather than linearly searching the whole range, use the fact that any
  // divisor <= sqrt(numel) will be paired with another divisor >= sqrt(numel).
  // Therefore we can stop at sqrt(numel) since we want to minimize the split
  // divisor.
  int64_t sqrt_n = int64_t(std::ceil(std::sqrt(double(numel))));
  for (int64_t d = split_lower; d <= std::min(split_upper, sqrt_n); d++) {
    if (numel % d == 0) {
      int64_t tma_elems = numel / d;
      if (tma_elems % alignment == 0) {
        return d;
      }
    }
  }

  // The previous loop searched where the small divisor is within the range
  // [split_lower, split_upper]. Now we check for cases where the large divisor
  // is within that range.
  for (int64_t d = sqrt_n; d >= 1; d--) {
    if (numel % d == 0) {
      int64_t paired = numel / d;
      if (split_lower <= paired && paired <= split_upper) {
        int64_t tma_elems = numel / paired;
        if (tma_elems % alignment == 0) {
          return paired;
        }
      }
    }
  }

  return 0;
}
} // namespace

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

  uint64_t dtype_bytes = props.max_dtype_size_bit_for_vectorization / 8;
  uint64_t smem_elems = dev_prop->sharedMemPerBlockOptin / dtype_bytes;

  // Heuristics: Require TMA loads are at least 16KB, and consume up to half of
  // shared memory.
  constexpr int64_t min_tma_bytes = 16384;
  const int64_t lower_elem_bound = min_tma_bytes / dtype_bytes;
  const int64_t upper_elem_bound = smem_elems / 2;

  // TMA requires 16-byte alignment after any splits
  const int64_t aligned_elems = 16 / dtype_bytes;

  // Search for a suitable split factor
  const int64_t tma_split_factor = getTmaSplit(
      props.inner_most_dimension_numel,
      aligned_elems,
      lower_elem_bound,
      upper_elem_bound);

  // If no valid split factor was found, fallback to non-TMA
  if (tma_split_factor == 0) {
    return nullptr;
  }

  int64_t threads_per_block = 256;
  int64_t unroll_factor = scheduler_utils::lastPow2(target_vect_unroll);

  auto params = std::make_unique<TmaInnerReductionParams>();
  params->tma_split_factor = tma_split_factor;
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

  scheduler_utils::cacheAndForkOutputs(fusion, true);

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
      scheduler_utils::canonicalizeReduction(fusion, reduction_tv, false);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;
  NVF_ERROR(has_iter_axis && has_red_axis);

  if (rparams->tma_split_factor > 1) {
    reduction_tv->split(1, rparams->tma_split_factor, false);
    reduction_tv->axis(1)->parallelize(ParallelType::Serial);

    for (auto tma_tv : tma_tvs) {
      tma_tv->split(1, rparams->tma_split_factor, false);
    }
  }

  // Propagate the merges to all TMA TVs
  TransformPropagator tma_propagator(reduction_tv);
  SetSelector tma_selector({tma_tvs.begin(), tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reduction_tv, &tma_selector)
      .traverse(&tma_propagator);

  int64_t iter_axis = 0;
  int64_t inner_reduce_axis = rparams->tma_split_factor > 1 ? 2 : 1;

  // Schedule TMA tvs as [BIDx, Bulk]
  tma_tvs[0]->axis(iter_axis)->parallelize(ParallelType::BIDx);
  tma_tvs[0]->axis(inner_reduce_axis)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(tma_tvs[0], tma_tvs);

  // Non-TMA scheduling
  //
  // Apply splits following the pattern:
  // [I, R] -> [I, R/tidx, tidx]
  //        -> [I, R/tidx/unroll, unroll, tidx]

  // Split 1: TIDx (always applied)
  reduction_tv->split(inner_reduce_axis, rparams->threads_per_block);
  reduction_tv->axis(inner_reduce_axis + 1)->parallelize(ParallelType::TIDx);
  reduction_tv->axis(inner_reduce_axis + 1)->padToMultipleOfWarp();

  // Split 2: Inner unroll (outside of TIDx)
  if (rparams->unroll_factor > 1) {
    reduction_tv->split(inner_reduce_axis, rparams->unroll_factor);
    reduction_tv->axis(inner_reduce_axis + 1)
        ->parallelize(ParallelType::Unroll);
  }

  // Split 3: Unswitch (always applied)
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
