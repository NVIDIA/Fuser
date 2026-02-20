// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/reduction_outer_tma.h"

#include <ATen/cuda/CUDAContext.h>

#include "scheduler/cache_policy_refiner.h"
#include "scheduler/reduction_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/utils.h"

namespace nvfuser {
namespace reduction {
namespace outer_tma {

std::unique_ptr<TmaOuterReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props) {
  FusionGuard fg(fusion);

  // TODO: These heuristics are stubbed out based on the manual test
  // (TmaOuterReductionManualTest::Basic). They need proper tuning.

  // 2D thread block: TIDx covers iteration, TIDy covers reduction
  const int64_t bdimx = 32;
  const int64_t bdimy = 16;

  // TMA tile sizes. Unroll factors are derived from these and thread dims.
  const int64_t tma_tile_i = 128;
  const int64_t tma_tile_r = 128;

  NVF_ERROR(tma_tile_i % bdimx == 0);
  NVF_ERROR(tma_tile_r % bdimy == 0);

  const int64_t iter_unroll_factor = tma_tile_i / bdimx;

  // Grid dimension for parallelizing the outer reduction across CTAs.
  // Modeled after the manual test: clamp lastPow2(outer_size / 256) to [1, 8].
  const int64_t outer_size = props.total_reduction_numel;
  int64_t grdim = std::max<int64_t>(
      1, std::min<int64_t>(8, scheduler_utils::lastPow2(outer_size / 256)));

  auto params = std::make_unique<TmaOuterReductionParams>();
  params->bdimx = bdimx;
  params->bdimy = bdimy;
  params->tma_tile_i = tma_tile_i;
  params->tma_tile_r = tma_tile_r;
  params->iter_unroll_factor = iter_unroll_factor;
  params->grdim = grdim;

  params->tag = "Outer Reduction TMA heuristics";
  params->cparams.index_type = runtime_info.getIndexType();

  return params;
}

void scheduleReduction(Fusion* fusion, const TmaOuterReductionParams* rparams) {
  FusionGuard fg(fusion);

  const int64_t bdimx = rparams->bdimx;
  const int64_t bdimy = rparams->bdimy;
  const int64_t tma_tile_i = rparams->tma_tile_i;
  const int64_t tma_tile_r = rparams->tma_tile_r;
  const int64_t iter_unroll_factor = rparams->iter_unroll_factor;
  const int64_t grdim = rparams->grdim;

  NVF_ERROR(tma_tile_i % bdimx == 0);
  NVF_ERROR(tma_tile_r % bdimy == 0);

  // Phase 1: Cache inputs into shared memory via TMA
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  std::vector<TensorView*> tma_tvs;
  for (auto [tv, input_idx] : cached_inputs) {
    if (auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition())) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulkTensorTile);
      tv->setMemoryType(MemoryType::Shared);
      tma_tvs.push_back(tv);
    }
  }
  NVF_ERROR(!tma_tvs.empty());

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  NVF_ERROR(!reduction_tvs.empty());
  TensorView* reduction_tv = reduction_tvs.at(0);

  // canonicalizeReduction with schedule_3d=false merges into [I, R] form.
  // For outer reduction, we want [R, I], so we reorder after canonicalization.
  auto dim_analysis =
      scheduler_utils::canonicalizeReduction(fusion, reduction_tv, false);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;
  NVF_ERROR(has_iter_axis && has_red_axis);

  // Reorder from [I, R] -> [R, I] for outer reduction pattern
  reduction_tv->reorder({{0, 1}, {1, 0}});

  // The reduction_tv already has a cacheBefore from cacheAndForkOutputs.
  // Use reduction_tv directly as our reduction reference for scheduling.
  TensorView* redu_tv = reduction_tv;

  // After canonicalization + reorder: [R, I]
  const int64_t outer_reduce_axis = 0;

  // Phase 2: Schedule TMA tensor with 2D TMA tiling
  // Apply transforms to the TMA smem TV.
  // We start from the TMA TV, which shares the same logical domain as the
  // reduction TV's producer.
  TensorView* tma_tv = tma_tvs[0];

  // [R, I] -> [R/tma_tile_r, tma_tile_r, I]
  tma_tv->split(outer_reduce_axis, tma_tile_r);

  // [R/tma_tile_r, tma_tile_r, I] -> [R/tma_tile_r, tma_tile_r, I/tma_tile_i,
  // tma_tile_i]
  tma_tv->split(2, tma_tile_i);

  // Split outer reduction for grid parallelization (BIDy)
  // [R/tma_tile_r, tma_tile_r, I/tma_tile_i, tma_tile_i]
  // -> [grdim, R', tma_tile_r, I/tma_tile_i, tma_tile_i]
  //      0      1    2           3              4
  tma_tv->split(0, grdim, false);

  // Phase 3: Propagate TMA tiling to all tensors
  TransformPropagator propagator(tma_tv);
  MaxLogicalDomainInfoSpanningTree(tma_tv).traverse(&propagator);

  // Phase 4: Parallelize TMA tensor
  tma_tv->axis(0)->parallelize(ParallelType::BIDy);
  tma_tv->axis(1)->parallelize(ParallelType::Serial);
  tma_tv->axis(2)->parallelize(ParallelType::Bulk); // reduction tile
  tma_tv->axis(3)->parallelize(ParallelType::BIDx);
  tma_tv->axis(4)->parallelize(ParallelType::Bulk); // iteration tile

  // Parallelize remaining TMA tvs to match
  scheduler_utils::parallelizeAllLike(tma_tv, tma_tvs);

  // Phase 5: Sub-split TMA tiles into thread dims
  // Split tma_tile_i (axis 4) into [bdimx, iter_unroll]
  redu_tv->split(4, iter_unroll_factor);

  // Split tma_tile_r (axis 2) into [redu_unroll, bdimy]
  redu_tv->split(2, bdimy);
  // Now: [grdim, R', redu_unroll, bdimy, I/tma_tile_i, bdimx, iter_unroll]
  //        0      1   2            3      4              5      6

  // Phase 6: Parallelize reduction tensor
  redu_tv->axis(0)->parallelize(ParallelType::BIDy);
  redu_tv->axis(1)->parallelize(ParallelType::Serial);
  redu_tv->axis(2)->parallelize(ParallelType::Unroll); // redu_unroll
  redu_tv->axis(3)->parallelize(ParallelType::TIDy); // bdimy
  redu_tv->axis(4)->parallelize(ParallelType::BIDx);
  redu_tv->axis(5)->parallelize(ParallelType::TIDx); // bdimx
  // Use Vectorize so it gets converted to Group for iterGroupedGridReduce
  redu_tv->axis(6)->parallelize(ParallelType::Vectorize); // iter_unroll

  // Phase 7: rFactor for grid reduction
  // rFactor reduction axes that are not thread-parallelized
  std::vector<int64_t> rfactor_axes;
  for (int64_t i = 0; i < redu_tv->nDims(); i++) {
    if (redu_tv->axis(i)->isReduction() && !redu_tv->axis(i)->isThread()) {
      rfactor_axes.push_back(i);
    }
  }

  TensorView* reference_tv = redu_tv;
  if (!rfactor_axes.empty()) {
    reference_tv = redu_tv->rFactor(rfactor_axes);
  }

  // Phase 8: Propagate thread-level splits to non-TMA TVs
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector non_tma_selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &non_tma_selector)
      .traverse(&non_tma_propagator);

  // Phase 9: Propagate parallelization with iter-grouped reduction
  const bool use_iter_grouped_reduction = true;

  if (reference_tv != redu_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, redu_tv, reduction_tvs);
    non_tma_tvs =
        ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  }

  // Collect output TVs for unroll_vectorizable_cached_tvs
  std::unordered_set<TensorView*> output_tvs;
  for (auto output : fusion->outputs()) {
    if (auto tv = dynamic_cast<TensorView*>(output)) {
      output_tvs.insert(tv);
    }
  }

  reduction_scheduler_utils::propagateParallelization(
      redu_tv,
      reference_tv,
      /*is_unroll_or_vectorization=*/true,
      use_iter_grouped_reduction,
      reduction_tvs,
      /*unroll_vectorizable_cached_tvs=*/output_tvs,
      /*selected_tvs=*/non_tma_tvs);

  // Phase 10: Inline and refine
  inlineMost();

  refineCachePolicy(fusion);
}

} // namespace outer_tma
} // namespace reduction
} // namespace nvfuser
