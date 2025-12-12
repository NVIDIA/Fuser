// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

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
  auto params = std::make_unique<TmaInnerReductionParams>();
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

  [[maybe_unused]] auto serialize = [&reduction_tv](
                                        int64_t axis, int64_t factor) {
    reduction_tv->split(axis, factor);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Serial);
  };

  [[maybe_unused]] auto vectorize = [&reduction_tv](
                                        int64_t axis, int64_t factor) {
    reduction_tv->split(axis, factor);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Vectorize);
  };

  [[maybe_unused]] auto inner_unswitch = [&reduction_tv](int64_t axis) {
    reduction_tv->split(axis, 1);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Unswitch);
  };

  [[maybe_unused]] auto inner_unroll = [&reduction_tv](
                                           int64_t axis, int64_t factor) {
    reduction_tv->split(axis, factor);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Unroll);
  };

  [[maybe_unused]] auto inner_parallel_static =
      [&reduction_tv](int64_t axis, ParallelType ptype, int64_t factor) {
        reduction_tv->split(axis, factor);
        reduction_tv->axis(axis + 1)->parallelize(ptype);
      };

  [[maybe_unused]] auto inner_parallel = [&reduction_tv](
                                             int64_t axis, ParallelType ptype) {
    reduction_tv->split(axis, NamedScalar::getParallelDim(ptype));
    reduction_tv->axis(axis + 1)->parallelize(ptype);
  };

  // TODO: Compute as heuristic on TmaInnerReductionParams
  [[maybe_unused]] const int64_t vect = 4;
  [[maybe_unused]] const int64_t tidx = 256;
  [[maybe_unused]] const int64_t unroll = 4;

  serialize(inner_reduce_axis, vect);

  inner_parallel_static(inner_reduce_axis, ParallelType::TIDx, tidx);

  inner_unroll(inner_reduce_axis, unroll);

  inner_unswitch(inner_reduce_axis);

  reduction_tv->axis(inner_reduce_axis)->parallelize(ParallelType::Serial);
  reduction_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);

  // TODO: rFactor reduction tv

  // Schedule non-TMA tvs based on reduction tv
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reduction_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reduction_tv, &selector)
      .traverse(&non_tma_propagator);

  scheduler_utils::parallelizeAllLike(reduction_tv, non_tma_tvs);

  // TODO: Vectorize load from smem
  // TODO: TMA or vectorize store

  inlineMost();
}
} // namespace tma
} // namespace reduction
} // namespace nvfuser
