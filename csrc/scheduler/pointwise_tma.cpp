// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/pointwise_tma.h>

#include <ATen/cuda/CUDAContext.h>
#include <ir/utils.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

namespace nvfuser {
namespace pointwise_tma {

namespace {

// schedule the reference tv
// propagate transformation to non-tma tvs
void scheduleTma1DTile(
    Fusion* fusion,
    TensorView* reference_tv,
    const PointwiseParams* pparams,
    const std::vector<std::pair<TensorView*, int64_t>>& cached_inputs,
    const std::vector<std::pair<TensorView*, int64_t>>& cached_outputs) {
  {
    TransformPropagator propagator(reference_tv);
    MaxLogicalDomainInfoSpanningTree spanning_tree(reference_tv);
    spanning_tree.traverse(&propagator);
  }

  int opos = 0;
  int ipos = pparams->break_point == 0 ? 0 : 1;
  std::cout << "ipos: " << ipos << ", opos: " << opos << std::endl;
  std::cout << "reference_tv: " << reference_tv->toString() << std::endl;
  // collect all tma tvs and manual parallelize them
  auto load_type = pparams->is_1d_tma ? LoadStoreOpType::CpAsyncBulk
                                      : LoadStoreOpType::CpAsyncBulkTensorTile;
  std::vector<TensorView*> tma_tvs;
  std::vector<TensorView*> ldg_tvs;
  for (const auto& p : cached_inputs) {
    if (p.first->nDims() == 1 && ipos != opos) {
      ldg_tvs.push_back(p.first);
      continue;
    }
    auto tma_tv = p.first;
    tma_tv->definition()->as<LoadStoreOp>()->setOpType(load_type);
    tma_tv->setMemoryType(MemoryType::Shared);
    tma_tv->cacheAfter();
    tma_tvs.push_back(tma_tv);
    tma_tv->split(ipos, pparams->tma_tile_inner);
    tma_tv->axis(ipos)->parallelize(ParallelType::BIDx);
    if (pparams->is_1d_tma || pparams->tma_tile_inner <= 256) {
      tma_tv->axis(ipos + 1)->parallelize(ParallelType::Bulk);
    } else {
      tma_tv->split(ipos + 1, 256);
      tma_tv->axis(ipos + 1)->parallelize(ParallelType::TIDx);
      tma_tv->axis(ipos + 2)->parallelize(ParallelType::Bulk);
    }
    if (ipos != opos) {
      tma_tv->axis(opos)->parallelize(ParallelType::BIDy);
    }
  }
  std::vector<TensorView*> reg_tvs;
  if (pparams->vectorize_smem_to_regs_load) {
    for (auto tma_tv : tma_tvs) {
      auto reg_tv = tma_tv->cacheAfter();
      reg_tvs.push_back(reg_tv);
    }
  }

  // [I] -> [I, TMA] -> [I, Serial, TIDx, Vectorization]
  reference_tv->split(ipos, pparams->tma_tile_inner);
  reference_tv->split(ipos + 1, pparams->vectorization_factor);
  reference_tv->split(ipos + 1, pparams->lparams.bdimx());
  reference_tv->axis(ipos + 2)->parallelize(ParallelType::TIDx);
  // [I, Serial, TIDx, Vectorization] -> [I, Unswitch, Serial, TIDx,
  // Vectorization] unswitch_computation may increase register usage and
  // decrease performance
  int vect_pos = ipos + 3;
  if (pparams->unswitch_computation) {
    vect_pos++;
    reference_tv->split(ipos, 1);
    reference_tv->axis(ipos + 1)->parallelize(ParallelType::Unswitch);
  }
  reference_tv->axis(ipos)->parallelize(ParallelType::BIDx);
  if (ipos != opos) {
    reference_tv->axis(opos)->parallelize(ParallelType::BIDy);
  }
  // propagate transformation and parallelize non-tma tvs
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator propagator(reference_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  // vectorize regs -> global
  if (pparams->vectorization_factor > 1) {
    for (auto [_, original] : cached_outputs) {
      auto output_tv = fusion->outputs().at(original)->as<TensorView>();
      output_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }
  // vectorize shared -> regs
  if (pparams->vectorize_smem_to_regs_load) {
    for (auto reg_tv : reg_tvs) {
      reg_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }
  // ininle all except ldg_tvs
  std::vector<TensorView*> non_ldg_tvs =
      ir_utils::allTvsExcept(fusion, {ldg_tvs.begin(), ldg_tvs.end()});
  inlineMost(non_ldg_tvs);
}

void scheduleTma2DTile(
    Fusion* fusion,
    TensorView* reference_tv,
    const PointwiseParams* pparams,
    const std::vector<std::pair<TensorView*, int64_t>>& cached_inputs,
    const std::vector<std::pair<TensorView*, int64_t>>& cached_outputs) {
  {
    TransformPropagator propagator(reference_tv);
    MaxLogicalDomainInfoSpanningTree spanning_tree(reference_tv);
    spanning_tree.traverse(&propagator);
  }
  if (reference_tv->isFusionOutput()) {
    reference_tv = ir_utils::getSoleProducerTv(reference_tv);
  }

  struct TileMN {
    int64_t m;
    int64_t n;
  };
  TileMN tma_tile = {pparams->tma_tile_outer, pparams->tma_tile_inner};
  TileMN blk_tile = {pparams->lparams.bdimy(), pparams->lparams.bdimx()};
  TileMN tid_tile = {
      pparams->unroll_factor_outer, pparams->vectorization_factor};

  std::vector<TensorView*> tma_tvs;
  std::vector<TensorView*> ldg_tvs;
  for (const auto& p : cached_inputs) {
    if (p.first->nDims() == 1) {
      ldg_tvs.push_back(p.first);
      continue;
    }
    auto tma_tv = p.first;
    tma_tv->definition()->as<LoadStoreOp>()->setOpType(
        LoadStoreOpType::CpAsyncBulkTensorTile);
    tma_tv->setMemoryType(MemoryType::Shared);
    tma_tv->cacheAfter();
    tma_tvs.push_back(tma_tv);
    //[O, I] -> [O/m, m, I/n, n] -> [O/m, I/n, m, n] -> [O/m, I/n, m, n]
    tma_tv->split(1, tma_tile.n);
    tma_tv->split(0, tma_tile.m);
    tma_tv->reorder({{1, 2}});
    tma_tv->axis(0)->parallelize(ParallelType::BIDy);
    tma_tv->axis(1)->parallelize(ParallelType::BIDx);
    tma_tv->axis(2)->parallelize(ParallelType::Bulk);
    tma_tv->axis(3)->parallelize(ParallelType::Bulk);
  }
  if (pparams->use_tma_store) {
    for (auto [_, original] : cached_outputs) {
      auto output_tv = fusion->outputs().at(original)->as<TensorView>();
      auto output_smem_tv =
          output_tv->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
      output_smem_tv->setMemoryType(MemoryType::Shared);
      tma_tvs.push_back(output_tv);
      // [O, I] -> [O/m, m, I/n, n] -> [O/m, I/n, m, n] -> [O/m, I/n, m, n]
      output_tv->split(1, tma_tile.n);
      output_tv->split(0, tma_tile.m);
      output_tv->reorder({{1, 2}});
      output_tv->axis(0)->parallelize(ParallelType::BIDy);
      output_tv->axis(1)->parallelize(ParallelType::BIDx);
      output_tv->axis(2)->parallelize(ParallelType::Bulk);
      output_tv->axis(3)->parallelize(ParallelType::Bulk);
    }
  }

  // cache regs
  std::vector<TensorView*> reg_tvs;
  if (pparams->vectorize_smem_to_regs_load) {
    for (auto tma_tv : tma_tvs) {
      if (tma_tv->isFusionOutput()) {
        continue;
      }
      auto reg_tv = tma_tv->cacheAfter();
      reg_tvs.push_back(reg_tv);
    }
  }

  //[O, I] -> [O/m, m, I/n, n] -> [O/m, I/n, m, n]
  reference_tv->split(1, tma_tile.n);
  reference_tv->split(0, tma_tile.m);
  reference_tv->reorder({{1, 2}});

  // [O/m, I/n, m, n] -> [O/m, I/n, m, n/v/x, x, v]
  reference_tv->split(3, tid_tile.n);
  reference_tv->split(3, blk_tile.n);

  // [O/m, I/n, m, n/v/x, x, v] ->[O/m, I/n, m/y, y, n/v/x, x, v]
  reference_tv->split(2, blk_tile.m);
  // [O/m, I/n, m/y, y, n/v/x, x, v] -> [O/m, I/n, mnyxv, y, x, v]
  reference_tv->reorder({{3, 4}});
  reference_tv->merge(2);

  // [O/m, I/n, mnyxv, y, x, v]
  reference_tv->axis(0)->parallelize(ParallelType::BIDy);
  reference_tv->axis(1)->parallelize(ParallelType::BIDx);
  reference_tv->axis(3)->parallelize(ParallelType::TIDy);
  reference_tv->axis(4)->parallelize(ParallelType::TIDx);
  int vect_pos = 5;

  // propagate transformation and parallelize non-tma tvs
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator propagator(reference_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  // vectorize regs -> global
  if (pparams->vectorization_factor > 1 && !pparams->use_tma_store) {
    for (auto [_, original] : cached_outputs) {
      auto output_tv = fusion->outputs().at(original)->as<TensorView>();
      output_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }
  // vectorize shared -> regs
  if (pparams->vectorize_smem_to_regs_load) {
    for (auto reg_tv : reg_tvs) {
      reg_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }
  // ininle all except ldg_tvs
  std::vector<TensorView*> non_ldg_tvs =
      ir_utils::allTvsExcept(fusion, {ldg_tvs.begin(), ldg_tvs.end()});
  inlineMost(non_ldg_tvs);
  for (auto ldg_tv : ldg_tvs) {
    std::cout << "ldg_tv: " << ldg_tv->toString() << std::endl;
    // outer bcast: [I/n, n/v/x, x, v]
    // inner bcast: [O/m, m/y, y]
    inlineSelectedAt({ldg_tv}, ldg_tv, 1);
    if (std::any_of(
            ldg_tv->getLoopDomain().begin(),
            ldg_tv->getLoopDomain().end(),
            [](IterDomain* id) {
              return id->getParallelType() == ParallelType::TIDx;
            })) {
      ldg_tv->axis(vect_pos - 2)->parallelize(ParallelType::Vectorize);
    }
  }
}

} // namespace

void getHeuristics(
    PointwiseParams* pparams,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  // TMA-specific heuristics are already set in the main getPointwiseHeuristics
  // This function can be used for additional TMA-specific tuning if needed
}

void scheduleFusion(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  TensorView* reference_tv = pointwise_utils::getReferenceTensor(fusion);
  NVF_ERROR(
      reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");

  if (pparams->break_point == 0) {
    // 1D TMA
    scheduleTma1DTile(
        fusion, reference_tv, pparams, cached_inputs, cached_outputs);
  } else if (pparams->tma_tile_outer > 1) {
    // 2D TMA
    scheduleTma2DTile(
        fusion, reference_tv, pparams, cached_inputs, cached_outputs);
  } else {
    // 1D TMA with break point
    scheduleTma1DTile(
        fusion, reference_tv, pparams, cached_inputs, cached_outputs);
  }
}

} // namespace pointwise_tma
} // namespace nvfuser
