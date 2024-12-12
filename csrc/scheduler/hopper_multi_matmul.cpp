// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/circular_buffer.h>
#include <disjoint_set.h>
#include <id_model/schedule.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/hopper_multi_matmul.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <utils.h>
#include <val_graph.h>
#include <val_graph_visitor.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <runtime/executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

MatmulDimRole HopperMultipleMatmulScheduler::findMatmulDimRole(IterDomain* id) {
  ValGroup vg = graph_->toGroup(id);
  auto it = id_roles_.find(vg);
  NVF_ERROR(it != id_roles_.end());
  return it->second;
}

void HopperMultipleMatmulScheduler::run() {
  // Clears memory spaces on intermediate tensors, calls
  // cache{After,Before,Fork} on inputs and outputs
  cacheInputsAndOutputs();

  // Finds matmul patterns and translates them to MmaOps, then finds tensor
  // and dimension roles for all tensors in the fusion
  findPatterns();
  translatePatterns();
  findRoles();

  // Defines acw_smem/bcw_smem and acr/bcr by possibly calling cacheAfter.
  // This also collects mma_results_
  defineOperandCaches();

  inspectPrologues();

  setCGADims();

  scheduleOperands();

  // schedule mma instruction output (mma_result)
  scheduleMmaResults();

  // schedule epilogue
  scheduleEpilogue();

  // schedule splitk_sum
  scheduleSplitKSum();

  setUpInlining();

  // set up circular buffering. This must come after everything up to
  // mma_result is scheduled, since everything in the main loop will need to
  // be rotated
  setUpCircularBuffering();
}

void HopperMultipleMatmulScheduler::cacheInputsAndOutputs() {
  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion_);

  // Cache inputs
  scheduler_utils::cacheInputs(fusion_, /*unroll=*/true);

  // Cache and fork outputs
  scheduler_utils::cacheAndForkOutputs(fusion_, /*unroll=*/true);
}

void HopperMultipleMatmulScheduler::defineOperandCaches() {
  cacheOperandsToSmem(as_, acw_smems_);
  cacheOperandsToSmem(bs_, bcw_smems_);

  // Now that we are finished possibly redefining the inputs to the MmaOps,
  // we can set the macro for those ops
  for (TensorView* mma_result : mma_results_) {
    MmaOp* mma = dynamic_cast<MmaOp*>(mma_result->definition());
    NVF_ERROR(mma != nullptr);
    mma->setMacro(params_->mma_macro);
  }
}

void HopperMultipleMatmulScheduler::cacheOperandsToSmem(
    const std::vector<TensorView*>& operands,
    std::vector<TensorView*>& smem_operands) {
  // Use cp.async.bulk (tma) as requested in scheduler params.
  smem_operands.resize(operands.size(), nullptr);
  for (size_t i : c10::irange(operands.size())) {
    TensorView* operand = operands[i];

    NVF_ERROR(operand->uses().size() == 1);
    smem_operands[i] = ir_utils::consumerTvsOf(operand).at(0);

    LoadStoreOpType load_op = params_->async_gmem_load_operands
        ? LoadStoreOpType::CpAsyncBulkTensorTile
        : LoadStoreOpType::Set;

    smem_operands[i]->definition()->as<LoadStoreOp>()->setOpType(load_op);
    smem_operands[i]->setMemoryType(MemoryType::Shared);
  }
}

void HopperMultipleMatmulScheduler::swizzleBlockTiles(
    TensorView* tv,
    std::vector<MatmulDimRole>& outer_dim_roles) {
  if (params_->grid_swizzle_factor != 1) {
    // Find position of outer M and N dims in schedule_.tiled
    int64_t Mo_pos = -1, No_pos = -1;
    for (size_t i : c10::irange(outer_dim_roles.size())) {
      if (outer_dim_roles[i] == MatmulDimRole::M) {
        Mo_pos = (int64_t)i;
      } else if (outer_dim_roles[i] == MatmulDimRole::N) {
        No_pos = (int64_t)i;
      }
    }

    int factor = std::max(1, params_->grid_swizzle_factor); // must be >=1
    switch (params_->cta_order) {
      case MatmulParams::TileRasterizationOrder::RowMajor:
        // split   [I1, I2/factor, factor]
        // reorder [I1, factor, I2/factor]
        // merge   [I1*factor, I2/factor]
        // where I1 and I2 are the outer M and N dimensions, respectively
        if (No_pos >= 0) {
          tv->split(No_pos, factor);
          // If No_pos < Mo_pos, then the split above shifts Mo_pos by one
          if (No_pos < Mo_pos) {
            Mo_pos++;
          }
          tv->reorder({{No_pos, No_pos + 1}});
          if (Mo_pos >= 0) {
            tv->merge(Mo_pos, No_pos);
          } else {
            // M is missing, so we skip the merge above. In this case we
            // should update the dim roles to reflect the new split axis.
            outer_dim_roles.insert(
                outer_dim_roles.begin() + No_pos, MatmulDimRole::N);
          }
        }
        break;

      case MatmulParams::TileRasterizationOrder::ColumnMajor:
        // split   [I1/factor, factor, I2]
        // reorder [I1/factor, I2, factor]
        // merge   [I1/factor, I2*factor]
        // where I1 and I2 are the outer M and N dimensions, respectively
        if (Mo_pos >= 0) {
          tv->split(Mo_pos, factor);
          // If No_pos < Mo_pos, then the split above shifts Mo_pos by one
          if (No_pos > Mo_pos) {
            No_pos++;
          }
          if (No_pos >= 0) {
            tv->reorder({{Mo_pos + 1, No_pos}});
            tv->merge(Mo_pos + 1, No_pos);
          } else {
            // N is missing, so we skip the merge above. In this case we
            // should update the dim roles to reflect the new split axis.
            outer_dim_roles.insert(
                outer_dim_roles.begin() + Mo_pos, MatmulDimRole::M);
          }
        }
    }
  }
}

TensorView* HopperMultipleMatmulScheduler::cacheAfter(
    TensorView* orig,
    LoadStoreOpType op_type,
    CacheOp cache_op,
    bool propagate_allocation_domain) {
  const std::vector<IterDomain*> orig_alloc = orig->getMaybeAllocationDomain();

  TensorView* c =
      orig->cacheAfter(op_type, cache_op, propagate_allocation_domain);

  if (propagate_allocation_domain) {
    const std::vector<IterDomain*> cache_alloc = c->getMaybeAllocationDomain();
    NVF_ERROR(orig_alloc.size() == cache_alloc.size());
    for (size_t i : c10::irange(orig_alloc.size())) {
      ValGroup vg = graph_->toGroup(orig_alloc[i]);
      graph_->initializeVal(cache_alloc[i], vg);
    }
  }

  const std::vector<IterDomain*> orig_logical =
      TensorDomain::noReductions(orig->getLogicalDomain());
  const std::vector<IterDomain*> cache_logical = c->getLogicalDomain();
  // in split-K we do rFactor which gives us a full = sum(partial)
  // where partial has root domain that matches the logical domain of the
  // original tensor. The logical domain contains Iteration transforms of the
  // Reduction axis in the original mma output.
  NVF_ERROR(orig_logical.size() == cache_logical.size());
  for (size_t i : c10::irange(orig_logical.size())) {
    ValGroup vg = graph_->toGroup(orig_logical[i]);
    graph_->initializeVal(cache_logical[i], vg);
  }

  return c;
}

std::vector<std::vector<MatmulDimRole>> HopperMultipleMatmulScheduler::
    blockTileTensors(const std::vector<TensorView*>& tvs) {
  if (canonical_dim_ordering_.empty()) {
    canonical_dim_ordering_ =
        mma_utils::canonicalDimOrdering(tensor_roles_, id_roles_, *graph_);
  }

  std::vector<std::vector<MatmulDimRole>> all_merged_roles;
  for (TensorView* tv : tvs) {
    // Find dimensions in canonical_dim_ordering_ that exist in tv's loop
    // domain. Reorder those according to the canonical dim ordering then
    std::unordered_map<ValGroup, IterDomain*> tv_dims;
    std::unordered_set<MatmulDimRole> axis_roles;
    for (IterDomain* id : tv->getLoopDomain()) {
      ValGroup vg = graph_->toGroup(id);
      tv_dims.emplace(vg, id);
      // track axis roles in this tensor to use in makeTile
      auto it = id_roles_.find(vg);
      NVF_ERROR(it != id_roles_.end());
      axis_roles.insert(it->second);
    }
    std::vector<IterDomain*> new_loop;
    new_loop.reserve(tv->nDims());
    for (const ValGroup& vg : canonical_dim_ordering_) {
      auto it = tv_dims.find(vg);
      if (it != tv_dims.end()) {
        new_loop.push_back(it->second);
      }
    }
    NVF_ERROR((int64_t)new_loop.size() == tv->nDims());
    tv->setLoopDomain(new_loop);

    // There could be multiple dimensions with the same role at this point, so
    // now we collect them. After this, tv will be at most 4 dimensions e.g.
    // BMNK based on canonical_dim_ordering_, with any of these dimensions
    // possibly missing.
    mma_utils::mergeConsecutiveAxesWithSameRole(tv, id_roles_, graph_);

    // Find order the axes that are present in the merged tensor
    std::vector<MatmulDimRole> merged_roles;
    merged_roles.reserve(tv->nDims());
    for (const ValGroup& vg : canonical_dim_ordering_) {
      MatmulDimRole role = id_roles_[vg];
      if (axis_roles.count(role) != 0) {
        if (merged_roles.empty() || merged_roles.back() != role) {
          merged_roles.push_back(role);
        }
      }
    }
    NVF_ERROR(merged_roles.size() == axis_roles.size());

    // TODO: (to be pursued after the multi-matmul refactor is fully merged)
    // this currently creates a separate AbstractMatmulTensor for each
    // TensorView. Instead, we should create a single AbstractMatmulTensor
    // then apply it (with "forwarding") to each TV instead. We already cache
    // a vector<ValGroup> as canonical_dim_ordering_ so AbstractTensor
    // scheduling is the next step in this modernization.
    mma_utils::makeTile(tv, params_->tile_sizes.cta_tile, merged_roles);

    swizzleBlockTiles(tv, merged_roles);

    all_merged_roles.push_back(merged_roles);

    if (params_->splitk_factor > 1) {
      // Outer K dimension in tv is in same position found in merged_roles
      for (size_t i : c10::irange(merged_roles.size())) {
        if (merged_roles[i] == MatmulDimRole::K) {
          tv->split((int64_t)i, params_->splitk_factor, /*inner*/ false);
        }
      }
    }
  }
  return all_merged_roles;
}

void HopperMultipleMatmulScheduler::inspectPrologues() const {
  for (TensorView* mma_result : mma_results_) {
    for (Val* v : mma_result->definition()->inputs()) {
      TensorView* op_input = v->as<TensorView>();

      // We currently require all operands to lie in smem, meaning we cannot yet
      // handle any prologue computation. This includes `BroadcastOp` which
      // might be introduced when translating a MatmulOp or LinearOp to MmaOp.
      Expr* def = op_input->definition();
      NVF_ERROR(def != nullptr && def->isA<LoadStoreOp>());
      NVF_ERROR(def->input(0)->isFusionInput());
    }
  }
}

void HopperMultipleMatmulScheduler::scheduleOperands() {
  NVF_CHECK(
      params_->async_gmem_load_operands,
      "Hopper matmul scheduler currently requires TMA to be enabled");
  auto scheduleBranch = [&](const std::vector<TensorView*>& gmem_operands,
                            const std::vector<TensorView*>& smem_operands,
                            MmaOperand operand_type) {
    blockTileTensors(smem_operands);
    for (TensorView* tv : smem_operands) {
      if (params_->promote_prologue_smem_reuse) {
        tv->promoteReuse();
      }
      mma_utils::orderTiledConcreteIdAsMaybeAllocationDomain(tv);
      MmaInputSmemSwizzle swizzle_type = mma_utils::tmaSwizzleSharedMemory(tv);
      tv->applyMmaSwizzleForTMALoad(swizzle_type);
    }
  };
  scheduleBranch(as_, acw_smems_, MmaOperand::A);
  scheduleBranch(bs_, bcw_smems_, MmaOperand::B);
}

void HopperMultipleMatmulScheduler::parallelizeBlocks(
    const std::vector<TensorView*>& tvs) const {
  for (TensorView* tv : tvs) {
    switch (params_->cta_order) {
      // TODO: Should we instead check the roles of these dimensions to take the
      // outermost two M or N axes?
      case MatmulParams::TileRasterizationOrder::RowMajor:
        tv->axis(num_device_and_batch_dims_)->parallelize(ParallelType::BIDx);
        tv->axis(num_device_and_batch_dims_ + 1)
            ->parallelize(ParallelType::BIDy);
        break;
      case MatmulParams::TileRasterizationOrder::ColumnMajor:
        tv->axis(num_device_and_batch_dims_)->parallelize(ParallelType::BIDy);
        tv->axis(num_device_and_batch_dims_ + 1)
            ->parallelize(ParallelType::BIDx);
        break;
      default:
        NVF_THROW("Invalid TileRasterizationOrder passed to Matmul scheduler");
    }
  }
}

void HopperMultipleMatmulScheduler::scheduleMmaResults() {
  GemmTile instruction_tile = getMmaOpShape(params_->mma_macro);
  NVF_CHECK(
      params_->tile_sizes.cta_tile.k == params_->tile_sizes.warp_tile.k,
      "CTA tile must match warp tile K dimension for Hopper matmul but found ",
      toString(params_->tile_sizes));
  // If cta_tile is not divisible by instruction tile the mma instruction will
  // be predicated.
  NVF_CHECK(
      params_->tile_sizes.cta_tile.m % instruction_tile.m == 0 &&
          params_->tile_sizes.cta_tile.n % instruction_tile.n == 0 &&
          params_->tile_sizes.cta_tile.k % instruction_tile.k == 0,
      "CTA tile must be divisible by macro size but found cta_tile: ",
      toString(params_->tile_sizes.cta_tile),
      " and macro: ",
      toString(params_->mma_macro));

  // Schedule mma results and propagate forward
  auto all_merged_roles = blockTileTensors(mma_results_);
  parallelizeBlocks(mma_results_);
  for (size_t i : c10::irange(mma_results_.size())) {
    TensorView*& mma_result = mma_results_[i];
    const std::vector<MatmulDimRole>& merged_roles = all_merged_roles[i];

    // Test that mma_result logical is MNK
    // TODO: This currently checks leaf domain only which does not necessarily
    // match logical
    // TODO: Lift this constraint. Use commitLeafToLogical if necessary. We
    // might just want to match using id_roles_
    NVF_ERROR(merged_roles.size() >= 3);
    const auto checkSingleDimRole =
        [&merged_roles](int64_t pos, MatmulDimRole expected_role) {
          if (pos < 0) {
            pos += (int64_t)merged_roles.size();
          }
          NVF_ERROR(pos >= 0);
          NVF_ERROR(pos < (int64_t)merged_roles.size());
          const auto& actual_role = merged_roles[(size_t)pos];
          NVF_ERROR(actual_role == expected_role);
        };
    checkSingleDimRole(-3, MatmulDimRole::M);
    checkSingleDimRole(-2, MatmulDimRole::N);
    checkSingleDimRole(-1, MatmulDimRole::K);

    // do split-K rFactor to define splitk_sum and smem_epilogue
    if (params_->splitk_factor != 1) {
      // Note that the split-K split is already done in blockTileTensors
      TensorView* splitk_sum = mma_result->rFactor({-4, -1});
      std::swap(splitk_sum, mma_result);
      splitk_sums_.push_back(splitk_sum);
    }

    mma_result->split(-3, getM(params_->mma_macro));
    mma_result->split(-2, getN(params_->mma_macro));
    // [Mo, No, Ko, Mio, Mii, Nio, Nii, Ki]
    // -> [Mo, No, Ko, Mio, Nio, Mii, Nii, Ki]
    mma_result->reorder({{-4, -3}});
    mma_result->merge(-5);
    mma_result->axis(-4)->parallelize(ParallelType::TIDy);

    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        mma_result->getLoopDomain());
    mma_result->setAllocationDomain(s.as<IterDomain*>(), true);
    mma_result->axis(-1)->parallelize(ParallelType::Mma);
    mma_result->axis(-2)->parallelize(ParallelType::Mma);
    mma_result->axis(-3)->parallelize(ParallelType::Mma);
  }
}

void HopperMultipleMatmulScheduler::scheduleOutputTensor(TensorView* c) {
  const MatMulTileOptions& gemm_tile = params_->tile_sizes;
  const int64_t vectorization_factor = params_->supported_vec_size.epilogue;
  // input tensor is in the form of [Mo,No,cta_tile_m,cta_tile_n]
  mma_utils::checkConcreteStaticDim(c->axis(-2));
  mma_utils::checkConcreteStaticDim(c->axis(-1));
  const int64_t tile_size_m = c->axis(-2)->extent()->evaluate().as<int64_t>();
  const int64_t tile_size_n = c->axis(-1)->extent()->evaluate().as<int64_t>();
  NVF_ERROR(
      tile_size_m == gemm_tile.cta_tile.m,
      "Actual tile size at axis(-2) in output tensor is different from CTA tile size! Expected: ",
      gemm_tile.cta_tile.m,
      ", actual: ",
      tile_size_m);
  NVF_ERROR(
      tile_size_n == gemm_tile.cta_tile.n,
      "Actual tile size at axis(-1) in output tensor is different from CTA tile size! Expected: ",
      gemm_tile.cta_tile.n,
      ", actual: ",
      tile_size_n);
  const int64_t tot_elements = tile_size_m * tile_size_n;
  constexpr int64_t warp_size = 32l;
  const int64_t tidx = warp_size;
  const int64_t tidy = gemm_tile.cta_tile.n / gemm_tile.warp_tile.n;
  const int64_t tidz = gemm_tile.cta_tile.m / gemm_tile.warp_tile.m;
  // step-1, merge last 2 dims
  c->merge(-2);
  // [Mo, No, m*n]

  // step-2, set vectorization to maximum
  // We have fixed tidx, tidy, and tidz, so we need to make sure that the
  // output tensor is divisible by tidx * tidy * tidz * vectorization_factor
  NVF_ERROR(
      tot_elements % (tidx * tidy * tidz * vectorization_factor) == 0,
      "Output tensor cannot be fully vectorized! tot_elements:",
      tot_elements,
      ", tidx: ",
      tidx,
      ", tidy: ",
      tidy,
      ", tidz: ",
      tidz,
      ", vectorization_factor: ",
      vectorization_factor);
  c->split(-1, vectorization_factor);
  c->axis(-1)->parallelize(ParallelType::Vectorize);
  // [Mo, No, m*n/vect, vect]

  // step-3, Split out a warp for TIDx
  c->split(-2, tidx);
  c->axis(-2)->parallelize(ParallelType::TIDx);
  // [Mo, No, m*n/vect/TIDx, TIDx, vect]

  // step-4, Split out for TIDy and TIDz
  // TIDy = cta_tile_n/warp_tile_n
  // TIDz = cta_tile_m/warp_tile_m
  c->split(-3, tidy);
  c->axis(-3)->parallelize(ParallelType::TIDy);

  c->split(-4, tidz);
  c->axis(-4)->parallelize(ParallelType::TIDz);
  // [Mo, No, m*n/vect/TIDx/TIDy/TIDz, TIDz, TIDy, TIDx, vect]

  for (TensorView* mma_result : mma_results_) {
    // step-5, Parallel first 2 dims same as mma_result
    scheduler_utils::parallelizeAllLike(
        mma_result,
        2,
        {c},
        {ParallelType::BIDx, ParallelType::BIDy, ParallelType::BIDz});
  }
}

void HopperMultipleMatmulScheduler::scheduleEpilogue() {
  // TODO: schedule epilogue by propagation backward from dc
  if (!params_->use_smem_epilogue) {
    for (Val* dv : fusion_->outputs()) {
      auto* d = dv->as<TensorView>();
      NVF_ERROR(d->definition() && d->definition()->isA<LoadStoreOp>());
      auto* dc = d->definition()->input(0)->as<TensorView>();

      // Block Schedule and Parallelize
      blockTileTensors({dc, d});
      parallelizeBlocks({dc, d});

      // Apply mma common transformation
      for (auto tv : {dc, d}) {
        // [..., Mo, No, Mi, Ni]
        tv->split(-2, getM(params_->mma_macro));
        tv->split(-1, getN(params_->mma_macro));
        // [..., Mo, No, Mio, Mii, Nio, Nii]
        // -> [..., Mo, No, Mio, Nio, Mii, Nii]
        tv->reorder({{-3, -2}});
        tv->merge(-4);
        auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
            tv->getLoopDomain());
        tv->setLoopDomain(s.as<IterDomain*>());
        tv->axis(-5)->parallelize(ParallelType::TIDy);
      }
      d->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  } else {
    constexpr int64_t stmatrix_tile_m = 16;
    constexpr int64_t stmatrix_tile_n = 16;

    // TODO: Support tma tile sizes that are a multiple of mma_macro.
    // The wgmma operation creates an output matrix of mma_macro size. The TMA
    // tile is a multiple of the macro size because stmatrix stores results from
    // wgmma to shared memory. For maximum inlining and to reduce shared memory
    // usage, the tma tile is mma_macro size.
    const int64_t tma_m = getM(params_->mma_macro);
    const int64_t tma_n = getN(params_->mma_macro);

    fusion_->manage("st_matrix_m_tile", stmatrix_tile_m);
    fusion_->manage("st_matrix_n_tile", stmatrix_tile_n);
    fusion_->manage("st_matrix_m", tma_m);
    fusion_->manage("st_matrix_n", tma_n);

    // Manually schedule register cache and output TensorView
    for (Val* dv : fusion_->outputs()) {
      auto* d = dv->as<TensorView>();
      NVF_ERROR(d->definition() && d->definition()->isA<LoadStoreOp>());
      auto* dc = d->definition()->input(0)->as<TensorView>();

      // NOTE: cacheBefore does not work with blockTileTensors
      TensorView* d_smem = cacheAfter(dc, LoadStoreOpType::Set);

      // Set MemoryType
      dc->setMemoryType(MemoryType::Local);
      d_smem->setMemoryType(MemoryType::Shared);

      // Set LoadStoreOp
      d_smem->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::StMatrix);
      d->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::CpAsyncBulkTensorTile);

      // Block Schedule and Parallelize
      blockTileTensors({dc, d_smem, d});
      parallelizeBlocks({dc, d_smem, d});

      // Apply mma common transformation
      for (auto tv : {dc, d_smem, d}) {
        // Original: [..., Mo, No, Mi, Ni]
        tv->split(-2, getM(params_->mma_macro));
        tv->split(-1, getN(params_->mma_macro));
        // After Split: [..., Mo, No, Mio, Mii, Nio, Nii]
        tv->reorder({{-3, -2}});
        // After Reorder: [..., Mo, No, Mio, Nio, Mii, Nii]
        tv->merge(-4);
        // After Merge: [..., Mo, No, Mio * Nio, Mii, Nii]
        tv->axis(-3)->parallelize(ParallelType::TIDy);
        // After Parallelize: [..., Mo, No, Mio * Nio (TIDy), Mii, Nii]
      }

      // Schedule register cache; Output from epilogue
      {
        auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
            dc->getLoopDomain());
        dc->setLoopDomain(s.as<IterDomain*>());
        dc->setAllocationDomain(s.as<IterDomain*>(), true);
      }

      MmaInputSmemSwizzle swizzle = mma_utils::tmaSwizzleSharedMemory(d_smem);

      // Schedule shared memory cache; Output from StMatrix
      mma_utils::scheduleStMatrixForMmaOutput(
          d_smem, swizzle, stmatrix_tile_m, stmatrix_tile_n);

      // Schedule global memory output; Output from TMA Store
      mma_utils::scheduleTMAStoreForMmaOutput(d, swizzle);
    }
  }
}

//! Propagates transformations from fusion output to fusion tv inputs that are
//!  producers in the epilogue. Transformations' propagation aims at input tvs
//!  which are not assigned to core roles, that is, are not MMA inputs.
void HopperMultipleMatmulScheduler::scheduleFusionInputsForEpilogue() {
  std::vector<TensorView*> cached_tvs;

  // Handling transformations in fusion input tvs with assigned EPILOGUE_INPUT
  //  role by propagating fusion output transformations through cached views
  //  of EPILOGUE_INPUT fusion input tvs and by setting vectorization of the
  //  inner most iterdomain of these cached views
  if (tensor_roles_.count(MatmulTensorRole::EPILOGUE_INPUT)) {
    auto& c_tvs = tensor_roles_.at(MatmulTensorRole::EPILOGUE_INPUT);

    // The system supports only scenario where there is only one fusion output
    //  with assigned OUTPUT role, this condition is already verified so there
    //  is no need for an additional checks here
    auto output_d = tensor_roles_.at(MatmulTensorRole::OUTPUT).front();
    for (auto* c : c_tvs) {
      cached_tvs.push_back(c->cacheAfter());
    }

    scheduler_utils::BoundedDirectionalTransformPropagator::backward(
        output_d, -1, c_tvs);

    std::unordered_set<ParallelType> parallel_types = {};
    if (params_->use_smem_epilogue) {
      // In cases where smem epilogue feature is enabled, the vectorization
      //  of domains will be propagated to fusion inputs that are epilogue
      //  inputs, this may result in unaligned memory reads. Vectorization is
      //  explicitly excluded form parallelization types to avoid this issue.
      // This should be changed when vectorization analysis is available and
      //  enabled for matmul scheduler.
      parallel_types = allParallelTypesExcept({ParallelType::Vectorize});
    }
    scheduler_utils::parallelizeAllLike(
        output_d, -1, cached_tvs, parallel_types);

    // The cached EPILOGUE_INPUT tvs are not needed anymore
    cached_tvs.clear();
  }
}

void HopperMultipleMatmulScheduler::scheduleSplitKSum() {
  if (params_->splitk_factor == 1) {
    return;
  }
  for (TensorView* splitk_sum : splitk_sums_) {
    // Always use serial grid reduction for split-K sum
    splitk_sum->definition()->as<ReductionOp>()->requestSerialGridReduction();

    // [..., Mo, No, Mi, Ni]
    splitk_sum->split(-2, getM(params_->mma_macro));
    splitk_sum->split(-1, getN(params_->mma_macro));
    // [..., Mo, No, Mio, Mii, Nio, Nii]
    // -> [..., Mo, No, Mio, Nio, Mii, Nii]
    splitk_sum->reorder({{-3, -2}});
    splitk_sum->merge(-4);
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        splitk_sum->getLoopDomain());
    splitk_sum->setLoopDomain(s.as<IterDomain*>());
    splitk_sum->axis(-5)->parallelize(ParallelType::TIDy);

    // splitk_sum->reorder({{2, -2}});
    splitk_sum->axis(2)->parallelize(ParallelType::BIDz);
    splitk_sum->axis(-1)->parallelize(ParallelType::Vectorize);
  }
}

void HopperMultipleMatmulScheduler::setUpInlining() {
  // auto inline for all tensors except register tensors
  std::unordered_set<TensorView*> smem_loads_and_mma_inputs;
  smem_loads_and_mma_inputs.insert(acrs_.begin(), acrs_.end());
  smem_loads_and_mma_inputs.insert(bcrs_.begin(), bcrs_.end());
  smem_loads_and_mma_inputs.insert(abs_.begin(), abs_.end());
  smem_loads_and_mma_inputs.insert(bbs_.begin(), bbs_.end());
  inlineMost(ir_utils::allTvsExcept(fusion_, smem_loads_and_mma_inputs));

  // if auto inline, will inline to position-7, leads to performance
  // regression
  for (TensorView* mma_result : mma_results_) {
    inlineSelectedAt(
        smem_loads_and_mma_inputs,
        mma_result,
        num_device_and_batch_dims_ + 6 + num_splitk_dims_);
  }
}

void HopperMultipleMatmulScheduler::setUpCircularBuffering() {
  // Propagate mma output swizzle and parallelization down the DAG
  if (params_->circular_buffer_options.circular_buffer_smem_write) {
    NVF_ERROR(
        params_->circular_buffer_options.smem_circular_buffer_stage > 1,
        "Invalid buffer stage config")
    if (params_->circular_buffer_options.smem_circular_buffer_stage > 2) {
      NVF_ERROR(
          params_->async_gmem_load_operands,
          "Circular buffer only supports async load");
    }
    NVF_CHECK(
        params_->circular_buffer_options.smem_circular_buffer_prefetch_gap >
                0 &&
            params_->circular_buffer_options
                    .smem_circular_buffer_prefetch_gap <=
                params_->circular_buffer_options.smem_circular_buffer_stage,
        "smem_circular_buffer_prefetch_gap is ",
        params_->circular_buffer_options.smem_circular_buffer_prefetch_gap,
        " but is expected to be positive and not greater than number of stages: ",
        params_->circular_buffer_options.smem_circular_buffer_stage);

    for (TensorView* acw_smem : acw_smems_) {
      acw_smem->circularBuffer(
          params_->circular_buffer_options.smem_circular_buffer_stage,
          /*prefetch_distance=*/
          params_->circular_buffer_options.smem_circular_buffer_stage -
              params_->circular_buffer_options
                  .smem_circular_buffer_prefetch_gap);
    }
    for (TensorView* bcw_smem : bcw_smems_) {
      bcw_smem->circularBuffer(
          params_->circular_buffer_options.smem_circular_buffer_stage,
          /*prefetch_distance=*/
          params_->circular_buffer_options.smem_circular_buffer_stage -
              params_->circular_buffer_options
                  .smem_circular_buffer_prefetch_gap);
    }
  }

  // NOTE: circular_buffer_smem_read is ignored for Hopper matmul since we do
  // not do any cache reads

  /*
  // TODO Investigate. Disable loop rotation with tma circular buffering
  if (params_->circular_buffer_options.circular_buffer_smem_read &&
      params_->circular_buffer_options.circular_buffer_smem_write) {
    // rotate Kg loop
    // This assumes we have a single main loop. If there were multiple main
    // loops, then we would need to rotate each of them separately.
    std::unordered_set<Statement*> all_smem_loads;
    all_smem_loads.insert(acrs_.begin(), acrs_.end());
    all_smem_loads.insert(bcrs_.begin(), bcrs_.end());
    scheduler_utils::rotateLoop(
        mma_results_.front(),
        num_device_and_batch_dims_ + 2 + num_splitk_dims_,
        all_smem_loads);
  }
  */
}

} // namespace nvfuser
