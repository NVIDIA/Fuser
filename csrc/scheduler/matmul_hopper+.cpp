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
#include <scheduler/matmul.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/matmul_hopper+.h>
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

namespace schedule_matmul {

namespace {

// Find the first MatmulDimRole from left to right in a vector of roles
int64_t findFirstRole(
    std::vector<MatmulDimRole>& roles,
    MatmulDimRole role_to_find) {
  auto role_iter =
      std::find_if(roles.begin(), roles.end(), [&](MatmulDimRole role) {
        return role == role_to_find;
      });
  if (role_iter == roles.end()) {
    return -1;
  }
  return std::distance(roles.begin(), role_iter);
}

} // namespace

void HopperPlus::transformLikeMmaOutputWithK(TensorView* tv) {
  NVF_ERROR(tv->axis(-1)->isReduction(), "Inner axis should be Reduction.");
  // The input is originally block tiled so that the inner dims are the CTA tile
  // size
  //
  // We split this into warp tiles then instruction tiles
  // Original: [..., M, N, K]
  tv->split(-3, params_->tile_sizes.warp_tile.m);
  tv->split(-3, getM(params_->mma_macro));
  tv->split(-2, params_->tile_sizes.warp_tile.n);
  tv->split(-2, getN(params_->mma_macro));
  // K dimension is present for mma_result
  // We don't need to split by warp_tile.k, since we always have
  // cta_tile.k == warp_tile.k
  tv->split(-1, getK(params_->mma_macro));
  // After Split: [..., Mo, Mw, Mi, No, Nw, Ni, Kw, Ki]
  tv->reorder({
      {-8, -8}, // Mo
      {-7, -6}, // Mw
      {-6, -3}, // Mi
      {-5, -7}, // No
      {-4, -5}, // Nw
      {-3, -2}, // Ni
      {-2, -4}, // Kw
      {-1, -1}, // Ki
  });
  // After Reorder: [..., Mo, No, Mw, Nw, Kw, Mi, Ni, Ki]
  tv->merge(-8);
  // After Merge: [..., Mo * No, Mw, Nw, Kw, Mi, Ni]
  if (isCooperative()) {
    tv->axis(-7)->parallelize(ParallelType::TIDy);
    // After Parallelize: [..., Mo * No (TIDy), Mw, Nw, Kw, Mi, Ni, Ki]
  }
}

void HopperPlus::transformLikeMmaOutputWithoutK(TensorView* tv) {
  NVF_ERROR(
      tv->domain()->loop().size() >= 4,
      "transformLikeMmaOutputWithoutK requires at least four iterDomains but ",
      tv->toString(),
      " only has ",
      tv->domain()->loop().size(),
      ".");
  NVF_ERROR(
      !tv->axis(-1)->isReduction(), "Inner axis should not be Reduction.");

  // The input is originally block tiled so that the inner dims are the CTA tile
  // size
  // Original: [..., M, N]
  // We split this into warp tiles then instruction tiles
  tv->split(-2, params_->tile_sizes.warp_tile.m);
  tv->split(-2, getM(params_->mma_macro));
  tv->split(-1, params_->tile_sizes.warp_tile.n);
  tv->split(-1, getN(params_->mma_macro));
  // After Split: [..., Mo, Mw, Mi, No, Nw, Ni]
  tv->reorder({
      {-3, -5},
      {-2, -3},
  });
  // After Reorder: [..., Mo, No, Mw, Nw, Mi, Ni]
  tv->merge(-6);
  // After Merge: [..., Mo * No, Mw, Nw, Mi, Ni]
  if (isCooperative()) {
    tv->axis(-5)->parallelize(ParallelType::TIDy);
    // After Parallelize: [..., Mo * No (TIDy), Mw, Nw, Mi, Ni]
  }
}

MatmulDimRole HopperPlus::findMatmulDimRole(IterDomain* id) {
  ValGroup vg = graph_->toGroup(id);
  auto it = id_roles_.find(vg);
  NVF_ERROR(it != id_roles_.end());
  return it->second;
}

void HopperPlus::validate() const {
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const int cc = device_prop->major * 10 + device_prop->minor;
  NVF_ERROR(
      cc >= 90, "This matmul scheduler is restricted to Hopper & Blackwell.");

  if (params_->tiling_strategy != MatmulParams::TilingStrategy::OneTilePerCTA) {
    NVF_CHECK(
        params_->splitk_factor == 1,
        "Hopper+ matmul scheduler does not support scheduling persistent "
        "split-K kernels");
  }

  NVF_CHECK(
      params_->tiling_strategy !=
          MatmulParams::TilingStrategy::DistributeStagesAcrossSMs,
      "Hopper+ matmul scheduler does not support distributing stages across "
      "SMs a la stream-K");

  NVF_CHECK(
      isCooperative(),
      "Hopper+ matmul scheduler only supports cooperatively buffering at the "
      "CTA level (no ping-pong)");
  if (isCooperative()) {
    NVF_CHECK(
        params_->tile_sizes.cta_tile.m % params_->tile_sizes.warp_tile.m == 0,
        "Expected m dimension for cta_tile to be divisble by warp_tile.");
    NVF_CHECK(
        params_->tile_sizes.cta_tile.n % params_->tile_sizes.warp_tile.n == 0,
        "Expected m dimension for cta_tile to be divisble by warp_tile.");
    NVF_CHECK(
        params_->tile_sizes.cta_tile.k % params_->tile_sizes.warp_tile.k == 0,
        "Expected m dimension for cta_tile to be divisble by warp_tile.");
  } else if (isPingPong()) {
    NVF_CHECK(
        params_->tile_sizes.cta_tile == params_->tile_sizes.warp_tile,
        "Expected cta_tile and warp_tile to be the same for Ping-Pong Matmul "
        "Kernels");
  }
}

void HopperPlus::run() {
  // Finds matmul patterns and translates them to MmaOps, then finds tensor
  // and dimension roles for all tensors in the fusion
  findPatterns();
  translatePatterns();
  // We use the tensor roles to cache operands and epilogue inputs differently
  findRoles();

  // Clears memory spaces on intermediate tensors, calls
  // cache{After,Before,Fork} on inputs and outputs.
  // Defines acw_smem/bcw_smem and acr/bcr by possibly calling cacheAfter.
  cacheInputsAndOutputs(/*skip_intermediates=*/true);

  // We need to find roles again after caching, since we will need to rebuild
  // the IdModel.
  // TODO: update the val graph on the fly in cacheInputsAndOutputs using
  // cacheAfter and missing cacheFork and cacheBefore utilities instead of doing
  // a full rebuild here
  findRoles();

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

void HopperPlus::reorderBlockTileTraversal(
    TensorView* tv,
    std::vector<MatmulDimRole>& outer_dim_roles) {
  NVF_ERROR(params_->grid_traversal_factor.first >= 1);
  NVF_ERROR(params_->grid_traversal_factor.second >= 1);

  // short-circuit: If grid traversal factor is 1x1, we don't need to reorder.
  if (params_->grid_traversal_factor.first == 1 &&
      params_->grid_traversal_factor.second == 1) {
    return;
  }

  // Find position of outer M and N dims in schedule_.tiled
  int64_t Mo_pos = findFirstRole(outer_dim_roles, MatmulDimRole::M);
  int64_t No_pos = findFirstRole(outer_dim_roles, MatmulDimRole::N);

  // Multi-factor grid traversal.
  // M and N roles must be present and consecutive.
  if (params_->grid_traversal_factor.first > 1 &&
      params_->grid_traversal_factor.second > 1) {
    NVF_ERROR(
        Mo_pos >= 0 || No_pos >= 0, "Either M or N role must be present.");
    NVF_ERROR(
        Mo_pos != No_pos, "The position of M and N roles must be different.");
    NVF_ERROR(abs(Mo_pos - No_pos) == 1, "M and N roles must be consecutive.");

    bool is_M_present = Mo_pos >= 0;
    bool is_N_present = No_pos >= 0;
    bool is_N_right_of_M = No_pos > Mo_pos;
    const int64_t min_axis_pos = std::min(Mo_pos, No_pos);

    // original: [M, N]
    // split:   [M, N/second_factor, second_factor]
    // split: [M/first_factor, first_factor, N/second_factor, second_factor]
    // reorder: [M/first_factor, N/second_factor, first_factor,
    // second_factor]
    // merge:
    // [M/first_factor * N/second_factor, first_factor, second_factor]
    // merge:
    // [M/first_factor * N/second_factor, first_factor * second_factor]

    // If N axis exists, then split by second grid traversal factor.
    if (is_N_present) {
      // split:   [M, N/second_factor, second_factor]
      tv->split(No_pos, params_->grid_traversal_factor.second);
    }
    // If N is to the left of M, then shift M by 1 because of second factor.
    if (!is_N_right_of_M) {
      Mo_pos++;
    }

    // If M axis exists, then split by first grid traveral factor.
    if (is_M_present) {
      // split: [M/first_factor, first_factor, N/second_factor, second_factor]
      tv->split(Mo_pos, params_->grid_traversal_factor.first);
    }
    // If N is to the right of M, then shift M by 1 because of the first factor.
    if (is_N_right_of_M) {
      No_pos++;
    }

    if (is_N_present && is_M_present) {
      NVF_ERROR(min_axis_pos >= 0, "Both M and N roles must exist.");
      // reorder: [M/first_factor, N/second_factor, first_factor,
      // second_factor]
      tv->reorder(
          {{min_axis_pos + 1, min_axis_pos + 2},
           {min_axis_pos + 2, min_axis_pos + 1}});
      // merge:
      // [M/first_factor * N/second_factor, first_factor, second_factor]
      tv->merge(min_axis_pos, min_axis_pos + 1);
      // merge:
      // [M/first_factor * N/second_factor, first_factor * second_factor]
      tv->merge(min_axis_pos + 1, min_axis_pos + 2);
    } else if (is_N_present) {
      // M is missing, so we skip the merge above. In this case we
      // should update the dim roles to reflect the new split axis.
      outer_dim_roles.insert(
          outer_dim_roles.begin() + No_pos, MatmulDimRole::N);
    } else if (is_M_present) {
      // N is missing, so we skip the merge above. In this case we
      // should update the dim roles to reflect the new split axis.
      outer_dim_roles.insert(
          outer_dim_roles.begin() + Mo_pos, MatmulDimRole::M);
    }
    return;
  }

  // Single factor grid traversal.
  NVF_ERROR(params_->grid_traversal_factor.first > 1);
  NVF_ERROR(params_->grid_traversal_factor.second == 1);
  int factor = params_->grid_traversal_factor.first;
  switch (params_->cta_order) {
    case MatmulParams::TileRasterizationOrder::ColumnMajor: {
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
    }

    case MatmulParams::TileRasterizationOrder::RowMajor: {
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
      break;
    }
    default:
      NVF_THROW("Invalid TileRasterizationOrder passed to Matmul scheduler");
  }
}

std::vector<std::vector<MatmulDimRole>> HopperPlus::blockTileTensors(
    const std::vector<TensorView*>& tvs) {
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

    reorderBlockTileTraversal(tv, merged_roles);

    if (params_->splitk_factor > 1) {
      // Outer K dimension in tv is in same position found in merged_roles
      for (size_t i : arange(merged_roles.size())) {
        if (merged_roles[i] == MatmulDimRole::K) {
          tv->split((int64_t)i, params_->splitk_factor, /*inner*/ false);
        }
      }
    }

    // Merge in batch dims to the BIDy dim for non-persistent
    if (params_->tiling_strategy ==
        MatmulParams::TilingStrategy::OneTilePerCTA) {
      if (num_local_batch_dims_ > 0) {
        NVF_ERROR(merged_roles.front() == MatmulDimRole::Batch);
        // Merge batch dim into the dimension that will be parallelized BIDy
        if (params_->cta_order ==
            MatmulParams::TileRasterizationOrder::ColumnMajor) {
          int64_t outer_grid_dim = num_device_dims_ + 2L;
          // [..., Batch, M, N, ...]
          tv->merge(num_device_dims_, outer_grid_dim);
          // [..., Batch*N, M, ...]
          // Now we need to transpose so that Batch*N is to the right of M
          tv->reorder({{num_device_dims_, num_device_dims_ + 1}});
        } else { // row major
          int64_t outer_grid_dim = num_device_dims_ + 1L;
          tv->merge(num_device_dims_, outer_grid_dim);
        }
        merged_roles.erase(merged_roles.begin());
      }
    } else if (
        params_->tiling_strategy ==
        MatmulParams::TilingStrategy::DistributeTilesAcrossSMs) {
      // Persistent kernel scheduling
      if (params_->cta_order ==
          MatmulParams::TileRasterizationOrder::ColumnMajor) {
        tv->reorder(
            {{num_device_and_batch_dims_, num_device_and_batch_dims_ + 1}});
      }
      tv->merge(num_device_and_batch_dims_, num_device_and_batch_dims_ + 1);

      if (num_local_batch_dims_ > 0) {
        NVF_ERROR(merged_roles.front() == MatmulDimRole::Batch);
        // Merge batch dims before doing the persistent split
        tv->merge(num_device_dims_);
        merged_roles.erase(merged_roles.begin());
      }

      const int64_t num_sms =
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
      tv->split(num_device_dims_, num_sms);
    } else {
      NVF_THROW("Unsupported tiling strategy");
    }

    all_merged_roles.push_back(merged_roles);
  }
  return all_merged_roles;
}

void HopperPlus::inspectPrologues() const {
  for (TensorView* mma_result : mma_results_) {
    for (Val* v : mma_result->definition()->inputs()) {
      TensorView* op_input = v->as<TensorView>();

      // We currently require all operands to lie in smem, meaning we cannot yet
      // handle any prologue computation. This includes `BroadcastOp` which
      // might be introduced when translating a MatmulOp or LinearOp to MmaOp.
      Expr* def = op_input->definition();
      NVF_ERROR(def != nullptr && def->isA<LoadStoreOp>());
      NVF_ERROR(
          def->input(0)->as<TensorView>()->getMemoryType() ==
          MemoryType::Global);
    }
  }
}

void HopperPlus::scheduleOperands() {
  NVF_CHECK(
      params_->async_gmem_load_operands,
      "Hopper+ matmul scheduler currently requires TMA to be enabled");
  auto scheduleBranch = [&](const std::vector<TensorView*>& gmem_operands,
                            const std::vector<TensorView*>& smem_operands,
                            MmaOperand operand_type) {
    blockTileTensors(smem_operands);
    parallelizeBlocks(smem_operands);
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

void HopperPlus::parallelizeBlocks(const std::vector<TensorView*>& tvs) const {
  for (TensorView* tv : tvs) {
    switch (params_->tiling_strategy) {
      case MatmulParams::TilingStrategy::OneTilePerCTA:
        // Data-parallel kernels are parallelized BIDx BIDy
        switch (params_->cta_order) {
          // TODO: Should we instead check the roles of these dimensions to take
          // the outermost two M or N axes?
          case MatmulParams::TileRasterizationOrder::ColumnMajor:
            tv->axis(num_device_dims_)->parallelize(ParallelType::BIDx);
            tv->axis(num_device_dims_ + 1)->parallelize(ParallelType::BIDy);
            break;
          case MatmulParams::TileRasterizationOrder::RowMajor:
            tv->axis(num_device_dims_)->parallelize(ParallelType::BIDy);
            tv->axis(num_device_dims_ + 1)->parallelize(ParallelType::BIDx);
            break;
          default:
            NVF_THROW(
                "Invalid TileRasterizationOrder passed to Matmul scheduler");
        }
        break;
      case MatmulParams::TilingStrategy::DistributeTilesAcrossSMs:
      case MatmulParams::TilingStrategy::DistributeStagesAcrossSMs:
        // For persistent kernels, we just parallelize the SM dimension
        tv->axis(num_device_dims_ + 1)->parallelize(ParallelType::BIDx);
        break;
    }
  }
}

void HopperPlus::setMmaResultAllocationDomain(TensorView* mma_result) {
  if (isBlackwell(params_->mma_macro)) {
    mma_result->setMemoryType(MemoryType::Tensor);
    // So far, we only support M128 Blackwell MMA macros. For these macros,
    // Rows of the accumulator span all 128 lanes of TMem. That is, the
    // allocation domain should be [Mi, (DimSep), ...other]
    // We want to move Mi to the front of the domain.
    std::vector<IterDomain*> allocation_domain = mma_result->getLoopDomain();
    auto item = allocation_domain[allocation_domain.size() - 3];
    allocation_domain.erase(
        allocation_domain.begin() + allocation_domain.size() - 3);
    allocation_domain.insert(allocation_domain.begin(), item);
    mma_result->setAllocationDomain(allocation_domain, true);
    mma_result->setTMemDimSepPos(1);
  } else {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        mma_result->getLoopDomain());
    mma_result->setAllocationDomain(s.as<IterDomain*>(), true);
  }
}

void HopperPlus::scheduleMmaResults() {
  GemmTile instruction_tile = getMmaOpShape(params_->mma_macro);
  NVF_CHECK(
      params_->tile_sizes.cta_tile.k == params_->tile_sizes.warp_tile.k,
      "CTA tile must match warp tile K dimension for Hopper+ matmul but found ",
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
  for (auto&& [i, mma_result] : enumerate(mma_results_)) {
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

    transformLikeMmaOutputWithK(mma_result);
    setMmaResultAllocationDomain(mma_result);

    mma_result->axis(-1)->parallelize(ParallelType::Mma);
    mma_result->axis(-2)->parallelize(ParallelType::Mma);
    mma_result->axis(-3)->parallelize(ParallelType::Mma);
  }
}

std::vector<TensorView*> HopperPlus::createTMemLoad() {
  if (!isBlackwell(params_->mma_macro)) {
    return {};
  }
  std::vector<TensorView*> tmem_ld_tvs;
  for (auto mma_result : mma_results_) {
    TensorView* tmem_ld_tv = cacheAfter(mma_result);
    tmem_ld_tv->definition()->as<LoadStoreOp>()->setOpType(
        LoadStoreOpType::LdTMem);
    tmem_ld_tvs.push_back(tmem_ld_tv);
  }
  return tmem_ld_tvs;
}

int64_t HopperPlus::getLdTMemVectorizeFactor() const {
  const int64_t n_mma = getN(params_->mma_macro);
  int64_t tmem_vectorize_factor = 1;
  while (n_mma % tmem_vectorize_factor == 0 && tmem_vectorize_factor <= 128) {
    tmem_vectorize_factor *= 2;
  }
  return tmem_vectorize_factor / 2;
}

void HopperPlus::scheduleEpilogueWithoutSmemEpilogueBlackwell() {
  const bool has_splitk = params_->splitk_factor != 1;
  std::vector<TensorView*> cached_tvs;
  std::vector<TensorView*> propagate_to =
      splitk_sums_.empty() ? mma_results_ : splitk_sums_;
  // When there is a split-K, the TMem load happens before split-K sum,
  // when there is no split-K, the TMem load happens in the epilogue.
  std::vector<TensorView*> tmem_ld_tvs =
      !has_splitk ? createTMemLoad() : std::vector<TensorView*>{};
  for (auto& [c, c_cache] : cached_epilogue_inputs_) {
    cached_tvs.push_back(c_cache);
    propagate_to.push_back(c);
  }
  for (Val* dv : fusion_->outputs()) {
    TensorView* d = dv->as<TensorView>();
    NVF_ERROR(d->definition() && d->definition()->isA<LoadStoreOp>());

    // Apply the default scheduling that is common to all register
    // TensorViews after wgmma.
    blockTileTensors({d});
    parallelizeBlocks({d});
    transformLikeMmaOutputWithoutK(d);

    // TIDx is 128, so we use it for lanes of the accumulator. Also, we
    // vectorize the TMem load with a factor of v (tmem_vectorize_factor).
    // [..., Mo * No, Mw, Nw, Mi (TIDx), Ni / v, v (Vectorize)]
    d->axis(-2)->parallelize(ParallelType::TIDx);
    int64_t tmem_vectorize_factor = getLdTMemVectorizeFactor();
    if (tmem_vectorize_factor < getN(params_->mma_macro)) {
      d->split(-1, tmem_vectorize_factor);
    }

    // TODO: We need to check bank conflicts in this path.
    // Propagate schedule changes back to the outputs of the Mma op.
    scheduler_utils::BoundedDirectionalTransformPropagator::backward(
        d,
        -1,
        propagate_to,
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType());

    // Vectorize the epilogue input load and output store. TMem load can
    // be vectorized to 512 byte, but gmem load/store can only be vectorized
    // to 16 bytes. So we need to further split the last dimension and use
    // multiple vector loads/stores. for each TMem load/store.
    // After split and parallelization:
    // (v = tmem_vectorize_factor, vv = params_->supported_vec_size.epilogue)
    // [..., Mo * No, Mw, Nw, Mi (TIDx), Ni / v, v/vv, vv]
    // TODO: Support vectorization_factor in MatmulParams
    if (tmem_vectorize_factor > params_->supported_vec_size.epilogue) {
      d->split(-1, params_->supported_vec_size.epilogue);
      for (auto c : cached_tvs) {
        bool is_2d_epilogue_input =
            TensorDomain::noBroadcasts(c->domain()->logical()).size() == 2;
        if (is_2d_epilogue_input) {
          c->split(-1, params_->supported_vec_size.epilogue);
        }
      }
    }
    d->axis(-1)->parallelize(ParallelType::Vectorize);
    if (!cached_tvs.empty()) {
      scheduler_utils::parallelizeAllLike(d, -1, cached_tvs);
    }
  }
  // Vectorize the TMem load, if any.
  for (auto tmem_ld_tv : tmem_ld_tvs) {
    tmem_ld_tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }
}

void HopperPlus::scheduleEpilogueWithoutSmemEpilogueHopper() {
  std::vector<TensorView*> cached_tvs;
  std::vector<TensorView*> propagate_to =
      splitk_sums_.empty() ? mma_results_ : splitk_sums_;
  for (auto& [c, c_cache] : cached_epilogue_inputs_) {
    cached_tvs.push_back(c_cache);
    propagate_to.push_back(c);
  }
  for (Val* dv : fusion_->outputs()) {
    TensorView* d = dv->as<TensorView>();
    NVF_ERROR(d->definition() && d->definition()->isA<LoadStoreOp>());

    // Apply the default scheduling that is common to all register
    // TensorViews after wgmma.
    blockTileTensors({d});
    parallelizeBlocks({d});
    transformLikeMmaOutputWithoutK(d);

    const AbstractTensor s =
        mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(d->getLoopDomain());
    d->setLoopDomain(s.as<IterDomain*>());

    // TODO: We need to check bank conflicts in this path.
    // Propagate schedule changes back to the outputs of the Mma op.
    scheduler_utils::BoundedDirectionalTransformPropagator::backward(
        d,
        -1,
        propagate_to,
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType());

    // We do not respect the vectorization_factor parameter, but always
    // vectorize the inner-dim with extent 2.
    NVF_ERROR(params_->supported_vec_size.epilogue >= 2);
    // TODO: Support vectorization_factor in MatmulParams
    d->axis(-1)->parallelize(ParallelType::Vectorize);
    if (!cached_tvs.empty()) {
      scheduler_utils::parallelizeAllLike(d, -1, cached_tvs);
    }
  }
}

void HopperPlus::scheduleEpilogueWithoutSmemEpilogue() {
  if (isBlackwell(params_->mma_macro)) {
    scheduleEpilogueWithoutSmemEpilogueBlackwell();
  } else {
    scheduleEpilogueWithoutSmemEpilogueHopper();
  }
}

void HopperPlus::scheduleEpilogueWithSmemEpilogue() {
  constexpr int64_t ldst_matrix_tile_m = 16;
  constexpr int64_t ldst_matrix_tile_n = 16;
  fusion_->manage("ldst_matrix_m_tile", ldst_matrix_tile_m);
  fusion_->manage("ldst_matrix_n_tile", ldst_matrix_tile_n);
  fusion_->manage("ldst_matrix_m_smem", params_->tile_sizes.warp_tile.m);
  fusion_->manage("ldst_matrix_n_smem", params_->tile_sizes.warp_tile.n);

  // Apply LdMatrix to any epilogue inputs loaded to smem with TMA.
  std::vector<TensorView*> tma_load_epilogue_inputs;

  // Propagate to (not including) the splitk output if there is a splitk
  // else this is just mma_results_
  std::vector<TensorView*> propagate_to =
      splitk_sums_.empty() ? mma_results_ : splitk_sums_;
  for (auto& [c, c_cache] : cached_epilogue_inputs_) {
    bool load_with_ldmatrix =
        params_->use_ldst_matrix && dataTypeSize(c_cache->dtype()) == 2;
    bool is_2d_epilogue_input =
        TensorDomain::noBroadcasts(c_cache->domain()->logical()).size() == 2;
    if (load_with_ldmatrix && is_2d_epilogue_input &&
        params_->async_gmem_load_operands) {
      // Schedule TMA load into shared memory for epilogue input
      c_cache->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::CpAsyncBulkTensorTile);
      c_cache->setMemoryType(MemoryType::Shared);

      // Apply the default scheduling that is common to all register
      // TensorViews after wgmma.
      blockTileTensors({c_cache});
      parallelizeBlocks({c_cache});
      transformLikeMmaOutputWithoutK(c_cache);

      // Swizzle to avoid shared memory bank conflicts
      MmaInputSmemSwizzle swizzle_type =
          mma_utils::tmaSwizzleSharedMemory(c_cache);
      c_cache->applyMmaSwizzleForTMALoad(swizzle_type);

      TensorView* reg_tv = cacheAfter(c_cache);
      reg_tv->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::LdMatrix);

      // Apply the default scheduling that is common to all register
      // TensorViews after wgmma.
      blockTileTensors({reg_tv});
      parallelizeBlocks({reg_tv});
      transformLikeMmaOutputWithoutK(reg_tv);

      // Schedule the loop and allocation domain of LdMatrix like the
      // accumulation register TensorView of wgmma.
      AbstractTensor s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
          reg_tv->getLoopDomain());
      reg_tv->setLoopDomain(s.as<IterDomain*>());
      reg_tv->setAllocationDomain(
          reg_tv->getLoopDomain(), /*new_contiguity=*/true);

      // Apply LdStMatrix scheduling to the wgmma loop domain
      mma_utils::scheduleLdStMatrixForMmaOutput(
          reg_tv, ldst_matrix_tile_m, ldst_matrix_tile_n);

      // Vectorize last iterDomain because LdMatrix loads all eight values with
      // a single LdMatrix.x4 operation
      reg_tv->axis(-1)->parallelize(ParallelType::Vectorize);

      // Do not propagate any other changes to LdMatrix.
      propagate_to.push_back(reg_tv);
    } else {
      // Propagate changes to the cache_after tensor if not using TMA load.
      propagate_to.push_back(c);
    }
  }

  // Manually schedule register cache and output TensorView
  for (Val* dv : fusion_->outputs()) {
    TensorView* d = dv->as<TensorView>();
    NVF_ERROR(d->definition() && d->definition()->isA<LoadStoreOp>());
    TensorView* dc = d->definition()->input(0)->as<TensorView>();

    // The chain of operations storing data to global memory:
    //   registers -> (stmatrix) -> smem -> (tma_store) -> gmem
    TensorView* d_smem = cacheBefore(d, LoadStoreOpType::Set);

    std::vector<TensorView*> tvs_to_schedule{d, d_smem};
    bool dc_is_mma_result =
        std::find(mma_results_.begin(), mma_results_.end(), dc) !=
        mma_results_.end();
    bool dc_is_splitk_sum = params_->splitk_factor > 1 &&
        std::find(splitk_sums_.begin(), splitk_sums_.end(), dc) !=
            splitk_sums_.end();

    if (!dc_is_mma_result && !dc_is_splitk_sum) {
      // Skip scheduling dc if it is an mma_result. This can happen if we are
      // not casting back to half-precision in the output
      tvs_to_schedule.push_back(dc);
    }

    // Set MemoryType
    dc->setMemoryType(MemoryType::Local);
    d_smem->setMemoryType(MemoryType::Shared);

    // Set LoadStoreOpType
    bool store_with_stmatrix =
        params_->use_ldst_matrix && dataTypeSize(dc->dtype()) == 2;
    if (store_with_stmatrix) {
      d_smem->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::StMatrix);
    }
    d->definition()->as<LoadStoreOp>()->setOpType(
        LoadStoreOpType::CpAsyncBulkTensorTile);

    // Apply the common transforms to dc, d_smem, d
    // After these transforms we schedule the inner two non-reduction loops
    // (instruction tile) of dc and propagate is back till the outputs of mma.
    blockTileTensors(tvs_to_schedule);
    parallelizeBlocks(tvs_to_schedule);
    for (auto tv : tvs_to_schedule) {
      transformLikeMmaOutputWithoutK(tv);
    }

    // Should not propagate if the dc is a mma output as the mma output has
    // already been scheduled.
    if (!dc_is_mma_result && !dc_is_splitk_sum) {
      auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
          dc->getLoopDomain());
      dc->setLoopDomain(s.as<IterDomain*>());
      dc->setAllocationDomain(s.as<IterDomain*>(), true);

      scheduler_utils::BoundedDirectionalTransformPropagator::backward(
          dc,
          -1,
          propagate_to,
          scheduler_utils::BoundedDirectionalTransformPropagator::Options()
              .propagateParallelType());
    }

    // Determine swizzle for TMA Store
    MmaInputSmemSwizzle swizzle = mma_utils::tmaSwizzleSharedMemory(d_smem);

    // First, create loop domain that matches wgmma register accumulator using
    // original loop domain.
    const AbstractTensor s =
        mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
            d_smem->getLoopDomain());
    // Create allocation domain with swizzle for TMA Store.
    // This step modifies loop domain and the creates a new allocation domain.
    if (swizzle != MmaInputSmemSwizzle::None) {
      mma_utils::scheduleTMAStoreForMmaOutput(d_smem, swizzle);
    }
    // Finally, set loop domain using saved AbstractTensor.
    d_smem->setLoopDomain(s.as<IterDomain*>());

    if (store_with_stmatrix) {
      // Apply LdStMatrix scheduling to the wgmma loop domain
      mma_utils::scheduleLdStMatrixForMmaOutput(
          d_smem, ldst_matrix_tile_m, ldst_matrix_tile_n);
    }
    d_smem->axis(-1)->parallelize(ParallelType::Vectorize);

    // Schedule global memory output; Output from TMA Store
    mma_utils::scheduleTMAStoreForMmaOutput(d, swizzle);
  }
}

void HopperPlus::scheduleEpilogue() {
  if (params_->use_smem_epilogue) {
    scheduleEpilogueWithSmemEpilogue();
  } else {
    scheduleEpilogueWithoutSmemEpilogue();
  }
}

void HopperPlus::scheduleSplitKSumHopper() {
  if (params_->splitk_factor == 1) {
    return;
  }
  for (TensorView* splitk_sum : splitk_sums_) {
    // Always use serial grid reduction for split-K sum
    splitk_sum->definition()->as<ReductionOp>()->requestSerialGridReduction();
    transformLikeMmaOutputWithoutK(splitk_sum);
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        splitk_sum->getLoopDomain());
    splitk_sum->setLoopDomain(s.as<IterDomain*>());
    splitk_sum->axis(2)->parallelize(ParallelType::BIDz);
    splitk_sum->axis(-1)->parallelize(ParallelType::Vectorize);
  }
}

constexpr int64_t hardcoded_blackwell_splitk_vectorization_factor = 4;

// Schedule TMem load tv and splitk_sum tv as follows:
//   v = vectorization factor for TMem load
//   vv = vectorization factor for splitk_sum, hardcoded to 4
// TMem load tv:
// [..., Mo * No (TIDy), Mw, Nw, Mi (TIDx), Ni / v, v (Vectorize)]
// Splitk_sum tv:
// [..., Mo * No (TIDy), Mw, Nw, Mi (TIDx), Ni / v, v/vv, vv (Vectorize)]
void HopperPlus::scheduleSplitKSumBlackwell() {
  if (params_->splitk_factor == 1) {
    return;
  }
  std::vector<TensorView*> tmem_ld_tvs = createTMemLoad();

  for (TensorView* splitk_sum : splitk_sums_) {
    // Always use serial grid reduction for split-K sum
    splitk_sum->definition()->as<ReductionOp>()->requestSerialGridReduction();
    transformLikeMmaOutputWithoutK(splitk_sum);
    splitk_sum->axis(2)->parallelize(ParallelType::BIDz);
    splitk_sum->split(-1, getLdTMemVectorizeFactor());
    scheduler_utils::BoundedDirectionalTransformPropagator::backward(
        splitk_sum,
        -1,
        mma_results_,
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType());
    splitk_sum->split(-1, hardcoded_blackwell_splitk_vectorization_factor);
    splitk_sum->axis(-1)->parallelize(ParallelType::Vectorize);
  }
  for (TensorView* tmem_ld_tv : tmem_ld_tvs) {
    tmem_ld_tv->axis(-3)->parallelize(ParallelType::TIDx);
    tmem_ld_tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }
}

void HopperPlus::scheduleSplitKSum() {
  if (isHopper(params_->mma_macro)) {
    scheduleSplitKSumHopper();
  } else {
    scheduleSplitKSumBlackwell();
  }
}

void HopperPlus::setUpInlining() {
  // auto inline for all tensors except register tensors
  std::unordered_set<TensorView*> smem_loads_and_mma_inputs;
  inlineMost(ir_utils::allTvsExcept(fusion_, smem_loads_and_mma_inputs));

  // if auto inline, will inline to position-7, leads to performance
  // regression
  for (TensorView* mma_result : mma_results_) {
    inlineSelectedAt(
        smem_loads_and_mma_inputs,
        mma_result,
        num_device_dims_ + 6 + num_splitk_dims_);
  }
}

void HopperPlus::setUpCircularBuffering() {
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
        " but is expected to be positive and not greater than number of "
        "stages: ",
        params_->circular_buffer_options.smem_circular_buffer_stage);

    CircularBufferType cb_type;
    switch (params_->circular_buffering_strategy) {
      case MatmulParams::CircularBufferingStrategy::Pipelined: {
        cb_type = (CircularBufferType)Pipelined(false);
        break;
      }
      case MatmulParams::CircularBufferingStrategy::WarpSpecialized: {
        int64_t num_math_warp_groups = 1L;
        NVF_ERROR(!mma_results_.empty());
        for (IterDomain* id : mma_results_.front()->getLoopDomain()) {
          if (id->getParallelType() == ParallelType::TIDy) {
            num_math_warp_groups *= id->extent()->evaluate().as<int64_t>();
          }
        }
        if (num_math_warp_groups != 2) {
          // Disable register sharing when there is only one math warp group.
          // In such case we will have 128 math threads and 128 dma threads,
          // for a total of 256 threads per CTA. The register file size on
          // Hopper is 64K registers, which is filled when a 256-thread CTA
          // has 256 registers per thread. Since 256 is already the maximum
          // number of registers per thread even with register sharing, there
          // is no point in doing register sharing to try and increase it.
          //
          // When there is more than one math warp group, we also disable
          // register sharing, since we don't currently compute the number of
          // register properly in that case.
          cb_type = (CircularBufferType)WarpSpecialized(ParallelType::TIDy);
        } else {
          constexpr int64_t num_registers_async_warp = 40;
          constexpr int64_t num_registers_compute_warp = 232;
          cb_type = (CircularBufferType)WarpSpecialized(
              ParallelType::TIDy,
              std::make_pair(
                  num_registers_async_warp, num_registers_compute_warp));
        }
        break;
      }
    }
    for (TensorView* acw_smem : acw_smems_) {
      acw_smem->circularBuffer(
          params_->circular_buffer_options.smem_circular_buffer_stage,
          /*prefetch_distance=*/
          params_->circular_buffer_options.smem_circular_buffer_stage -
              params_->circular_buffer_options
                  .smem_circular_buffer_prefetch_gap,
          /*type=*/cb_type);
    }
    for (TensorView* bcw_smem : bcw_smems_) {
      bcw_smem->circularBuffer(
          params_->circular_buffer_options.smem_circular_buffer_stage,
          /*prefetch_distance=*/
          params_->circular_buffer_options.smem_circular_buffer_stage -
              params_->circular_buffer_options
                  .smem_circular_buffer_prefetch_gap,
          /*type=*/cb_type);
    }
  }

  // NOTE: circular_buffer_smem_read is ignored for Hopper+ matmul since we do
  // not do any cache reads
}

void HopperPlus::setOperandSmemLoadAndCacheOps(
    TensorView* operand,
    int64_t vec_size) {
  auto* lsop = operand->definition()->as<LoadStoreOp>();
  LoadStoreOpType load_op = params_->async_gmem_load_operands
      ? LoadStoreOpType::CpAsyncBulkTensorTile
      : LoadStoreOpType::Set;
  lsop->setOpType(load_op);
}

} // namespace schedule_matmul
} // namespace nvfuser
