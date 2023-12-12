// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <ir/printer.h>
#include <root_domain_map.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <variant>
#include "mma_type.h"
namespace nvfuser {

namespace mma_utils {

//! A wrapper to get MMA Tensor data types
//!   The order of returned types: INPUT_A, INPUT_B, OUTPUT_D
inline mma_utils::MmaDataTypes getMmaDataTypes(
    const std::map<MatmulRole, std::vector<TensorView*>>& roles_map) {
  auto getMMADataType = [&](MatmulRole role) {
    auto entry = roles_map.find(role);
    if (entry != roles_map.end() && !entry->second.empty()) {
      return entry->second.front()->dtype();
    }
    NVF_ERROR(false, "Get MMA Tensor data type failed!");
  };
  const auto a_type = getMMADataType(MatmulRole::INPUT_A);
  const auto b_type = getMMADataType(MatmulRole::INPUT_B);
  const auto c_type = getMMADataType(MatmulRole::OUTPUT_D);
  return mma_utils::MmaDataTypes{a_type, b_type, c_type};
}

std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    const int smem_double_buffer_stage,
    const MmaDataTypes& data_types,
    bool smem_a_reuse_guaranteed,
    bool smem_b_reuse_guaranteed,
    bool ignore_occupancy_drop) {
  const auto properties = at::cuda::getCurrentDeviceProperties();
  const size_t device_smem_limit = properties->sharedMemPerBlockOptin;
  const size_t shared_memory_overhead = properties->reservedSharedMemPerBlock;
  const size_t shared_memory_available =
      device_smem_limit - shared_memory_overhead;

  auto warp_dims = gemm_tile.cta_tile / gemm_tile.warp_tile;
  const auto threads_per_block =
      warp_dims.m * warp_dims.n * warp_dims.k * properties->warpSize;

  // see scheduleContiguousVectorLoad
  const int vector_word = 8;
  const int round_to_factor = warp_dims.m * warp_dims.n * warp_dims.k *
      properties->warpSize * vector_word;
  const int mk = gemm_tile.cta_tile.m * gemm_tile.cta_tile.k;
  const int nk = gemm_tile.cta_tile.n * gemm_tile.cta_tile.k;
  const size_t smem_a = (size_t)(ceilDiv(mk, round_to_factor) *
                                 round_to_factor * smem_double_buffer_stage) *
      dataTypeSize(data_types[0]);
  const size_t smem_b = (size_t)(ceilDiv(nk, round_to_factor) *
                                 round_to_factor * smem_double_buffer_stage) *
      dataTypeSize(data_types[1]);
  const size_t smem_c = (size_t)(gemm_tile.cta_tile.m * gemm_tile.cta_tile.n) *
      dataTypeSize(data_types[2]);

  // NOTE: we can simply add these sizes since they should be integer multiples
  // of 16 bytes, so they will automatically be aligned. This may change with
  // FP8, in which case the expressions below should be updated to insert
  // alignment expressions, using the expected stack ordering in
  // StackBasedSharedMemAllocator.
  NVF_CHECK(smem_a % 16 == 0 && smem_b % 16 == 0 && smem_b % 16 == 0);

  const size_t total_without_smem_epilogue = smem_a + smem_b;
  const size_t total_with_noreuse_smem_epilogue = smem_a + smem_b + smem_c;
  // Even if we actually do wind up re-claiming smem_a and smem_b, if we
  // cannot prove it at this point then we have to assume it will not be
  // reclaimed.
  const size_t total_with_reused_smem_epilogue = std::max(
      smem_a + smem_b,
      (smem_a_reuse_guaranteed ? 0 : smem_a) +
          (smem_b_reuse_guaranteed ? 0 : smem_b) + smem_c);

  // shortcut where occupancy change is ignored.
  if (ignore_occupancy_drop) {
    if (shared_memory_available >= total_with_noreuse_smem_epilogue) {
      return {true, false};
    } else {
      return {shared_memory_available >= total_with_reused_smem_epilogue, true};
    }
  }

  // use additional shared memory for epilogue if occupancy is not changed.
  // occupancy is estimated using register and shared memory usage.
  const auto threads_per_sm = getThreadsPerSMGivenRegPerThread(255);
  const auto blocks_per_sm_by_register = threads_per_sm / threads_per_block;
  const auto blocks_per_sm_without_smem_epilogue = std::min(
      shared_memory_available / total_without_smem_epilogue,
      (size_t)blocks_per_sm_by_register);
  const auto blocks_per_sm_with_reused_smem_epilogue = std::min(
      shared_memory_available / total_with_reused_smem_epilogue,
      (size_t)blocks_per_sm_by_register);
  const auto blocks_per_sm_with_noreuse_smem_epilogue = std::min(
      shared_memory_available / total_with_noreuse_smem_epilogue,
      (size_t)blocks_per_sm_by_register);

  // Return whether we should use smem for epilogue, and whether syncing for
  // re-use is desired. We avoid the sync if omitting it does not decrease
  // occupancy.
  auto promote_prologue_smem_reuse = blocks_per_sm_with_reused_smem_epilogue !=
      blocks_per_sm_with_noreuse_smem_epilogue;

  return {
      blocks_per_sm_with_reused_smem_epilogue ==
          blocks_per_sm_without_smem_epilogue,
      promote_prologue_smem_reuse};
}

std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    const int smem_double_buffer_stage,
    const RolesMap& roles_map,
    const bool ignore_occupancy_drop) {
  const auto data_types = getMmaDataTypes(roles_map);

  // smem_a and smem_b are guaranteed to be re-used for smem_c as long as:
  //   - they are marked for re-use using promoteReuse
  //   - they are not aliased by another tensor whose lifetime extends past the
  //   start of smem_epilogue's.
  //   - their lifetimes do not overlap smem_epilogue
  //
  // We can guarantee the first condition by calling tv->promoteReuse() in
  // scheduleProlog.
  //
  // The second condition would only be the case if another smem tensor had the
  // same indexing and its lifetime did not overlap. Matmul scheduler only uses
  // smem for these three arrays, so the only candidate for aliasing is C. If C
  // aliases either A or B, the following expression is still valid.
  //
  // The third condition is satisfied in the simple cases where the inputs to
  // the matmul have only this use. However, it could be violated if a or b has
  // other uses that get ordered after the matmul; for example when computing
  // matmul(A, B) + A for square matrices A and B. In that case, the smem tensor
  // resulting from A->cacheAfter() will be used in both the matmul as well as
  // the addition that occurs in the epilogue, extending the lifetime such that
  // it violates the third condition above. In order to avoid errors in these
  // cases, we check that there is no re-use when there is more than one use of
  // either a or b. If there are multiple uses we might wind up re-using memory,
  // but in that case the calculation below will be overly conservative.
  TensorView* a = roles_map.at(MatmulRole::INPUT_A).front();
  TensorView* b = roles_map.at(MatmulRole::INPUT_B).front();
  bool smem_a_reuse_guaranteed = a->uses().size() == 1;
  bool smem_b_reuse_guaranteed = b->uses().size() == 1;

  return generateSharedMemoryEpilogueHeuristics(
      gemm_tile,
      smem_double_buffer_stage,
      data_types,
      smem_a_reuse_guaranteed,
      smem_b_reuse_guaranteed,
      ignore_occupancy_drop);
}

void scheduleWarpTileWithReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  NVF_CHECK(
      cta_tile.k % warp_tile.k == 0,
      "Number of warp on k dimension need to be integer");

  int num_warp_k = cta_tile.k / warp_tile.k;

  mma_utils::checkDimSize(
      tv, {-3, -2, -1}, {cta_tile.m, cta_tile.n, cta_tile.k});

  if (num_warp_k == 1) {
    // Non split K over warp case:

    //       -3   -2  -1
    //[...    M,   N,  K]
    // Distribute warp tile:
    tv->split(-3, warp_tile.m);
    tv->split(-2, warp_tile.n);

    //  -5   -4   -3   -2   -1
    // [Mwo  Mw  Nwo   Nw   K]
    tv->split(-4, instruction_tile.m);
    tv->split(-2, instruction_tile.n);
    tv->split(-1, instruction_tile.k);

    //   -8  -7 -6 -5 -4 -3  -2 -1
    // [Mwo Mw Mi Nwo Nw Ni Kwo Ki]

    tv->reorder({{-7, -5}, {-6, -3}, {-5, -6}, {-3, -2}, {-2, -8}, {-8, -7}});
    //   -8  -7 -6  -5 -4 -3 -2 -1
    // [Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  } else {
    // Split K over warp case:
    // Main difference is that an additional
    //  thread dimension needs to be reserved
    //  for cross warp reduction:
    //       -3   -2  -1
    //[...    M,   N,  K]
    // Distribute warp tile:
    tv->split(-3, warp_tile.m);
    tv->split(-2, warp_tile.n);
    tv->split(-1, warp_tile.k);

    //   -6  -5   -4   -3   -2   -1
    // [Mwo  Mw  Nwo   Nw   Kwo  Kw]
    tv->split(-5, instruction_tile.m);
    tv->split(-3, instruction_tile.n);
    tv->split(-1, instruction_tile.k);

    //  -9  -8  -7 -6 -5 -4 -3 -2 -1
    // [Mwo Mw Mi Nwo Nw Ni Kwo Kw Ki]

    tv->reorder({{-8, -6}, {-7, -3}, {-6, -8}, {-4, -2}, {-3, -7}, {-2, -4}});
    //  -9   -8  -7 -6 -5 -4 -3 -2 -1
    // [Mwo  Nwo Ko Mw Nw Kw, Mi Ni Ki]

    tv->merge(-9);
    //  -8  -7 -6 -5 -4   -3 -2 -1
    // [MNwo Ko Mw Nw Kw, Mi Ni Ki]
  }
}

void scheduleWarpTileWithNoReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  mma_utils::checkDimSize(tv, {-2, -1}, {cta_tile.m, cta_tile.n});

  NVF_CHECK(
      cta_tile.k % warp_tile.k == 0,
      "Number of warp on k dimension need to be integer");

  int num_warp_k = cta_tile.k / warp_tile.k;

  //        -2  -1
  //[...    M,   N]

  // Distribute warp tile:
  tv->split(-2, warp_tile.m);
  tv->split(-1, warp_tile.n);

  //  -4   -3   -2   -1
  // [Mwo  Mw  Nwo   Nw ]
  tv->split(-3, instruction_tile.m);
  tv->split(-1, instruction_tile.n);

  //  -6 -5  -4 -3 -2 -1
  // [Mwo Mw Mi Nwo Nw Ni]

  tv->reorder({{-5, -4}, {-4, -2}, {-3, -5}, {-2, -3}});

  //  -6   -5  -4 -3 -2 -1
  // [Mwo  Nwo Mw Nw Mi Ni]

  if (num_warp_k != 1) {
    // The non reduction warps are merged together
    //  to save one thread dim for cross dim reduce.
    tv->merge(-6);
    //  -5  -4 -3 -2 -1
    // [MNo Mw Nw Mi Ni]
  }
}

//! Split the innermost dim to a vectorized load
void scheduleContiguousVectorLoad(
    TensorView* tv,
    MatMulTileOptions tile,
    int vector_word,
    bool vectorize) {
  auto warp_dims = tile.cta_tile / tile.warp_tile;
  int num_of_thread = warp_dims.m * warp_dims.n * warp_dims.k * 32;

  tv->split(-1, num_of_thread * vector_word);
  tv->split(-1, vector_word);
  // [..., thread, vec]
  // distribute to warp: for tidx
  tv->split(-2, 32);

  //      -3    -2    -1
  // [...warp, lane, vec]

  if (warp_dims.k == 1) {
    //      -4     -3    -2    -1
    // [...warpM, warpN, lane, vec]
    tv->split(-3, warp_dims.n);
  } else {
    //      -4     -3    -2    -1
    // [...warpMN, warpR, lane, vec]
    tv->split(-3, warp_dims.k);
  }

  if (vectorize) {
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  tv->axis(-2)->parallelize(ParallelType::TIDx);
  tv->axis(-3)->parallelize(ParallelType::TIDy);
  tv->axis(-4)->parallelize(ParallelType::TIDz);
}

void makeTile(TensorView* tv, std::vector<int> tile_sizes) {
  NVF_CHECK(
      tv->getLeafDomain().size() >= tile_sizes.size(),
      "Tensor dimension less than tile dimension!");

  // Number of inner dimensions we are tiling.
  const int64_t tile_dimension_size = (int64_t)tile_sizes.size();

  // Split the inner dimensions:
  for (int64_t idx : c10::irange(tile_dimension_size)) {
    // Using negative indexing to accomodate potential batching
    //  dimensions on the further left. Eg.:
    //  0, 1, 2   ->         -3,-2,-1
    // [M, N, K]  -> [B0, B1, M, N, K]
    tv->split((int)(idx - tile_dimension_size), (int)tile_sizes.at(idx));
  }

  // The transformation happened should look like:
  //   Before               After
  // [..., M, N, K] -> [..., Mo, Mi, No, Ni, Ko, Ki]

  // Re-order the tiles so that all the outer tiles are
  //  on the left of all the inner tiles
  std::unordered_map<int, int> reorder_map_old_to_new;

  // Number of tiled inner dimensions after we split.
  const auto split_tile_dimension_size = 2 * tile_dimension_size;
  for (auto idx : c10::irange(split_tile_dimension_size)) {
    // We want to reorder as follows:
    //           Before
    //
    // [..., Mo, Mi, No, Ni, Ko, Ki] ->
    //                 After
    //      vvv group0 vvv  vvv group1 vvv
    // [..., Mo, No, Ko,     Mi, Ni, Ki]

    // The index offset within group of current
    //  iterdomain, with grouping specified above.
    auto index_within_group = idx / 2;

    // The index of the group the current id belongs
    //  to, as specified above.
    auto group_index = idx % 2;

    // Calculate the actual index after reordering
    auto index_after_reorder =
        group_index * tile_dimension_size + index_within_group;

    // Add pair {idx_before, idx_after} to re-order map.
    reorder_map_old_to_new.insert(std::make_pair(
        idx - split_tile_dimension_size,
        index_after_reorder - split_tile_dimension_size));
  }

  // Apply the re-order map to tensor
  tv->reorder(reorder_map_old_to_new);
}

namespace {

std::optional<IterDomain*> getMaybeRootIfInnermostTiled(
    IterDomain* id,
    const std::unordered_set<IterDomain*>& maybe_rfactor_id_set) {
  // Root id defaults to an "innermost id".
  while (id->definition() && !maybe_rfactor_id_set.count(id)) {
    if (auto split = dynamic_cast<Split*>(id->definition())) {
      if (id == split->inner()) {
        id = split->in();
        continue;
      }
    }
    // Didn't pass the inner most check, return empty.
    return std::nullopt;
  }

  return id;
}

} // namespace

void orderTiledConcreteIdAsRoot(TensorView* tv) {
  auto ndims = tv->nDims();

  // Keep track of the left most position where we will
  //  be reordering the axes.
  auto leftmost_pos = ndims;

  // Pull the root id's of the given tv.
  std::unordered_set<IterDomain*> maybe_rfactor_id_set{
      tv->getMaybeRFactorDomain().begin(), tv->getMaybeRFactorDomain().end()};

  // Keep track of leaf positions that is either a reduction
  //  or a broadcast.
  // Note: Currently don't really see a case where this function
  //  should be called on a reduction output tv, but adding them
  //  here for completeness.
  std::deque<int> broadcast_or_reduction_pos;

  // Map the root id's to their innermost concrete id's
  //  on the leaf.
  std::unordered_map<IterDomain*, int> root_id_to_inner_leaf_pos;

  // Try to re-order inner iterdomains from the innermost
  //  position backward. This utility only tries to re-order
  //  inner tiles on the innermost positions, like the resulting
  //  tensor from makeTile utility.
  // The re-ordering would first try to decide the inner iterdomains
  //  we want to re-order. For this we start from the innermost position
  //  and move back and collect all the iterdomains that we know
  //  are inner tiles of some root domain or broadcast/reduction domains
  //  that won't affect the concrete id layout.
  // The collection process would stop whenever a iterdomain that is
  //  neither an inner tile nor reduction/broadcast is found, and would
  //  not re-order any iterdomain beyond that point to keep the
  //  outer loop structure unchanged.
  for (int64_t i = static_cast<int64_t>(ndims) - 1; i >= 0; i--) {
    auto leaf_id = tv->axis((int)i);
    if (leaf_id->isBroadcast() || leaf_id->isReduction()) {
      // Register this reduction or broadcast axis
      //  to reorder.
      broadcast_or_reduction_pos.push_front((int)i);
      leftmost_pos = i;
      continue;
    }
    auto maybe_root =
        getMaybeRootIfInnermostTiled(leaf_id, maybe_rfactor_id_set);

    if (maybe_root.has_value()) {
      // Found an innermost id, add them to the
      //  axes to reorder.
      NVF_ERROR(
          root_id_to_inner_leaf_pos
              .insert(std::make_pair(maybe_root.value(), i))
              .second,
          "Multiple \"innermost\" id seen for root id :",
          maybe_root.value()->toString(),
          " on ",
          tv->toString(),
          " very likely an invariant is broken.");
      leftmost_pos = i;
    } else {
      break;
    }
  }

  // Calculate the ordering:

  // pointer to the current target postion after
  //  repordering
  int current_pos = (int)leftmost_pos;
  std::unordered_map<int, int> reorder_map_old_to_new;

  // first place all the broadcast and reduction on the left:
  for (auto original_broadcast_or_reduction_pos : broadcast_or_reduction_pos) {
    reorder_map_old_to_new[original_broadcast_or_reduction_pos] = current_pos++;
  }

  // Next put all the innermost leaf id's, we make sure that
  //  the inner tile ordering follows the corresponding root
  //  domain ordering by iterating on the root domain and
  //  find their corresponding inner tile iterdomains from
  //  the populated root_id_to_inner_leaf_pos.
  for (auto root_id : tv->getMaybeRFactorDomain()) {
    auto leaf_id_pos_it = root_id_to_inner_leaf_pos.find(root_id);
    if (leaf_id_pos_it != root_id_to_inner_leaf_pos.end()) {
      reorder_map_old_to_new[leaf_id_pos_it->second] = current_pos++;
    }
  }

  // Validate that we have processed all inner ids or broadcast/reduction
  //  ids we have registered.
  NVF_ERROR(current_pos == (int)ndims, "Inconsistent ordering logic");

  // Apply the new order:
  tv->reorder(reorder_map_old_to_new);
}

namespace {

// Utility for mma dimension matching
enum class MmaDimension { M = 0, N, K };

// Preliminary checks to try to validate that leaf is
//  a innermost dim of root of exactly the given size.
bool canValidateIsInnerDim(
    IterDomain* root,
    IterDomain* leaf,
    int inner_dim_size) {
  auto expr = leaf->definition();
  if (!leaf->extent()->isConstInt()) {
    return false;
  }
  if (leaf->extent()->evaluate() != inner_dim_size) {
    return false;
  }

  while (expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      // Inner split only
      if (leaf != split->inner()) {
        return false;
      }
      // Const split only
      if (!split->factor()->isConstInt()) {
        return false;
      }
      leaf = split->in();
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // Might consider just rejecting merge.
      auto outer = merge->outer();
      if (outer->isBroadcast()) {
        return false;
      }

      // Only support merging with constant sized dims
      if (!leaf->extent()->isConstInt()) {
        return false;
      }
      leaf = merge->inner();
    } else {
      // No support for swizzled inner dim for now.
      //  Might need to add transpose swizzle here.
      return false;
    }
    expr = leaf->definition();
  }
  return leaf == root;
}

} // namespace

void checkDimSize(
    TensorView* tv,
    std::vector<int> axis,
    std::vector<int> expect) {
  NVF_ERROR(
      axis.size() == expect.size(),
      "CheckDimSize: Mismatched axis and expect size");
  for (auto axis_index : c10::irange(axis.size())) {
    NVF_ERROR(
        ((axis[axis_index] + static_cast<int>(tv->nDims())) >= 0) &&
            (axis[axis_index] < (int)tv->nDims()),
        "CheckDimSize: axis position out of bound ",
        axis[axis_index],
        " ",
        tv->nDims());
    auto id = tv->axis(axis[axis_index]);
    NVF_CHECK(
        id->extent()->isConstInt(),
        "Mma warp mapping: instruction tile has to be constant");
    NVF_CHECK(
        id->extent()->evaluate() == expect[axis_index],
        "Mma warp mapping: unexpected tile size at",
        axis_index,
        ":",
        id->extent()->evaluate(),
        "vs",
        expect[axis_index],
        "\n for tv: ",
        tv->toString());
  }
}

static void setWarpMapped(TensorView* tv, int number_of_dims) {
  for (int id : c10::irange(number_of_dims)) {
    tv->axis(-id - 1)->toMmaSwizzled();
  }
}

namespace {

// Utility function for mma domain mapping:
//  returns the Iterdomain from the accumulator tv that corresponds
//  to the given mma dimension. See [MMA dimension matching].
std::vector<IterDomain*> getMmaDomains(MmaOp* mma, MmaDimension dimension) {
  // This utility is user facing so shouldn't ever see tensor index here.

  // Note: [Use Root Domain in Accumulator TV]
  //  Have to use root domain for accumulator tv since the operands do not have
  //  root/rfactor domains that map to the rfactor domain of output.
  //  For example:
  //   C[I,I,R,R] = mma (A[I,B,I,I], B[B,I,I,I]),
  //  if we do
  //    c->split(-1,4);
  //    c->rfactor(-1);
  //  on the mma stage we get:
  //   C[I,I,R,Io,R(4)] = mma (A[I,B,I,I], B[B,I,I,I]),
  //  and in this case Io and R(4) would not be able to find root mapping
  //  in A or B.
  //
  //  Essentially in the case of rfactor, this utility does producer side
  //   matching so looking at root domain would be required.
  //  This matching pattern should support most common matmul applications,
  //   but in follow ups we may need to extend RFactor matching if there
  //   are more complex scheduling patterns that we want to support.
  auto accumulator_domain = mma->out()->as<TensorView>()->getRootDomain();
  auto a_domain = TensorDomain::noReductions(
      mma->inA()->as<TensorView>()->getMaybeRFactorDomain());
  auto b_domain = TensorDomain::noReductions(
      mma->inB()->as<TensorView>()->getMaybeRFactorDomain());
  NVF_CHECK(
      a_domain.size() == b_domain.size() &&
          a_domain.size() == accumulator_domain.size(),
      "Inconsistent dimensions in mma op",
      a_domain.size(),
      " ",
      b_domain.size(),
      " ",
      accumulator_domain.size());

  std::vector<IterDomain*> result;

  for (auto id_idx : c10::irange(a_domain.size())) {
    // checks if this id should be included in the result
    bool include_this_id = false;
    bool is_broadcast_in_a = a_domain[id_idx]->isBroadcast();
    bool is_broadcast_in_b = b_domain[id_idx]->isBroadcast();
    bool is_reduction_id = accumulator_domain[id_idx]->isReduction();

    switch (dimension) {
      case MmaDimension::K:
        // K dimension is the dimension that is concrete in
        //  operands, and is reduced by mma. This complies with
        //  tensor contraction definition.
        include_this_id =
            !is_broadcast_in_a && !is_broadcast_in_b && is_reduction_id;
        break;
      // M and N dimension below are defined as the iterdomains
      //  that are not reduced by mma, and are concretized in this stage.
      case MmaDimension::M:
        include_this_id =
            !is_broadcast_in_a && is_broadcast_in_b && !is_reduction_id;
        break;
      case MmaDimension::N:
        include_this_id =
            is_broadcast_in_a && !is_broadcast_in_b && !is_reduction_id;
        break;

      default:
        NVF_ERROR(false, "unreachable");
    }

    if (include_this_id) {
      result.push_back(accumulator_domain.at(id_idx));
    }
  }

  return result;
}

//! Variant of getMmaDomains that returns a set
std::unordered_set<IterDomain*> getMmaDomainSet(
    MmaOp* mma,
    MmaDimension dimension) {
  auto mma_domains = getMmaDomains(mma, dimension);
  return {mma_domains.begin(), mma_domains.end()};
}

} // namespace

void WarpMmaSwizzler::scheduleLdMatrix(TensorView* tv, MmaOperand operand) {
  bool transpose = tv->definition()->as<LoadStoreOp>()->opType() ==
      LoadStoreOpType::LdMatrixTranspose;
  // For A, we have an extra outer dim (-6), which is the "warp group". For
  // Hopper, mma instructions executes on warp group level. For Turing/Ampere,
  // this dim will just have extent 1.

  //               A                                   B
  //  -6    -5  -4   -3   -2   -1     or     -5  -4   -3   -2   -1
  //[4moo, 8mi, 4k, 2ko, 2mo, 2ki]         [8ni, 4k, 2ko, 1no, 2ki]
  tv->reorder({{-2, -4}, {-3, -5}});
  //                A                                   B
  //  -6    -5   -4   -3  -2   -1     or     -5   -4   -3  -2   -1
  //[4moo, 2ko, 2mo, 8mi, 4k, 2ki]         [2ko, 1no, 8ni, 4k, 2ki]
  tv->merge(-2);
  //              A                                      B
  //  -5    -4   -3   -2  -1         or          -4   -3   -2   -1
  //[4moo, 2ko, 2mo, 8mi, 8k]                  [2ko, 1no, 8ni, 8k]
  if (transpose) {
    tv->reorder({{-2, -1}});
    //              A                                     B
    //  -5    -4   -3  -2   -1        or          -4   -3  -2   -1
    //[4moo, 2ko, 2mo, 8k, 8mi]                 [2ko, 1no, 8k, 8ni]
  }

  tv->merge(-4);
  tv->merge(-3);
  if (operand == MmaOperand::A) {
    // For A, we have an extra outer dim which is the warp group. Merge it back
    // here so that TIDx represent a warp group, instead of a single warp.
    tv->merge(-3);
  }
  //    A                         B
  // -2  -1         or          -2 -1
  //[128, 8]                   [16, 8]

  // The extent of axis(-2) is the number of threads that contains useful
  // addresses. We can not parallelize axis(-2) directly if the extent is less
  // than 32. Instead, we should split axis(-1) and merge it to axis(-2) to
  // get a complete warp of 32 threads. This makes sure that, during lowering,
  // our system can correctly compute the buffer size.
  int64_t num_tidx_with_addr = tv->axis(-2)->extent()->evaluate().as<int64_t>();
  if (num_tidx_with_addr < 32) {
    int64_t factor = 32 / num_tidx_with_addr;
    tv->split(-1, factor, false);
    tv->reorder({{-2, -3}, {-3, -2}});
    //    -3           -2              -1
    // [factor, num_tidx_with_addr, 8/factor]
    // For indexing, we only care about what we get when the index of axis(-3)
    // is 0. For higher values, they are garbage, and abandoned.
    tv->merge(-3);
  }

  //    A                      B
  // -2  -1        or        -2 -1
  //[128, 8]                [32, 4]

  tv->axis(-2)->parallelize(ParallelType::TIDx);
  // TODO: this is not really vectorization. Change its parallel type to Mma.
  tv->axis(-1)->parallelize(ParallelType::Vectorize);
  setWarpMapped(tv, 2);
}

void WarpMmaSwizzler::scheduleOperandRead(TensorView* tv, MmaOperand operand) {
  // This function works for all mma ops, regardless of the architecture.
  // Operand A and B are slightly different in the sense that operand A can be
  // (>=16)x16 matrix, but operand B can only be 8x16 or 16x16. For operand A,
  // the Hopper one is the most general one. For earlier architectures, we will
  // have some dimensions with size 1 after split, this is fine. Memory format
  // for hopper mma:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-a
  NVF_ERROR(tv->nDims() >= 2);

  //     A                            B
  //  -2   -1          or          -2   -1
  //[64m, 16k]                    [8n, 16k]
  tv->split(-2, 8);
  tv->split(-1, 2);
  tv->split(-2, 4);

  //          A                               B
  // -5  -4  -3  -2  -1      or      -5  -4  -3  -2  -1
  //[8m, 8m, 2k, 4k, 2k']           [1n, 8n, 2k, 4k, 2k']

  if (operand == MmaOperand::A) {
    // For A, we need to have an extra outer dim (-6) for warp group.
    tv->split(-5, 2);
    // On Ampere and Turing, the extent of dim -6 after the split below will be
    // just 1. On Hopper, the dim -6 will be 4 because Hopper warp group
    // instructions have 4x larger m extend than Ampere/Turing.
  }

  //            A                                 B
  // -6  -5  -4  -3  -2  -1      or      -5  -4  -3  -2  -1
  //[4m, 2m, 8m, 2k, 4k, 2k']           [1n, 8n, 2k, 4k, 2k']

  tv->reorder({{-4, -5}, {-5, -2}, {-2, -4}});

  //            A                                B
  // -6  -5  -4  -3  -2  -1     or      -5  -4  -3  -2  -1
  //[4m, 8m, 4k, 2k, 2m, 2k']          [8n, 4k, 2k, 1n, 2k']

  // ldmatrix loads multiple 8x8 matrices from shared memory to registers in a
  // swizzled memory format.
  //   +--------+--------+
  //   |        |        |
  //   |  8x8   |  8x8   |
  //   |        |        |
  //   +--------+--------+
  //   |        |        |
  //   |  8x8   |  8x8   |
  //   |        |        |
  //   +--------+--------+
  // If n_major is true, these 8x8 matrices are visited in the order of:
  // top left -> top right -> bottom left -> bottom right.
  // If n_major is false, these 8x8 matrices are visited in the order of:
  // top left -> bottom left -> top right -> bottom right.
  //
  // In principle, only `n_major = false` should be needed. But unfortunately,
  // we are taking advantage of the ldmatrix large load in a pretty hacky way.
  // For example, for Turing, only m16n8k8 is supported by hardware. But we are
  // also using a fake m16n8k16 and m16n16k16, which uses a single large
  // ldmatrix to load data to register, and run multiple mma instructions to
  // consume these data. In the future, we should only keep the m16n8k8 macro,
  // and schedule m16n8k16 and m16n16k16 more correctly than this current way.
  bool n_major =
      operand == MmaOperand::B && tv->axis(-2)->extent()->evaluate() > 1;
  if (n_major) {
    tv->reorder({{-2, -3}, {-3, -2}});
    // -5  -4  -2  -3  -1
    //[8n, 4k, 1n, 2k, 2k']
  }

  bool set_allocation = ir_utils::isLdMatrixOp(tv->definition());
  if (!set_allocation) {
    for (auto u : tv->uses()) {
      if (u->isA<MmaOp>()) {
        set_allocation = true;
        break;
      }
    }
  }
  if (set_allocation) {
    tv->setAllocationDomain(tv->getLeafDomain(), true);
  }
}

// Reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-swizzling-modes
void WarpMmaSwizzler::scheduleOperandRead(
    TensorView* tv,
    MmaInputSmemSwizzle swizzle,
    bool transpose) {
  if (transpose) {
    tv->reorder({{-2, -1}});
  }
  if (swizzle == MmaInputSmemSwizzle::None) {
    // For no-swizzle case, the entire tile are divided into 8x8 core matrices,
    // and each core matrix resides in a contiguous 8*8*2 bytes region in shared
    // memory. [K, M]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, Mo, M8]
    tv->reorder({{-2, -3}});
    // [Ko, Mo, K8, M8]
  } else if (swizzle == MmaInputSmemSwizzle::B128) {
    NVF_ERROR(false, "Not implemented yet");
  } else if (swizzle == MmaInputSmemSwizzle::B64) {
    NVF_ERROR(false, "Not implemented yet");
  } else if (swizzle == MmaInputSmemSwizzle::B32) {
    // For example [K, M]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, Mo, M8]
    tv->reorder({{-2, -3}});
    // [Ko, Mo, K8, M8]
    tv->merge(-4);
    // [KMo, K8, M8]
    tv->split(-3, 2);
    // [KMoo, KMo2, K8, M8]
    tv->reorder({{-2, -3}});
    // [KMoo, K8, KMo2, M8]
    tv->split(-3, 4);
    // [KMoo, K2, K4, KMo2, M8]
    tv->swizzle(SwizzleType::XOR, -4, -2);
    tv->merge(-4);
    tv->merge(-3);
    // [KMoo, KKMo16, M8]
  } else {
    NVF_ERROR(false, "Unsupported smem swizzle");
  }
  tv->setAllocationDomain(tv->getLeafDomain(), true);
}

void WarpMmaSwizzler::scheduleMmaWarpOutput(TensorView* tv) {
  // This function works for all mma ops, regardless of the architecture. The
  // Hopper one is the most general one. For earlier architectures, we will have
  // some dimensions with size 1 after split, this is fine.
  // Memory format for hopper mma:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d

  // Assume last 2 dims, for example [M64, N24] or [M64, N24, R]
  NVF_ERROR(tv->nDims() >= 2);
  bool is_mma_output = tv->definition()->isA<MmaOp>();

  int m_pos = is_mma_output ? -3 : -2;
  int n_pos = is_mma_output ? -2 : -1;

  //   m    n
  // [M64, N24  (,R)]
  tv->split(m_pos--, 8);
  tv->split(m_pos--, 2);
  //   m           n
  // [M4, M2, M8, N24  (,R)]
  tv->split(n_pos, 8);
  tv->split(n_pos, 2);

  n_pos -= 2;
  m_pos -= 2;
  //  m           n
  // [M4, M2, M8, N3, N4, N2  (,R)]

  tv->reorder({{m_pos + 1, n_pos + 1}, {n_pos + 1, m_pos + 2}});
  //  m           n
  // [M4, M8, N4, N3, M2, N2  (,R)]
  tv->merge(m_pos++);
  tv->merge(m_pos++);

  //       m
  // [WarpGroup128, N3, M2, N2  (,R)]

  if (is_mma_output) {
    tv->split(-1, 2);
    tv->split(-2, 4);
    m_pos -= 2;
    //       m
    // [WarpGroup128, N3, M2, N2, Ro, R4, R2]
  }

  NVF_CHECK(tv->definition() != nullptr);

  tv->axis(m_pos)->parallelize(ParallelType::TIDx);

  if (is_mma_output) {
    // Set instruction loops for mma reduce
    int pos = -1;
    while (pos > m_pos) {
      tv->axis(pos--)->parallelize(ParallelType::Mma);
    }
    setWarpMapped(tv, 7);
  }
}

void canonicalizeMmaTvOrdering(TensorView* tv) {
  std::unordered_set<IterDomain*> root_id_set{
      tv->getMaybeRFactorDomain().begin(), tv->getMaybeRFactorDomain().end()};

  auto mma = dynamic_cast<MmaOp*>(tv->definition());
  NVF_CHECK(
      mma != nullptr, "canonicalizeMmaTvOrdering : only support mma op output");

  auto m_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::M);
  auto n_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::N);
  auto k_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::K);

  std::vector<int> batch_pos, prev_reduction_pos, m_pos, n_pos, k_pos;

  int ndims = (int)tv->nDims();

  for (auto idx : c10::irange(ndims)) {
    auto id = tv->axis(idx);
    NVF_CHECK(root_id_set.count(id), id->toString(), " not a root id.");

    // Categorize each original iterdomain position
    if (m_id_set.count(id)) {
      m_pos.push_back(idx);
    } else if (n_id_set.count(id)) {
      n_pos.push_back(idx);
    } else if (k_id_set.count(id)) {
      k_pos.push_back(idx);
    } else if (id->isReduction()) {
      prev_reduction_pos.push_back(idx);
    } else {
      batch_pos.push_back(idx);
    }
  }

  // Collect all mma id's, other id's would be either
  //  batch or incoming reduction.

  // Ordering map from old position to new position
  //  that we wil build using the position vectors.
  std::unordered_map<int, int> order_map;

  // Running position counter keeping track of the
  //  current insert position in order_map.
  int current_pos = 0;

  // Utility to insert the ordered pos sequences to
  //  the ordering map.
  auto insert_to_order_map =
      [&order_map, &current_pos](const std::vector<int>& original_pos) {
        for (auto pos : original_pos) {
          order_map[pos] = current_pos++;
        }
      };

  // Order the categories, while keeping the original
  //  intra-category ordering.
  insert_to_order_map(batch_pos);
  insert_to_order_map(prev_reduction_pos);
  insert_to_order_map(m_pos);
  insert_to_order_map(n_pos);
  insert_to_order_map(k_pos);

  // Validate that all of the root ids are covered by
  //  the inserted categories.
  NVF_ERROR(current_pos == ndims, "Id not completely categorized");

  // Apply the new ordering
  tv->reorder(order_map);
}

namespace {

inline void resolveTvToMatmulDomainsMapping(
    DependenciesMap& deps_map,
    const std::vector<TensorView*>& tensors,
    IterDomain* m,
    IterDomain* n,
    IterDomain* k,
    const ComputeAtMap& ca_map) {
  for (const auto tv : tensors) {
    for (const auto domain : tv->getLeafDomain()) {
      if (ca_map.areMapped(m, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDomain::M);
        continue;
      }
      if (ca_map.areMapped(n, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDomain::N);
        continue;
      }
      if (ca_map.areMapped(k, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDomain::K);
        continue;
      }
    }
  }
}

} // anonymous namespace

ProblemIterDomainsOpt getProblemIterDomains(Fusion* fusion) {
  auto mma_exprs = ir_utils::getOpsOfType<MmaOp>(fusion);
  if (mma_exprs.size() != 1) {
    std::stringstream ss;
    ss << "Invalid number of MmaOp instances in fusion, expected 1, got "
       << mma_exprs.size();
    return ss.str();
  }
  const auto mma_output = mma_exprs.front()->out();

  // NOTE: the iter domains of MMA output should be [...,M,K,N]
  IterDomain* m = nullptr;
  IterDomain* n = nullptr;
  IterDomain* k = nullptr;

  const auto leaf_domains =
      static_cast<const TensorView*>(mma_output)->getLeafDomain();
  const auto concrete =
      TensorDomain::noReductions(TensorDomain::noBroadcasts(leaf_domains));
  if (concrete.size() < MIN_MATMUL_INPUTS_NUMBER) {
    std::stringstream ss;
    ss << "Failed to find the minimum number of MMA input candidates, expected "
       << MIN_MATMUL_INPUTS_NUMBER << ", got " << concrete.size();
    return ss.str();
  }

  // M,N are inner most concrete iter domains
  m = concrete.rbegin()[1];
  n = concrete.rbegin()[0];

  // K is a reduction domain, search for the inner most reduction domain
  for (auto iter_domain = leaf_domains.rbegin();
       iter_domain != leaf_domains.rend();
       ++iter_domain) {
    if ((*iter_domain)->isReduction()) {
      k = *iter_domain;
      break;
    }
  }
  NVF_ERROR(k != nullptr, "Failed to find K domain in MMA output");

  return ProblemIterDomains{m, n, k};
}

MatmulProblemLayoutOpt getMmaLayout(Fusion* fusion) {
  ComputeAtMap ca_map(fusion);
  const auto mma_input_candidates =
      ir_utils::filterByType<TensorView>(fusion->inputs()).vector();
  if (mma_input_candidates.empty()) {
    return {"Failed to find any TV that is fusion input"};
  }

  const auto mma_output_domains = getProblemIterDomains(fusion);
  if (!mma_output_domains.isValid()) {
    return mma_output_domains.getErrorMsg();
  }

  const auto domains_data = mma_output_domains.getData();
  const auto m = domains_data[(size_t)MatmulDomain::M];
  const auto n = domains_data[(size_t)MatmulDomain::N];
  const auto k = domains_data[(size_t)MatmulDomain::K];

  DependenciesMap deps_map;
  resolveTvToMatmulDomainsMapping(
      deps_map, mma_input_candidates, m, n, k, ca_map);

  bool mk_found = false;
  bool km_found = false;
  bool nk_found = false;
  bool kn_found = false;
  const static DomainsDesc mk_desc = {MatmulDomain::M, MatmulDomain::K};
  const static DomainsDesc km_desc = {MatmulDomain::K, MatmulDomain::M};
  const static DomainsDesc nk_desc = {MatmulDomain::N, MatmulDomain::K};
  const static DomainsDesc kn_desc = {MatmulDomain::K, MatmulDomain::N};

  for (const auto& item : deps_map) {
    if (item.second == mk_desc) {
      if (mk_found) {
        return {
            "Failed to find MMA input, more than one fusion input has [..., M, ..., K, ...] iter domains"};
      }
      mk_found = true;
    }
    if (item.second == km_desc) {
      if (km_found) {
        return {
            "Failed to find MMA input, more than one fusion input has [..., K, ..., M, ...] iter domains"};
      }
      km_found = true;
    }
    if (item.second == nk_desc) {
      if (nk_found) {
        return {
            "Failed to find MMA input, more than one fusion input has [..., N, ..., K, ...] iter domains"};
      }
      nk_found = true;
    }
    if (item.second == kn_desc) {
      if (kn_found) {
        return {
            "Failed to find MMA input, more than one fusion input has [..., K, ..., N, ...] iter domains"};
      }
      kn_found = true;
    }
  }

  if ((mk_found && kn_found) && !(km_found || nk_found)) {
    return MmaLayout::TT;
  }
  if ((km_found && kn_found) && !(mk_found || nk_found)) {
    return MmaLayout::NT;
  }
  if ((mk_found && nk_found) && !(km_found || kn_found)) {
    return MmaLayout::TN;
  }
  if ((km_found && nk_found) && !(mk_found || kn_found)) {
    return MmaLayout::NN;
  }

  return {"Failed to decide fusion inputs' data layout."};
}

RolesMapOpt getTensorsRoles(Fusion* fusion) {
  ComputeAtMap ca_map(fusion);
  const auto mma_input_candidates =
      ir_utils::filterByType<TensorView>(fusion->inputs()).vector();
  if (mma_input_candidates.empty()) {
    return {"Failed to find any TV that is fusion input"};
  }
  const auto mma_output_candidates =
      ir_utils::filterByType<TensorView>(fusion->outputs()).vector();
  if (mma_output_candidates.empty()) {
    return {"Failed to find any TV that is fusion output"};
  }

  const auto mma_output_domains = getProblemIterDomains(fusion);
  if (!mma_output_domains.isValid()) {
    return mma_output_domains.getErrorMsg();
  }

  const auto findInputRolesByDomains = [](const DependenciesMap& deps_map,
                                          RolesMap& roles_map) {
    for (const auto& entry : deps_map) {
      const auto& domains = entry.second;
      const auto begin = domains.begin();
      const auto end = domains.end();

      bool has_m = (end != std::find(begin, end, MatmulDomain::M));
      bool has_n = (end != std::find(begin, end, MatmulDomain::N));
      bool has_k = (end != std::find(begin, end, MatmulDomain::K));

      if (has_m && has_k && !has_n) {
        roles_map[MatmulRole::INPUT_A].push_back(entry.first);
        continue;
      }
      if (has_n && has_k && !has_m) {
        roles_map[MatmulRole::INPUT_B].push_back(entry.first);
        continue;
      }
      if (has_m && has_n && !has_k) {
        roles_map[MatmulRole::INPUT_C].push_back(entry.first);
        continue;
      }
      // Bias vectors are assigned to INPUT_C role
      if (has_m && !has_n && !has_k) {
        roles_map[MatmulRole::INPUT_C].push_back(entry.first);
        continue;
      }
    }

    for (auto& [role, tvs] : roles_map) {
      // NOTE: sort input roles in descending order by uses() size, and
      //  if equal then by name() to ensure the stable ordering of tensor
      //  views in collections assigned to the supported roles
      std::sort(tvs.begin(), tvs.end(), [](TensorView* a, TensorView* b) {
        return (a->uses().size() == b->uses().size())
            ? (a->name() < b->name())
            : (a->uses().size() > b->uses().size());
      });
    }
  };

  const auto findOutputRolesByDomains = [](const DependenciesMap& deps_map,
                                           RolesMap& roles_map) {
    std::vector<TensorView*> storage;
    storage.reserve(deps_map.size());

    for (const auto& entry : deps_map) {
      const auto& domains = entry.second;
      const auto begin = domains.begin();
      const auto end = domains.end();

      bool has_m = (end != std::find(begin, end, MatmulDomain::M));
      bool has_n = (end != std::find(begin, end, MatmulDomain::N));

      // NOTE: depending on fusion definition k domain may appear in the output:
      //  - for mma_output == fusion output k domain is present
      //  - for mma_output != fusion output (fusion with epilogue) k domain
      //    is not present

      // NOTE: the core fusion output tensors are the ones with m and n
      //  domains
      if (has_m && has_n) {
        storage.push_back(entry.first);
      }
    }

    // NOTE: sort output roles in descending order by uses() size, and
    //  if equal then by name() to ensure the stable ordering of tensor
    //  views in collections assigned to the supported roles
    std::sort(storage.begin(), storage.end(), [](TensorView* a, TensorView* b) {
      return (a->uses().size() == b->uses().size())
          ? (a->name() < b->name())
          : (a->uses().size() > b->uses().size());
    });

    if (!storage.empty()) {
      // NOTE: currently, we pick as a reference tensor one with `m` and `n`
      //       IterDomains and the most uses
      auto pos = storage.begin();
      roles_map[MatmulRole::OUTPUT_D].push_back(*pos);
      for (++pos; pos != storage.end(); ++pos) {
        roles_map[MatmulRole::OUTPUT_AUX].push_back(*pos);
      }
    }
  };

  const auto domains_data = mma_output_domains.getData();
  const auto m = domains_data[(size_t)MatmulDomain::M];
  const auto n = domains_data[(size_t)MatmulDomain::N];
  const auto k = domains_data[(size_t)MatmulDomain::K];

  DependenciesMap deps_map;
  RolesMap roles_map;

  // Handle fusion input TensorView objects
  resolveTvToMatmulDomainsMapping(
      deps_map, mma_input_candidates, m, n, k, ca_map);
  findInputRolesByDomains(deps_map, roles_map);

  deps_map.clear();

  // Handle fusion output TensorView objects
  resolveTvToMatmulDomainsMapping(
      deps_map, mma_output_candidates, m, n, k, ca_map);
  findOutputRolesByDomains(deps_map, roles_map);

  return roles_map;
}

namespace {

void addMMAOp(Fusion* fusion_, std::vector<MulSumAsMmaProps>& props) {
  auto* init = IrBuilder::create<Val>(0.0);
  for (auto prop : props) {
    IrBuilder::create<MmaOp>(prop.out, prop.a, prop.b, init);
  }
}

// Check the val (in) is the output of broadcast.
// Then check the output of the broadcast is 3D (4D for bmm).
bool hasValidBroadcastOp(TensorView* bcast_out) {
  // First check the tensorsview is 3D (4D)
  // and has one broadcast dim.
  auto dims = bcast_out->domain()->nDims();
  if (!((dims == 3 || dims == 4) &&
        bcast_out->domain()->noBroadcasts().size() == dims - 1)) {
    return false;
  }

  // Check if the definition is a broadcast op.
  if (dynamic_cast<BroadcastOp*>(bcast_out->definition())) {
    return true;
  }

  return false;
}

// This function checks if the mul-sum can be replace with a mma op. The checks
// are:
// 1. The inputs to the muls are broadcast ops.
// 2. The broadcasts have 2D or 3D(bmm) inputs.
// 3. The broadcasts only broadcast one dim and the dims are different for the 2
// muls.
// 4. There is a single reduction dim, and that dim that is not either of the
// broadcast dims.
bool broadcastsAreValid(
    TensorView* left,
    TensorView* right,
    unsigned int reduction_axis) {
  if (!(hasValidBroadcastOp(left) && hasValidBroadcastOp(right))) {
    return false;
  }

  auto bcast_l = dynamic_cast<BroadcastOp*>(left->definition());
  auto bcast_r = dynamic_cast<BroadcastOp*>(right->definition());

  // Ensure that only one dim is getting broadcast.
  auto bcastFlags_l = bcast_l->getBroadcastDimFlags();
  auto bcastFlags_r = bcast_r->getBroadcastDimFlags();
  auto count_l = std::count(bcastFlags_l.begin(), bcastFlags_l.end(), true);
  auto count_r = std::count(bcastFlags_r.begin(), bcastFlags_r.end(), true);
  if ((count_l != 1) || (count_l != count_r)) {
    return false;
  }

  // Also ensure that it's not the same dim for the two muls. that's
  // getting broadcast.
  auto idx_l = std::find(bcastFlags_l.begin(), bcastFlags_l.end(), true) -
      bcastFlags_l.begin();
  auto idx_r = std::find(bcastFlags_r.begin(), bcastFlags_r.end(), true) -
      bcastFlags_r.begin();
  if (idx_l == idx_r) {
    return false;
  }

  // Also ensure that the reduction dim is not either of the broadcast dim.
  if (reduction_axis == idx_l || reduction_axis == idx_r) {
    return false;
  }

  // Check different dimensions are the broadcast dims.
  return true;
}

// If the tensorview is a output of a cast operation, then
// return the input to the cast operation, else return the tensorview.
TensorView* getTensorviewPriorToCast(TensorView* in) {
  if (auto uCastOp = dynamic_cast<UnaryOp*>(in->definition());
      uCastOp && uCastOp->getUnaryOpType() == UnaryOpType::Cast) {
    return static_cast<TensorView*>(uCastOp->in());
  }
  return in;
}

// Check if the Mul-Sum pair represents a matmul. If so, add the properties
// of the mma op which can be a tentatice substitue. This checks that the output
// of sum has on reduction axis, and the inputs to mul are valid broadcasts.
std::optional<MulSumAsMmaProps> getMulSumInsOutsBcasts(
    BinaryOp* mop,
    ReductionOp* redop) {
  auto a = getTensorviewPriorToCast(static_cast<TensorView*>(mop->lhs()));
  auto b = getTensorviewPriorToCast(static_cast<TensorView*>(mop->rhs()));

  // Get the dimension of the reduction in the output. If not present, bail.
  // Also ensure there is only only reduction axis.
  auto red_axis = static_cast<TensorView*>(redop->out())->getReductionAxis();
  auto num_reduction_dims =
      static_cast<TensorView*>(redop->out())->domain()->nDims() -
      static_cast<TensorView*>(redop->out())->domain()->noReductions().size();
  if (!red_axis.has_value() || num_reduction_dims > 1) {
    return std::nullopt;
  }

  if (broadcastsAreValid(a, b, *red_axis)) {
    return MulSumAsMmaProps(
        mop,
        redop,
        a,
        b,
        static_cast<TensorView*>(redop->output(0)),
        dynamic_cast<BroadcastOp*>(a->definition()),
        dynamic_cast<BroadcastOp*>(b->definition()));
  }
  return std::nullopt;
}
} // namespace

void CombineMulSum::handle(ReductionOp* stmt) {
  // Check if operation is a sum.
  if (stmt->getReductionOpType() == BinaryOpType::Add) {
    auto* inputOfSum = stmt->in();
    if (inputOfSum != nullptr) {
      auto* expr = inputOfSum->definition();
      // Then check if the prodcer of the sum is a mul.
      if (auto bOp = dynamic_cast<BinaryOp*>(expr)) {
        // If it'a mul followed by a sum, put this in a list.
        if (bOp->getBinaryOpType() == BinaryOpType::Mul) {
          // If the Mul-Sum is a valid representation of a matmul,
          // then get the properties of the replacement Mma op.
          auto props = getMulSumInsOutsBcasts(bOp, stmt);
          if (props.has_value()) {
            mul_sum_props_.push_back(*props);
          }
        }
      }
    }
  }
};

std::vector<MulSumAsMmaProps> CombineMulSum::generateMulSumCanidates(
    bool use_cached_results) {
  if (use_cached_results && !mul_sum_props_.empty()) {
    return mul_sum_props_;
  }
  traverse(fusion_);
  return mul_sum_props_;
}

void CombineMulSum::replaceWithMmaOp() {
  // Recreate the mul-sum pairs since someone
  // may run this function more than once.
  generateMulSumCanidates();
  addMMAOp(fusion_, mul_sum_props_);
  return;
}

} // namespace mma_utils

} // namespace nvfuser
