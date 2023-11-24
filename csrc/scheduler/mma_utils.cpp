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

void WarpMmaSwizzler::scheduleMmaOutput(TensorView* tv, MmaOptions options) {
  // This function works for all mma ops, regardless of the architecture. The
  // Hopper one is the most general one. For earlier architectures, we will have
  // multiple dimensions with size 1 after split, this is fine.
  // Memory format for hopper mma:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d

  // Assume last 2 dims, for example [M64, N24] or [M64, N24, R]
  // Locate instruction m
  bool is_mma_output = tv->definition()->isA<MmaOp>();

  int m_pos = is_mma_output ? -3 : -2;
  int n_pos = is_mma_output ? -2 : -1;

  //   m    n
  // [M64, N24  (,R)]
  tv->split(m_pos--, 8);
  tv->split(m_pos--, 2);
  //   m           n
  // [M4, M2, M8, N24  (,R)]
  tv->split(n_pos--, 8);
  tv->split(n_pos-- + 1, 2);
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
    // TODO: I am not sure if this part works for Hopper, need to revisit this.
    tv->split(-1, 2);
    tv->split(-2, 4);
    m_pos -= 2;
    //       m
    // [WarpGroup128, N3, M2, N2, Ro, R4, R2]
  }

  NVF_CHECK(tv->definition() != nullptr);

  if (is_mma_output) {
    // Set instruction loops for mma reduce
    int pos = -1;
    while (pos > m_pos) {
      tv->axis(pos--)->parallelize(ParallelType::Mma);
    }
  }

  tv->axis(m_pos)->parallelize(ParallelType::TIDx);

  if (is_mma_output) {
    setWarpMapped(tv, 7);
  }
}

void WarpMmaSwizzler::scheduleOperandRead(TensorView* tv, MmaOptions options) {
  // Schedules operand for inner most 3 contiguous dimensions
  // Assumes M, N, K

  switch (options.macro) {
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
    case MmaOptions::MacroType::Turing_16_16_16:
    case MmaOptions::MacroType::Ampere_16_16_16:
      scheduleTuringOperandRead(tv);
      break;
    default:
      NVF_CHECK(false, "WarpMmaSwizzler: please specify macro");
      break;
  }
}

void WarpMmaSwizzler::setWarpMapped(TensorView* tv, int number_of_dims) {
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

// [MMA dimension matching]
// Returns all the axes that correspond to the given mma dimension. This is the
//   first relaxation step on the mma check.
// Mma operations concerns 3 dimensions, namely, the M, N,
//  and K dimension, more details see [Operand Layout Convention] in mma_type.h.
//  The current implementation, for best effort safety, supports the patterns
//  where the root axes can be classified into one of the 3 dimension types.
//  This is a helpful initial step into defining tensor contraction
//  optimizations.
//
// A concrete example:
//  T0 [I0, I1, I2, R3, I4, I5] = mma(T1[I01, B11, B21, I31, I41, B51], T2[B02,
//  I12, B22, I32, I42, I52], {3};
// In this case some example querries:
//  K dimension of T0 = {R3}
//  M dimension of T1 = {I01}
//  N dimension of T2 = {I52}
//  etc.
std::vector<IterDomain*> getMmaRootDimensions(
    TensorView* tv,
    MmaOp* mma,
    MmaDimension dimension) {
  // Build a fusion-level root domain map
  //  so we can use the mma swizzles on non-immediate tensor operands, for
  //  example loadstore staging ops.
  ComputeAtRootDomainMap root_map;
  root_map.build();

  // FIXME:
  // Several optimization is possible at this stage but assuming we don't have
  //  a lot of mma ops in a fusion this could be lower priority.
  // First it'd be nice not having to build root map every time this function
  //  is called. That'd require some explicit boundary where we "lock" the
  //  compute in the fusion so the root map stays valid.
  // Second it'd reduce complexity of the below matching by an order if we have
  //  something similar to "disjointSetOf" in idGraph, for just the root domains
  //  at scheduler composing time.
  auto mma_root_dimensions = getMmaDomains(mma, dimension);
  auto mma_accumulator_tv = mma->out()->as<TensorView>();

  std::vector<IterDomain*> result;

  // Need to use root domain for accumulator tv and maybe rfactor domain
  //  otherwise. See [Use Root Domain in Accumulator TV].
  auto is_mma_output =
      tv->definition() != nullptr && tv->definition()->isA<MmaOp>();
  const auto& tv_root_domain =
      is_mma_output ? tv->getRootDomain() : tv->getMaybeRFactorDomain();

  // Loop through tensorview's root domains and accumulate all the
  //  root domain IterDomain's that maps to any of the collected
  //  mma root dimension from the mma accumulator tv.
  for (auto tv_id : tv_root_domain) {
    if (std::any_of(
            mma_root_dimensions.begin(),
            mma_root_dimensions.end(),
            [&](IterDomain* mma_id) {
              return root_map.canMap(
                  tv->domain(), tv_id, mma_accumulator_tv->domain(), mma_id);
            })) {
      result.push_back(tv_id);
    }
  }

  return result;
}

//! Utility function to help check that the innermost 3 iterdomains
//!  are also the corresponding innermost {m,n,k} dimensions of
//!  the root id's that are participating in the mma operation.
//! This is a format check before the warp mma swizzler applies mma
//!  swizzles to make sure that the swizzler is applying the right
//!  swizzles to the right axes.
//! This check will be relaxed as we build out the mma usage patterns.
void validateMmaRootInnerMNK(
    TensorView* tv,
    MmaOptions options,
    int m,
    int n,
    int k) {
  auto mma = options.mmaOp();
  auto m_dims = getMmaRootDimensions(tv, mma, MmaDimension::M);
  auto n_dims = getMmaRootDimensions(tv, mma, MmaDimension::N);
  auto k_dims = getMmaRootDimensions(tv, mma, MmaDimension::K);

  NVF_CHECK(
      !m_dims.empty() && !n_dims.empty() && !k_dims.empty(),
      "validateMmaRootInnerMNK: MMA Axes incomplete");

  // Still check the innermost dims of each at the current state:
  NVF_ERROR(tv->nDims() >= 3);
  NVF_ERROR(
      canValidateIsInnerDim(m_dims.back(), tv->axis(-3), m),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  NVF_ERROR(
      canValidateIsInnerDim(n_dims.back(), tv->axis(-2), n),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  NVF_ERROR(
      canValidateIsInnerDim(k_dims.back(), tv->axis(-1), k),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
}

//! Utility function to help check that the innermost 3 iterdomains
//!  are also the corresponding innermost {m,n} dimensions of
//!  the root id's that are participating in the mma operation.
//! This is a format check before the warp mma swizzler applies mma
//!  swizzles to make sure that the swizzler is applying the right
//!  swizzles to the right axes.
//! This check will be relaxed as we build out the mma usage patterns.
void validateMmaRootInnerMN(TensorView* tv, MmaOptions options, int m, int n) {
  auto mma = options.mmaOp();
  auto m_dims = getMmaRootDimensions(tv, mma, MmaDimension::M);
  auto n_dims = getMmaRootDimensions(tv, mma, MmaDimension::N);

  NVF_CHECK(
      !m_dims.empty() && !n_dims.empty(),
      "validateMmaRootInnerMNK: MMA Axes incomplete");

  // Still check the innermost dims of each at the current state:
  NVF_ERROR(tv->nDims() >= 2);
  NVF_ERROR(
      canValidateIsInnerDim(m_dims.back(), tv->axis(-2), m),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  NVF_ERROR(
      canValidateIsInnerDim(n_dims.back(), tv->axis(-1), n),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
}

} // namespace

void WarpMmaSwizzler::scheduleTuringOperandRead(TensorView* tv) {
  NVF_ERROR(tv->nDims() >= 2);
  //  -2   -1          or          -2   -1
  //[16m, 16k]                    [8n, 16k]
  tv->split(-2, 8);
  tv->split(-1, 2);
  tv->split(-2, 4);

  // -5  -4  -3  -2  -1      or      -5  -4  -3  -2  -1
  //[2m, 8m, 2k, 4k, 2k']           [1n, 8n, 2k, 4k, 2k']
  tv->reorder({{-4, -5}, {-5, -2}, {-2, -4}});

  // -5  -4   -3  -2  -1    or      -5  -4  -3  -2  -1
  //[8m, 4k, 2k, 2m, 2k']          [8n, 4k, 2k, 1n, 2k']
  tv->setAllocationDomain(tv->getLeafDomain(), true);
}

void WarpMmaSwizzler::scheduleLdMatrix(TensorView* tv, bool mn_major) {
  bool transpose = tv->definition()->as<LoadStoreOp>()->opType() ==
      LoadStoreOpType::LdMatrixTranspose;
  //  -5   -4   -3   -2    -1          or          -5   -4   -3   -2   -1
  //[8mni, 4k, 2ko, 2mno, 2ki]                   [8mni, 4k, 2ko, 1mno, 2ki]
  tv->reorder({{-2, -4}, {-3, -5}});
  //  -5   -4    -3   -2   -1          or          -5   -4    -3   -2   -1
  //[2ko, 2mno, 8mni, 4k, 2ki]                   [2ko, 1mno, 8mni, 4k, 2ki]
  tv->merge(-2);
  //  -4   -3    -2   -1         or          -4   -3    -2   -1
  //[2ko, 2mno, 8mni, 8k]                  [2ko, 1mno, 8mni, 8k]
  if (transpose) {
    tv->reorder({{-2, -1}});
    //  -4   -3   -2   -1        or          -4    -3   -2   -1
    //[2ko, 2mno, 8k, 8mni]                 [2ko, 1mno, 8k, 8mni]
  }
  if (mn_major) {
    tv->reorder({{-4, -3}, {-3, -4}});
    //  -4    -3  -2   -1        or           -4    -3  -2   -1
    //[2mno, 2ko, 8k, 8mni]                 [1mno, 2ko, 8k, 8mni]
  }
  tv->merge(-4);
  tv->merge(-3);
  // -2  -1         or          -2  -1
  //[32, 8k]                   [16, 8k]

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

  // -2 -1        or          -2 -1
  //[32, 8k]                [32, 4k]

  tv->axis(-2)->parallelize(ParallelType::TIDx);
  // TODO: this is not really vectorization. Change its parallel type to Mma.
  tv->axis(-1)->parallelize(ParallelType::Vectorize);
  setWarpMapped(tv, 2);
}

namespace {

bool isMmaInitLoop(const kir::Scope& loop_body) {
  for (auto expr : loop_body.exprs()) {
    if (auto inner_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      if (!isMmaInitLoop(inner_loop->body())) {
        return false;
      }
    } else if (auto ldst = dynamic_cast<LoadStoreOp*>(expr)) {
      if (!ir_utils::isTvOp(ldst)) {
        return false;
      }
      if (auto ti = dynamic_cast<kir::TensorIndex*>(ldst->output(0))) {
        if (!ti->view()->definition() ||
            !ti->view()->definition()->isA<MmaOp>()) {
          return false;
        }
      }
      if (auto tv = dynamic_cast<TensorView*>(ldst->output(0))) {
        if (!tv->definition() || !tv->definition()->isA<MmaOp>()) {
          return false;
        }
      }
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      if (!isMmaInitLoop(ite->thenBody())) {
        return false;
      }
      if (!isMmaInitLoop(ite->elseBody())) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

} // namespace

bool isMmaInitLoop(const kir::ForLoop* loop) {
  return isMmaInitLoop(loop->body());
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

MatmulProblemLayoutOpt getMatmulLayout(Fusion* fusion) {
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
    return MmaOptions::MmaLayout::TT;
  }
  if ((km_found && kn_found) && !(mk_found || nk_found)) {
    return MmaOptions::MmaLayout::NT;
  }
  if ((mk_found && nk_found) && !(km_found || kn_found)) {
    return MmaOptions::MmaLayout::TN;
  }
  if ((km_found && nk_found) && !(mk_found || kn_found)) {
    return MmaOptions::MmaLayout::NN;
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

  const auto findRolesByDomains = [](const DependenciesMap& deps_map,
                                     RolesMap& roles_map,
                                     const bool processing_output) {
    for (const auto& entry : deps_map) {
      const auto& domains = entry.second;
      const auto begin = domains.begin();
      const auto end = domains.end();

      bool has_m = (end != std::find(begin, end, MatmulDomain::M));
      bool has_n = (end != std::find(begin, end, MatmulDomain::N));
      bool has_k = (end != std::find(begin, end, MatmulDomain::K));

      if (!processing_output && has_m && has_k && !has_n) {
        roles_map[MatmulRole::INPUT_A].push_back(entry.first);
        continue;
      }
      if (!processing_output && has_n && has_k && !has_m) {
        roles_map[MatmulRole::INPUT_B].push_back(entry.first);
        continue;
      }
      if (!processing_output && has_m && has_n && !has_k) {
        roles_map[MatmulRole::INPUT_C].push_back(entry.first);
        continue;
      }
      // Bias vectors are assigned to INPUT_C role
      if (!processing_output && has_m && !has_n && !has_k) {
        roles_map[MatmulRole::INPUT_C].push_back(entry.first);
        continue;
      }

      // NOTE: depending on fusion definition k domain may appear in the output:
      //  - for mma_output == fusion output k domain is present
      //  - for mma_output != fusion output (fusion with epilogue) k domain
      //    is not present
      if (processing_output && has_m && has_n) {
        roles_map[MatmulRole::OUTPUT_D].push_back(entry.first);
        continue;
      }
    }
    for (auto& [role, tvs] : roles_map) {
      // sort tvs by name()
      std::sort(tvs.begin(), tvs.end(), [](TensorView* a, TensorView* b) {
        return a->name() < b->name();
      });
    }
  };

  const auto domains_data = mma_output_domains.getData();
  const auto m = domains_data[(size_t)MatmulDomain::M];
  const auto n = domains_data[(size_t)MatmulDomain::N];
  const auto k = domains_data[(size_t)MatmulDomain::K];

  DependenciesMap deps_map;
  RolesMap roles_map;

  // Handle fusion input TensorView objects
  bool handling_output = false;
  resolveTvToMatmulDomainsMapping(
      deps_map, mma_input_candidates, m, n, k, ca_map);
  findRolesByDomains(deps_map, roles_map, handling_output);

  deps_map.clear();

  // Handle fusion output TensorView objects
  handling_output = true;
  resolveTvToMatmulDomainsMapping(
      deps_map, mma_output_candidates, m, n, k, ca_map);
  findRolesByDomains(deps_map, roles_map, handling_output);

  return roles_map;
}

} // namespace mma_utils

} // namespace nvfuser
