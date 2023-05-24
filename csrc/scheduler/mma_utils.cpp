// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <ir/printer.h>
#include <root_domain_map.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>

namespace nvfuser {

namespace mma_utils {

void scheduleWarpTileWithReduction(TensorView* tv, MatMulTileOptions tile) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = tile.instruction_tile;

  TORCH_CHECK(
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

  TORCH_CHECK(
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
  TORCH_CHECK(
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

c10::optional<IterDomain*> getMaybeRootIfInnermostTiled(
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
    return c10::nullopt;
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
      TORCH_INTERNAL_ASSERT(
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
  TORCH_INTERNAL_ASSERT(
      current_pos == (int)ndims, "Inconsistent ordering logic");

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
  // Accept boundary case for Volta.
  if (leaf == root && leaf->isBroadcast()) {
    return true;
  }
  auto expr = leaf->definition();
  if (!leaf->extent()->isConstInt()) {
    return false;
  }
  if (leaf->extent()->evaluateInt() != inner_dim_size) {
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
      if (split->factor()->evaluateInt() < inner_dim_size) {
        // This might be too restrictive. Would need more
        //   bookkeeping to relax.
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
      if (leaf->extent()->evaluateInt() != inner_dim_size) {
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
  TORCH_INTERNAL_ASSERT(
      axis.size() == expect.size(),
      "CheckDimSize: Mismatched axis and expect size");
  for (auto axis_index : c10::irange(axis.size())) {
    TORCH_INTERNAL_ASSERT(
        ((axis[axis_index] + static_cast<int>(tv->nDims())) >= 0) &&
            (axis[axis_index] < (int)tv->nDims()),
        "CheckDimSize: axis position out of bound ",
        axis[axis_index],
        " ",
        tv->nDims());
    auto id = tv->axis(axis[axis_index]);
    TORCH_CHECK(
        id->extent()->isConstInt(),
        "Mma warp mapping: instruction tile has to be constant");
    TORCH_CHECK(
        id->extent()->evaluateInt() == expect[axis_index],
        "Mma warp mapping: unexpected tile size at",
        axis_index,
        ":",
        id->extent()->evaluateInt(),
        "vs",
        expect[axis_index]);
  }
}

void WarpMmaSwizzler::scheduleMmaWarpOutput(
    TensorView* tv,
    MmaOptions options) {
  auto macro = options.macro;
  switch (macro) {
    case MmaOptions::MacroType::Volta_16_16_4:
      scheduleVoltaM16N16K4Fp32Output(tv, options);
      if (tv->definition()->isA<MmaOp>()) {
        setWarpMapped(tv, 5);
      }
      break;
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
      scheduleTuringM16N8K16MmaWarpOutput(tv, options);
      if (tv->definition()->isA<MmaOp>()) {
        setWarpMapped(tv, 4);
      }
      break;
    case MmaOptions::MacroType::Turing_16_16_16:
    case MmaOptions::MacroType::Ampere_16_16_16:
      scheduleTuringM16N16K16MmaWarpOutput(tv, options);
      if (tv->definition()->isA<MmaOp>()) {
        setWarpMapped(tv, 4);
      }
      break;
    default:
      TORCH_CHECK(
          false, "scheduleMmaWarp: unsupported mma option ", toString(macro));
      break;
  }
}

void WarpMmaSwizzler::scheduleOperandRead(TensorView* tv, MmaOptions options) {
  // Schedules operand for inner most 3 contiguous dimensions
  // Assumes M, N, K

  switch (options.macro) {
    case MmaOptions::MacroType::Volta_16_16_4:
      scheduleVoltaOperandRead(tv, options);
      break;
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
    case MmaOptions::MacroType::Turing_16_16_16:
    case MmaOptions::MacroType::Ampere_16_16_16:
      scheduleTuringOperandRead(tv, options);
      break;
    default:
      TORCH_CHECK(false, "WarpMmaSwizzler: please specify macro");
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
  TORCH_CHECK(
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
        TORCH_INTERNAL_ASSERT(false, "unreachable");
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

  TORCH_CHECK(
      !m_dims.empty() && !n_dims.empty() && !k_dims.empty(),
      "validateMmaRootInnerMNK: MMA Axes incomplete");

  // Still check the innermost dims of each at the current state:
  TORCH_INTERNAL_ASSERT(tv->nDims() >= 3);
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(m_dims.back(), tv->axis(-3), m),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(n_dims.back(), tv->axis(-2), n),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  TORCH_INTERNAL_ASSERT(
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

  TORCH_CHECK(
      !m_dims.empty() && !n_dims.empty(),
      "validateMmaRootInnerMNK: MMA Axes incomplete");

  // Still check the innermost dims of each at the current state:
  TORCH_INTERNAL_ASSERT(tv->nDims() >= 2);
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(m_dims.back(), tv->axis(-2), m),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
  TORCH_INTERNAL_ASSERT(
      canValidateIsInnerDim(n_dims.back(), tv->axis(-1), n),
      "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
}

//! Performs checks on tv given to schedule ld matrix.
//!  Currently only allowed ones are either:
//!    1. direct output of an ldmatrix op  or
//!    2. direct output of a broadcast op following a ldmatrix op
//!  Returns true if the tv is an immediate output of ldmatrix op
//!
//! TODO: this check is a WAR with pattern matching for now.
//!  The two patterns mentioned above are the only supported use
//!    cases of ldmatrix currently. This restriction can be greatly
//!    relaxed after the iterdomain swizzle infrastructure, which
//!    will provide the capability to directly model the exact
//!    data format of ldmatrix output.
bool checkLdMatrixTv(TensorView* tv) {
  // First check if tv is an ldmatrix output:
  auto tv_def = tv->definition();
  TORCH_CHECK(tv_def != nullptr, "ldmatrix : invalid tv");
  bool is_immediate_output = true;
  if (!ir_utils::isLdMatrixOp(tv_def)) {
    // Only allow one broadcast in between tv and the ldmatrix op
    TORCH_CHECK(
        tv_def->isA<BroadcastOp>(),
        "ldmatrix: only allow serial broadcast between ldmatrix and mma");
    tv_def = tv_def->input(0)->definition();
    TORCH_CHECK(tv_def != nullptr, "ldmatrix : invalid tv");
    is_immediate_output = false;
  }
  TORCH_CHECK(ir_utils::isLdMatrixOp(tv_def), "ldmatrix : invalid op type");
  TORCH_CHECK(
      tv->nDims() >= 2,
      "ldmatrix: scheduled tv needs to be at least 2 dimensional");
  TORCH_CHECK(
      !tv->axis(-1)->isBroadcast(), "ldmatrix: unsupported scheduled axes");
  TORCH_CHECK(
      !tv->axis(-1)->isReduction(), "ldmatrix: unsupported scheduled axes");
  TORCH_CHECK(
      !tv->axis(-2)->isBroadcast(), "ldmatrix: unsupported scheduled axes");
  TORCH_CHECK(
      !tv->axis(-2)->isReduction(), "ldmatrix: unsupported scheduled axes");
  return is_immediate_output;
}

void scheduleVoltaA(TensorView* tv, MmaOptions options) {
  // Assumed:
  // [..., 16, 16 ,4]
  // [..., M,  BN, K]
  // Some validation:
  validateMmaRootInnerMNK(tv, options, 16, 16, 4);
  bool transposed = isOperandTransposed(options);

  tv->split(-3, 4);

  // Split out 16 from the bcast
  tv->split(-2, 16);
  tv->split(-2, 8);

  // -6   -5    -4  -3   -2  -1
  //[Mo4, Mi4, Noo, No2, Ni8, K]

  if (transposed) {
    tv->reorder({{-5, -3}, {-3, -5}});
    // -6   -5    -4  -3   -2  -1
    //[Mo4, No2, Noo, Mi4, Ni8, K]

  } else {
    tv->reorder({{-5, -1}, {-3, -5}, {-1, -3}});
    // -6   -5    -4  -3  -2  -1
    //[Mo4, No2, Noo,  K, Ni8, Mi4]
  }

  tv->merge(-6);
  tv->merge(-5);
  tv->merge(-4);

  //[Warp, Ni8, K/Mi4]
  tv->axis(-3)->parallelize(ParallelType::TIDx);
}

void scheduleVoltaB(TensorView* tv, MmaOptions options) {
  // Assumed:
  // [..., 16,16,4]
  // [..., BM, N, K]
  // Some validation:
  validateMmaRootInnerMNK(tv, options, 16, 16, 4);

  bool transposed = isOperandTransposed(options);
  tv->split(-3, 16);
  tv->split(-3, 8);

  tv->split(-2, 8);
  tv->split(-2, 4);

  // -7   -6   -5   -4   -3    -2   -1
  //[Moo, Mo2, Mi8, No2, Nio2, Nii4, K]
  tv->reorder({{-6, -4}, {-5, -6}, {-4, -3}, {-3, -5}});

  // -7   -6   -5   -4    -3    -2   -1
  //[Moo, Mi8, Nio2, Mo2, No2,  Nii4, K ]
  if (transposed) {
    tv->reorder({{-2, -1}, {-1, -2}});
    //  -7   -6   -5   -4    -3  -2   -1
    //[Moo, Mi8, Nio2, Mo2, No2, K, Nii4]
  }

  tv->merge(-5);
  tv->merge(-4);
  tv->merge(-3);

  //[Moo, Mi8, Warp, K/Nii4]
  tv->axis(-2)->parallelize(ParallelType::TIDx);
}

void scheduleLdMatrix(TensorView* tv, MmaOptions options) {
  // Check if tv should use ldmatrix layout and
  //   if tv is immediate output of ldmatrix
  bool is_immediate_output = checkLdMatrixTv(tv);

  // Check mma option is supported
  TORCH_CHECK(
      options.macro == MmaOptions::MacroType::Ampere_16_8_16 ||
          options.macro == MmaOptions::MacroType::Ampere_16_16_16 ||
          options.macro == MmaOptions::MacroType::Turing_16_8_16 ||
          options.macro == MmaOptions::MacroType::Turing_16_16_16,
      "scheduleLdMatrix: unknown macro for ldmatrix");

  if (options.operand == MmaOptions::Operand::A) {
    TORCH_INTERNAL_ASSERT(tv->nDims() >= 2);
    // validation:
    auto mma = options.mmaOp();
    auto m_dims = getMmaRootDimensions(tv, mma, MmaDimension::M);
    auto k_dims = getMmaRootDimensions(tv, mma, MmaDimension::K);
    bool transposed =
        (options.layout == MmaOptions::MmaLayout::NN ||
         options.layout == MmaOptions::MmaLayout::NT);

    TORCH_INTERNAL_ASSERT(
        canValidateIsInnerDim(m_dims.back(), tv->axis(-2), 16),
        "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");
    TORCH_INTERNAL_ASSERT(
        canValidateIsInnerDim(k_dims.back(), tv->axis(-1), 16),
        "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain",
        tv->toString());

    //[16m, 16k]
    tv->split(-2, 8);
    tv->split(-1, 8);

    // -4  -3  -2  -1
    //[2o, 8o, 2i, 8i]
    tv->reorder({{-4, -3}, {-3, -2}, {-2, -4}});

    //  -4  -3 -2  -1
    // [2i, 2o, 8o, 8i]

    if (transposed) {
      tv->reorder({{-1, -2}, {-2, -1}});
    }

    tv->merge(-4);
    tv->merge(-3);
    // [warp, 8i/o]

    tv->axis(-2)->parallelize(ParallelType::TIDx);
  } else if (options.operand == MmaOptions::Operand::B) {
    auto mma = options.mmaOp();
    auto n_dims = getMmaRootDimensions(tv, mma, MmaDimension::N);
    auto k_dims = getMmaRootDimensions(tv, mma, MmaDimension::K);
    bool transposed =
        (options.layout == MmaOptions::MmaLayout::NT ||
         options.layout == MmaOptions::MmaLayout::TT);

    TORCH_INTERNAL_ASSERT(
        canValidateIsInnerDim(k_dims.back(), tv->axis(-1), 16),
        "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");

    // Each ldmatrix 4 would be loading an effective 16x16x16 tile, which is 2x
    // the
    //  size of regular 16x8x16 tile supported by largest mma operation. The
    //  swizzle also needs to be different to take this into account.
    // TODO:
    //  Using an emulated 16x16x16 mma tile is a temporary step to enable the
    //   widest load possible for scheduler bring up phase.
    //  A unifying step would be needed in a follow up to support all these
    //  swizzles
    //   with a single affine utility.
    bool use_ldmatrix4 = canValidateIsInnerDim(n_dims.back(), tv->axis(-2), 16);

    if (use_ldmatrix4) {
      // [... N16, K16]
      tv->split(-2, 8);
      tv->split(-1, 8);

      //       -4   -3  -2  -1
      // [... N2o, N8, K2o, K8]
      tv->reorder({{-3, -2}, {-2, -3}});
      // [... N2o, K2o, N8, K8]

      if (transposed) {
        tv->reorder({{-1, -2}, {-2, -1}});
      }

      tv->merge(-4);
      tv->merge(-3);

      // [Warp, K8]
      tv->axis(-2)->parallelize(ParallelType::TIDx);
    } else {
      // validation:
      TORCH_INTERNAL_ASSERT(
          canValidateIsInnerDim(n_dims.back(), tv->axis(-2), 8),
          "MMA swizzle: requires instruction tile iterdomains on the innermost side of the tensordomain");

      if (transposed) {
        // [8, 16]
        tv->split(-2, 4);

        // [2i, 4i, 16]
        tv->reorder({{-1, -2}, {-2, -1}});
        // [2i, 16, 4i]

        tv->merge(-3);
        // [warp, 4i]
      } else {
        //[8, 16]
        tv->split(-1, 4);
        tv->split(-2, 2);

        // 0  1   2   3
        //[8, oo2,oi2,i4]
        tv->reorder({{-4, -2}, {-2, -4}});

        // 0     1   2  3
        //[oi2, oo2, 8,i4]

        tv->merge(-4);
        tv->merge(-3);
        //  0    1
        //[warp, i4]
      }

      tv->axis(-2)->parallelize(ParallelType::TIDx);
    }
  } else {
    TORCH_INTERNAL_ASSERT(false, "unreachable");
  }

  if (is_immediate_output) {
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }
}

} // namespace

void WarpMmaSwizzler::scheduleVoltaOperandRead(
    TensorView* tv,
    MmaOptions options) {
  switch (options.operand) {
    case MmaOptions::Operand::A:
      scheduleVoltaA(tv, options);
      setWarpMapped(tv, 3);
      break;
    case MmaOptions::Operand::B:
      scheduleVoltaB(tv, options);
      setWarpMapped(tv, 4);
      break;
    default:
      TORCH_CHECK(false, "WarpMmaSwizzler: please specify operand");
  }
}

// Fp32 and Fp16 outputs have different layouts on volta,
//   but we only support fp32 accumulate at this stage.
void WarpMmaSwizzler::scheduleVoltaM16N16K4Fp32Output(
    TensorView* tv,
    const MmaOptions& options) {
  // Assume last 2 dims [M16, N16] or [M16, N16, R]
  bool is_reduction = tv->axis(-1)->isReduction();

  // Make sure instruction tile size is correct.
  if (is_reduction) {
    validateMmaRootInnerMNK(tv, options, 16, 16, 4);
  } else {
    validateMmaRootInnerMN(tv, options, 16, 16);
  }

  int m_pos = is_reduction ? -3 : -2;

  // Assumed:
  //       m
  // [..., 16,16, (4)]
  // [..., M, N,  (R)]
  tv->split(m_pos, 4);
  tv->split(m_pos, 2);
  tv->split(m_pos + 1, 8);
  tv->split(m_pos + 1, 4);
  tv->split(m_pos + 1, 2);

  //        m-5  m-4   m-3   m-2   m-1    m     m+1   m+2
  // [..., Mo4, Mio2, Mii2,  No2, Nio2, Niio2, Niii2, (R)]
  tv->reorder(
      {{m_pos - 4, m_pos - 1},
       {m_pos - 3, m_pos - 2},
       {m_pos - 2, m_pos - 4},
       {m_pos - 1, m_pos},
       {m_pos, m_pos - 3}});

  //        m-5  m-4   m-3   m-2   m-1    m     m+1   m+2
  //  [..., Mo4, No2, Niio2, Mii2, Mio2, Nio2, Niii2, (R)]

  tv->merge(m_pos - 5);
  tv->merge(m_pos - 4);
  tv->merge(m_pos - 3);

  //  m-2   m-1   m     m+1   m+2
  //[Warp, Mio2, Nio2, Niii2, (R)]
  tv->axis(m_pos - 2)->parallelize(ParallelType::TIDx);

  if (is_reduction && tv->definition()->isA<MmaOp>()) {
    // Set instruction loops for mma reduce output
    for (int pos : c10::irange(5)) {
      if (!tv->axis(-pos - 1)->isThread()) {
        tv->axis(-pos - 1)->parallelize(ParallelType::Mma);
      }
      tv->axis(-pos - 1)->toMmaSwizzled();
    }
  }
}

void WarpMmaSwizzler::scheduleTuringOperandRead(
    TensorView* tv,
    MmaOptions options) {
  scheduleLdMatrix(tv, options);
  setWarpMapped(tv, 2);
}

void WarpMmaSwizzler::scheduleTuringM16N8K16MmaWarpOutput(
    TensorView* tv,
    const MmaOptions& options) {
  // Assume last 2 dims [M16, N8] or [M16, N8, R]
  // Locate instruction m
  bool is_reduction = tv->axis(-1)->isReduction();

  // Make sure instruction tile size is correct.
  if (is_reduction) {
    validateMmaRootInnerMNK(tv, options, 16, 8, 16);
  } else {
    validateMmaRootInnerMN(tv, options, 16, 8);
  }

  int m_pos = is_reduction ? -3 : -2;

  //  m
  // [16, 8  (,R)]
  tv->split(m_pos, 8);
  tv->split(m_pos + 1, 2);

  //          m
  // [2o, 8o, 4i, 2i (,R)]
  tv->merge(m_pos - 1);

  //       m
  // [2o, Warp, 2i (,R)]
  TORCH_CHECK(tv->definition() != nullptr);

  if (is_reduction && tv->definition()->isA<MmaOp>()) {
    // Set instruction loops for mma reduce
    for (int pos : c10::irange(4)) {
      tv->axis(-pos - 1)->parallelize(ParallelType::Mma);
    }
  }

  tv->axis(m_pos)->parallelize(ParallelType::TIDx);
}

void WarpMmaSwizzler::scheduleTuringM16N16K16MmaWarpOutput(
    TensorView* tv,
    const MmaOptions& options) {
  // Assume last 2 dims [M16, N8] or [M16, N8, R]
  // Locate instruction m
  bool is_reduction = tv->axis(-1)->isReduction();

  // Make sure instruction tile size is correct.
  if (is_reduction) {
    validateMmaRootInnerMNK(tv, options, 16, 16, 16);
  } else {
    validateMmaRootInnerMN(tv, options, 16, 16);
  }

  int m_pos = is_reduction ? -3 : -2;
  //  m
  // [16, 16  (,R)]

  tv->split(m_pos + 1, 8);
  //       m
  // [16, n2, 8 (,R)]
  tv->reorder({{m_pos, m_pos - 1}, {m_pos - 1, m_pos}});

  //       m
  // [n2, 16, 8  (,R)]
  tv->split(m_pos, 8);
  tv->split(m_pos + 1, 2);

  //          m
  // [2o, 8o, 4i, 2i (,R)]
  tv->merge(m_pos - 1);

  //       m
  // [2o, Warp, 2i (,R)]
  TORCH_CHECK(tv->definition() != nullptr);

  if (is_reduction && tv->definition()->isA<MmaOp>()) {
    // Set instruction loops for mma reduce
    for (int pos : c10::irange(5)) {
      tv->axis(-pos - 1)->parallelize(ParallelType::Mma);
    }
  }

  tv->axis(m_pos)->parallelize(ParallelType::TIDx);
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
  TORCH_CHECK(
      mma != nullptr, "canonicalizeMmaTvOrdering : only support mma op output");

  auto m_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::M);
  auto n_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::N);
  auto k_id_set = mma_utils::getMmaDomainSet(mma, mma_utils::MmaDimension::K);

  std::vector<int> batch_pos, prev_reduction_pos, m_pos, n_pos, k_pos;

  int ndims = (int)tv->nDims();

  for (auto idx : c10::irange(ndims)) {
    auto id = tv->axis(idx);
    TORCH_CHECK(root_id_set.count(id), id->toString(), " not a root id.");

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
  TORCH_INTERNAL_ASSERT(current_pos == ndims, "Id not completely categorized");

  // Apply the new ordering
  tv->reorder(order_map);
}

} // namespace mma_utils

} // namespace nvfuser
