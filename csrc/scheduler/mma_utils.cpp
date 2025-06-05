// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>

#include <device_lower/utils.h>
#include <disjoint_set.h>
#include <id_model/id_model.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <options.h>
#include <scheduler/mma_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/utils.h>
#include <type.h>
#include <val_graph.h>
#include <variant>

namespace nvfuser {

namespace mma_utils {

//! A wrapper to get MMA Tensor data types
//!   The order of returned types: A, B, OUTPUT
inline mma_utils::MmaDataTypes getMmaDataTypes(
    const TensorRolesMap& tensor_roles) {
  auto getMMADataType = [&](MatmulTensorRole role) {
    auto entry = tensor_roles.find(role);
    if (entry != tensor_roles.end() && !entry->second.empty()) {
      return entry->second.front()->dtype();
    }
    NVF_THROW("Get MMA Tensor data type failed!");
  };
  const auto a_type = getMMADataType(MatmulTensorRole::OPERAND_A);
  const auto b_type = getMMADataType(MatmulTensorRole::OPERAND_B);
  const auto c_type = getMMADataType(MatmulTensorRole::OUTPUT);
  return mma_utils::MmaDataTypes{a_type, b_type, c_type};
}

//! Return sizes of smem_a, smem_b, smem_c in bytes
std::tuple<int64_t, int64_t, int64_t> computeSharedMemorySizes(
    const MatMulTileOptions& gemm_tile,
    const MatmulParams::CircularBufferOptions& circular_buffer_options,
    const MmaDataTypes& data_types) {
  const auto properties = at::cuda::getCurrentDeviceProperties();

  auto warp_dims = gemm_tile.cta_tile / gemm_tile.warp_tile;

  int64_t ab_factor = circular_buffer_options.circular_buffer_smem_write
      ? circular_buffer_options.smem_circular_buffer_stage
      : 1;

  // see scheduleContiguousVectorLoad
  const int64_t vector_word = 8;
  const int64_t round_to_factor = warp_dims.m * warp_dims.n * warp_dims.k *
      properties->warpSize * vector_word;
  const int64_t mk = gemm_tile.cta_tile.m * gemm_tile.cta_tile.k;
  const int64_t nk = gemm_tile.cta_tile.n * gemm_tile.cta_tile.k;
  const int64_t smem_a = ceilDiv(mk, round_to_factor) * round_to_factor *
      ab_factor * dataTypeSize(data_types[0]);
  const int64_t smem_b = ceilDiv(nk, round_to_factor) * round_to_factor *
      ab_factor * dataTypeSize(data_types[1]);
  const int64_t smem_c =
      gemm_tile.cta_tile.m * gemm_tile.cta_tile.n * dataTypeSize(data_types[2]);

  return {smem_a, smem_b, smem_c};
}

int64_t computeExpectedSharedMemoryUsage(
    const MatmulParams* mparams,
    const MmaDataTypes& data_types,
    bool smem_a_reuse_guaranteed,
    bool smem_b_reuse_guaranteed) {
  const auto [smem_a, smem_b, smem_c] = computeSharedMemorySizes(
      mparams->tile_sizes, mparams->circular_buffer_options, data_types);

  if (mparams->use_smem_epilogue) {
    if (mparams->promote_prologue_smem_reuse) {
      return (int64_t)std::max(
          smem_c + (smem_a_reuse_guaranteed ? 0 : smem_a) +
              (smem_b_reuse_guaranteed ? 0 : smem_b),
          smem_a + smem_b);
    } else {
      return (int64_t)(smem_a + smem_b + smem_c);
    }
  } else {
    return (int64_t)(smem_a + smem_b);
  }
}

std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    int smem_circular_buffer_stage,
    const MmaDataTypes& data_types,
    bool smem_a_reuse_guaranteed,
    bool smem_b_reuse_guaranteed,
    bool ignore_occupancy_drop) {
  const size_t shared_memory_available = deviceAvailableSharedMemoryBytes();

  // We clip smem_circular_buffer_stage to 1 since we will always load operands
  // to smem even if stages=0. That is, we interpret stages <= 1 as requesting
  // "no circular-buffering", but we still stage incoming data to smem.
  if (smem_circular_buffer_stage < 1) {
    smem_circular_buffer_stage = 1;
  }

  // Create a temporary CircularBufferOptions with full circular buffering, for
  // estimating shared memory size.
  MatmulParams::CircularBufferOptions circular_buffer_options{
      /*circular_buffer_smem_write=*/true,
      /*circular_buffer_smem_read=*/true,
      smem_circular_buffer_stage};

  const auto [smem_a, smem_b, smem_c] =
      computeSharedMemorySizes(gemm_tile, circular_buffer_options, data_types);

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

  // Regardless of occupancy considerations, if we cannot fit an smem epilogue
  // without reuse then we must promote reuse
  bool must_reuse = shared_memory_available < total_with_noreuse_smem_epilogue;

  // shortcut where occupancy change is ignored.
  if (ignore_occupancy_drop) {
    if (must_reuse) {
      return {shared_memory_available >= total_with_reused_smem_epilogue, true};
    } else {
      return {true, false};
    }
  }

  // use additional shared memory for epilogue if occupancy is not changed.
  // occupancy is estimated using register and shared memory usage.
  auto warp_dims = gemm_tile.cta_tile / gemm_tile.warp_tile;
  const auto warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;
  const auto threads_per_block =
      warp_dims.m * warp_dims.n * warp_dims.k * warp_size;
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
  bool promote_prologue_smem_reuse = must_reuse ||
      blocks_per_sm_with_reused_smem_epilogue !=
          blocks_per_sm_with_noreuse_smem_epilogue;

  return {
      blocks_per_sm_with_reused_smem_epilogue ==
          blocks_per_sm_without_smem_epilogue,
      promote_prologue_smem_reuse};
}

TensorView* getOperandTv(
    const TensorRolesMap& tensor_roles,
    MatmulTensorRole role) {
  const auto it = tensor_roles.find(role);
  NVF_ERROR(it != tensor_roles.end(), "Could not find any tensors with role");
  const std::vector<TensorView*>& operands = it->second;
  NVF_ERROR(
      isOptionEnabled(EnableOption::FuseMultipleMatmuls) ||
          operands.size() == 1,
      "Exactly one operand is expected in each A and B role");
  return operands.front();
}

std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    const int smem_circular_buffer_stage,
    const TensorRolesMap& tensor_roles,
    const bool ignore_occupancy_drop) {
  auto data_types = getMmaDataTypes(tensor_roles);
  // getMmaDataTypes provides the dtypes of A, B, and OUTPUT.
  // These are the problem types that indicate the gmem IO. We use smem to load
  // A and B, but instead of OUTPUT which is the result of the epilogue, we
  // store mma_result which is the _input_ to the epilogue. In cases where the
  // epilogue contains a cast back down to reduced precision, we will still use
  // Float for the epilogue smem. If we support Double or Complex in the future
  // then we might need a better way to determine this data type.
  data_types[2] = DataType::Float;

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
  const TensorView* a = getOperandTv(tensor_roles, MatmulTensorRole::OPERAND_A);
  const TensorView* b = getOperandTv(tensor_roles, MatmulTensorRole::OPERAND_B);
  bool smem_a_reuse_guaranteed = a->uses().size() == 1;
  bool smem_b_reuse_guaranteed = b->uses().size() == 1;

  return generateSharedMemoryEpilogueHeuristics(
      gemm_tile,
      smem_circular_buffer_stage,
      data_types,
      smem_a_reuse_guaranteed,
      smem_b_reuse_guaranteed,
      ignore_occupancy_drop);
}

void scheduleWarpTileWithReduction(
    TensorView* tv,
    MatMulTileOptions tile,
    MmaMacro macro) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = getMmaOpShape(macro);

  // Do not split K dimension of CTA tile into multiple warp tiles
  NVF_CHECK(
      cta_tile.k == warp_tile.k,
      "CTA tile and warp tile must have same K dimension");

  mma_utils::checkDimSize(
      tv, {-3, -2, -1}, {cta_tile.m, cta_tile.n, cta_tile.k});

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
}

void scheduleWarpTileWithNoReduction(
    TensorView* tv,
    MatMulTileOptions tile,
    MmaMacro macro) {
  // Assumes
  // [M, N, K]
  auto cta_tile = tile.cta_tile;
  auto warp_tile = tile.warp_tile;
  auto instruction_tile = getMmaOpShape(macro);

  mma_utils::checkDimSize(tv, {-2, -1}, {cta_tile.m, cta_tile.n});

  NVF_CHECK(
      cta_tile.k % warp_tile.k == 0,
      "Number of warp on k dimension need to be integer");

  int64_t num_warp_k = cta_tile.k / warp_tile.k;

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
    int64_t vector_word,
    bool vectorize) {
  auto warp_dims = tile.cta_tile / tile.warp_tile;
  int64_t num_of_thread = warp_dims.m * warp_dims.n * warp_dims.k * 32;

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

void makeTile(
    AbstractMatmulTensor& abten,
    const std::vector<int64_t>& tile_sizes) {
  NVF_CHECK(
      abten.size() >= tile_sizes.size(),
      "Tensor dimension less than tile dimension!");

  // Split the inner dimensions
  size_t num_split_axes = 0;
  for (int64_t i = (int64_t)abten.size() - 1; i >= 0; --i) {
    if (num_split_axes > 2) {
      break;
    }
    const std::optional<MatmulDimRole> id_role_opt = abten.getTag(i);
    if (!id_role_opt.has_value()) {
      continue;
    }
    const MatmulDimRole id_role = id_role_opt.value();
    // Assumes tile_sizes are given in m,n,k order
    switch (id_role) {
      case MatmulDimRole::M:
        abten.split(i, tile_sizes.at(0));
        break;
      case MatmulDimRole::N:
        abten.split(i, tile_sizes.at(1));
        break;
      case MatmulDimRole::K:
        abten.split(i, tile_sizes.at(2));
        break;
      default:
        continue;
    }
    num_split_axes++;
  }
  // The transformation above is:
  //   Before               After
  // [..., M, N, K] -> [..., Mo, Mi, No, Ni, Ko, Ki]

  // Now we re-order the tiles so that all the outer tiles are
  //  on the left of all the inner tiles
  std::unordered_map<int64_t, int64_t> reorder_map_old_to_new;

  // Number of tiled inner dimensions after we split.
  const auto split_tile_dimension_size = 2 * num_split_axes;
  for (auto idx : arange(split_tile_dimension_size)) {
    // We want to reorder as follows:
    //           Before                               After
    //                                      vvv group0 vvv  vvv group1 vvv
    // [..., Mo, Mi, No, Ni, Ko, Ki] -> [..., Mo, No, Ko,     Mi, Ni, Ki]

    // The index offset within group of current
    //  iterdomain, with grouping specified above.
    auto index_within_group = idx / 2;

    // The index of the group the current id belongs
    //  to, as specified above.
    auto group_index = idx % 2;

    // Calculate the actual index after reordering
    auto index_after_reorder =
        group_index * num_split_axes + index_within_group;

    // Add pair {idx_before, idx_after} to re-order map.
    reorder_map_old_to_new.insert(std::make_pair(
        idx - split_tile_dimension_size,
        index_after_reorder - split_tile_dimension_size));
  }

  // Apply the re-order map to abstract tensor
  abten.reorder(reorder_map_old_to_new);
}

void makeTile(TensorView* tv, const std::vector<int64_t>& tile_sizes) {
  // We will create an AbstractMatmulTensor so that we can use the abstract
  // makeTile implementation above.

  // Set tags for the innermost axes corresponding to m,n,k (omitting some
  // axes if tile_sizes.size() < 3
  std::vector<std::unordered_set<MatmulDimRole>> axis_roles(tv->nDims());
  NVF_ERROR(axis_roles.size() >= tile_sizes.size());
  for (size_t i : arange(tile_sizes.size())) {
    size_t pos = axis_roles.size() - tile_sizes.size() + i;
    switch (i) {
      case 0:
        axis_roles[pos].insert(MatmulDimRole::M);
        break;
      case 1:
        axis_roles[pos].insert(MatmulDimRole::N);
        break;
      case 2:
        axis_roles[pos].insert(MatmulDimRole::K);
        break;
      default:
        NVF_THROW("Length tile_sizes must be 3 or less");
    }
  }
  AbstractMatmulTensor abten(tv->getLoopDomain(), axis_roles);
  makeTile(abten, tile_sizes);
  tv->setLoopDomain(abten.as<IterDomain*>());
}

void makeTile(
    TensorView* tv,
    const GemmTile& mnk_tile_sizes,
    const std::vector<MatmulDimRole>& axis_roles) {
  NVF_ERROR(
      tv->getLoopDomain().size() == axis_roles.size(),
      "Tensor dimension must equal number of provided axis roles");

  std::unordered_set<MatmulDimRole> axis_set(
      axis_roles.begin(), axis_roles.end());
  NVF_ERROR(
      axis_set.size() == axis_roles.size(),
      "Repeated axis roles are not allowed");
  // Here we fill out tile_sizes to match the given axis roles. For example
  // axis_roles might be something like [N, M], in which case we should use
  // {mnk_tile_sizes.n, mnk_tile_sizes.m}.
  std::vector<int64_t> tile_sizes;
  for (MatmulDimRole role : axis_roles) {
    switch (role) {
      case MatmulDimRole::Batch:
        NVF_ERROR(tile_sizes.empty(), "Batch dimension must be first");
        break;
      case MatmulDimRole::M:
        tile_sizes.push_back(mnk_tile_sizes.m);
        break;
      case MatmulDimRole::N:
        tile_sizes.push_back(mnk_tile_sizes.n);
        break;
      case MatmulDimRole::K:
        tile_sizes.push_back(mnk_tile_sizes.k);
        break;
    }
  }

  makeTile(tv, tile_sizes);
}

namespace {

std::optional<IterDomain*> getMaybeAllocationIfInnermostTiled(
    IterDomain* id,
    const std::unordered_set<IterDomain*>& maybe_allocation_id_set) {
  // Root id defaults to an "innermost id".
  while (id->definition() && !maybe_allocation_id_set.count(id)) {
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

void orderTiledConcreteIdAsMaybeAllocationDomain(TensorView* tv) {
  int64_t ndims = tv->nDims();

  // Keep track of the left most position where we will
  //  be reordering the axes.
  int64_t leftmost_pos = ndims;

  // Pull the maybe allocation domain id's of the given tv.
  std::unordered_set<IterDomain*> id_set{
      tv->getMaybeAllocationDomain().begin(),
      tv->getMaybeAllocationDomain().end()};

  // Keep track of loop positions that is either a reduction
  //  or a broadcast.
  // Note: Currently don't really see a case where this function
  //  should be called on a reduction output tv, but adding them
  //  here for completeness.
  std::deque<int64_t> broadcast_or_reduction_pos;

  // Map the id's to their innermost concrete id's
  //  on the loop.
  std::unordered_map<IterDomain*, int64_t> id_to_inner_loop_pos;

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
  for (int64_t i = ndims - 1; i >= 0; i--) {
    auto loop_id = tv->axis(i);
    if (loop_id->isBroadcast() || loop_id->isReduction()) {
      // Register this reduction or broadcast axis
      //  to reorder.
      broadcast_or_reduction_pos.push_front(i);
      leftmost_pos = i;
      continue;
    }
    auto maybe_alloc_domain =
        getMaybeAllocationIfInnermostTiled(loop_id, id_set);

    if (maybe_alloc_domain.has_value()) {
      // Found an innermost id, add them to the
      //  axes to reorder.
      NVF_ERROR(
          id_to_inner_loop_pos
              .insert(std::make_pair(maybe_alloc_domain.value(), i))
              .second,
          "Multiple \"innermost\" id seen for id :",
          maybe_alloc_domain.value()->toString(),
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
  int64_t current_pos = (int64_t)leftmost_pos;
  std::unordered_map<int64_t, int64_t> reorder_map_old_to_new;

  // first place all the broadcast and reduction on the left:
  for (auto original_broadcast_or_reduction_pos : broadcast_or_reduction_pos) {
    reorder_map_old_to_new[original_broadcast_or_reduction_pos] = current_pos++;
  }

  // Next put all the innermost loop id's, we make sure that
  //  the inner tile ordering follows the corresponding root
  //  domain ordering by iterating on the root domain and
  //  find their corresponding inner tile iterdomains from
  //  the populated root_id_to_inner_loop_pos.
  for (auto id : tv->getMaybeAllocationDomain()) {
    auto loop_id_pos_it = id_to_inner_loop_pos.find(id);
    if (loop_id_pos_it != id_to_inner_loop_pos.end()) {
      reorder_map_old_to_new[loop_id_pos_it->second] = current_pos++;
    }
  }

  // Validate that we have processed all inner ids or broadcast/reduction
  //  ids we have registered.
  NVF_ERROR(current_pos == ndims, "Inconsistent ordering logic");

  // Apply the new order:
  tv->reorder(reorder_map_old_to_new);
}

namespace {

// Utility for mma dimension matching
enum class MmaDimension { M = 0, N, K };

// Preliminary checks to try to validate that loop is
//  a innermost dim of root of exactly the given size.
bool canValidateIsInnerDim(
    IterDomain* root,
    IterDomain* loop,
    int inner_dim_size) {
  auto expr = loop->definition();
  if (!loop->extent()->isConstInt()) {
    return false;
  }
  if (loop->extent()->evaluate() != inner_dim_size) {
    return false;
  }

  while (expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      // Inner split only
      if (loop != split->inner()) {
        return false;
      }
      // Const split only
      if (!split->factor()->isConstInt()) {
        return false;
      }
      loop = split->in();
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      // Might consider just rejecting merge.
      auto outer = merge->outer();
      if (outer->isBroadcast()) {
        return false;
      }

      // Only support merging with constant sized dims
      if (!loop->extent()->isConstInt()) {
        return false;
      }
      loop = merge->inner();
    } else {
      // No support for swizzled inner dim for now.
      //  Might need to add transpose swizzle here.
      return false;
    }
    expr = loop->definition();
  }
  return loop == root;
}

} // namespace

void checkDimSize(
    TensorView* tv,
    std::vector<int64_t> axis,
    std::vector<int64_t> expect) {
  NVF_ERROR(
      axis.size() == expect.size(),
      "CheckDimSize: Mismatched axis and expect size");
  for (auto axis_index : arange(axis.size())) {
    NVF_ERROR(
        ((axis[axis_index] + tv->nDims()) >= 0) &&
            (axis[axis_index] < tv->nDims()),
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

namespace {

// Utility function for mma domain mapping:
//  returns the Iterdomain from the accumulator tv that corresponds
//  to the given mma dimension. See [MMA dimension matching].
std::vector<IterDomain*> getMmaDomains(MmaOp* mma, MmaDimension dimension) {
  // This utility is user facing so shouldn't ever see tensor index here.

  // Note: [Use Root Domain in Accumulator TV]
  //  Have to use root domain for accumulator tv since the operands do not have
  //  root/logical domains that map to the logical domain of output.
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
  auto accumulator_domain = mma->out()->as<TensorView>()->getMaybeRootDomain();
  auto a_domain = TensorDomain::noReductions(
      mma->inA()->as<TensorView>()->getLogicalDomain());
  auto b_domain = TensorDomain::noReductions(
      mma->inB()->as<TensorView>()->getLogicalDomain());
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

  for (auto id_idx : arange(a_domain.size())) {
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
        NVF_THROW("unreachable");
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

// Function to travel up the DAG along the innermost path from the inner-most ID
// of allocation domain. Assumption: There are only splits and merges from the
// root to the allocation. We only handle going up splits when the producer is
// the inner output of a split. If this was not a case, then given merges,
// splits and reorders, we could have a case that we start from a ID derived
// from K, but while going back up the DAG we end up with a non-K ID. Eg: (K, M)
// -> Merge -> () -> Split -> (K, M) -> Reorder -> (M , K) -> Split -> M, K_o,
// K_in. If we start with K_in we can end up with M.
IterDomain* getIDinConsumerRoot(IterDomain* id, TensorView* tv) {
  while (!tv->domain()->isMaybeRoot(id)) {
    Expr* expr = id->definition();
    NVF_ERROR(expr != nullptr);
    NVF_CHECK(expr->isA<Merge>() || expr->isA<Split>());
    if (expr->isA<Split>()) {
      NVF_CHECK(
          id == expr->as<Split>()->inner(),
          "We only handle cases where the inner-most ID"
          "of the allocation domain was the inner output of a split");
      id = expr->as<Split>()->in();
    } else {
      id = expr->as<Merge>()->inner();
    }
  }
  return id;
}

} // namespace

// The assumption made in this function is that we have set the allocation in
// the register (acr/bb). The inner-most ID in the allocation domain of the
// consumer (register) is derived from a series of scheduling operations on the
// 'k' ID (for now only splits, but there could be merges in the future).
// So starting from the inner-most ID of the consumer's allocation we go up the
// DAG (along the innermost path) to ID this came from in the root domain.  We
// then map this ID in the root domain to producer's (shared memory) logical
// domain. Once we have the ID in the producer's logical domain, we check if
// that's the innermost dimension in its allocation domain. Here, the other
// assumption we have is that the producer's allocation domain is a permutation
// of the logical domain. If the ID is the innermost of the allocation no
// transpose is needed.
bool isLdMatrixTranspose(const LoadStoreOp* ldst) {
  const auto consumer = ir_utils::getTvOutput(ldst);
  const auto producer = ir_utils::getTvInput(ldst);

  // TODO Fix analysis for this case.
  // LdMatrix Transpose is not supported for LoadStoreOp expression with
  // a TMA load definition and without any MmaOp use.
  if (producer->definition() != nullptr &&
      ir_utils::isCpAsyncBulkLoad(producer->definition())) {
    NVF_ERROR(std::all_of(
        consumer->uses().begin(), consumer->uses().end(), [](Expr* e) {
          return !e->isA<MmaOp>();
        }));
    return false;
  }

  // Get the innermost ID and go back up the DAG to the root domain.
  auto corresponding_id_in_consumer_root = getIDinConsumerRoot(
      consumer->getMaybeAllocationDomain().back(), consumer);

  // This gives us the ID in the consumer root domain.
  // We'll later map this ID to one in the producer.
  const PairwiseLogicalDomainMap map_across_ldst(producer, consumer);
  const auto c2p_map = map_across_ldst.mapConsumerToProducer();
  const auto id_in_proc_rfactor = c2p_map.at(corresponding_id_in_consumer_root);

  // If the innermost ID of the (maybe)Allocation domain
  // is not the same as the mapped ID in the producer, then
  // we need to transpose.
  return producer->getMaybeAllocationDomain().back() != id_in_proc_rfactor;
}

void MmaSwizzler::scheduleLdMatrix(TensorView* tv, MmaOperand operand) {
  NVF_CHECK(tv->definition()->isA<LoadStoreOp>());
  bool transpose = isLdMatrixTranspose(tv->definition()->as<LoadStoreOp>());
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
}

void MmaSwizzler::parallelizeAsBulkSkippingFirstIDs(
    TensorView* tv,
    int64_t first_ids_to_skip) {
  auto skip = 0;
  for (auto id : tv->getLoopDomain()) {
    if (skip < first_ids_to_skip) {
      skip++;
      continue;
    }
    id->parallelize(ParallelType::Bulk);
  }
}

void MmaSwizzler::scheduleTMALoadForMma(
    TensorView* tv,
    MmaInputSmemSwizzle swizzle) {
  // In the comments below I have kept K as the outer dimension. That is
  // just to have a concrete running example - it can be inner or outer.

  int64_t num_ids_to_skip =
      static_cast<int64_t>(tv->getLoopDomain().size() - 2);
  NVF_ERROR(num_ids_to_skip >= 0);
  if (swizzle == MmaInputSmemSwizzle::None) {
    // For no-swizzle case, the entire tile are divided into 8x8 core matrices,
    // and each core matrix resides in a contiguous 8*8*2 bytes region in shared
    // memory. [K, N]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, No, N8]
    tv->reorder({{-2, -3}});
    // [Ko, No, K8, N8]
    num_ids_to_skip += 2;
  } else {
    auto dtype = tv->getDataType().value();

    // In the comments below I assume K=16, N=32, swizzle=32, dtype = half.

    // split the inner-dim
    // [K(16), N(32)] -> [K(16), NO(2), NI(16)]
    tv->split(-1, getBytesFromSwizzle(swizzle) / dataTypeSize(dtype));

    // [NO, K, NI] - the TMA Box is [K, NI]
    tv->reorder({{-2, -3}});

    // [NO, K, NI] ->
    // [NO, KO(2), KIO(2), KII(4), NIO(2), NII(8)]
    tv->swizzleTMABox(swizzle);
    num_ids_to_skip += 1;
  }

  parallelizeAsBulkSkippingFirstIDs(tv, num_ids_to_skip);

  // Set the allocation to the loop domain.
  tv->setAllocationDomain(tv->getLoopDomain(), true);
}

void MmaSwizzler::scheduleOperandRead(TensorView* tv, MmaOperand operand) {
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
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }
}

// Reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-swizzling-modes
void MmaSwizzler::scheduleOperandRead(
    TensorView* tv,
    MmaInputSmemSwizzle swizzle) {
  if (swizzle == MmaInputSmemSwizzle::None) {
    // For no-swizzle case, the entire tile are divided into 8x8 core matrices,
    // and each core matrix resides in a contiguous 8*8*2 bytes region in shared
    // memory. [K, M]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, Mo, M8]
    tv->reorder({{-2, -3}});
    // [Ko, Mo, K8, M8]
  } else {
    auto swizzle_size = getBytesFromSwizzle(swizzle) / 16;
    // For example, [K, M]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, Mo, M8]
    // Note: the extent of Mo may not be a multiple of swizzle_size, but we
    // still split swizzle_size. If this is the case, effectively we are
    // padding it to a multiple of swizzle_size.
    tv->split(-2, swizzle_size);
    // For example, swizzle_size = 2
    // [Ko, K8, Moo, Mo2, M8]
    tv->split(-4, 8 / swizzle_size);
    // [Ko, K2, K4, Moo, Mo2, M8]
    tv->swizzle(SwizzleType::XOR, -5, -2);
    tv->reorder({{-3, -5}});
    // [Ko, Moo, K2, K4, Mo2, M8]
  }
  tv->setAllocationDomain(tv->getLoopDomain(), true);
}

AbstractTensor MmaSwizzler::scheduleMmaOutputAllocation(AbstractTensor t) {
  // This function works for all mma ops, regardless of the architecture. The
  // Hopper one is the most general one. For earlier architectures, we will have
  // some dimensions with size 1 after split, this is fine.
  // Memory format for hopper mma:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d

  // Assume last 2 dims, for example [M64, N24] or [M64, N24, R]
  NVF_ERROR(t.size() >= 2);
  bool has_reduction = t[-1]->isReduction();

  int64_t m_pos = has_reduction ? -3 : -2;
  int64_t n_pos = has_reduction ? -2 : -1;

  //   m    n
  // [M64, N24  (,R)]
  t.split(m_pos--, 8);
  t.split(m_pos--, 2);
  //   m           n
  // [M4, M2, M8, N24  (,R)]
  t.split(n_pos, 8);
  t.split(n_pos, 2);

  n_pos -= 2;
  m_pos -= 2;
  //  m           n
  // [M4, M2, M8, N3, N4, N2  (,R)]

  t.reorder({{m_pos + 1, n_pos + 1}, {n_pos + 1, m_pos + 2}});
  //  m           n
  // [M4, M8, N4, N3, M2, N2  (,R)]
  t.merge(m_pos++);
  t.merge(m_pos++);

  //       m
  // [WarpGroup128, N3, M2, N2  (,R)]

  if (has_reduction) {
    t.split(-1, 2);
    t.split(-2, 4);
    m_pos -= 2;
    //       m
    // [WarpGroup128, N3, M2, N2, Ro, R4, R2]
  }

  t.parallelize(m_pos, ParallelType::TIDx);

  if (has_reduction) {
    // Set instruction loops for mma reduce
    int64_t pos = -1;
    while (pos > m_pos) {
      t.parallelize(pos--, ParallelType::Mma);
    }
  }
  return t;
}

// TODO: Remove this in favor of mergeConsecutiveAxesWithSameRole once
// multi-matmul refactor is finished.
std::vector<MatmulDimRole> canonicalizeMmaTvOrdering(
    TensorView* tv,
    const ValGraph& graph,
    const DimRolesMap& dim_roles,
    const std::vector<ValGroup>& ordering) {
  std::unordered_map<int64_t, int64_t> old2new;

  for (size_t i : arange(tv->nDims())) {
    IterDomain* id = tv->axis((int64_t)i);
    const ValGroup& g = graph.toGroup(id);
    auto order_it = std::find(ordering.begin(), ordering.end(), g);
    NVF_ERROR(
        order_it != ordering.end(),
        "Couldn't find ordering entry for ",
        id->toString());
    size_t pos = std::distance(ordering.begin(), order_it);
    old2new[(int64_t)i] = (int64_t)pos;
  }

  tv->reorder(old2new);

  // Now merge dims that have the same role
  NVF_ERROR(tv->nDims() > 0);
  const auto getRole = [&dim_roles, &graph](IterDomain* id) {
    const ValGroup& g = graph.toGroup(id);
    const auto it = dim_roles.find(g);
    NVF_ERROR(it != dim_roles.end());
    return it->second;
  };
  // Loop from inner to outer, merging when needed
  MatmulDimRole prev_role = getRole(tv->axis(-1));
  std::vector<MatmulDimRole> roles{prev_role};
  for (int64_t dim = tv->nDims() - 2; dim >= 0; --dim) {
    MatmulDimRole role = getRole(tv->axis(dim));
    if (role == prev_role) {
      tv->merge(dim);
    } else {
      roles.push_back(role);
    }
    prev_role = role;
  }
  // Roles are inserted in reverse order
  std::reverse(roles.begin(), roles.end());
  return roles;
}

void mergeConsecutiveAxesWithSameRole(
    TensorView* tv,
    const DimRolesMap& dim_roles,
    const ValGraph* graph) {
  const auto getRole = [&](const int64_t pos) {
    const ValGroup& vg = graph->toGroup(tv->axis(pos));
    auto it = dim_roles.find(vg);
    NVF_ERROR(it != dim_roles.end());
    return it->second;
  };
  // Loop from inner to outer, merging when needed
  NVF_ERROR(tv->nDims() > 0);
  MatmulDimRole prev_role = getRole(-1);
  for (int64_t dim = (int64_t)tv->nDims() - 2; dim >= 0; --dim) {
    MatmulDimRole role = getRole(dim);
    if (role == prev_role) {
      tv->merge(dim);
    }
    prev_role = role;
  }
}

namespace {
inline void resolveTvToMatmulDimRolesMapping(
    DependenciesMap& deps_map,
    const std::vector<TensorView*>& tensors,
    IterDomain* m,
    IterDomain* n,
    IterDomain* k,
    const ComputeAtMap& ca_map) {
  for (const auto tv : tensors) {
    // This ensures all inputs are added to the deps_map.
    // There could be inputs such as a zero-dimensional bias which
    // would otherwise be skipped.
    deps_map[tv] = {};
    for (const auto domain : tv->getLoopDomain()) {
      if (ca_map.areMapped(m, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDimRole::M);
        continue;
      }
      if (ca_map.areMapped(n, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDimRole::N);
        continue;
      }
      if (ca_map.areMapped(k, domain, IdMappingMode::EXACT)) {
        deps_map[tv].push_back(MatmulDimRole::K);
        continue;
      }
    }
  }
}

} // anonymous namespace

void scheduleTMAStoreForMmaOutput(TensorView* tv, MmaInputSmemSwizzle swizzle) {
  // [BDX, BDY, TDY, MI, NI]
  // skip all but last 2 iterDomains
  int64_t num_ids_to_skip =
      static_cast<int64_t>(tv->getLoopDomain().size() - 2);

  NVF_ERROR(num_ids_to_skip >= 0);
  if (swizzle == MmaInputSmemSwizzle::None) {
    // For no-swizzle case, the entire tile are divided into 8x8 core matrices,
    // and each core matrix resides in a contiguous 8*8*2 bytes region in shared
    // memory. [K, N]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, No, N8]
    tv->reorder({{-2, -3}});
    // [Ko, No, K8, N8]
    num_ids_to_skip += 2;
  } else {
    auto dtype = tv->getDataType().value();

    // In the comments below I assume K=16, N=32, swizzle=32, dtype = half.

    // split the inner-dim
    // [K(16), N(32)] -> [K(16), NO(2), NI(16)]
    tv->split(-1, getBytesFromSwizzle(swizzle) / dataTypeSize(dtype));

    // [NO, K, NI] - the TMA Box is [K, NI]
    tv->reorder({{-2, -3}});

    // [NO, K, NI] ->
    // [NO, KO(2), KIO(2), KII(4), NIO(2), NII(8)]
    tv->swizzleTMABox(swizzle);
    num_ids_to_skip += 1;
  }

  // The shared memory producer must have the swizzled allocation domain.
  // The global memory consumer must have the ParallelType::Bulk iterDomains.
  if (tv->getMemoryType() == MemoryType::Shared) {
    // Set the allocation to the loop domain.
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  } else {
    mma_utils::MmaSwizzler::parallelizeAsBulkSkippingFirstIDs(
        tv, num_ids_to_skip);
  }
}

void scheduleLdStMatrixForMmaOutput(
    TensorView* tv,
    int64_t tile_m,
    int64_t tile_n) {
  NVF_ERROR(
      ((tile_m == 16 && tile_n == 16) || (tile_m == 16 && tile_n == 8)),
      "We only support 16x16 and 16x16 stmatrix now");

  NVF_CHECK(
      dataTypeSize(tv->dtype()) == 2,
      "we only support 16-bit types in stmatrix");

  // NOTE: There can be iterDomains to left of the mma output if there is cta
  // or warp tiling.
  if (tile_m == 16 && tile_n == 16) {
    // Let [M, N] be [64, 32]
    // After scheduleMmaOutputAllocation: [128(TIDx), 4, 2, 2]
    // [128(TIDx), 4(n), 2, 2] ->  [128(TIDx), 2(no), 2(ni), 2, 2]
    tv->split(-3, 2);
    // [128(TIDx), 2(no), 2(ni), 2, 2] -> [2(no), 128(TIDx), 2(ni), 2, 2]
    tv->reorder({{-4, -5}});
    // [128(TIDx), 2(no), 2(ni), 2, 2] -> [2(no), 128(TIDx), 8 (vectorize)]
    tv->merge(-3);
    tv->merge(-2);
  } else if (tile_m == 16 && tile_n == 8) {
    // Let [M, N] be [64, 16]
    // After scheduleMmaOutputAllocation: [128(TIDx), 2, 2, 2]
    // [128(TIDx), 2, 2, 2] -> [2, 128(TIDx), 2, 2]
    tv->reorder({{-3, -4}});
    // [2, 128(TIDx), 2, 2] -> [2, 128(TIDx), 4(vectorize)]
    tv->merge(-2);
  }
}

// Apply to the TensorView after block tiling and parallelization, but before
// applying mma allocation to loop domain.
AbstractTensor scheduleLdStMatrixSharedMemory(
    TensorView* tv,
    int64_t smem_tile_n,
    int64_t ldst_tile_n) {
  // Assume the input TensorView is block tiled. e.g., The last two iterDomains
  // are the warp tile except for k dimension.
  // The CTA tile is (128, 256).
  // The Warp tile is (64, 256).
  // The TMA box is (64, 64).
  // The LdStMatrix.x4 tile is (16, 16).
  // The core matrix for wgmma and LdStMatrix is (8, 8).

  AbstractTensor abstract_tensor(tv->getLoopDomain());
  // (GM, GN, cta_m * cta_n, warp_m, warp_n)

  // Split by TMA shared memory box
  abstract_tensor.split(-1, smem_tile_n);
  abstract_tensor.reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), no(4), m(64), ni(64))

  // Split by (16, 16) matrix for LdStMatrix.x4
  abstract_tensor.split(-2, 16);
  abstract_tensor.split(-1, ldst_tile_n);
  abstract_tensor.reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), no(4), mo(4), nio(4), mi(16), nii(16))
  // Omit (GM, GN, cta_m(2), cta_n(1)) after this for brevity.

  // For shared memory addressing, each thread specifies a row for each (8, 8)
  // matrix. e.g., For stmatrix.x4, 32 threads move a (16, 16) matrix.

  // Inside the tile box [16, 16], we can think of it as 4 8x8 tiles:
  // *****************
  // *       *       *
  // *       *       *
  // *  T0   *  T2   *
  // *       *       *
  // *       *       *
  // *****************
  // *       *       *
  // *       *       *
  // *  T1   *  T3   *
  // *       *       *
  // *       *       *
  // *****************

  // Split inner-dimension by 8 to traverse the rows of the (8, 8) matrices.
  abstract_tensor.split(-1, 8);
  // (no(4), mo(4), nio(4), mi(16), niio(2), niii(8))

  // The tile is stored in row-major order, so issue four stmatrix.x4
  // operations along the M dimension for a 128 thread warp group.
  // Also, traverse along 16 rows first before moving along column dimension.
  abstract_tensor.reorder({{-5, -4}, {-4, -5}, {-3, -2}, {-2, -3}});
  // (no(4), nio(4), mo(4), niio(2), mi(16), niii(8))

  if (ldst_tile_n == 8) {
    // When MmaInputSmemSwizzle::None, the fusion uses ldstmatrix.x2.
    // mo * niio * mi = 64, so split inner-most dimension by 4 and move the
    // remainder to TIDx axis.

    // (no(32), nio(1), mo(4), niio(1), mi(16), niii(8))
    abstract_tensor.split(-1, 4);
    // (no(32), nio(1), mo(4), niio(1), mi(16), niiio(2), niiii(4))
    abstract_tensor.reorder({{-3, -2}, {-2, -3}});
    // (no(32), nio(1), mo(4), niio(1), niiio(2), mi(16), niiii(4))
    abstract_tensor.merge(-5, -4);
    abstract_tensor.merge(-4, -3);
    abstract_tensor.merge(-3, -2);
    // (no(32), nio(1), (mo * niio * niiio * mi)(128), niiii(4))
  } else {
    abstract_tensor.merge(-4, -3);
    abstract_tensor.merge(-3, -2);
    // (no(4), nio(4), (mo * niio * mi)(128), niii(8))
  }

  // Merge no and nio to create a single serial IterDomain
  // This ^^^ is an artifact of matmul scheduling functions.
  abstract_tensor.merge(-4, -3);
  // (no * nio)(16), (niio * mo * mi)(128), niii(8))

  return abstract_tensor;
}

MatmulOperandInnerDimsOpt getOperandInnerDims(Fusion* fusion) {
  const std::vector<MatmulPattern> patterns = findMatmulPatterns(fusion);
  if (patterns.size() != 1) {
    if (!isOptionEnabled(EnableOption::FuseMultipleMatmuls)) {
      std::stringstream ss;
      ss << "Invalid number of MmaOp instances in fusion, expected 1, got "
         << patterns.size();
      return ss.str();
    }
    TORCH_WARN("TODO: Update getOperandInnerDims for multiple patterns");
  }
  const MatmulPattern& pattern = patterns[0];
  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto id_roles = pattern.getDimRoles(id_model);
  const auto tensor_roles_opt = getTensorRoles(fusion, id_model, id_roles);
  if (!tensor_roles_opt.isValid()) {
    return {tensor_roles_opt.getErrorMsg()};
  }
  return getOperandInnerDims(id_model, id_roles, tensor_roles_opt.getData());
}

MatmulOperandInnerDimsOpt getOperandInnerDims(
    const IdModel& id_model,
    const DimRolesMap& dim_roles,
    const TensorRolesMap& tensor_roles) {
  // Assumes the Broadcast graph has already been built, since we've been
  // provided dim_roles
  const ValGraph& graph = id_model.idGraph(IdMappingMode::BROADCAST);

  // Note: using DataWrapperOpt<MatmulDimRole> would be preferable here.
  // However, using DataWrapperOpt<MatmulDimRole>(std::move(dom)) leads to a
  // clang-tidy warning because MatmulDimRole is trivially movable. There is
  // only a move constructor for DataWrapperOpt to prevent inadvertent copying.
  // To avoid this complication I'm using an unwrapped variant for the lambda's
  // result type.
  using MatmulDimRoleOpt = std::variant<std::string, MatmulDimRole>;
  const auto findInnerDim = [&dim_roles,
                             &graph](TensorView* tv) -> MatmulDimRoleOpt {
    IterDomain* inner_id =
        TensorDomain::noReductions(tv->getMaybeAllocationDomain()).back();
    const ValGroup& g = graph.toGroup(inner_id);
    auto g_it = dim_roles.find(g);
    if (g_it == dim_roles.end()) {
      return "Inner domain of tensor was not mapped to a MatmulDimRole";
    }
    return g_it->second;
  };
  TensorView* a = getOperandTv(tensor_roles, MatmulTensorRole::OPERAND_A);
  TensorView* b = getOperandTv(tensor_roles, MatmulTensorRole::OPERAND_B);

  const MatmulDimRoleOpt innerdim_a_opt = findInnerDim(a);
  if (std::holds_alternative<std::string>(innerdim_a_opt)) {
    std::string err = std::get<std::string>(innerdim_a_opt);
    return err;
  }
  const MatmulDimRoleOpt innerdim_b_opt = findInnerDim(b);
  if (std::holds_alternative<std::string>(innerdim_b_opt)) {
    std::string err = std::get<std::string>(innerdim_b_opt);
    return err;
  }
  const MatmulDimRole innerdim_a = std::get<MatmulDimRole>(innerdim_a_opt);
  const MatmulDimRole innerdim_b = std::get<MatmulDimRole>(innerdim_b_opt);

  return std::vector<MatmulDimRole>{innerdim_a, innerdim_b};
}

TensorRolesMapOpt getTensorRoles(
    Fusion* fusion,
    const IdModel& id_model,
    const DimRolesMap& dim_roles) {
  NVF_ERROR(fusion != nullptr);
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

  TensorRolesMap tensor_roles;

  // Assumes the Broadcast graph has already been built, since we've been
  // provided dim_roles
  const ValGraph& graph = id_model.idGraph(IdMappingMode::BROADCAST);

  struct DimPresence {
    bool m = false;
    bool n = false;
    bool k = false;
    bool unmapped = false;
  };

  const auto findDims = [&dim_roles, &graph](TensorView* tv) {
    DimPresence has;
    for (IterDomain* id : TensorDomain::noReductions(tv->getLogicalDomain())) {
      if (id->isBroadcast() || id->isDeviceDim()) {
        continue;
      }
      const ValGroup& g = graph.toGroup(id);
      auto it = dim_roles.find(g);
      if (it == dim_roles.end()) {
        // tv has an unmapped non-broadcast and non-reduction dimension
        has.unmapped = true;
        continue;
      }
      has.m = has.m || it->second == MatmulDimRole::M;
      has.n = has.n || it->second == MatmulDimRole::N;
      has.k = has.k || it->second == MatmulDimRole::K;
    }
    return has;
  };

  for (TensorView* tv : mma_input_candidates) {
    DimPresence has = findDims(tv);
    if (has.unmapped) {
      // Don't map TVs to roles if they have unmapped dims
      continue;
    }
    if (has.k) {
      tensor_roles
          [has.m ? MatmulTensorRole::OPERAND_A : MatmulTensorRole::OPERAND_B]
              .push_back(tv);
    } else {
      tensor_roles[MatmulTensorRole::EPILOGUE_INPUT].push_back(tv);
      continue;
    }
  }

  std::vector<TensorView*> storage;
  for (TensorView* tv : mma_output_candidates) {
    DimPresence has = findDims(tv);
    // NOTE: depending on fusion definition k domain may appear in the output:
    //  - for mma_output == fusion output k domain is present
    //  - for mma_output != fusion output (fusion with epilogue) k domain
    //    is not present
    if (has.k || has.unmapped) {
      // Don't map TVs to output roles if they have unmapped dims, or if they
      // have K dimension
      continue;
    }

    // NOTE: the core fusion output tensors are the ones with m and n
    //  domains
    if (has.m && has.n) {
      storage.push_back(tv);
    }
  }

  if (!storage.empty()) {
    tensor_roles[MatmulTensorRole::OUTPUT] = storage;
  }

  for (auto& [role, tvs] : tensor_roles) {
    // NOTE: sort role tvs in descending order by uses() size, and
    //  if equal then by name() to ensure the stable ordering of tensor
    //  views in collections assigned to the supported roles
    std::sort(tvs.begin(), tvs.end(), [](TensorView* a, TensorView* b) {
      return (a->uses().size() == b->uses().size())
          ? (a->name() < b->name())
          : (a->uses().size() > b->uses().size());
    });
  }

  return tensor_roles;
}

namespace {
// Check the val (in) is the output of broadcast.
// Then check the output of the broadcast is 3D (4D for bmm).
bool hasValidBroadcastOp(TensorView* bcast_out) {
  // First check the tensorsview is 3D (4D)
  // and has one broadcast dim.
  // Ignore device dimensions in this analysis.
  auto non_device_dims =
      TensorDomain::noDevices(bcast_out->getLoopDomain()).size();
  if (!((non_device_dims == 3 || non_device_dims == 4) &&
        TensorDomain::noDevices(bcast_out->domain()->noBroadcasts()).size() ==
            non_device_dims - 1)) {
    return false;
  }

  // Check if the definition is a broadcast op.
  if (dynamic_cast<BroadcastOp*>(bcast_out->definition())) {
    return true;
  }

  return false;
}

int64_t numBroadcastDeviceDims(TensorView* tv) {
  return std::count_if(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      [](IterDomain* id) { return id->isDeviceDim() && id->isBroadcast(); });
}

// This function checks if the mul-sum can be replace with a mma op. The
// checks are:
// 1. The inputs to the muls are broadcast ops.
// 2. The broadcasts have 2D or 3D(bmm) inputs.
// 3. The broadcasts only broadcast one dim and the dims are different for the
// 2 muls.
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

  // Ensure that only one non-device dim is getting broadcast.
  auto bcastFlags_l = bcast_l->getBroadcastDimFlags();
  auto bcastFlags_r = bcast_r->getBroadcastDimFlags();
  auto bcast_l_devices = numBroadcastDeviceDims(left);
  auto bcast_r_devices = numBroadcastDeviceDims(right);
  auto count_l = std::count(bcastFlags_l.begin(), bcastFlags_l.end(), true) -
      bcast_l_devices;
  auto count_r = std::count(bcastFlags_r.begin(), bcastFlags_r.end(), true) -
      bcast_r_devices;
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

} // namespace

char dtypeToChar(const DataType& dtype) {
  if (dtype == DataType::Half) {
    return 'H';
  } else if (dtype == DataType::BFloat16) {
    return 'T';
  } else if (dtype == DataType::Float) {
    return 'S';
  } else if (dtype == DataType::Double) {
    return 'D';
  }
  NVF_THROW("Unsupported dtype for matmul: ", dtype);
  return 0;
}

namespace {
class MatmulPatternMatcher : IterVisitor {
 public:
  static std::vector<MatmulPattern> run(Fusion* fusion) {
    MatmulPatternMatcher matcher;
    matcher.traverse(fusion);
    return matcher.patterns_;
  }

 private:
  using IterVisitor::handle;

  // TODO: These methods currently assume the output will have allocation
  // domain equal to its logical. However, if the logical domain is specified,
  // or if there is a transpose operation in the epilogue, then this
  // assumption will be violated. In such cases we should actually swap and
  // transpose A and B.

  // Match all LinearOps and MatmulOps as MatmulPatterns. This includes ops
  // whose inputs are not 2D, i.e. matrix-vector products. The matmul
  // scheduler will decide whether or not it can fuse a given pattern based on
  // the dimensionality of its inputs.
  void handle(LinearOp* lop) override {
    MatmulPattern& pattern = patterns_.emplace_back();
    pattern.A = lop->inA()->as<TensorView>();
    pattern.B = lop->inB()->as<TensorView>();
    pattern.output = lop->out()->as<TensorView>();
  }

  void handle(MatmulOp* mop) override {
    MatmulPattern& pattern = patterns_.emplace_back();
    pattern.A = mop->inA()->as<TensorView>();
    pattern.B = mop->inB()->as<TensorView>();
    pattern.output = mop->out()->as<TensorView>();
  }

  // Handle the case when no translation is needed.
  void handle(MmaOp* mop) override {
    MatmulPattern& pattern = patterns_.emplace_back();
    pattern.A = mop->inA()->as<TensorView>();
    pattern.B = mop->inB()->as<TensorView>();
    pattern.output = mop->out()->as<TensorView>();
  }

  void handle(ReductionOp* rop) override {
    // Check if operation is a sum.
    if (rop->getReductionOpType() != BinaryOpType::Add) {
      return;
    }
    // Then check if the producer of the sum is a mul.
    if (auto bop = dynamic_cast<BinaryOp*>(rop->in()->definition())) {
      if (bop->getBinaryOpType() != BinaryOpType::Mul) {
        return;
      }
      // Remember that we are just gathering the immediate inputs to the
      // matmul, so there should be no prologue between a, b and the mul/sum.

      // Check that the inputs have broadcasts that are not all in common,
      // i.e. that there is at least one M and at least one N dimension.

      // Note that there might be a cast to Float just before the multiply.
      // This happens when using the `mul` op with reduced precision inputs.
      // It can also happen if the inputs to `mul` in the definition were
      // Float, but the Fusion was segmented and casts to half precision were
      // inserted at the segmentation edge (see
      // castInputOutputToLowerPrecision in fusion_segmenter.cpp).
      TensorView* ltv = dynamic_cast<TensorView*>(bop->lhs());
      TensorView* rtv = dynamic_cast<TensorView*>(bop->rhs());
      if (ltv == nullptr || rtv == nullptr) {
        // Found a scalar input
        return;
      }
      ltv = getTensorviewPriorToCast(ltv);
      rtv = getTensorviewPriorToCast(rtv);

      std::vector<IterDomain*> lrf = TensorDomain::noDevices(
          TensorDomain::noReductions(ltv->getLogicalDomain()));
      std::vector<IterDomain*> rrf = TensorDomain::noDevices(
          TensorDomain::noReductions(rtv->getLogicalDomain()));

      // These sizes should match since ops::maybeBroadcast places
      // BroadcastOps for implicit broadcasting.
      NVF_ERROR(lrf.size() == rrf.size());
      const std::vector<IterDomain*>& red_root = TensorDomain::noDevices(
          rop->out()->as<TensorView>()->getMaybeRootDomain());
      NVF_ERROR(red_root.size() == lrf.size());
      // Find innermost M or N dimension in output
      // We will assume for now that the output logical domain matches the
      // fusion output's allocation domain; in particular that the innermost
      // dimension is an N dimension. This allows us to determine which of lhs
      // and rhs is A and B.
      // TODO: analyze fusion outputs to determine N dimensions
      bool lhs_is_A = true;
      bool has_m = false, has_n = false;
      // Loop backwards to find inner-most Iteration domain in output
      for (int64_t i = (int64_t)red_root.size() - 1; i >= 0; --i) {
        IterDomain* lhs_id = lrf[(size_t)i];
        IterDomain* rhs_id = rrf[(size_t)i];
        IterDomain* out_id = red_root[(size_t)i];
        if (out_id->isIteration()) {
          if (lhs_id->isBroadcast() != rhs_id->isBroadcast()) {
            // This is either an M or N dimension

            // Operand domains must be Broadcast and Iteration
            NVF_ERROR(lhs_id->isIteration() || rhs_id->isIteration());

            if (!has_n) {
              // This is the inner-most output non-batch dim, so it is N
              has_n = true;
              // rhs is B if it has this dimension
              lhs_is_A = rhs_id->isIteration();
              continue;
            }
            // We have found the inner-most N dim, so we can now use lhs_is_A
            // to tell whether this is M or N
            has_m = has_m || (lhs_is_A && lhs_id->isIteration()) ||
                (!lhs_is_A && (rhs_id->isIteration()));
          }
          // out_id could also be a batch dim
        } else if (out_id->isReduction()) {
          // matmul must be contraction of non-broadcast dimensions
          if (!lhs_id->isIteration() || !rhs_id->isIteration()) {
            return;
          }
        } else if (!out_id->isBroadcast()) {
          // Reduction output ID should be iteration, reduction, or broadcast
          return;
        }
      }
      if (!has_m || !has_n) {
        // This is an ordinary reduction or mat-vec, not a matmul
        return;
      }

      MatmulPattern& pattern = patterns_.emplace_back();
      pattern.A = lhs_is_A ? ltv : rtv;
      pattern.B = lhs_is_A ? rtv : ltv;
      pattern.output = rop->out()->as<TensorView>();
    }
  }

 private:
  std::vector<MatmulPattern> patterns_;
};

} // namespace

std::vector<MatmulPattern> findMatmulPatterns(Fusion* fusion) {
  return MatmulPatternMatcher::run(fusion);
}

std::string MatmulPattern::toString() const {
  std::stringstream ss;
  ss << "MatmulPattern{";
  ss << "\n  A=" << A->toString();
  ss << "\n  B=" << B->toString();
  ss << "\n  output=" << output->toString() << "\n}";
  return ss.str();
}

namespace {

// The `MatmulTranslator` helper class is used to map different matrix
// multiplication patterns to `MmaOp`. The `MmaOp` expression maps to the
//  TensorCore ptx function.
//
//    1. `MmaOp` --This expression is what we need, so no changes required.
// 2. `ReductionOp` -- This expression corresponds with the sum operation in
// the `broadcast->multiply->sum` pattern.
// 3. `LinearOp` -- This expression is `y = w @ x + beta`, so it is replaced
// with `MmaOp` and pointwise `add`.
// 4. `MatmulOp` -- This expression is `y = A[M, K] @ B[K, N]`. The `MmaOp`
// expression requires `[M, N, K]` ordering, so it requires transposing the
// `B` operand. It also support batch matrix multiplication.
//
// `finalizeMatmulOrLinearOp`
//  * Fused-Multiply-Sum (FMS) is the output from MmaOp.
//  * The output dtype can be different than the original output dtype.
//  * This function casts the FMS TensorView to the original output dtype if
//   necessary.
//
// `OptInDispatch` is used to catch any unsupported `MatmulPattern`. It is
// preferred to throw an error than to fallback to a sub-optimal default
// `MmaOp` translation.
class MatmulTranslator : public OptInDispatch {
 public:
  static MatmulPattern::TranslationResult translate(MatmulPattern& pattern) {
    MatmulTranslator trans(pattern);
    trans.dispatch(pattern.output->definition());
    return MatmulPattern::TranslationResult{trans.mma_, trans.replacements_};
  }

 private:
  MatmulTranslator(MatmulPattern& pattern) : pattern_(pattern) {}

  using OptInDispatch::handle;

  void handle(MmaOp* mma) final {
    mma_ = mma;
  }

  void handle(ReductionOp* rop) final {
    Val* init = IrBuilder::create<Val>(0.0, pattern_.output->dtype());
    // This replaces the mul and sum by overwriting output->definition()
    mma_ =
        IrBuilder::create<MmaOp>(pattern_.output, pattern_.A, pattern_.B, init);
  }

  //! Replace a TV, recording it in replacements_
  void replaceTV(TensorView* old_tv, TensorView* new_tv) {
    replacements_[old_tv] = new_tv;
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(old_tv, new_tv);
  }

  //! In a horizontal fusion, we will translate each MatmulPattern in turn. When
  //! we do so, if there are repeated operands, we do not want to repeat the
  //! broadcasting and transposing. This function only replays a broadcast if
  //! there is no pre-existing use that matches. Otherwise it returns the
  //! existing consumer.
  TensorView* getBroadcast(
      TensorView* tv,
      const std::vector<bool>& bcast_flags) {
    for (Expr* use : tv->uses()) {
      if (auto* bop = dynamic_cast<BroadcastOp*>(use);
          bop != nullptr && bop->getBroadcastDimFlags() == bcast_flags) {
        return bop->out()->as<TensorView>();
      }
    }
    return broadcast(tv, bcast_flags);
  }

  TensorView* getBroadcastInnerToRank(TensorView* tv, int64_t rank) {
    size_t tv_rank = TensorDomain::noReductions(tv->getLogicalDomain()).size();

    // broadcast inner on inp to match rank with other.
    if ((int64_t)tv_rank < rank) {
      const int num_bcast = static_cast<int>(rank - tv_rank);
      std::vector<bool> inner_bcast_dims(rank, false);
      std::fill(
          inner_bcast_dims.begin(), inner_bcast_dims.begin() + num_bcast, true);
      return getBroadcast(tv, inner_bcast_dims);
    }
    return tv;
  }

  TensorView* getUnsqueeze(TensorView* tv, int64_t dim) {
    size_t tv_rank = TensorDomain::noReductions(tv->getLogicalDomain()).size();
    std::vector<bool> bcast_dims(tv_rank + 1, false);
    if (dim < 0) {
      dim += (int64_t)tv_rank + 1;
    }
    bcast_dims.at(dim) = true;
    return getBroadcast(tv, bcast_dims);
  }

  TensorView* getTranspose(TensorView* tv, int64_t dim0, int64_t dim1) {
    for (Expr* use : tv->uses()) {
      if (!use->isA<LoadStoreOp>()) {
        continue;
      }
      auto* out_tv = use->output(0)->as<TensorView>();
      std::optional<std::vector<int64_t>> perm_opt =
          ir_utils::computePermutation(
              out_tv->getMaybeRootDomain(), out_tv->getLogicalDomain());
      if (!perm_opt.has_value()) {
        continue;
      }
      std::vector<int64_t> perm = perm_opt.value();
      std::vector<int64_t> target_perm;
      target_perm.reserve(out_tv->getLogicalDomain().size());
      for (size_t i : c10::irange(out_tv->getLogicalDomain().size())) {
        target_perm.push_back((int64_t)i);
      }
      target_perm.at((size_t)dim0) = dim1;
      target_perm.at((size_t)dim1) = dim0;
      if (perm == target_perm) {
        return out_tv;
      }
    }
    return transpose(tv, dim0, dim1);
  }

  void handle(LinearOp* lop) final {
    // This will hold the translated output from MatmulOp or LinearOp
    TensorView* fms = nullptr;
    // Linear takes inputs input, weight(, bias)
    //   - input can be any dimension > 0. We assert that it must be at least
    //   2 and refuse to translate if dimension is 1.
    //   - weight can be one or two dimensional. We refuse to translate if
    //   dimension is 1.
    //   - bias, if present, can be zero or one dimensional. Bias can only be
    //   present if weight is 2D
    //
    // When A has dimension greater than two, all the preceding dimensions
    // are essentially also M dimensions. The output is shaped like
    //
    // A [ ... iS0{M} iS1{K} ]
    // B [ iS2{N} iS3{K} ]
    // out [ ... iS3{M} iS3{N} rS3{K} ]
    //
    // We translate by broadcasting input, weight, and bias such that the
    // contracted dimension K is in the last position (this is true of the
    // logical domains in input and weight already). Then we form an MmaOp and
    // optionally add the bias tensor followed by a cast back to the input
    // dtype.
    int64_t a_dims = (int64_t)pattern_.A->getLogicalDomain().size();
    int64_t b_dims = (int64_t)pattern_.B->getLogicalDomain().size();
    NVF_ERROR(
        a_dims > 1 && b_dims > 1, "Cannot translate LinearOp with 1D input");
    NVF_ERROR(
        b_dims == 2, "Cannot translate LinearOp without 2D weight tensor");
    std::vector<bool> bcast_dim(pattern_.A->nDims() + 1, false);
    bcast_dim[bcast_dim.size() - 2] = true; // N
    pattern_.A = getBroadcast(pattern_.A, bcast_dim);

    bcast_dim[bcast_dim.size() - 2] = false; // reset N
    std::fill(bcast_dim.begin(), bcast_dim.end() - 2, true);
    pattern_.B = getBroadcast(pattern_.B, bcast_dim);

    fms = fusedMultiplySum(pattern_.A, pattern_.B, {-1});
    mma_ = fms->definition()->as<MmaOp>();

    auto* bias = dynamic_cast<TensorView*>(lop->bias());
    if (bias != nullptr) {
      fms = add(fms, bias);
    }
    finalizeMatmulOpOrLinearOp(fms);
  }

  void handle(MatmulOp* mop) final {
    // MatmulOp takes inputs whose sizes are [..., M, K] and [..., K, N], so
    // we must transpose B then broadcast both operands before creating the
    // final op.
    //
    // Also note that the output of MatmulOp is a tensor of shape [..., M, N]
    // whose dtype matches that of the inputs. We will most commonly then also
    // need to cast the output of the MmaOp to produce the output TensorView.
    //
    // There are two possibilities:
    //
    // Case 1: A->nDims() > B->nDims():
    //
    //   A [ ..., B1, ..., Bn, M, K ]
    //   B [ B1, ..., Bn, K, N ]
    //
    //   All the preceding dimensions in A are additional M dimensions. There
    //   are batch dimensions in between those and "M".
    //
    // Case 2: A->nDims() <= B->nDims():
    //
    //   A [ B1, ..., Bn, M, K ]
    //   B [ ..., B1, ..., Bn, K, N ]
    //
    //   All the preceding dimensions in B are additional N dimensions. There
    //   are batch dimensions in between those and "N".
    //
    // In either case, to form the output we transpose B in the last two dims,
    // and prepend broadcasts to the lower dimensional input as needed.
    NVF_ERROR(
        pattern_.A->nDims() > 1 && pattern_.B->nDims() > 1,
        "Cannot translate MatmulOp with 1D input");
    TensorView* fms = nullptr;
    TensorView* Btrans = getTranspose(pattern_.B, -2, -1);
    pattern_.A = getUnsqueeze(pattern_.A, -2);
    pattern_.B = getUnsqueeze(Btrans, -3);
    // A and B might have different dimensions. If so, broadcast the smaller
    // one up to the size of the larger.
    int64_t out_dims = std::max(pattern_.A->nDims(), pattern_.B->nDims());
    // Add new outer broadcast dimensions if necessary
    pattern_.A = getBroadcastInnerToRank(pattern_.A, out_dims);
    pattern_.B = getBroadcastInnerToRank(pattern_.B, out_dims);
    fms = fusedMultiplySum(pattern_.A, pattern_.B, {-1});
    mma_ = fms->definition()->as<MmaOp>();
    finalizeMatmulOpOrLinearOp(fms);
  }

  // The following is common to both MatmulOp and LinearOp translation
  void finalizeMatmulOpOrLinearOp(TensorView* fms) {
    NVF_ERROR(fms != nullptr);

    // TODO: skip downcasting if the only uses of `output` are casts back to
    // higher precision in order avoid the round trip cast in defining an
    // epilogue that starts with MatmulOp.
    if (pattern_.output->dtype() != fms->dtype()) {
      // When fms is a different dtype from output, it means we _might_ need
      // to insert a cast. However, we can skip inserting that cast for any
      // uses of output that are simply casts back to Float.

      // This vector holds tensors that would be round-trip cast to the same
      // dtype as fms. We first collect these Vals then we do the replacements
      // separately in order to avoid dereferencing an Expr* that has already
      // been replaced.
      std::vector<TensorView*> round_trip_tvs;
      for (Expr* use : pattern_.output->uses()) {
        if (auto* uop = dynamic_cast<UnaryOp*>(use); uop != nullptr &&
            uop->getUnaryOpType() == UnaryOpType::Cast &&
            uop->out()->dtype() == fms->dtype()) {
          round_trip_tvs.push_back(uop->out()->as<TensorView>());
        }
      }
      // If there are any uses that were not round-trip casts, then we should
      // insert the castOp.
      //
      // We also always need to insert the cast if the MatmulOp output is a
      // Fusion output.
      if (pattern_.output->isFusionOutput() ||
          pattern_.output->uses().size() > round_trip_tvs.size()) {
        TensorView* old_output = pattern_.output;
        pattern_.output = castOp(pattern_.output->dtype(), fms);
        replaceTV(old_output, pattern_.output);
      }
      // if any casts are skipped, then we reset output to point to the Float
      // output fms instead of the downcast.
      if (!round_trip_tvs.empty()) {
        pattern_.output = fms;
      }
      // Finally, replace the round_trip_tvs with fms
      for (TensorView* tv : round_trip_tvs) {
        replaceTV(tv, fms);
      }
    } else {
      // No cast needed, for example the inputs might be Float
      ir_utils::transferDefinitionToNewOutputs(
          fms->definition(), {pattern_.output});
    }
  }

 private:
  MatmulPattern& pattern_;
  MatmulPattern::TranslationResult result_;
  MmaOp* mma_ = nullptr;
  std::unordered_map<TensorView*, TensorView*> replacements_;
};

} // namespace

MatmulPattern::TranslationResult MatmulPattern::translateToMmaOp() {
  return MatmulTranslator::translate(*this);
}

namespace {
// Determine dim roles for either a MatmulOp or a LinearOp, given IterDomain
// mappings
DimRolesMap matmulOrLinearOpDimRoles(
    const ValGraph& graph,
    const std::vector<IterDomain*>& out_logical,
    const std::vector<IterDomain*>& mapping_a,
    const std::vector<IterDomain*>& mapping_b) {
  std::unordered_map<ValGroup, MatmulDimRole> dim_roles;
  NVF_ERROR(mapping_a.size() == out_logical.size());
  NVF_ERROR(mapping_a.size() == mapping_b.size());
  for (size_t i : arange(out_logical.size())) {
    IterDomain* id_out = out_logical[i];
    const ValGroup& g = graph.toGroup(id_out);

    if (id_out->isReduction()) {
      dim_roles[g] = MatmulDimRole::K;
      continue;
    }

    bool has_a = mapping_a[i] != nullptr && mapping_a[i]->isIteration();
    bool has_b = mapping_b[i] != nullptr && mapping_b[i]->isIteration();

    NVF_ERROR(has_a || has_b);
    // If both operand IterDomains are Broadcast, treat as Batch dimension
    // If they mismatch, then one must be broadcast which determines M or N
    if (has_a == has_b) {
      dim_roles[g] = MatmulDimRole::Batch;
    } else if (has_a) {
      dim_roles[g] = MatmulDimRole::M;
    } else if (has_b) {
      dim_roles[g] = MatmulDimRole::N;
    }
  }
  return dim_roles;
}
} // namespace

DimRolesMap MatmulPattern::getDimRoles(IdModel& id_model) const {
  const ValGraph& graph = id_model.maybeBuildGraph(IdMappingMode::BROADCAST);

  // There are four types of ValGroup involved in a MatmulPattern: M, N, K, and
  // Batch. These are enumerated in the MatmulDimRole enum class. They are
  // defined by their membership as follows:
  //   M: present in A and output, but not B
  //   N: present in B and output, but not A
  //   K: present in A and B, but not output
  //   Batch: present in all A, B, and output
  // If there are other patterns, for example a ValGroup present in only A, then
  // we should raise an exception here.

  if (output->definition()->isA<MatmulOp>()) {
    const std::vector<IterDomain*>& out_logical = output->getLogicalDomain();
    return matmulOrLinearOpDimRoles(
        graph,
        out_logical,
        ops::mapMatmulOpIterDomains(
            A->getLogicalDomain(), 0, out_logical.size()),
        ops::mapMatmulOpIterDomains(
            B->getLogicalDomain(), 1, out_logical.size()));

  } else if (output->definition()->isA<LinearOp>()) {
    const std::vector<IterDomain*>& out_logical = output->getLogicalDomain();
    bool k_bcast = A->getLogicalDomain().back()->isBroadcast();
    return matmulOrLinearOpDimRoles(
        graph,
        out_logical,
        ops::mapLinearOpIterDomains(
            A->getLogicalDomain(), 0, out_logical.size(), k_bcast),
        ops::mapLinearOpIterDomains(
            B->getLogicalDomain(), 1, out_logical.size(), k_bcast));
  }

  // The code below handles MmaOp or mul-sum patterns

  // Indicates whether a ValGroup is present in A (bit 0), B (bit 1), or output
  // (bit 2)
  using DimPresence = std::bitset<3>;

  // for each valgroup, store a pair of flags. The first records whether the
  // group is present at all in the tv. The second records whether the value is
  // concrete (i.e. not reduction, broadcast, or device).
  std::unordered_map<ValGroup, DimPresence> flags;
  const auto recordPresence = [&graph, &flags](
                                  TensorView* tv, size_t tensor_num) {
    for (IterDomain* id : tv->getLogicalDomain()) {
      const ValGroup& g = graph.toGroup(id);
      DimPresence& group_flags = flags[g];
      // Note: broadcast or device dims will be initialized to have all false
      // flags above
      if (id->isReduction() || id->isBroadcast() || id->isDeviceDim()) {
        continue;
      }
      group_flags.set(tensor_num);
    }
  };
  recordPresence(A, 0);
  recordPresence(B, 1);
  recordPresence(output, 2);

  DimRolesMap dim_roles;

  for (const auto& [g, concrete_flags] : flags) {
    if (concrete_flags.all() || concrete_flags.none()) {
      // Batch dimensions are any of those that are not concretized or reduced.
      // These could be all Iteration or all Broadcast
      dim_roles[g] = MatmulDimRole::Batch;
    } else if (concrete_flags == 0b011) {
      dim_roles[g] = MatmulDimRole::K;
    } else if (concrete_flags == 0b101) {
      dim_roles[g] = MatmulDimRole::M;
    } else if (concrete_flags == 0b110) {
      dim_roles[g] = MatmulDimRole::N;
    } else {
      NVF_THROW(
          "IterDomain ValGroup should be concrete in at least two of A, B, output.",
          " concrete_flags: ",
          concrete_flags);
    }
  }

  // NOTE: For Hopper, we create loop broadcasts to mimic logical broadcasts
  // when translating MatmulOp and LinearOp. Here we detect these and map them
  // appropriately.
  for (IterDomain* id : A->getLoopDomain()) {
    const ValGroup& g = graph.toGroup(id);
    if (dim_roles.count(g) == 0) {
      dim_roles[g] = MatmulDimRole::N;
    }
  }
  for (IterDomain* id : B->getLoopDomain()) {
    const ValGroup& g = graph.toGroup(id);
    if (dim_roles.count(g) == 0) {
      dim_roles[g] = MatmulDimRole::M;
    }
  }

  return dim_roles;
}

std::vector<ValGroup> canonicalDimOrdering(
    const mma_utils::TensorRolesMap& tensor_roles,
    const mma_utils::DimRolesMap& dim_roles,
    const ValGraph& graph) {
  VectorOfUniqueEntries<ValGroup> batch_dims, m_dims, n_dims, k_dims,
      other_dims, device_dims;
  // This is +1 if N should come before M and -1 otherwise. It is zero until the
  // M/N ordering has been determined.
  int64_t n_inside_m = 0;
  for (MatmulTensorRole tv_role :
       {MatmulTensorRole::OUTPUT,
        MatmulTensorRole::OPERAND_A,
        MatmulTensorRole::OPERAND_B,
        MatmulTensorRole::EPILOGUE_INPUT}) {
    const auto it = tensor_roles.find(tv_role);
    if (it == tensor_roles.end()) {
      continue;
    }
    for (TensorView* tv : it->second) {
      // We iterate in reverse through the allocation domain of tv so that we
      // can find the inner-most dimensions
      for (auto id_it = tv->getMaybeAllocationDomain().rbegin();
           id_it != tv->getMaybeAllocationDomain().rend();
           id_it++) {
        IterDomain* id = *id_it;
        if (id->isDeviceDim()) {
          // save device dim groups since they must be outermost
          const ValGroup& g = graph.toGroup(id);
          device_dims.pushBack(g);
          continue;
        } else if (id->isBroadcast() || id->isReduction()) {
          // all-broadcast and all-reduction groups will be outermost within
          // their roles
          continue;
        }
        const ValGroup& g = graph.toGroup(id);
        const auto it = dim_roles.find(g);
        if (it == dim_roles.end()) {
          other_dims.pushBack(g);
        } else {
          switch (it->second) {
            case MatmulDimRole::Batch:
              batch_dims.pushBack(g);
              break;
            case MatmulDimRole::M:
              if (n_inside_m == 0) {
                // We encountered an M dimension before an N dimension
                n_inside_m = -1;
              }
              m_dims.pushBack(g);
              break;
            case MatmulDimRole::N:
              if (n_inside_m == 0) {
                // We encountered an N dimension before an M dimension
                n_inside_m = 1;
              }
              n_dims.pushBack(g);
              break;
            case MatmulDimRole::K:
              // Order K dimensions like operands, and all others like outputs
              if (tv_role == MatmulTensorRole::OPERAND_A ||
                  tv_role == MatmulTensorRole::OPERAND_B) {
                k_dims.pushBack(g);
              }
              break;
          }
        }
      }
    }
  }
  NVF_ERROR(other_dims.empty(), "Found unrecognized dims in matmul tensors");

  // At this point we might not have included all the ValGroups for each role,
  // since some might be all Broadcast (e.g. broadcast batch dims). We will add
  // any skipped groups outside the ones that are already included, i.e. we
  // place broadcast dims outside non-broadcast within the same role.
  for (const auto& [g, role] : dim_roles) {
    VectorOfUniqueEntries<ValGroup>* inserted = nullptr;
    switch (role) {
      case MatmulDimRole::Batch:
        inserted = &batch_dims;
        break;
      case MatmulDimRole::M:
        inserted = &m_dims;
        break;
      case MatmulDimRole::N:
        inserted = &n_dims;
        break;
      case MatmulDimRole::K:
        inserted = &k_dims;
        break;
    }
    auto it = inserted->set().find(g);
    if (it == inserted->set().end()) {
      // We did not insert this group yet. Insert it at the outer position
      // (inserted is reverse-ordered).
      inserted->pushBack(g);
    }
  }

  // See https://github.com/NVIDIA/Fuser/pull/2303#discussion_r1626587836
  NVF_ERROR(
      n_inside_m,
      "Currently N must be the innermost dimension. This constraint will be lifted in the future");

  // Insert the reverse-ordered groups in order
  std::vector<ValGroup> ordering;
  ordering.reserve(
      batch_dims.size() + m_dims.size() + n_dims.size() + k_dims.size());
  const auto insert = [&ordering, &device_dims](
                          const VectorOfUniqueEntries<ValGroup>& v,
                          bool skip_device_dims = true) {
    for (auto it = v.vector().rbegin(); it != v.vector().rend(); ++it) {
      const ValGroup& g = *it;
      if (skip_device_dims && device_dims.has(g)) {
        continue;
      }
      ordering.push_back(g);
    }
  };
  // Insert the device dims first, then skip them when inserting dims from each
  // other role
  insert(device_dims, /*skip_device_dims=*/false);
  insert(batch_dims);
  if (n_inside_m == 1) {
    insert(m_dims);
    insert(n_dims);
  } else {
    NVF_ERROR(
        n_inside_m == -1 || (n_dims.empty() && m_dims.empty()),
        "Could not determine order of M and N dims");
    insert(n_dims);
    insert(m_dims);
  }
  insert(k_dims);

  return ordering;
}

std::optional<std::pair<DimRolesMap, TensorRolesMap>> allPatternRoles(
    IdModel& id_model,
    const std::vector<MatmulPattern>& patterns) {
  Fusion* fusion = nullptr;
  DimRolesMap id_roles;
  for (const MatmulPattern& pattern : patterns) {
    if (fusion == nullptr) {
      fusion = pattern.output->fusion();
    } else {
      NVF_ERROR(fusion == pattern.output->fusion());
    }
    mma_utils::DimRolesMap pattern_id_roles = pattern.getDimRoles(id_model);
    for (const auto& [g, role] : pattern_id_roles) {
      const auto& [it, inserted] = id_roles.try_emplace(g, role);
      if (!inserted && it->second != role) {
        return std::nullopt;
      }
    }
  }
  const auto tensor_roles_opt =
      mma_utils::getTensorRoles(fusion, id_model, id_roles);
  if (!tensor_roles_opt.isValid()) {
    return std::nullopt;
  }
  return std::pair<DimRolesMap, TensorRolesMap>{
      id_roles, tensor_roles_opt.getData()};
}

namespace {
// This function returns a pair of integers. The first integer is the gcd
// between megabanks and row stride. The second integer is the repeat pattern
// size. If the gcd is 1, then no swizzle is necessary to resolve bank
// conflicts. In that case, the second integer is irrelevant and -1 is returned.
std::pair<int64_t, int64_t> analyzeSwizzleSharedMemory(
    TensorView* shared_mem_tv) {
  NVF_ERROR(shared_mem_tv->getMemoryType() == MemoryType::Shared);
  AbstractTensor swizzle_domain(shared_mem_tv->getLoopDomain());

  // Check that the innermost 2 dimensions are concrete and static
  //  sized so that the swizzle function can be defined.
  NVF_ERROR(
      (int64_t)swizzle_domain.size() >= 2,
      "At least 2D input (excluding consecutive reduction domains starting from the innermost dim) needed for swizzling, but get ",
      shared_mem_tv->toString());
  mma_utils::checkConcreteStaticDim(swizzle_domain[-2]);
  mma_utils::checkConcreteStaticDim(swizzle_domain[-1]);

  // Extract the constant sizes of the swizzled tile
  const int64_t tile_size_x =
      swizzle_domain[-2]->extent()->evaluate().as<int64_t>();
  const int64_t tile_size_y =
      swizzle_domain[-1]->extent()->evaluate().as<int64_t>();

  // Only tested for (1) ldmatrix access with sizeof(T) == 16bit (i.e.
  // half/bfloat16) and (2) epilogue general access with sizeof(T) == 32bit
  // (i.e. float)
  const int64_t data_type_size = dataTypeSize(*shared_mem_tv->getDataType());
  NVF_ERROR(data_type_size == 2 || data_type_size == 4);

  // For main loop, ldmatrix loads a n_rows x n_cols = 8 x 8 matrix each time.
  // For epilogue, threads in a warp is organized as 8 rows x 4 columns.
  // Each thread vectorized write 2 items, so 8 items per row.
  //--0--1--2--3
  //--4--5--6--7
  //--8--9--10-11
  //--12-13-14-15
  //--16-17-18-19
  //--20-21-22-23
  //--24-25-26-27
  //--28-29-30-31
  constexpr int64_t n_rows = 8;
  constexpr int64_t n_cols = 8;

  // Column size of the tile needs to be multiples of 8 for ldmatrix to work.
  NVF_ERROR(
      tile_size_x >= n_rows && tile_size_x % n_rows == 0 &&
          tile_size_y >= n_cols && tile_size_y % n_cols == 0,
      "Prolog swizzle for ldmatrix, illegal tile size for prolog swizzle",
      tile_size_x,
      "x",
      tile_size_y);

  /* Note [How to remove bank conflict for ldmatrix?]
   *
   * **This note is interleaved with code, I suggest reading this note like
   *   reading a jupyter notebook**
   *
   * Our task is to make sure different rows does not fall into the same
   * bank of shared memory.
   *
   * Introduction to bank conflict can be found at page 54-72 of:
   * https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf
   *
   * When we talk about bank conflict removal, we are talking about the
   * following task:
   *   "there are 32 banks, and each bank contains one 4-byte word, we want to
   *    make sure different lanes in a warp does not access different word
   *    addresses in the same bank"
   * For example, if thread 0 is accessing word address 1, and thread 1 is
   * accessing word address 33, then these two threads will have a bank
   * conflict because they are accessing different word addresses in the same
   * bank. However, if thread 0 is accessing byte address 4 and thread 1 is
   * accessing byte address 6 then there will be no bank conflict because 4
   * and 6 both belong to word 1.
   */

  constexpr int64_t smem_bytes_per_word = 4;
  constexpr int64_t smem_banks = 32;

  /* but here, for our convenience, because ldmatrix always use vectorized
   * access of 8 items = 16 bytes = 4 words, we further group words into
   * units: we consider each 4 words as a "unit", and each 4 banks as a
   * "megabank". So we can rephrase our task as:
   *   "there are 8 megabanks, and each megabanks contains one 4-word unit, we
   *    want to make sure different lanes in a warp does not access different
   *    unit addresses in the same megabank"
   * In this terminology, matrices are in the row major format, each matrix
   * has 8 rows, and each row has exactly one unit.
   */

  constexpr int64_t items_per_unit = n_cols;
  const int64_t bytes_per_unit = items_per_unit * data_type_size;
  const int64_t words_per_unit = bytes_per_unit / smem_bytes_per_word;
  const int64_t num_megabanks = smem_banks / words_per_unit;

  /* In the following example, each CTA tile contains 2 rows and 3 colums of
   * matrices, each 8x8 size:
   *   +----------+----------+----------+
   *   | matrix 0 | matrix 1 | matrix 2 |
   *   +----------+----------+----------+
   *   | matrix 3 | matrix 4 | matrix 5 |
   *   +----------+----------+----------+
   * The addresses of different rows in the same matrix are offset by 3 units.
   * In this perspective, loading a matrix is a strided memory access with the
   * following stride (in units):
   */

  // number of units per row
  int64_t row_stride = tile_size_y / items_per_unit;

  /* So the bank conflicting problem is now converted to the following game:
   *   I have a clock that has one pointer and `num_megabanks` ticks. I start
   *   my game by making my pointer pointing to somewhere, and turn forward
   *   the pointer `n_rows` times, each time by `row_stride` ticks.
   * This problem can be well modeled by modular arithmetic in number theory
   * using the concept "integers modulo n" a.k.a. "Z/nZ"[1].
   * Take n = 6 as an example, Z/6Z only has 6 elements: 0, 1, 2, 3, 4, 5.
   * Additions and multiplications are defined in a cyclic manner:
   *   5 + 1 = 0, 5 + 2 = 1, 5 + 3 = 2, 5 + 4 = 3, ...
   *   2 * 1 = 2, 2 * 2 = 4, 2 * 3 = 0, 2 * 4 = 2, ...
   * With this definition, Z is mapped to Z/nZ naturally by i -> i % n [2]
   *
   * It worth mention that Z/nZ is a "commutative ring", that is, we can use
   * addition and multiplication rules just like using normal integers:
   *   a + b = b + a, a * (b + c) = a * b + a * c, ...
   * In short, we can reason about Z/nZ just like we are reasoning about
   * integers, except that every number is automatically "% n".
   *
   * Reference:
   * [1] https://en.wikipedia.org/wiki/Modular_arithmetic#Integers_modulo_n
   * [2] The % is under Euclidean definition, that is -1 % 6 is 5 instead of
   *     -1, see [The Mathematics of Integer Arithmetic] for more detail. But
   *     we are only interested in non-negative numbers here, so there is no
   *     need to worry about this problem
   */

  // row_stride in Z/nZ, where n is num_megabanks:
  // assert(row_stride >= 0);
  // assert(num_megabanks >= 0);
  int64_t row_stride_znz = row_stride % num_megabanks;
  /* Consider the following function in Z/nZ:
   *   f(i; init) = init + i * stride
   * where init is the initial position of the pointer in the clock when we
   * start the game, and stride is the number of ticks we move forward each
   * time, and i is the number of times we move forward. For a fixed init, we
   * abbrivate f(i; init) as f(i).
   *
   * In our problem, f(i) is the megabank of the `i`th row of the matrix, and
   * `init` is the megabank of the 0th row of the matrix.
   *
   * One very important property of f(i) is:
   * - if f(i1) == f(i2), then for every j, f(i1 + j) = f(i2 + j)
   * This property is true because:
   *   f(i1 + j) = f(i1) + j * stride = f(i2) + j * stride = f(i2 + j)
   *
   * The above property tells us, as we turn the clock forward:
   * - initially, we will go to a never-visited tick in each turn, but,
   * - at some point, we will return back to our original position, and,
   * - after we return, we start repeat the pervious pattern again and again.
   *
   * As an example, consider f(i) where init = 0, stride = 6, under Z/8Z:
   *     i  0 1 2 3 4 5 6 7
   *   f(i) 0 6 4 2 0 6 4 2
   * We can see that f(i) is repeating a pattern of four unique numbers
   * "0 6 4 2" twice. In our bank conflict problem, this means we are using 4
   * different megabanks, and we have a 2-way conflict.
   *
   * The question of interest is, does the above observation generalize? That
   * is, does f(i) always repeat a pattern of p unique numbers q times? Note
   * that p and q must satisfy p * q = n.
   *
   * The answer to the above question is: yes! Consider the following
   * equation:
   *    f(i1 + j) == f(i1)
   * We want to know what is the smallest positive number j that makes the
   * above equation true. Because this tells us in how many steps we will see
   * repeat. This equation can be simplified as:
   *   f(i1 + j) == f(i1) + j * stride == f(i1)
   *   ==> j * stride == 0
   *
   * An important tool to study this equation is multiplicative inverse:
   * https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
   * A number i has multiplicative inverse `minv(i)` in Z/nZ if and only if it
   * coprime with n. `minv(i)` is the number that `i * minv(i) == 1`. So in
   * Z/nZ, the equation `ax = b` has solution `x = minv(a)*b` if a has
   * multiplicative inverse. For example, in Z/15Z, `minv(2) = 8` because
   *   (2 * 8) % 15 = 1
   *
   * stride has an multiplicative inverse if and only if stride coprime with
   * n, that is, g := gcd(stride, n) == 1. In such case, the solution to our
   * equation j * stride == 0 is j = minv(stride) * 0 = 0, that is: f(i) does
   * not repeat, that is: there is no bank conflict.
   */

  int64_t g = std::gcd(num_megabanks, row_stride_znz);
  if (g == 1) {
    return {g, -1}; // No need to swizzle in this case.
  }

  /* For the case where stride does not coprime with n, we note that
   * j * stride == 0 in Z/nZ is equivalent to (j * stride) % n = 0 in Z. We
   * can write stride and n as:
   *   stride = s * g, n = m * g
   * According to Theorem 4.13 in [The Mathematics of Integer Arithmetic], we
   * have:
   *   (j * stride) % n = 0
   *   ==> (j * s) % m * g = 0
   *   ==> (j * s) % m = 0
   * which is equivalent to j * s == 0 in Z/mZ. Because s coprime with m, we
   * further get:
   *   j == 0 (in Z/mZ)
   * That is, j is a multiple of m in Z. So the smallest positive j that make
   * the equation hold is n / g.
   *
   * That is: f(i) always repeat a pattern of n/g unique numbers g times.
   * In other word: we are using n/g megabanks, and we have a g-way bank
   * conflict.
   *
   * Let's use the word "pattern" to refer to the set of values of `f` at
   * different `i`, that is:
   *   pattern k = { f(i; init=k) | i in Z/nZ }
   * For the example of stride = 6 under Z/8Z, we have the following patterns
   *        f(i): 01234567
   *   pattern 0: x_x_x_x_
   *   pattern 1: _x_x_x_x
   *   (x => occupied, _ => unoccupied)
   */

  int64_t repeated_pattern_size = num_megabanks / g;

  if (repeated_pattern_size >= n_rows) {
    return {g, -1}; // No need to swizzle in this case.
  }

  return {g, repeated_pattern_size};
}
} // namespace

MmaInputSmemSwizzle tmaSwizzleSharedMemory(TensorView* shared_mem_tv) {
  auto&& [g, repeated_pattern_size] = analyzeSwizzleSharedMemory(shared_mem_tv);

  if (g == 1) {
    return MmaInputSmemSwizzle::None; // No need to swizzle in this case.
  }

  // 128B swizzle results in 8 x 8 matrix given half precision inputs.
  constexpr int64_t n_rows = 8;

  NVF_ERROR(
      n_rows % repeated_pattern_size == 0,
      "Can not partition matrix into megarows");
  int64_t num_gigarows = n_rows / repeated_pattern_size;
  int64_t num_gigabanks = g; // also = num_megabanks / repeated_pattern_size

  /* To further simplify the problem, if we assume: */
  NVF_ERROR(
      num_gigarows % num_gigabanks == 0,
      "Requires non-square swizzle, which is not supported yet");

  AbstractTensor swizzle_domain(shared_mem_tv->getLoopDomain());
  // Extract the constant sizes of the swizzled tile
  const int64_t inner_dim_size =
      swizzle_domain[-1]->extent()->evaluate().as<int64_t>();

  auto dtype = shared_mem_tv->getDataType().value();
  const int64_t B128_elements = 128 / dataTypeSize(dtype);
  const int64_t B64_elements = 64 / dataTypeSize(dtype);
  const int64_t B32_elements = 32 / dataTypeSize(dtype);

  if (inner_dim_size % B128_elements == 0) {
    return MmaInputSmemSwizzle::B128;
  } else if (inner_dim_size % B64_elements == 0) {
    return MmaInputSmemSwizzle::B64;
  } else if (inner_dim_size % B32_elements == 0) {
    return MmaInputSmemSwizzle::B32;
  } else {
    NVF_THROW("Unsupported swizzle size for TMA shared memory mma inputs");
  }
}

} // namespace mma_utils

std::string toString(const mma_utils::AbstractMatmulTensor& abten) {
  std::ostringstream ss;
  ss << "AbstractMatmulTensor (" << abten.size() << "):" << std::endl;
  for (size_t i : arange(abten.size())) {
    const AbstractId& abs_id = abten[(int64_t)i];
    const std::optional<MatmulDimRole> role = abten.getTag((int64_t)i).value();
    ss << "  " << (role.has_value() ? toString(role.value()) : "no role");
    if (abs_id.is<ValGroupAndItsGraph>()) {
      const ValGroup& g = abs_id.as<ValGroupAndItsGraph>().group;
      for (Val* v : g->vector()) {
        ss << " " << v->toString();
      }
    }
    ss << std::endl;
  }
  return ss.str();
}

} // namespace nvfuser
