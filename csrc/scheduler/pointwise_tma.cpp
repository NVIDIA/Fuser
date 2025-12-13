// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/pointwise_tma.h>

#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/pointwise_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include <transform_iter.h>
#include <transform_replay.h>

namespace nvfuser {
namespace pointwise {
namespace tma {

// ============================================================================
// TMA TERMINOLOGY GUIDE:
// ============================================================================
// The TMA scheduler uses a 2-level hierarchy to organize data:
//
// 1. TMA DOMAIN: The logical split of the entire problem space into two parts:
//    - tma_domain_inner: Size of the inner (contiguous) dimension
//    - tma_domain_outer: Size of the outer dimension (n_elems /
//    tma_domain_inner)
//    - requirement: n_elems % tma_domain_inner == 0
//    Example: For 2048 elements with tma_domain_inner=512:
//      TMA Domain structure: [4, 512]
//
// 2. TMA TILE: The actual hardware tile size for each TMA load operation
//    - tma_tile_inner: Number of elements along the inner dimension per tile
//    - tma_tile_outer: Number of elements along the outer dimension per tile
//    Example: For TMA domain [4, 512] with tiles [2, 128]:
//      Each TMA load fetches a [2 x 128] tile, requiring 2 x 4 = 8 tiles total
//
//    Note: In general TMA terminology, a "box" is the dense rectangular region
//    loaded from global memory, while a "tile" is a potentially strided subset
//    selected from that box. The pointwise scheduler always uses dense tiles
//    (tile = box), so we use "TMA tile" throughout to refer to both concepts.
//
// Transformation sequence: logical domain -> TMA domain -> TMA tile
//  [I0, I1, ...] -- > [I] -> [Do, Di] -> [Do/to, to, Di/ti, ti]
//   where Do=tma_domain_outer, Di=tma_domain_inner,
//         to=tma_tile_outer, ti=tma_tile_inner
// ============================================================================

// TODO: This can be further relaxed to allow more tensor views with fewer
// dimensions, e.g., outer broadcast inputs [B, I] can also be loaded with TMA.
bool isTvSuitableForTma(const TensorView* tv, int64_t n_valid_dims) {
  return scheduler_utils::nLogicalDims(tv) == n_valid_dims;
};

// Returns the total bits required to load one element from each TMA-loaded
// input. This is used to determine how many elements should be loaded from each
// input to achieve the required bits in flight.
int64_t getInputBitsPerElement(
    const pointwise_utils::FusionRuntimeProperties& prop) {
  int64_t bits_per_element = 0;
  int64_t n_valid_dims = scheduler_utils::nLogicalDims(prop.largest_out);
  for (const auto& tv : prop.vectorizable_inputs_outputs) {
    if (tv->isFusionInput() && isTvSuitableForTma(tv, n_valid_dims)) {
      bits_per_element +=
          dataTypeSizeBit(tv->getDataType().value(), prop.index_type);
    }
  }
  return bits_per_element;
}

std::unique_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  constexpr int64_t threads_per_warp = 32;

  auto params = std::make_unique<PointwiseParams>();
  params->tag = "Pointwise TMA heuristics";
  params->cparams.index_type = prop.index_type;
  params->use_tma_load = true;

  // ========== Step 0: Determine Break Point ==========
  // The break point determines the scheduling strategy:
  //
  // break_point == 0: Use 1D scheduler
  //   - All iteration domains are flattened into a single dimension
  //   - This is then split into [tma_domain_outer, tma_domain_inner]
  //
  // break_point > 0: Use 2D scheduler
  //   - Iteration domains are merged at the break point into two dimensions
  //   - These directly correspond to [tma_domain_outer, tma_domain_inner]
  auto bp_info = pointwise_utils::getBreakPoint(
      fusion, prop, data_cache, /*is_tma =*/true);
  params->break_point = bp_info.break_point;

  // ========== Step 1: Compute TMA Domain Dimensions ==========
  // The TMA domain splits the entire problem into
  // [tma_domain_outer, tma_domain_inner]
  // Target: Make the inner domain ~512 elements for good memory locality

  // tma_domain_inner: Size of the innermost contiguous dimension
  // This represents how many consecutive elements are treated as a single "row"

  // tma_domain_outer: Number of "rows" after splitting
  // The full TMA domain is now [tma_domain_outer, tma_domain_inner]
  // where tma_domain_outer * tma_domain_inner = n_elems
  constexpr int64_t tma_domain_inner_target = 512;

  const int64_t tma_domain_inner = bp_info.break_point == 0
      ? scheduler_utils::getTmaDomainInner(
            prop.n_elems,
            tma_domain_inner_target,
            prop.min_dtype_size_bit_for_vectorization)
      : bp_info.right_elem_count;
  NVF_ERROR(
      tma_domain_inner > 1 && prop.n_elems % tma_domain_inner == 0,
      "Illegal tma_domain_inner size: ",
      tma_domain_inner,
      ", n_elems: ",
      prop.n_elems);

  // TMA requires 128-bit alignment. When break_point > 0, tma_domain_inner is
  // determined by right_elem_count, which may not satisfy this requirement.
  // Reject such cases early to avoid generating invalid TMA schedules.
  //
  // TODO: The break_point selection logic should account for alignment
  // requirements. For instance, a 512x127 tensor could use a 1D scheduler
  // (break_point=0) but fails with a 2D scheduler if 127 elements don't meet
  // 128-bit alignment. This limitation affects both TMA and non-TMA pointwise
  // schedulers. See TmaDomainBroadcastIllegal test for an example.
  if (bp_info.break_point > 0) {
    const int64_t ref_elem_size_bits = dataTypeSizeBit(
        prop.largest_out->getDataType().value(), prop.index_type);
    const int64_t tma_domain_inner_bits = tma_domain_inner * ref_elem_size_bits;
    if (tma_domain_inner_bits % 128 != 0) {
      return nullptr;
    }
  }

  const int64_t tma_domain_outer = prop.n_elems / tma_domain_inner;
  params->tma_domain_inner = tma_domain_inner;

  // ========== Step 2: Determine Target Elements Per CTA ==========
  // We calculate how many elements each CTA should load based on memory
  // bandwidth requirements.
  // bits_per_element: Total bits to load for one position across all TMA inputs
  // (e.g., if we have 2 float32 inputs, this would be 64 bits)
  // elements_per_cta: Target number of elements for each CTA to process
  // Each CTA has 128 threads, round up to 1024 for divisible by 128 and leave 8
  // for vectorization and unroll.
  constexpr int64_t cta_per_sm = 8;
  const int64_t bits_per_sm = scheduler_utils::getRequiredBitsInFlight();
  const int64_t bits_per_cta = bits_per_sm / cta_per_sm;
  const int64_t bits_per_element = getInputBitsPerElement(prop);
  if (bits_per_element == 0) {
    return nullptr;
  }
  const int64_t elements_per_cta = scheduler_utils::roundUpToN(
      ceilDiv(bits_per_cta, bits_per_element), 1024);

  // ========== Step 3: Compute TMA Tile Dimensions ==========
  // TMA tiles define the tile size loaded by each TMA operation:
  // [tma_tile_outer, tma_tile_inner]
  // Constraints:
  // 1. Hardware limit: Each tile dimension must be ≤ 256 elements
  // 2. 2D TMA requirement: Need at least 2 tiles along inner dimension
  //    (i.e., tma_tile_inner ≤ tma_domain_inner / 2) to maintain 2D structure
  // 3. Don't exceed domain boundaries

  // tma_tile_inner_max: Maximum size for tma tile inner dimension
  // Division by 2 ensures at least 2 tiles fit within tma_domain_inner
  // Don't exceed the hardware limit
  const int64_t tma_tile_inner_max =
      std::min(tma_domain_inner / 2, kMaxElementsPerTmaTileDim);

  // tma_tile_outer_max: Maximum size for tma tile outer dimension
  // Don't exceed the total number of "rows" in tma_domain_outer
  // Don't exceed the hardware limit
  const int64_t tma_tile_outer_max =
      std::min(tma_domain_outer, kMaxElementsPerTmaTileDim);

  // tma_tile_inner: Actual tma tile in inner dimension
  // Start with warp size, then grow by powers of 2 until reaching
  // tma_tile_inner_max e.g. 16, 32, 64, 128, 256
  int64_t tma_tile_inner = std::min(tma_tile_inner_max, threads_per_warp);
  while (tma_tile_inner * 2 <= tma_tile_inner_max) {
    tma_tile_inner *= 2;
  }

  // tma_tile_outer: Actual tma tile in outer dimension
  // Compute to achieve target elements_per_cta, capped by tma_tile_outer_max
  const int64_t tma_tile_outer = std::max(
      1L, std::min(elements_per_cta / tma_tile_inner, tma_tile_outer_max));

  params->tma_tile_inner = tma_tile_inner;
  params->tma_tile_outer = tma_tile_outer;

  // ========== Step 4: Configure Thread Block Dimensions ==========
  // bdimx strategy:
  // - Use min(32, tma_tile_inner) to avoid using more threads than elements
  // - For small tiles (e.g., tma_tile_inner=8 when input is 17x16), bdimx=8
  //   since it doesn't make sense to use more than 8 threads for 8 elements
  // - For most cases, tma_tile_inner is a multiple of 32, so bdimx=32
  //
  // Benefits of bdimx=32 (when possible):
  // (a) Avoids bank conflicts: 32 threads accessing 32 elements = no conflict
  //     Using fewer threads (e.g., 16) would cause multi-way bank conflicts if
  //     vectorized load from shared memory to registers is not used.
  // (b) Enables vectorization: With max tma_tile_inner=256, using 32 threads
  //     means each thread processes 8 elements, enabling vectorized writes to
  //     global memory (e.g., 8-element vectors for optimal performance)
  constexpr int64_t threads_per_cta = 128;
  const int64_t bdimx = std::min(threads_per_warp, tma_tile_inner);
  const int64_t bdimy = std::min(threads_per_cta / bdimx, tma_tile_outer);
  params->lparams.bindUnsafe(bdimx, ParallelType::TIDx);
  params->lparams.bindUnsafe(bdimy, ParallelType::TIDy);

  // ========== Step 5: Determine Vectorization Factor ==========
  // Don't limit vectorization factor by number of IO tensors or wave count
  // since some of inputs are TMA-loaded which uses shared memory instead of
  // registers.
  params->vectorization_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      prop.largest_out,
      data_cache,
      bp_info.break_point,
      /*max_vectorization_size_in_bit=*/128,
      prop.min_dtype_size_bit_for_vectorization,
      prop.max_dtype_size_bit_for_vectorization,
      /*n_vectorizable_tensors=*/-1,
      /*n_waves=*/-1,
      /*logical_reorder_map=*/
      pointwise_utils::getLogicalReorderMap(
          prop.largest_out, prop.has_reshapes, data_cache));

  // TMA store
  params->use_tma_store = false;

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n==== Pointwise TMA Scheduler Heuristics ====\n";
    debug() << "Domain sizes:\n";
    debug() << "  n_elems: " << prop.n_elems << "\n";
    debug() << "  break_point: " << bp_info.break_point << "\n";
    debug() << "  tma_domain_inner: " << tma_domain_inner << "\n";
    debug() << "  tma_domain_outer: " << tma_domain_outer << "\n";
    debug() << "\nMemory and CTA configuration:\n";
    debug() << "  cta_per_sm: " << cta_per_sm << "\n";
    debug() << "  bits_per_sm: " << bits_per_sm << "\n";
    debug() << "  bits_per_cta: " << bits_per_cta << "\n";
    debug() << "  bits_per_element: " << bits_per_element << "\n";
    debug() << "  elements_per_cta: " << elements_per_cta << "\n";
    debug() << "\nTMA tile configuration:\n";
    debug() << "  kMaxElementsPerTmaTileDim: " << kMaxElementsPerTmaTileDim
            << "\n";
    debug() << "  tma_tile_inner_max: " << tma_tile_inner_max << "\n";
    debug() << "  tma_tile_inner: " << tma_tile_inner << "\n";
    debug() << "  tma_tile_outer: " << tma_tile_outer << "\n";
    debug() << "  tma_tile_size: " << (tma_tile_inner * tma_tile_outer) << "\n";
    debug() << "  use_tma_load: " << params->use_tma_load << "\n";
    debug() << "  use_tma_store: " << params->use_tma_store << "\n";
    debug() << "\nThread block configuration:\n";
    debug() << "  blockDim.x (TIDx): " << bdimx << "\n";
    debug() << "  blockDim.y (TIDy): " << bdimy << "\n";
    debug() << "  threads_per_cta: " << (bdimx * bdimy) << "\n";
    debug() << "\nVectorization:\n";
    debug() << "  max_dtype_size_bit: "
            << prop.max_dtype_size_bit_for_vectorization << "\n";
    debug() << "  min_dtype_size_bit: "
            << prop.min_dtype_size_bit_for_vectorization << "\n";
    debug() << "  vectorization_factor: " << params->vectorization_factor
            << "\n";
    debug() << "============================================\n" << std::endl;
  }
  return params;
}

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  // ========== Phase 1: Common Setup and Caching ==========
  // Merge all dimensions into a single iteration domain. The TMA domain split
  // will be handled later via domain size parameters.
  auto schedule_info_opt =
      pointwise_utils::commonPointwiseSchedule(fusion, pparams->break_point);
  if (!schedule_info_opt.has_value()) {
    // Zero-dimensional tensors, nothing to schedule
    return;
  }
  auto& schedule_info = schedule_info_opt.value();

  auto& cached_inputs = schedule_info.cached_inputs;
  auto& cached_outputs = schedule_info.cached_outputs;

  // reference_tv: The tensor view used as a template for transformations
  TensorView* reference_tv = schedule_info.reference_tv;

  // Get inputs/outputs that have the innermost dimension and can be vectorized
  auto inputs_outputs =
      scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true, true);
  std::unordered_set<TensorView*> vectorizable_io_tvs(
      inputs_outputs.begin(), inputs_outputs.end());

  // ========== Phase 2: Classify Inputs for TMA vs LDG Loading ==========
  // TMA (Tensor Memory Accelerator): Hardware-accelerated global->shared loads
  // LDG (LoaD Global): Standard global memory loads for tensors not suitable
  // for TMA

  // n_valid_dims: Number of logical dimensions in the reference tensor
  int64_t n_valid_dims = scheduler_utils::nLogicalDims(reference_tv);

  // tma_tvs: Inputs that will use TMA load (same dimensionality as reference)
  std::vector<TensorView*> tma_tvs;

  // ldg_tvs: Inputs that will use standard global loads (different
  // dimensionality)
  std::vector<TensorView*> ldg_tvs;

  for (const auto& [tv, _] : cached_inputs) {
    if (!isTvSuitableForTma(tv, n_valid_dims)) {
      ldg_tvs.push_back(tv);
      continue;
    }
    // Configure the load operation to use TMA
    auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition());
    if (load_op) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulkTensorTile);
    }
    // TMA loads data into shared memory
    tv->setMemoryType(MemoryType::Shared);
    // Create a register cache after the shared memory tensor
    tv->cacheAfter();
    tma_tvs.push_back(tv);
  }

  // ========== Phase 3: Split Into TMA Domain and Tiles ==========
  // Transform the flattened domain through a two-level hierarchy:
  // Step 1: [I0] -> [tma_domain_outer, tma_domain_inner]
  //   Split into outer and inner domains based on tma_domain_inner
  // Note: Skip this split if break_point != 0, as the reference tensor has
  // already been transformed into [lhs, rhs] domains at the break point by
  // commonPointwiseSchedule, which provides the required 2D TMA structure.
  // TODO: consider device domain and non-concretized domain.
  if (pparams->break_point == 0) {
    NVF_ERROR_EQ(reference_tv->nDims(), 1);
    reference_tv->split(0, pparams->tma_domain_inner);
  } else {
    NVF_ERROR_EQ(reference_tv->nDims(), 2);
  }

  // Step 2: [tma_domain_outer, tma_domain_inner] ->
  //         [tma_domain_outer/tma_tile_outer, tma_tile_outer,
  //          tma_domain_inner/tma_tile_inner, tma_tile_inner]
  //   Split each domain into tiles based on tma_tile sizes
  //   This creates: [outer_grid, outer_tile, inner_grid, inner_tile]
  reference_tv->split(1, pparams->tma_tile_inner);
  reference_tv->split(0, pparams->tma_tile_outer);

  // reorder to [outer_grid, inner_grid, outer_tile, inner_tile] for better
  // inlining
  reference_tv->reorder({{1, 2}});

  // Propagate these transformations to all tensors in the fusion
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);

  // ========== Phase 4: Parallelize TMA Tensors ==========
  // Parallelization strategy for TMA:
  //   axis(0) [tma_domain_outer/tma_tile_outer]: Grid (BIDy or BIDx)
  //   axis(1) [tma_domain_inner/tma_tile_inner]: Grid (BIDx or BIDy)
  //   axis(2) [tma_tile_outer]:                  Bulk (TMA tile)
  //   axis(3) [tma_tile_inner]:                  Bulk (TMA tile)

  // outer_cord_pt/inner_cord_pt: Grid parallelization types (BIDx/BIDy)
  auto outer_cord_pt = ParallelType::BIDy;
  auto inner_cord_pt = ParallelType::BIDx;
  if (pparams->flip_grid_binding) {
    std::swap(outer_cord_pt, inner_cord_pt);
  }

  // Apply TMA parallelization to reference
  reference_tv->axis(0)->parallelize(outer_cord_pt); // Outer grid
  reference_tv->axis(1)->parallelize(inner_cord_pt); // Inner grid
  reference_tv->axis(2)->parallelize(ParallelType::Bulk); // Outer tile (TMA)
  reference_tv->axis(3)->parallelize(ParallelType::Bulk); // Inner tile (TMA)

  // Apply same parallelization to all TMA input tensors
  scheduler_utils::parallelizeAllLike(reference_tv, tma_tvs);

  // Reset reference tensor's tile axes to Serial for subsequent scheduling
  // (TMA tensors keep Bulk parallelization; reference is for non-TMA
  // scheduling)
  reference_tv->axis(2)->parallelize(ParallelType::Serial);
  reference_tv->axis(3)->parallelize(ParallelType::Serial);

  // ========== Phase 5: Schedule Non-TMA Tensors ==========
  // Starting structure: [outer_grid, inner_grid, outer_tile, inner_tile]
  // Target structure:   [outer_grid, inner_grid, outer_tile/y, y,
  // inner_tile/v/x, x, v]
  //   where y = TIDy (threads), x = TIDx (threads), v = vectorization

  int64_t opos = 2; // Position of outer tile dimension (outer_tile)
  int64_t ipos = 3; // Position of inner tile dimension (inner_tile)

  // Split inner tile: inner_tile -> [inner_tile/v/x, x, v]
  reference_tv->split(ipos, pparams->vectorization_factor);
  reference_tv->split(ipos, pparams->lparams.bdimx());

  // Split outer tile: tma_tile_outer -> [tma_tile_outer/y, y]
  reference_tv->split(opos, pparams->lparams.bdimy());

  // reorder to [outer_grid, inner_grid, outer_tile/y, inner_tile/v/x, y, x, v]
  reference_tv->reorder({{3, 4}});

  // Propagate these transformations to all non-TMA tensors
  // (TMA tensors already have their final schedule from Phase 4)
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&non_tma_propagator);

  // ========== Phase 6: Apply Thread Parallelization ==========
  // Final axis structure: [outer_grid, inner_grid, outer_tile/y,
  // inner_tile/v/x, y, x, v]
  //   axis(0): Outer grid dimension
  //   axis(1): Inner grid dimension
  //   axis(2): Outer tile / TIDy serial
  //   axis(3): Inner tile / vect / TIDx serial
  //   axis(4): Thread block Y dimension (TIDy)
  //   axis(5): Thread block X dimension (TIDx)
  //   axis(6): Vectorization dimension
  reference_tv->axis(0)->parallelize(outer_cord_pt); // Grid outer
  reference_tv->axis(1)->parallelize(inner_cord_pt); // Grid inner
  reference_tv->axis(4)->parallelize(ParallelType::TIDy); // Thread Y
  reference_tv->axis(5)->parallelize(ParallelType::TIDx); // Thread X

  int64_t vect_pos = 6; // Position of vectorization axis
  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  // ========== Phase 7: Apply Vectorization ==========
  // Vectorize register <-> global memory transfers for non-TMA tensors

  // Vectorize output stores (register -> global)
  if (!pparams->use_tma_store && pparams->vectorization_factor > 1) {
    for (const auto& [_, original_idx] : cached_outputs) {
      auto output_tv =
          dynamic_cast<TensorView*>(fusion->outputs().at(original_idx));
      if (output_tv && vectorizable_io_tvs.contains(output_tv)) {
        output_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  // Vectorize LDG input loads (global -> register)
  if (pparams->vectorization_factor > 1) {
    for (auto ldg_tv : ldg_tvs) {
      if (vectorizable_io_tvs.contains(ldg_tv)) {
        ldg_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  // ========== Phase 8: Inline ==========
  // Inline TMA and LDG loads right after the last block parallelization axis.
  // This ensures global memory loads are issued early, before computations.
  // LDG tensors are typically broadcasts with minimal register usage.
  auto getLastBlockParallelizationAxisPosition = [](TensorView* tv) -> int64_t {
    for (auto i : arange(tv->nDims()) | std::views::reverse) {
      if (tv->axis(i)->isBlockDim()) {
        return i + 1;
      }
    }
    return 0;
  };
  std::vector<TensorView*> tma_or_ldg_tvs(tma_tvs.begin(), tma_tvs.end());
  tma_or_ldg_tvs.insert(tma_or_ldg_tvs.end(), ldg_tvs.begin(), ldg_tvs.end());
  for (auto tv : tma_or_ldg_tvs) {
    int64_t inline_pos = getLastBlockParallelizationAxisPosition(tv);
    NVF_ERROR_LT(inline_pos, tv->nDims());
    tv->inlineAt(inline_pos);
  }
  // inline other tensors to minimize register pressure
  std::vector<TensorView*> compute_tvs = ir_utils::allTvsExcept(
      fusion, {tma_or_ldg_tvs.begin(), tma_or_ldg_tvs.end()});
  inlineMost(compute_tvs);
}

} // namespace tma
} // namespace pointwise
} // namespace nvfuser
