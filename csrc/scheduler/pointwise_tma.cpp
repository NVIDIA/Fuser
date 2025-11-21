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
#include <transform_iter.h>
#include <transform_replay.h>

namespace nvfuser {
namespace pointwise {
namespace tma {

int64_t getMinDtypeBitsOfTmaCompatibleInputs(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<int64_t>& tma_compatible_input_indices) {
  int64_t min_dtype_bits = std::numeric_limits<int64_t>::max();
  for (const auto& input_idx : tma_compatible_input_indices) {
    auto tv = dynamic_cast<TensorView*>(fusion->inputs().at(input_idx));
    min_dtype_bits = std::min(
        min_dtype_bits,
        dataTypeSizeBit(
            tv->getDataType().value(), runtime_info.getIndexType()));
  }
  return min_dtype_bits;
}
// Returns the total bits required to load one element from each TMA-loaded
// input. This is used to determine how many elements should be loaded from each
// input to achieve the required bits in flight.
int64_t getInputBitsPerElement(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<int64_t>& tma_compatible_input_indices) {
  int64_t bits_per_element = 0;
  for (const auto& input_idx : tma_compatible_input_indices) {
    auto tv = dynamic_cast<TensorView*>(fusion->inputs().at(input_idx));
    NVF_ERROR(
        tv != nullptr,
        "Input tensor at index ",
        input_idx,
        " is not a TensorView");
    bits_per_element +=
        dataTypeSizeBit(tv->getDataType().value(), runtime_info.getIndexType());
  }
  return bits_per_element;
}

std::vector<int64_t> getTmaCompatibleInputIndices(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* largest_out,
    int64_t break_point) {
  int64_t n_valid_dims = scheduler_utils::nLogicalDims(largest_out);
  std::vector<int64_t> tma_compatible_input_indices;
  for (auto [input_idx, input] : enumerate(fusion->inputs())) {
    auto tv = dynamic_cast<TensorView*>(input);
    if (!tv) {
      continue;
    }
    // must be cacheable
    if (scheduler_utils::getCacheableUses(tv).empty()) {
      continue;
    }

    // must be suitable for TMA based on the number of elements and dtype size
    if (!scheduler_utils::isTvSizeSuitableForTma(tv, runtime_info)) {
      continue;
    }

    // must have the same number of logical dimensions as the reference tensor
    // to avoid loading tensors that are smaller than the reference tensor
    if (scheduler_utils::nLogicalDims(tv) != n_valid_dims) {
      continue;
    }

    // must be contiguous
    const auto contiguity = tv->domain()->contiguity();
    if (std::any_of(
            contiguity.begin(),
            contiguity.end(),
            [](const std::optional<bool>& contiguity) {
              return !contiguity.has_value() || !contiguity.value();
            })) {
      continue;
    }

    // break point separates the reference tv into [lhs, rhs]
    // then we use 2D tile, [lhs/outer, outer, rhs/inner, inner]
    // To use TMA, this tv must have both lhs and rhs.
    // see PointwiseTest.BroadcastAddInner for example.
    if ((int64_t)tv->getLoopDomain().size() <= break_point) {
      continue;
    }
  }
  return tma_compatible_input_indices;
}

std::unique_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  // Hardware constants
  constexpr int64_t threads_per_warp = 32;
  constexpr int64_t max_size_per_tma_tile_dim = 256;
  auto params = std::make_unique<PointwiseParams>();
  params->tag = "Pointwise TMA heuristics";
  params->cparams.index_type = prop.index_type;
  params->use_tma_load = true;

  auto tma_compatible_input_indices = getTmaCompatibleInputIndices(
      fusion, runtime_info, prop.largest_out, params->break_point);
  if (tma_compatible_input_indices.empty()) {
    return nullptr;
  } else {
    params->tma_compatible_input_indices = tma_compatible_input_indices;
  }

  // If element count with the smallest dtype size is suitable for TMA, then
  // it is also suitable for the other dtype sizes.
  constexpr int64_t target_inner_tma_domain_size = 512;
  const int64_t min_dtype_bits = getMinDtypeBitsOfTmaCompatibleInputs(
      fusion, runtime_info, tma_compatible_input_indices);
  int64_t tma_domain_inner = scheduler_utils::getInnerTmaDomainSize(
      prop.n_elems, target_inner_tma_domain_size, min_dtype_bits);
  if (tma_domain_inner == 1 || prop.n_elems % tma_domain_inner != 0) {
    return nullptr;
  }
  // constexpr int64_t align_bytes = 16;
  // const int64_t min_size = 2 * align_bytes / min_dtype_bytes;
  // if(tma_domain_inner % min_size != 0){
  //   return nullptr;
  // }

  const int64_t tma_outer_domain_size = prop.n_elems / tma_domain_inner;
  params->tma_domain_inner = tma_domain_inner;

  auto bp_info = pointwise_utils::getBreakPoint(
      fusion, prop, data_cache, /*is_tma =*/true);
  params->break_point = bp_info.break_point;

  // Compute elements_per_cta: Each CTA issues one TMA load operation. We
  // calculate the number of elements per TMA load based on the required bits
  // in flight, assuming 8 CTAs per SM. This is a guideline; the actual tile
  // size is determined by tma_tile_inner and tma_tile_outer.
  // - Inner tile size: ensure at least 2 tiles in the inner TMA dimension
  // dimension. outer tile size: don't exceed the outer TMA dimension size Both
  // Both are subject to hardware constraints of 256 elements per dimension.
  constexpr int64_t cta_per_sm = 8;
  int64_t bits_per_sm = scheduler_utils::getRequiredBitsInFlight();
  int64_t bits_per_cta = bits_per_sm / cta_per_sm;
  int64_t bits_per_element = getInputBitsPerElement(
      fusion, runtime_info, params->tma_compatible_input_indices);
  int64_t elements_per_cta = ceilDiv(bits_per_cta, bits_per_element);
  elements_per_cta = scheduler_utils::roundUpToN(elements_per_cta, 1024);
  int64_t max_tma_tile_inner =
      std::min(tma_domain_inner / 2, max_size_per_tma_tile_dim);
  int64_t max_tma_tile_outer =
      std::min(tma_outer_domain_size, max_size_per_tma_tile_dim);
  int64_t tma_tile_inner = std::min(tma_domain_inner / 2, threads_per_warp);
  while (tma_tile_inner * 2 <= max_tma_tile_inner) {
    tma_tile_inner *= 2;
  }
  int64_t tma_tile_outer = std::max(
      1L, std::min(elements_per_cta / tma_tile_inner, max_tma_tile_outer));
  params->tma_tile_inner = tma_tile_inner;
  params->tma_tile_outer = tma_tile_outer;

  // Set block dimensions: typical configuration is 32 threads in x-dimension
  // and 4 threads in y-dimension, but constrain to TMA tile size in each
  // dimension.
  constexpr int64_t threads_per_cta = 128;
  int64_t bdimx = std::min(threads_per_warp, tma_tile_inner);
  int64_t bdimy = std::min(threads_per_cta / bdimx, tma_tile_outer);
  params->lparams.bindUnsafe(bdimx, ParallelType::TIDx);
  params->lparams.bindUnsafe(bdimy, ParallelType::TIDy);

  // Set vectorization factor for global memory <-> register transfers.
  // The [tma_tile_inner] dimension is scheduled as [S, TIDx, Vect], so the
  // vectorization factor cannot exceed tma_tile_inner / bdimx.
  NVF_ERROR(
      tma_tile_inner % bdimx == 0, "tma_tile_inner must be divisible by bdimx");
  constexpr int64_t max_vectorization_size_in_bit = 128;
  int64_t vect_factor_dtype =
      max_vectorization_size_in_bit / prop.max_dtype_size_bit_for_vectorization;
  int64_t vect_factor_tma_tile_size = tma_tile_inner / bdimx;
  params->vectorization_factor =
      std::min(vect_factor_dtype, vect_factor_tma_tile_size);

  // TMA store
  params->use_tma_store = false;

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n==== Pointwise TMA Scheduler Heuristics ====\n";
    debug() << "Domain sizes:\n";
    debug() << "  n_elems: " << prop.n_elems << "\n";
    debug() << "  break_point: " << bp_info.break_point << "\n";
    debug() << "  tma_domain_inner: " << tma_domain_inner << "\n";
    debug() << "  tma_outer_domain_size: " << tma_outer_domain_size << "\n";
    debug() << "\nMemory and CTA configuration:\n";
    debug() << "  cta_per_sm: " << cta_per_sm << "\n";
    debug() << "  bits_per_sm: " << bits_per_sm << "\n";
    debug() << "  bits_per_cta: " << bits_per_cta << "\n";
    debug() << "  bits_per_element: " << bits_per_element << "\n";
    debug() << "  elements_per_cta: " << elements_per_cta << "\n";
    debug() << "\nTMA tile configuration:\n";
    debug() << "  max_size_per_tma_tile_dim: " << max_size_per_tma_tile_dim
            << "\n";
    debug() << "  max_tma_tile_inner: " << max_tma_tile_inner << "\n";
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
    debug() << "  max_vectorization_size_in_bit: "
            << max_vectorization_size_in_bit << "\n";
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

  // Always merge all dimensions without considering the break point. The
  // break point effect can be equivalently handled by setting TMA domain sizes.
  auto schedule_info_opt =
      pointwise_utils::commonPointwiseSchedule(fusion, pparams->break_point);
  if (!schedule_info_opt.has_value()) {
    // Zero-dimensional tensors, nothing to schedule
    return;
  }
  auto& schedule_info = schedule_info_opt.value();

  auto& cached_inputs = schedule_info.cached_inputs;
  auto& cached_outputs = schedule_info.cached_outputs;
  TensorView* reference_tv = schedule_info.reference_tv;
  auto inputs_outputs =
      scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true, true);
  std::unordered_set<TensorView*> vectorizable_io_tvs(
      inputs_outputs.begin(), inputs_outputs.end());

  std::cout << "reference_tv: " << reference_tv->toString() << std::endl;
  reference_tv->printTransforms();

  // For each cached input: use TMA load if it has the same number of logical
  // dimensions as the reference tensor; otherwise, use LDG (standard load).
  int64_t n_valid_dims = scheduler_utils::nLogicalDims(reference_tv);
  std::vector<TensorView*> tma_tvs;
  std::vector<TensorView*> ldg_tvs;
  const auto& tma_indices = pparams->tma_compatible_input_indices;
  for (const auto& [tv, input_idx] : cached_inputs) {
    if (std::find(tma_indices.begin(), tma_indices.end(), input_idx) ==
        tma_indices.end()) {
      ldg_tvs.push_back(tv);
      continue;
    }
    auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition());
    if (load_op) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulkTensorTile);
    }
    tv->setMemoryType(MemoryType::Shared);
    tv->cacheAfter();
    tma_tvs.push_back(tv);
  }

  // Split the TMA domain: [I0] -> [Do, Di]
  if (pparams->break_point == 0) {
    reference_tv->split(0, pparams->tma_domain_inner);
  } else {
    NVF_ERROR(
        n_valid_dims >= 2,
        "Required at least 2 valid dimensions for Tma scheduling, but got ",
        n_valid_dims);
  }

  // Split into TMA tiles (box/tile sizes)
  // [Do, Di] -> [Do/to, to, Di/ti, ti] -> [Do/to, Di/ti, to, ti]
  reference_tv->split(1, pparams->tma_tile_inner);
  reference_tv->split(0, pparams->tma_tile_outer);
  // reorder for better inline
  reference_tv->reorder({{1, 2}});

  // Propagate the TMA-related transformations to all tensors
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);

  // Parallelize TMA tensors. After propagation, reset the reference tensor
  // back to serial mode to enable further scheduling of non-TMA parts.
  auto outer_cord_pt = ParallelType::BIDy;
  auto inner_cord_pt = ParallelType::BIDx;
  if (pparams->flip_grid_binding) {
    std::swap(outer_cord_pt, inner_cord_pt);
  }
  reference_tv->axis(0)->parallelize(outer_cord_pt);
  reference_tv->axis(1)->parallelize(inner_cord_pt);
  reference_tv->axis(2)->parallelize(ParallelType::Bulk);
  reference_tv->axis(3)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(reference_tv, tma_tvs);
  reference_tv->axis(2)->parallelize(ParallelType::Serial);
  reference_tv->axis(3)->parallelize(ParallelType::Serial);

  // Schedule the non-TMA parts, starting with [Do/to, to, Di/ti, ti]
  // Transform: [Do/to, Di/ti, to, ti] -> [Do/to, Di/ti, to/y, y, ti/v/x, x, v]
  int64_t opos = 2, ipos = 3;
  reference_tv->split(ipos, pparams->vectorization_factor);
  reference_tv->split(ipos, pparams->lparams.bdimx());
  reference_tv->split(opos, pparams->lparams.bdimy());
  // reorder for better inline
  // [Do/to, Di/ti, to/y, y, ti/v/x, x, v] -> [Do/to, Di/ti, to/y, ti/v/x, y, x,
  // v]
  reference_tv->reorder({{3, 4}});

  // Propagate transformations to non-TMA tensors
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&non_tma_propagator);

  reference_tv->axis(4)->parallelize(ParallelType::TIDy);
  reference_tv->axis(5)->parallelize(ParallelType::TIDx);
  int64_t vect_pos = 6; // Position for vectorization axis
  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  // Vectorize register -> global memory transfers
  if (!pparams->use_tma_store && pparams->vectorization_factor > 1) {
    for (const auto& [_, original_idx] : cached_outputs) {
      auto output_tv =
          dynamic_cast<TensorView*>(fusion->outputs().at(original_idx));
      if (output_tv && vectorizable_io_tvs.contains(output_tv)) {
        output_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  // Vectorize global memory -> register transfers
  for (auto ldg_tv : ldg_tvs) {
    if (vectorizable_io_tvs.contains(ldg_tv)) {
      ldg_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
    // unroll serial loop over tidx/y
    // inlineMost won't inline over unrolled domains, we can still use
    // inlineMost for ldg tvs. Issuing register loads at the beginning of the
    // kernel is more efficient compared to delaying them until they are used.
    // These non-TMA-loaded tensors are usually broadcast inputs with smaller
    // sizes, so they only slightly increase register usage. Performance
    // comparison (inlining most vs not inlining ldg_tvs):
    //   - Inline most: 29 registers, 100% occupancy, 53% SOL
    //   - Uninlined ldg_tvs: 32 registers, 100% occupancy, 88% SOL
    for (int idx = 0; idx < ldg_tv->nDims(); idx++) {
      if (ldg_tv->axis(idx)->isThreadDim() &&
          ldg_tv->axis(idx - 1)->getParallelType() == ParallelType::Serial) {
        ldg_tv->axis(idx - 1)->parallelize(ParallelType::Unroll);
      }
    }
  }

  inlineMost();
}

} // namespace tma
} // namespace pointwise
} // namespace nvfuser
