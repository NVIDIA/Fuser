// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/pointwise_utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ir/printer.h>
#include <multidevice/utils.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>

namespace nvfuser {
namespace pointwise_utils {

TensorView* getReferenceTensor(Fusion* fusion) {
  FusionGuard fg(fusion);
  scheduler_tools::PointwiseDomainMap domain_map(fusion);
  auto reference_tv = domain_map.findReferenceTensor();
  return reference_tv;
}

TensorView* getLargestOutputTensor(
    Fusion* fusion,
    HeuristicDataCache* data_cache) {
  auto domain_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::DomainMap>(
          data_cache, [fusion]() {
            return std::make_unique<scheduler_tools::PointwiseDomainMap>(
                fusion);
          });
  const auto& domain_map = dynamic_cast<scheduler_tools::PointwiseDomainMap&>(
      domain_map_entry.get());

  auto largest_out_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensors>(
          data_cache, [&domain_map]() {
            std::vector<TensorView*> data{domain_map.findReferenceTensor()};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  TensorView* largest_out = largest_out_entry.get()[0];

  NVF_ERROR(largest_out != nullptr);
  return largest_out;
}

std::vector<IterDomain*> getReferenceLoop(
    TensorView* largest_out,
    bool has_reshapes,
    HeuristicDataCache* data_cache) {
  std::unordered_map<int64_t, int64_t> loop_reorder_map;
  if (!has_reshapes) {
    loop_reorder_map = scheduler_utils::reorderLoopAsAllocationMap(largest_out);
  }

  std::vector<IterDomain*> ref_loop = largest_out->getLoopDomain();
  // reorder of root to align with logical map should always help with indexing,
  // even when vectorization isn't used.
  if (!loop_reorder_map.empty()) {
    ref_loop = TensorDomain::orderedAs(ref_loop, loop_reorder_map);
  }
  // We always cacheBefore output at the beginning of the scheduling. And after
  // cacheBefore, the reference tensor will have all reduction IDs removed.
  ref_loop = TensorDomain::noDevices(TensorDomain::noReductions(ref_loop));

  return ref_loop;
}

std::pair<std::vector<int64_t>, int64_t> getElementCounts(
    const std::vector<IterDomain*>& ref_loop,
    SchedulerRuntimeInfo& runtime_info) {
  std::vector<int64_t> elem_counts;
  elem_counts.reserve(ref_loop.size());
  int64_t n_elems = 1;
  for (IterDomain* ref_id : ref_loop) {
    auto extent_pvalue =
        runtime_info.expressionEvaluator().evaluate(ref_id->extent());
    NVF_ERROR(
        extent_pvalue.hasValue(),
        "Error inferring size for pointwise scheduler: ",
        ref_id->extent()->toInlineString());
    auto extent = extent_pvalue.as<int64_t>();
    elem_counts.push_back(extent);
    n_elems *= extent;
  }
  return {elem_counts, n_elems};
}

std::unordered_map<int64_t, int64_t> getLogicalReorderMap(
    TensorView* largest_out,
    bool has_reshapes,
    HeuristicDataCache* data_cache) {
  auto logical_reorder_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::LogicalReorderMap>(
          data_cache, [largest_out, has_reshapes]() {
            // NOTE: reorder_map is only applied for fusion without view
            // op yet.
            if (has_reshapes) {
              return std::make_unique<std::unordered_map<int64_t, int64_t>>();
            }
            return std::make_unique<std::unordered_map<int64_t, int64_t>>(
                scheduler_utils::reorderLogicalAsAllocationMap(largest_out));
          });
  return logical_reorder_map_entry.get();
}

std::optional<FusionRuntimeProperties> getFusionRuntimeProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionRuntimeProperties prop;

  prop.largest_out = getLargestOutputTensor(fusion, data_cache);

  prop.device_multiprocessor_count = static_cast<int64_t>(
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  prop.has_reshapes = !ir_utils::getReshapeOps(fusion).empty();

  prop.ref_loop =
      getReferenceLoop(prop.largest_out, prop.has_reshapes, data_cache);

  auto [elem_counts, n_elems] = getElementCounts(prop.ref_loop, runtime_info);
  prop.elem_counts = std::move(elem_counts);
  prop.n_elems = n_elems;

  // Check for zero dimensional or zero size tensors
  if (std::ranges::empty(
          prop.largest_out->getLoopDomain() | TensorDomain::kNoReductions |
          TensorDomain::kNoBroadcasts | TensorDomain::kNoDevices) ||
      prop.n_elems == 0) {
    return std::nullopt;
  }

  // Get vectorizable inputs/outputs using cache
  auto vectorizable_inputs_outputs_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizableInputsAndOutputs>(
      data_cache, [&prop]() {
        return std::make_unique<std::vector<TensorView*>>(
            scheduler_utils::getInputsOutputsWithInnerDim(
                prop.largest_out, true, true));
      });
  prop.vectorizable_inputs_outputs = vectorizable_inputs_outputs_entry.get();

  // Calculate max and min dtype size for vectorization
  const auto index_type = runtime_info.getIndexType();
  int64_t max_dtype_size_bit = 0;
  int64_t min_dtype_size_bit = std::numeric_limits<int64_t>::max();

  for (auto tv : prop.vectorizable_inputs_outputs) {
    int64_t dtype_size_bit =
        dataTypeSizeBit(tv->getDataType().value(), index_type);
    max_dtype_size_bit = std::max(max_dtype_size_bit, dtype_size_bit);
    min_dtype_size_bit = std::min(min_dtype_size_bit, dtype_size_bit);
  }

  // If no vectorizable inputs/outputs, set default values
  // This prevents having a too large vectorization factor.
  if (max_dtype_size_bit == 0) {
    max_dtype_size_bit = 8;
  }
  if (min_dtype_size_bit == std::numeric_limits<int64_t>::max()) {
    min_dtype_size_bit = 8;
  }

  prop.max_dtype_size_bit_for_vectorization = max_dtype_size_bit;
  prop.min_dtype_size_bit_for_vectorization = min_dtype_size_bit;
  prop.index_type = index_type;

  return prop;
}

BreakPointInfo getBreakPoint(
    Fusion* fusion,
    const FusionRuntimeProperties& prop,
    HeuristicDataCache* data_cache,
    int64_t max_vect_factor,
    int64_t kThreadX) {
  BreakPointInfo result;

  // Calculate dtype_sum_bit from fusion inputs/outputs
  int64_t dtype_sum_bit = 0;
  const auto index_type = prop.index_type;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    dtype_sum_bit += dataTypeSizeBit(inp->getDataType().value(), index_type);
  }
  for (auto out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    dtype_sum_bit += dataTypeSizeBit(out->getDataType().value(), index_type);
  }

  // Get broadcast information
  TensorView* largest_out = prop.largest_out;
  auto broadcast_info_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::BroadcastMultiples>(
      data_cache, [&largest_out, &index_type]() {
        return std::make_unique<scheduler_utils::BroadcastMultipleInformation>(
            scheduler_utils::getBroadcastMultiples(largest_out, index_type));
      });
  const auto& broadcast_info = broadcast_info_entry.get();

  const auto& ref_loop = prop.ref_loop;
  const auto& elem_counts = prop.elem_counts;
  const int64_t n_elems = prop.n_elems;
  const auto& view_disjoint_sets = broadcast_info.view_disjoint_set_ids;
  const auto& broadcast_bit_multiples = broadcast_info.broadcast_multiples;

  // Default values for 1D scheduling
  result.break_point = 0;
  result.flip_grid_binding = false;
  result.right_elem_count = 0;
  result.is_outer_broadcast_dominated = false;

  // Figure out break point position
  // How much would this transfer cost if it was done as a 1-D schedule
  int64_t transfer_size_1d_bit = 1;

  for (const auto i : arange(ref_loop.size())) {
    transfer_size_1d_bit =
        transfer_size_1d_bit * elem_counts[i] * dtype_sum_bit;
  }

  // Calculate optimal break point for 2D scheduling
  int64_t min_total_transfer_bit = std::numeric_limits<int64_t>::max();
  // Don't check the inner most dimension, scheduler assumes there's always
  // an rhs
  for (const auto break_point_i : arange((int64_t)ref_loop.size())) {
    // If break point is incoherent with view, don't consider breaking here.
    if (!scheduler_utils::breakIsDisjoint(view_disjoint_sets, break_point_i)) {
      continue;
    }

    // Number of elements in the right side of reference tv with
    // break_point_i
    int64_t cur_right_elem_count = 1;
    for (const auto right_i : arange(break_point_i, ref_loop.size())) {
      cur_right_elem_count = cur_right_elem_count * elem_counts[right_i];
    }

    auto cur_left_elem_count = n_elems / cur_right_elem_count;
    if (cur_left_elem_count <= 1) {
      continue;
    }

    auto lhs_bit_multiple = broadcast_bit_multiples[break_point_i].lhs_multiple;
    auto rhs_bit_multiple = broadcast_bit_multiples[break_point_i].rhs_multiple;

    // Estimate transfer cost with this break point
    int64_t cur_transfer_size_bit = 1;
    int64_t right_transfer_size_bit = 1;

    for (const auto left_i : arange(break_point_i)) {
      cur_transfer_size_bit =
          cur_transfer_size_bit * elem_counts[left_i] * lhs_bit_multiple;
    }

    for (const auto right_i : arange(break_point_i, ref_loop.size())) {
      right_transfer_size_bit =
          right_transfer_size_bit * elem_counts[right_i] * rhs_bit_multiple;
    }
    cur_transfer_size_bit *= right_transfer_size_bit;

    //  Continue if this break point doesn't save at least 10% of 1D
    //  scheduling or isn't better than previous break_points found.
    if (cur_transfer_size_bit >= min_total_transfer_bit ||
        cur_transfer_size_bit * 10 >= transfer_size_1d_bit * 9) {
      continue;
    }

    // Need to be able to parallelize, don't use break if there's not
    // at least an unrolled warp.
    if (ceilDiv(cur_right_elem_count, max_vect_factor) <=
        at::cuda::getCurrentDeviceProperties()->warpSize) {
      continue;
    }
    // If outer broadcast, or balanced broadcast:
    if (lhs_bit_multiple <= rhs_bit_multiple &&
        // If right transfer size is bigger than half of L2
        at::cuda::getCurrentDeviceProperties()->l2CacheSize * 8 <
            right_transfer_size_bit * 2) {
      // flip BIDx and BIDy bindings
      result.flip_grid_binding = true;
    } else {
      result.flip_grid_binding = false;
    }

    // Use this break point
    result.break_point = static_cast<int>(break_point_i);
    min_total_transfer_bit = cur_transfer_size_bit;
    result.right_elem_count = cur_right_elem_count;

    // when lhs byte multiple is smaller than rhs byte multiple,
    // there is broadcast in the lhs, which is outer broadcast.
    result.is_outer_broadcast_dominated = lhs_bit_multiple < rhs_bit_multiple;
  }

  return result;
}

BlockGridConfig getBlockGridConfig(
    const FusionRuntimeProperties& prop,
    const BreakPointInfo& bp_info,
    int64_t max_vect_factor,
    int64_t kThreadX) {
  BlockGridConfig result;

  // Copy break point information
  result.break_point = bp_info.break_point;
  result.flip_grid_binding = bp_info.flip_grid_binding;
  result.right_elem_count = bp_info.right_elem_count;
  result.is_outer_broadcast_dominated = bp_info.is_outer_broadcast_dominated;

  // Default thread/grid dimensions for 1D scheduling
  result.bdimx = kThreadX;
  result.bdimy = 1;
  result.gdim_left = 1;
  result.gdim_right = 1;

  // Calculate thread/grid dimensions if we have a 2D break point
  if (result.break_point > 0 && result.right_elem_count > 0) {
    const int64_t n_elems = prop.n_elems;
    int64_t threads_per_warp =
        (int64_t)at::cuda::getCurrentDeviceProperties()->warpSize;

    int64_t cur_left_elem_count = n_elems / result.right_elem_count;

    // Start bdimx with 1 warp, increase if split is divisible
    int64_t after_vect = ceilDiv(result.right_elem_count, max_vect_factor);
    result.bdimx = std::min(after_vect, threads_per_warp);
    while (result.bdimx * 2 <= kThreadX && result.bdimx * 2 <= after_vect &&
           after_vect % (result.bdimx * 2) == 0) {
      result.bdimx *= 2;
    }
    result.bdimy = kThreadX / result.bdimx;
    result.gdim_left = ceilDiv(cur_left_elem_count, result.bdimy);
    result.gdim_right =
        ceilDiv(result.right_elem_count, result.bdimx * max_vect_factor);
  }

  return result;
}

std::optional<CommonScheduleInfo> commonPointwiseSchedule(
    Fusion* fusion,
    int64_t break_point) {
  CommonScheduleInfo info;

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // Cache inputs
  info.cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  info.cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  refineCachePolicy(fusion);

  // Filter and collect input tensors (excluding hanging tensors)
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      info.input_tvs.push_back(tv);
    }
  }
  auto output_tvs_view = ir_utils::filterByType<TensorView>(fusion->outputs());
  info.output_tvs =
      std::vector<TensorView*>(output_tvs_view.begin(), output_tvs_view.end());

  // Find maximum dimensions across all inputs and outputs
  int64_t max_dims = 0;
  for (auto inp : info.input_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(inp), max_dims);
  }

  for (auto out : info.output_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(out), max_dims);
  }

  // If everything is zero dim tensors, return nullopt
  if (max_dims == 0) {
    return std::nullopt;
  }

  // Get reference tensor
  info.reference_tv = getReferenceTensor(fusion);
  NVF_ERROR(
      info.reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");
  auto ref_orig_loop = info.reference_tv->getLoopDomain();

  // Move non-concretized broadcasts innermost
  scheduler_utils::moveNonConcretizedBroadcastInnermost(
      fusion, {info.reference_tv});
  int64_t num_device_dims = numDeviceDims(info.reference_tv);
  int64_t device_aware_break_point = break_point + num_device_dims;

  // Positions of rhs and lhs after merging all dimensions.
  int64_t rhs_i = -1;
  int64_t lhs_i = -1;

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    info.reference_tv->reorder(
        scheduler_utils::domainReorderAsLogicalMap(info.reference_tv));
    // Reorder so that DeviceDims are in front
    reorderParallelizedToFront(info.reference_tv);

    // Break point is relative to logical domain, find the loop domain ID's in
    // the left/right side, we really need the values in domain, but easiest way
    // to do this is with Dependency check which will grab all intermediate
    // values too.
    auto lhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + num_device_dims,
         ref_orig_loop.begin() + device_aware_break_point},
        {info.reference_tv->getLoopDomain().begin() + num_device_dims,
         info.reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> lhs_all_vals_set(
        lhs_all_vals.begin(), lhs_all_vals.end());

    auto rhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + device_aware_break_point, ref_orig_loop.end()},
        {info.reference_tv->getLoopDomain().begin() + num_device_dims,
         info.reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> rhs_all_vals_set(
        rhs_all_vals.begin(), rhs_all_vals.end());

    // Make sure lhs and rhs groups are disjoint.
    for (auto lhs_val : lhs_all_vals) {
      if (rhs_all_vals_set.count(lhs_val) != 0) {
        std::ostringstream os;
        IrTransformPrinter printer(os);
        printer.printTransforms(info.reference_tv);
        NVF_THROW(
            "Error in pointwise scheduler. LHS and RHS of the 2D scheduler are "
            "not disjoint. ",
            lhs_val->toString(),
            " belongs to both. device_aware_break_point = ",
            device_aware_break_point,
            ". reference_tv = ",
            info.reference_tv->toString(),
            " and its transforms are:\n",
            os.str());
      }
    }
    NVF_ERROR(
        !rhs_all_vals.empty(),
        "Expecting at least one dimension in the RHS of the pointwise "
        "scheduler.");

    // Merge rhs, then lhs.
    IterDomain* rhs_id = nullptr;
    IterDomain* lhs_id = nullptr;
    for (int64_t pos = info.reference_tv->nDims() - 1; pos >= 0; pos--) {
      // Merge from right to left
      auto id = info.reference_tv->axis(pos);
      if (lhs_all_vals_set.count(id) > 0) {
        if (lhs_id == nullptr) {
          lhs_id = id;
          lhs_i = pos;
        } else {
          info.reference_tv->merge(pos, lhs_i);
          lhs_i = pos;
          if (rhs_i > lhs_i) {
            rhs_i--;
          }
        }
      } else if (rhs_all_vals_set.count(id) > 0) {
        if (rhs_id == nullptr) {
          rhs_id = id;
          rhs_i = pos;
        } else {
          info.reference_tv->merge(pos, rhs_i);
          rhs_i = pos;
          if (lhs_i > rhs_i) {
            lhs_i--;
          }
        }
      }
    }
    // Find the iter domains that should be in the lhs, and rhs.
  } else {
    // Don't need to worry about view transformations, just merge reference tv
    // as we normally would.

    std::unordered_map<int64_t, int64_t> loop_reorder_map =
        scheduler_utils::reorderLoopAsAllocationMap(info.reference_tv);
    if (!loop_reorder_map.empty()) {
      info.reference_tv->reorder(loop_reorder_map);
    }
    reorderParallelizedToFront(info.reference_tv);

    // Merge right side of break point
    for (int64_t i = info.reference_tv->nDims(); i > device_aware_break_point;
         i--) {
      auto axis_i = i - 1;
      if (rhs_i == -1) {
        rhs_i = axis_i;
      } else {
        info.reference_tv->merge(axis_i, rhs_i);
        rhs_i = axis_i;
      }
    }
    if (rhs_i >= 0) {
      // If there's an rhs
      info.reference_tv->reorder({{rhs_i, -1}});
    }

    // Merge left side of break point
    for (int64_t i = device_aware_break_point; i > num_device_dims; i--) {
      auto axis_i = i - 1;
      if (lhs_i == -1) {
        lhs_i = axis_i;
      } else {
        info.reference_tv->merge(axis_i, lhs_i);
        lhs_i = axis_i;
      }
    }
  }

  // Store the positions of lhs and rhs merged dimensions
  info.lhs_i = lhs_i;
  info.rhs_i = rhs_i;

  return info;
}

} // namespace pointwise_utils
} // namespace nvfuser
