// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <debug.h>
#include <instrumentation.h>
#include <ir/printer.h>
#include <multidevice/utils.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/pointwise.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <scheduler/transpose.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>

namespace nvfuser {

namespace {
// constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
// Unused at the moment, commenting for clang tidy
constexpr int64_t kThreadX = 128;

class DomainMap : public pointwise_utils::DomainMap {
 public:
  using pointwise_utils::DomainMap::DomainMap;

  // The pointwise scheduler heuristics requires a minimum number of axes.
  // The output reference tensor should respect this requirement.
  TensorView* findReferenceTensorView(int64_t minimum_num_axes = 0) const {
    TensorView* result = nullptr;
    int64_t max_dims = -1;
    for (auto output_tv :
         ir_utils::filterByType<TensorView>(fusion_->outputs())) {
      std::cerr << "findRef. " << output_tv->toString()
                << ": isValidReference: " << isValidReference(output_tv)
                << ", hasMinimum: "
                << hasMinimumSize(output_tv, minimum_num_axes)
                << ", !isInput:" << !output_tv->isFusionInput() << "\n";
      if (isValidReference(output_tv) &&
          hasMinimumSize(output_tv, minimum_num_axes) &&
          !output_tv->isFusionInput()) {
        int64_t n_dims = pointwise_utils::nRootDims(output_tv);
        if (n_dims > max_dims) {
          result = output_tv;
          max_dims = n_dims;
        }
      }
    }
    return result;
  }

 private:
  bool hasMinimumSize(TensorView* tv, int64_t num_axes) const {
    NVF_ERROR(tv != nullptr);
    return (num_axes == 0 || (int64_t)tv->getLogicalDomain().size() > num_axes);
  }
};

TensorView* getAllocationReferenceTensor(Fusion* fusion) {
  TensorView* reference = nullptr;
  for (auto input_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    if (reference == nullptr) {
      reference = input_tv;
      continue;
    }

    if (input_tv->getLogicalDomain().size() >
        reference->getLogicalDomain().size()) {
      reference = input_tv;
      continue;
    }

    if (TensorDomain::noBroadcasts(input_tv->getLogicalDomain()).size() >
        TensorDomain::noBroadcasts(reference->getLogicalDomain()).size()) {
      reference = input_tv;
      continue;
    }
  }

  std::cerr << "Reference TV: " << reference->toString() << ", allocation: "
            << toDelimitedString(reference->getMaybeAllocationDomain()) << "\n";
  return reference;
}

TensorView* getReferenceTensor(Fusion* fusion) {
  TensorView* ref = nullptr;
  for (auto expr : fusion->exprs()) {
    auto cat = dynamic_cast<CatOp*>(expr);
    if (cat == nullptr) {
      continue;
    }

    auto cat_output = cat->output(0)->as<TensorView>();

    ref = cat_output;
    break;
  }

  if (ref == nullptr) {
    ref = getAllocationReferenceTensor(fusion);
  }

  std::cerr << "Reference TV: " << ref->toString() << ", allocation: "
            << toDelimitedString(ref->getMaybeAllocationDomain()) << "\n";
  return ref;
}

} // namespace

std::unique_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  // Incase any buffer is of type DataType::Index
  const auto index_type = runtime_info.getIndexType();

  auto params = std::make_unique<PointwiseParams>();
  params->tag = "Pointwise heuristics";
  params->cparams.index_type = index_type;

  auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());

  auto domain_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::DomainMap>(
          data_cache,
          [fusion]() { return std::make_unique<DomainMap>(fusion); });
  const auto& domain_map = dynamic_cast<DomainMap&>(domain_map_entry.get());

  auto largest_out_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensors>(
          data_cache, [&domain_map]() {
            std::vector<TensorView*> data{domain_map.findReferenceTensorView()};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  TensorView* largest_out = largest_out_entry.get()[0];

  if (!getenv("USE_LARGEST_OUT")) {
    auto cur_ref = largest_out;
    largest_out = getReferenceTensor(fusion);
    std::cerr << "Reference: Using " << largest_out->toString()
              << " instead of " << cur_ref->toString() << "\n";
  }

  NVF_ERROR(largest_out != nullptr);

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // TODO: Set to 1?
  int64_t max_input_dtype_size = 2;

  for (auto inp : in_tvs) {
    max_input_dtype_size = std::max(
        max_input_dtype_size,
        (int64_t)dataTypeSize(inp->getDataType().value(), index_type));
  }

  auto logical_reorder_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::LogicalReorderMap>(
          data_cache, [&fusion, &largest_out]() {
            // NOTE: logical_reorder_map is only applied for fusion without view
            // op yet.
            if (!ir_utils::getViewOps(fusion).empty()) {
              return std::make_unique<std::unordered_map<int64_t, int64_t>>();
            }
            return std::make_unique<std::unordered_map<int64_t, int64_t>>(
                scheduler_utils::maybeLogicalReorderAsAllocationMap(
                    largest_out));
          });
  const std::unordered_map<int64_t, int64_t>& logical_reorder_map =
      logical_reorder_map_entry.get();

  auto ref_root = largest_out->getLogicalDomain();
  // reorder of root to align with logical map should always help with indexing,
  // even when vectorization isn't used.
  if (!logical_reorder_map.empty()) {
    ref_root = TensorDomain::orderedAs(ref_root, logical_reorder_map);
  }
  // We always cacheBefore output at the beginning of the scheduling. And after
  // cacheBefore, the reference tensor will have all reduction IDs removed.
  ref_root = TensorDomain::noDevices(TensorDomain::noReductions(ref_root));

  std::vector<int64_t> elem_counts(ref_root.size(), 1);
  int64_t n_elems = 1;
  for (size_t ref_i = 0; ref_i < ref_root.size(); ref_i++) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(ref_root[ref_i]->extent());
    NVF_ERROR(
        inferred_val.hasValue(),
        "Error inferring size for pointwise scheduler: ",
        ref_root[ref_i]->extent()->toInlineString());
    elem_counts[ref_i] = inferred_val.as<int64_t>();
    n_elems *= elem_counts[ref_i];
  }

  // If zero dimensional or zero size, return default parameters
  if (TensorDomain::noDevices(
          TensorDomain::noReductions(
              TensorDomain::noBroadcasts(largest_out->getLoopDomain())))
          .empty() ||
      n_elems == 0) {
    auto vectorizable_inputs_outputs_entry = HeuristicDataCacheEntry<
        HeuristicCompileTime::VectorizableInputsAndOutputs>(data_cache, []() {
      return std::make_unique<std::vector<TensorView*>>();
    });
    vectorizable_inputs_outputs_entry.get();

    auto broadcast_info = HeuristicDataCacheEntry<
        HeuristicCompileTime::BroadcastMultiples>(data_cache, []() {
      return std::make_unique<scheduler_utils::BroadcastMultipleInformation>();
    });
    broadcast_info.get();

    vectorize_helper::getVectorizationFactor(
        runtime_info, largest_out, data_cache, 0);

    // All cache entries that are expected to be generated in the pointwise
    // scheduler by registry.cpp::HeuristicDataCache::validate() must be created
    // before hitting this return.
    auto pwise_params = std::make_unique<PointwiseParams>();
    pwise_params->tag = "Pointwise heuristics";
    pwise_params->cparams.index_type = index_type;
    return pwise_params;
  }

  // Find all vectorizable inputs/outputs
  auto vectorizable_inputs_outputs_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizableInputsAndOutputs>(
      data_cache, [&largest_out]() {
        return std::make_unique<std::vector<TensorView*>>(
            scheduler_utils::getInputsOutputsWithInnerDim(
                largest_out, true, true));
      });

  constexpr int64_t kSixteen = 16; // clang tidy

  auto max_vect_unroll_factor = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)kSixteen / max_input_dtype_size,
      // Reduce max unrolling factor if we have many inputs/outputs to unroll
      // as it could start consuming a lot of registers.
      std::max(
          (scheduler_utils::lastPow2(
               (int64_t)vectorizable_inputs_outputs_entry.get().size()) >>
           2),
          (int64_t)1));

  // Don't unroll at the cost of getting a full wave on the GPU
  if (n_elems < device_multiprocessor_count * kThreadX &&
      max_vect_unroll_factor > 1) {
    max_vect_unroll_factor = std::min(
        max_vect_unroll_factor,
        ceilDiv(n_elems, device_multiprocessor_count * kThreadX));
  }

  auto max_vect_factor =
      std::min(kSixteen / max_input_dtype_size, max_vect_unroll_factor);

  // See pointwise.h to understand what we're doing for this 2D analysis.
  // Ideal break point location
  int break_point = 0;

  // If break_point, mark if BIDy and BIDx should be positionally reversed
  // relative to root domains
  bool flip_grid_binding = false;

  // Elements on the right of break point (without break point all are on the
  // right)
  int64_t right_elem_count = 0;

  int64_t bdimx = kThreadX;

  // bdimy may be used if the right side of the break point is not large and we
  // need to expand block level parallelism into the left side of the break
  // point.
  int64_t bdimy = 1;

  // In 2D scheduler gdim_left is used to parallelize the left side of the break
  // point.
  int64_t gdim_left = 1;

  // gdim_right is used if there's too much parallelization in the right side of
  // the break point. We will expand grid parallelization into the right side of
  // the break point with gdim_left and use gdim_right for the left side of the
  // break point.
  int64_t gdim_right = 1;

  auto broadcast_info = HeuristicDataCacheEntry<
      HeuristicCompileTime::BroadcastMultiples>(
      data_cache, [&largest_out, &index_type]() {
        return std::make_unique<scheduler_utils::BroadcastMultipleInformation>(
            scheduler_utils::getBroadcastMultiples(largest_out, index_type));
      });

  auto& view_disjoint_sets = broadcast_info.get().view_disjoint_set_ids;
  auto& broadcast_byte_multiples = broadcast_info.get().broadcast_multiples;
  NVF_ERROR(broadcast_byte_multiples.size() == ref_root.size());

  int64_t dtype_sum = 0;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    dtype_sum += (int64_t)dataTypeSize(inp->getDataType().value(), index_type);
  }
  for (auto out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    dtype_sum += (int64_t)dataTypeSize(out->getDataType().value(), index_type);
  }

  { // Figure out break point position. Empty scope, consider moving to a
    // separate function.
    //
    // How much would this transfer cost if it was done as a 1-D schedule
    int64_t transfer_size_1d = 1;

    for (const auto i : c10::irange(ref_root.size())) {
      transfer_size_1d = transfer_size_1d * elem_counts[i] * dtype_sum;
    }

    // If there isn't very much parallelism available, just use 1D scheduler
    if (n_elems * 2 > device_multiprocessor_count * kThreadX) {
      int64_t min_total_transfer = std::numeric_limits<int64_t>::max();

      // Don't check the inner most dimension, scheduler assumes there's always
      // an rhs
      for (const auto break_point_i : c10::irange((int64_t)ref_root.size())) {
        // If break point is incoherent with view, don't consider breaking here.
        if (!scheduler_utils::breakIsDisjoint(
                view_disjoint_sets, break_point_i)) {
          continue;
        }

        // Number of elements in the right side of reference tv with
        // break_point_i
        int64_t cur_right_elem_count = 1;
        for (const auto right_i : c10::irange(break_point_i, ref_root.size())) {
          cur_right_elem_count = cur_right_elem_count * elem_counts[right_i];
        }

        auto cur_left_elem_count = n_elems / cur_right_elem_count;
        if (cur_left_elem_count <= 1) {
          continue;
        }

        auto lhs_byte_multiple =
            broadcast_byte_multiples[break_point_i].lhs_multiple;
        auto rhs_byte_multiple =
            broadcast_byte_multiples[break_point_i].rhs_multiple;

        // Estimate transfer cost with this break point
        int64_t cur_transfer_size = 1;
        int64_t right_transfer_size = 1;

        for (const auto left_i : c10::irange(break_point_i)) {
          cur_transfer_size =
              cur_transfer_size * elem_counts[left_i] * lhs_byte_multiple;
        }

        for (const auto right_i : c10::irange(break_point_i, ref_root.size())) {
          right_transfer_size =
              right_transfer_size * elem_counts[right_i] * rhs_byte_multiple;
        }
        cur_transfer_size *= right_transfer_size;

        //  Continue if this break point doesn't save at least 10% of 1D
        //  scheduling or isn't better than previous break_points found.
        if (cur_transfer_size >= min_total_transfer ||
            cur_transfer_size * 10 >= transfer_size_1d * 9) {
          continue;
        }

        // Need to be able to parallelize, don't use break if there's not
        // at least an unrolled warp.
        if (ceilDiv(cur_right_elem_count, max_vect_unroll_factor) <=
            at::cuda::getCurrentDeviceProperties()->warpSize) {
          continue;
        }

        // If outer broadcast, or balanced broadcast:
        if (lhs_byte_multiple <= rhs_byte_multiple &&
            // If right transfer size is bigger than half of L2
            at::cuda::getCurrentDeviceProperties()->l2CacheSize <
                right_transfer_size * 2) {
          // flip BIDx and BIDy bindings
          flip_grid_binding = true;
        } else {
          flip_grid_binding = false;
        }
        // Min transfer found, start setting values
        bdimx = std::min(
            ceilDiv(cur_right_elem_count, max_vect_unroll_factor), kThreadX);
        bdimy = 1;
        // Put remainder in bdimy if there's at least a wave of grid level
        // parallelism.
        if (cur_left_elem_count > device_multiprocessor_count) {
          bdimy = kThreadX / bdimx;
        }
        auto remainder_left = ceilDiv(cur_left_elem_count, bdimy);
        auto remainder_right =
            ceilDiv(cur_right_elem_count, bdimx * max_vect_unroll_factor);
        // Use this break point
        break_point = static_cast<int>(break_point_i);
        min_total_transfer = cur_transfer_size;
        right_elem_count = cur_right_elem_count;

        gdim_left = remainder_left;
        gdim_right = remainder_right;
      }
    }
  }

  params->vectorization_factor = std::min(
      max_vect_factor,
      vectorize_helper::getVectorizationFactor(
          runtime_info,
          largest_out,
          data_cache,
          break_point,
          logical_reorder_map));

  // preserve the old heuristic where unroll is used only when vectorization is
  // not used. should allow to use both unroll and vectorization together in
  // heuristics tuning.
  if (params->vectorization_factor == 1) {
    params->unroll_factor = scheduler_utils::safeDiv(
        max_vect_unroll_factor, params->vectorization_factor);
  }

  NVF_ERROR(right_elem_count > 0 || break_point == 0);
  NVF_ERROR(!(bdimy > 1 && gdim_right > 1));

  params->break_point = break_point;
  params->flip_grid_binding = flip_grid_binding;
  params->split_block = bdimy > 1;

  params->lparams.bind(bdimx, ParallelType::TIDx);
  if (params->split_block) {
    params->lparams.bind(bdimy, ParallelType::TIDy);
  }
  if ((flip_grid_binding && gdim_right > 65535) ||
      (!flip_grid_binding && gdim_left > 65535)) {
    params->split_grid_y_dim = true;
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Pointwise Stats ========\n"
            << "num_elems: " << n_elems << "\n"
            << "elem_counts: " << elem_counts << "\n"
            << "max_input_dtype_size: " << max_input_dtype_size << "\n"
            << "unroll_factor: " << params->unroll_factor << std::endl
            << "vectorize_factor: " << params->vectorization_factor << std::endl
            << "\n"
            << "logical_reorder_map: ";
    for (auto [i, j] : logical_reorder_map) {
      debug() << "(" << i << ", " << j << "), ";
    }
    debug() << "\nbroadcast_byte_multiples: ";
    for (auto multiple : broadcast_byte_multiples) {
      debug() << "(" << multiple.lhs_multiple << ", " << multiple.rhs_multiple
              << "), ";
    }
    debug() << "\nLHS elems: "
            << (right_elem_count > 0 ? n_elems / right_elem_count : 0)
            << " RHS elems: " << right_elem_count << std::endl;
    debug() << std::endl;
    debug() << params->toString() << std::endl;
  }

  return params;
}

// Return reference tensor view.
TensorView* getReferenceTensorView(Fusion* fusion) {
  FusionGuard fg(fusion);
  DomainMap domain_map(fusion);
  auto reference_tv = domain_map.findReferenceTensorView();
  return reference_tv;
}

//! Utility for canSchedule interface to check if this fusion has
//!  a fully broadcasted reference tensor, which is necessary for
//!  the pointwise scheduler.
bool hasReferenceTensorView(Fusion* fusion) {
  return getReferenceTensorView(fusion) != nullptr;
}

bool PointWiseScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (scheduler_utils::isResharding(fusion)) {
    FUSER_PERF_SCOPE("PointWiseScheduler::canScheduleCompileTime");
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Currently using the same path as the scheduler
  // to eliminate mismatch between canSchedule and
  // schedule pointwise.
  if (!hasReferenceTensorView(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "cannot find reference tensor");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
    return false;
  }

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }
  }

  if (ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no support for reduction ops");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  return true;
}

bool PointWiseScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("PointWiseScheduler::canScheduleRunTime");
  std::cerr << "PW: canScheduleRunTime\n";
  auto can_schedule_transpose_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::CanScheduleTranspose>(
          data_cache, [fusion]() {
            return std::make_unique<bool>(
                TransposeScheduler().canScheduleCompileTime(fusion));
          });
  if (can_schedule_transpose_entry.get()) {
    return !TransposeScheduler().canScheduleRunTime(
        fusion, runtime_info, data_cache);
  }
  std::cerr << "PW: canScheduleRunTime done\n";
  return true;
}

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  std::cout << std::endl;
  std::cerr << "schedulePointwise\n";
  fusion->printMath();
  std::cout << std::endl;

  // Cache inputs
  // auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  // auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  // Reference for ordering
  TensorView* reference_order_tv = nullptr;
  {
    // TODO: Propagate slice first. DO it manually for now. Or use
    // scheduleLoopDomain in a forward fashion. It should be possible
    // to reverse the setting.

    std::cout << "Before resize scheduling" << std::endl;
    fusion->printMath();
    std::cout << std::endl;
    scheduler_tools::propagateCatToInputs(fusion);
    scheduler_tools::propagateSliceToOutputs(fusion);

    reference_order_tv = getAllocationReferenceTensor(fusion);
  }

  std::cout << "scheduilng done\n";
  fusion->printMath();
  std::cout << std::endl;

  for (auto tv : fusion->allTvs()) {
    std::cerr << "Scheduled TV: " << tv->toString() << "\n";
    if (tv->hasRoot()) {
      std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain()) << "\n";
    }
    std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
              << "\n";
    std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain()) << "\n";
    for (auto expr : tv->domain()->allExprs()) {
      std::cerr << expr->toString();
    }
  }

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  refineCachePolicy(fusion);

  std::vector<TensorView*> input_tvs;
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      input_tvs.push_back(tv);
    }
  }
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());

  int64_t max_dims = 0;
  for (auto inp : input_tvs) {
    max_dims = std::max(pointwise_utils::nRootDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(pointwise_utils::nRootDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  TensorView* reference_tv = getReferenceTensor(fusion);
  std::cerr << "Reference: " << reference_tv->toString() << "\n";

  // Make sure reference is ordered properly
  {
    IdModel id_model(fusion, /*build_models=*/false);
    const auto& graph = id_model.buildExactGraph();
    const auto ordered_domains =
        scheduler_utils::getIterationDomainsOrderedLike(
            graph,
            graph.toGroups(reference_tv->getLoopDomain()),
            graph.toGroups(reference_order_tv->getMaybeAllocationDomain()));
    std::unordered_map<int64_t, int64_t> old2new;
    for (const auto i : c10::irange(reference_tv->getLoopDomain().size())) {
      const auto& loop_group =
          graph.toGroup(reference_tv->getLoopDomain().at(i));
      auto it =
          std::find(ordered_domains.begin(), ordered_domains.end(), loop_group);
      NVF_ERROR(it != ordered_domains.end());
      auto new_pos = (int64_t)std::distance(ordered_domains.begin(), it);
      old2new.emplace((int64_t)i, new_pos);
    }

    reference_tv->reorder(old2new);
    std::cerr << "Reordered reference: " << reference_tv->toString() << "\n";
  }

  fusion->printMath();
  std::cout << std::endl;

  NVF_ERROR(
      reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");

  scheduler_utils::moveNonConcretizedBroadcastInnermost(fusion, {reference_tv});

  // Trivial scheduling
  reference_tv->flatten();
  reference_tv->split(0, 128);
  reference_tv->split(0, 1 << 15);
  reference_tv->axis(-1)->parallelize(ParallelType::TIDx);
  reference_tv->axis(-2)->parallelize(ParallelType::BIDx);

  std::cout << "Scheduled reference:\n";
  reference_tv->printTransforms();
  std::cout << std::endl;

#if 0
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree spanning_tree(reference_tv);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference_tv);
#else
  scheduler_tools::scheduleLoopDomainsLike(
      fusion->allTvs(), reference_tv->getLoopDomain());
#endif

  std::cout << "Reference scheduling propagated\n";
  fusion->printMath();
  fusion->print();
  std::cout << std::endl;

  for (auto tv : fusion->allTvs()) {
    std::cerr << "Final scheduled TV: " << tv->toString() << "\n";
    if (tv->hasRoot()) {
      std::cerr << "\tRoot: " << toDelimitedString(tv->getRootDomain()) << "\n";
    }
    std::cerr << "\tLogical: " << toDelimitedString(tv->getLogicalDomain())
              << "\n";
    std::cerr << "\tLoop: " << toDelimitedString(tv->getLoopDomain()) << "\n";
    for (auto expr : tv->domain()->allExprs()) {
      std::cerr << expr->toString();
    }
  }

  {
    IdModel id_model(fusion);
    std::cout << id_model.idGraph(IdMappingMode::EXACT).toString();
    std::cout << std::endl;
  }

  inlineMost();

  // scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  // TODO(#1401): We could let segmentation split a partially alias-producing
  // fusion into an alias-only segment and the rest. This way, the rest of the
  // fusion (which has fewer expressions) can potentially find a better
  // scheduler and we need to call markAliases only in NoOpScheduler.
  markAliases(fusion);

  std::cout << "All done\n";
  fusion->printMath();
  std::cout << std::endl;
}

std::unique_ptr<HeuristicParams> PointWiseScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("PointWiseScheduler::computeHeuristics");
  auto pparams = getPointwiseHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(pparams != nullptr);
  return pparams;
}

void PointWiseScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("PointWiseScheduler::schedule");
  auto pparams = dynamic_cast<const PointwiseParams*>(params);
  NVF_ERROR(
      pparams != nullptr,
      "Incorrect parameters sent to PointWiseScheduler::schedule",
      params);
  schedulePointwise(fusion, pparams);
}

} // namespace nvfuser
