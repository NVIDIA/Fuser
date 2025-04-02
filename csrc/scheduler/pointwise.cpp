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
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>

namespace nvfuser {

namespace {
// constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
// Unused at the moment, commenting for clang tidy
constexpr int64_t kThreadX = 128;

// Get number of vectorizable non-outer broadcast inputs
// vectorizable_inputs: all vectorizable inputs
// break_point: the break point of the broadcast flags.
// Used to determine the influence of inputs on unroll factor.
// outer broadcast inputs are not counted since outer unroll is used
// and they are only loaded once regardless of the unroll factor due to
// the re-use across different unrolled iterations.
int64_t getNumOfNonOuterBcastInputs(
    std::vector<TensorView*> vectorizable_inputs,
    int64_t break_point) {
  if (break_point == 0) {
    return std::max((int64_t)vectorizable_inputs.size(), 1L);
  }

  // Returns true if tv is outer broadcast tv or is used by outer broadcast
  // op.
  auto isUsedByOuterBcast = [&break_point](TensorView* tv) {
    // If all the dims to the left of the break point are broadcast, then
    // this tv is considered as an outer broadcast.
    const auto& domains = tv->getLogicalDomain();
    if (std::all_of(
            domains.begin(), domains.begin() + break_point, [](IterDomain* id) {
              return id->isBroadcast();
            })) {
      return true;
    }
    // check consumers
    const auto& all_consumers = DependencyCheck::getAllDependentVals({tv});
    for (auto tv : all_consumers) {
      if (tv->definition()->isA<BroadcastOp>()) {
        const auto& bcast_flags =
            tv->definition()->as<BroadcastOp>()->getBroadcastDimFlags();

        if (std::all_of(
                bcast_flags.begin(),
                bcast_flags.begin() + break_point,
                [](bool flag) { return flag; })) {
          return true;
        }
      }
    }
    return false;
  };
  int64_t n_non_bcast_inputs = 0;
  for (auto tv : vectorizable_inputs) {
    if (!isUsedByOuterBcast(tv)) {
      n_non_bcast_inputs++;
    }
  }
  // return 1 if no non-outer broadcast inputs to avoid division by 0
  return std::max(n_non_bcast_inputs, 1L);
}

// calculate unroll factor based on inputs and computations.
int64_t getEmpiricalUnrollFactor(
    Fusion* fusion,
    int64_t break_point,
    int64_t vectorization_bytes,
    std::vector<TensorView*> vectorizable_inputs) {
  // no need to unroll if no vectorizable inputs
  if (vectorizable_inputs.empty()) {
    return 1;
  }
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // calculate the required bytes in flight to cover the latency.
  // assuming 100% occupancy.
  int64_t required_bytes_per_thread =
      scheduler_utils::getRequiredBytesInFlight() /
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  int64_t unroll_factor =
      std::max(1L, required_bytes_per_thread / vectorization_bytes);
  // If unroll is required, further scale up with computation cost and scale
  // down with input counts. Won't be triggered on A100 and H100.
  if (unroll_factor > 1) {
    int64_t computation_factor =
        scheduler_utils::getComputationCostFactor(fusion);
    unroll_factor *= computation_factor;
    int64_t n_inputs_factor =
        getNumOfNonOuterBcastInputs(vectorizable_inputs, break_point);
    unroll_factor = scheduler_utils::safeDiv(unroll_factor, n_inputs_factor);
  }
  return unroll_factor;
}

// calculate unroll factor based on total blocks to ensure we still
// have 8 waves after unroll.
int64_t getElementBasedUnrollFactor(int64_t total_blocks) {
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t target_waves = 8L;
  int64_t max_block_per_sm =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor / kThreadX;
  int64_t n_waves_wo_unroll = ceilDiv(
      total_blocks, max_block_per_sm * (int64_t)dev_prop->multiProcessorCount);
  int64_t n_elems_limited_unroll =
      scheduler_utils::roundUpPow2(ceilDiv(n_waves_wo_unroll, target_waves));
  return n_elems_limited_unroll;
}

// returns unroll factor.
// The unroll factor is calculated based on the following:
// (1) ensure enough bytes in flight to cover gmem access latency
// (2) ensure enough threads for thread level parallelism (TLP)
// (3) when kernel doesn't have enough TLP and split is not divisible, don't
// unroll.
int64_t getUnrollFactor(
    Fusion* fusion,
    int64_t break_point,
    int64_t total_blocks,
    int64_t vectorization_bytes,
    bool divisible_split,
    std::vector<TensorView*> vectorizable_io_tvs) {
  // only consider vectorizable inputs,
  // needs to check if it's already in the list to avoid duplication since a tv
  // may be both input and output, e.g. NVFuserTest.FusionIssue2372_CUDA
  std::vector<TensorView*> vectorizable_inputs;
  for (auto* tv : vectorizable_io_tvs) {
    if (tv->isFusionInput() &&
        std::find(vectorizable_inputs.begin(), vectorizable_inputs.end(), tv) ==
            vectorizable_inputs.end()) {
      vectorizable_inputs.push_back(tv);
    }
  }

  int64_t empirical_unroll = getEmpiricalUnrollFactor(
      fusion, break_point, vectorization_bytes, vectorizable_inputs);

  // limit unroll factor when n_elems is small to ensure enough
  // blocks for thread level parallelism.
  int64_t n_elems_limited_unroll = getElementBasedUnrollFactor(total_blocks);

  // Avoid unrolling when the unroll factor is constrained by `n_elems` and the
  // split is not divisible. Why? While unrolling increases instruction-level
  // parallelism (ILP), it decreases thread-level parallelism (TLP). A
  // non-divisible split further reduces the number of effective threads, which
  // negatively impacts TLP. Therefore, if the kernel lacks sufficient TLP,
  // unrolling should be avoided.
  if (n_elems_limited_unroll < empirical_unroll && !divisible_split) {
    return 1;
  } else {
    return std::min(n_elems_limited_unroll, empirical_unroll);
  }
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

  const auto device_multiprocessor_count = static_cast<int64_t>(
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  auto reorder_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::LogicalReorderMap>(
          data_cache, [&fusion, &largest_out]() {
            // NOTE: reorder_map is only applied for fusion without view
            // op yet.
            if (!ir_utils::getViewOps(fusion).empty()) {
              return std::make_unique<std::unordered_map<int64_t, int64_t>>();
            }
            return std::make_unique<std::unordered_map<int64_t, int64_t>>(
                scheduler_utils::maybeReorderAsAllocationMap(largest_out));
          });
  const std::unordered_map<int64_t, int64_t>& reorder_map =
      reorder_map_entry.get();

  std::vector<IterDomain*> ref_loop = largest_out->getLoopDomain();
  // reorder of root to align with logical map should always help with indexing,
  // even when vectorization isn't used.
  if (!reorder_map.empty()) {
    ref_loop = TensorDomain::orderedAs(ref_loop, reorder_map);
  }
  // We always cacheBefore output at the beginning of the scheduling. And after
  // cacheBefore, the reference tensor will have all reduction IDs removed.
  ref_loop = TensorDomain::noDevices(TensorDomain::noReductions(ref_loop));

  std::vector<int64_t> elem_counts(ref_loop.size(), 1);
  int64_t n_elems = 1;
  for (size_t ref_i = 0; ref_i < ref_loop.size(); ref_i++) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(ref_loop[ref_i]->extent());
    NVF_ERROR(
        inferred_val.hasValue(),
        "Error inferring size for pointwise scheduler: ",
        ref_loop[ref_i]->extent()->toInlineString());
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

  int64_t max_dtype_size_for_vectorization = 1;
  for (auto inp : vectorizable_inputs_outputs_entry.get()) {
    max_dtype_size_for_vectorization = std::max(
        max_dtype_size_for_vectorization,
        (int64_t)dataTypeSize(inp->getDataType().value(), index_type));
  }

  constexpr int64_t kSixteen = 16; // clang tidy
  auto max_vect_factor = ceilDiv(
      // Available vectorization based on size of data type
      (int64_t)kSixteen / max_dtype_size_for_vectorization,
      // Reduce max vectorization factor if we have many inputs/outputs to
      // vectorize as it could start consuming a lot of registers.
      std::max(
          (scheduler_utils::lastPow2(
               (int64_t)vectorizable_inputs_outputs_entry.get().size()) >>
           2),
          (int64_t)1));
  // Don't vectorize at the cost of getting a full wave on the GPU
  if (n_elems < device_multiprocessor_count * kThreadX && max_vect_factor > 1) {
    max_vect_factor = std::min(
        max_vect_factor,
        ceilDiv(n_elems, device_multiprocessor_count * kThreadX));
  }

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
  NVF_ERROR(broadcast_byte_multiples.size() == TensorDomain::noDevices(TensorDomain::noReductions(largest_out->getLogicalDomain())).size(), "Broadcast byte multiples size mismatch: ", broadcast_byte_multiples.size(), " != ", largest_out->getLogicalDomain(), "Loop domain:", ref_loop);

  int64_t dtype_sum = 0;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    dtype_sum += (int64_t)dataTypeSize(inp->getDataType().value(), index_type);
  }
  for (auto out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    dtype_sum += (int64_t)dataTypeSize(out->getDataType().value(), index_type);
  }

  // Indicates whether the fusion is outer broadcast dominated or not.
  bool is_outer_broadcast_dominated = false;
  { // Figure out break point position. Empty scope, consider moving to a
    // separate function.
    //
    // How much would this transfer cost if it was done as a 1-D schedule
    int64_t transfer_size_1d = 1;

    for (const auto i : arange(ref_loop.size())) {
      transfer_size_1d = transfer_size_1d * elem_counts[i] * dtype_sum;
    }

    // If there isn't very much parallelism available, just use 1D scheduler
    if (n_elems * 2 > device_multiprocessor_count * kThreadX) {
      int64_t min_total_transfer = std::numeric_limits<int64_t>::max();
      int64_t threads_per_warp =
          (int64_t)at::cuda::getCurrentDeviceProperties()->warpSize;
      // Don't check the inner most dimension, scheduler assumes there's always
      // an rhs
      for (const auto break_point_i : arange((int64_t)ref_loop.size())) {
        // If break point is incoherent with view, don't consider breaking here.
        if (!scheduler_utils::breakIsDisjoint(
                view_disjoint_sets, break_point_i)) {
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

        auto lhs_byte_multiple =
            broadcast_byte_multiples[break_point_i].lhs_multiple;
        auto rhs_byte_multiple =
            broadcast_byte_multiples[break_point_i].rhs_multiple;

        // Estimate transfer cost with this break point
        int64_t cur_transfer_size = 1;
        int64_t right_transfer_size = 1;

        for (const auto left_i : arange(break_point_i)) {
          cur_transfer_size =
              cur_transfer_size * elem_counts[left_i] * lhs_byte_multiple;
        }

        for (const auto right_i : arange(break_point_i, ref_loop.size())) {
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
        if (ceilDiv(cur_right_elem_count, max_vect_factor) <=
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
        // Start bdimx with 1 warp, increase if split is divisible
        int64_t after_vect = ceilDiv(cur_right_elem_count, max_vect_factor);
        bdimx = std::min(after_vect, threads_per_warp);
        while (bdimx * 2 <= kThreadX && bdimx * 2 <= after_vect &&
               after_vect % (bdimx * 2) == 0) {
          bdimx *= 2;
        }
        bdimy = kThreadX / bdimx;
        auto remainder_left = ceilDiv(cur_left_elem_count, bdimy);
        auto remainder_right =
            ceilDiv(cur_right_elem_count, bdimx * max_vect_factor);
        // Use this break point
        break_point = static_cast<int>(break_point_i);
        min_total_transfer = cur_transfer_size;
        right_elem_count = cur_right_elem_count;

        gdim_left = remainder_left;
        gdim_right = remainder_right;

        // when lhs byte multiple is smaller than rhs byte multiple,
        // there is broadcast in the lhs, which is outer broadcast.
        is_outer_broadcast_dominated = lhs_byte_multiple < rhs_byte_multiple;
      }
    }
  }

  params->vectorization_factor = std::min(
      max_vect_factor,
      vectorize_helper::getVectorizationFactor(
          runtime_info, largest_out, data_cache, break_point, reorder_map));

  // get unroll factor:

  int64_t total_blocks = break_point > 0
      ? gdim_left * gdim_right
      : ceilDiv(n_elems / max_vect_factor, kThreadX);
  bool divisible_split = break_point > 0
      ? (right_elem_count % (params->vectorization_factor * bdimx) == 0)
      : (n_elems % (params->vectorization_factor * kThreadX) == 0);
  int64_t unroll_factor = getUnrollFactor(
      fusion,
      break_point,
      total_blocks,
      params->vectorization_factor * max_dtype_size_for_vectorization,
      divisible_split,
      vectorizable_inputs_outputs_entry.get());

  if (is_outer_broadcast_dominated) {
    params->unroll_factor_outer = unroll_factor;
  } else {
    params->unroll_factor_inner = unroll_factor;
  }
  gdim_left = ceilDiv(gdim_left, params->unroll_factor_outer);
  gdim_right = ceilDiv(gdim_right, params->unroll_factor_inner);

  NVF_ERROR(right_elem_count > 0 || break_point == 0);

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
            << "max_dtype_size_for_vectorization: "
            << max_dtype_size_for_vectorization << "\n"
            << "unroll_factor_inner: " << params->unroll_factor_inner
            << std::endl
            << "unroll_factor_outer: " << params->unroll_factor_outer
            << std::endl
            << "vectorize_factor: " << params->vectorization_factor << std::endl
            << "\n"
            << "reorder_map: ";
    for (auto [i, j] : reorder_map) {
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

//! Utility for canSchedule interface to check if this fusion has
//!  a fully broadcasted reference tensor, which is necessary for
//!  the pointwise scheduler.
bool hasReferenceTensorView(Fusion* fusion) {
  return pointwise_utils::getReferenceTensor(fusion) != nullptr;
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

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

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
    max_dims = std::max(scheduler_utils::nLogicalDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  TensorView* reference_tv = pointwise_utils::getReferenceTensor(fusion);
  std::vector<IterDomain*> ref_orig_loop = reference_tv->getLoopDomain();

  NVF_ERROR(
      reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");

  scheduler_utils::moveNonConcretizedBroadcastInnermost(fusion, {reference_tv});

  int64_t num_device_dims = numDeviceDims(reference_tv);
  int64_t device_aware_break_point = pparams->break_point + num_device_dims;

  // Positions of rhs and lhs after merging all dimensions.
  int64_t rhs_i = -1;
  int64_t lhs_i = -1;

  if (!ir_utils::getViewOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion, ca_map);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    reference_tv->reorder(
        scheduler_utils::domainReorderAsLogicalMap(reference_tv));
    // Reorder so that DeviceDims are in front
    reorderDIDToFront(reference_tv);

    // Break point is relative to logical domain, find the loop domain ID's in
    // the left/right side, we really need the values in domain, but easiest way
    // to do this is with Dependency check which will grab all intermediate
    // values too.
    auto lhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + num_device_dims,
         ref_orig_loop.begin() + device_aware_break_point},
        {reference_tv->getLoopDomain().begin() + num_device_dims,
         reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> lhs_all_vals_set(
        lhs_all_vals.begin(), lhs_all_vals.end());

    auto rhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + device_aware_break_point, ref_orig_loop.end()},
        {reference_tv->getLoopDomain().begin() + num_device_dims,
         reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> rhs_all_vals_set(
        rhs_all_vals.begin(), rhs_all_vals.end());

    // Make sure lhs and rhs groups are disjoint.
    for (auto lhs_val : lhs_all_vals) {
      if (rhs_all_vals_set.count(lhs_val) != 0) {
        std::ostringstream os;
        IrTransformPrinter printer(os);
        printer.printTransforms(reference_tv);
        NVF_THROW(
            "Error in pointwise scheduler. LHS and RHS of the 2D scheduler are "
            "not disjoint. ",
            lhs_val->toString(),
            " belongs to both. device_aware_break_point = ",
            device_aware_break_point,
            ". reference_tv = ",
            reference_tv->toString(),
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
    for (int64_t pos = reference_tv->nDims() - 1; pos >= 0; pos--) {
      // Merge from right to left
      auto id = reference_tv->axis(pos);
      if (lhs_all_vals_set.count(id) > 0) {
        if (lhs_id == nullptr) {
          lhs_id = id;
          lhs_i = pos;
        } else {
          reference_tv->merge(pos, lhs_i);
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
          reference_tv->merge(pos, rhs_i);
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

    std::unordered_map<int64_t, int64_t> reorder_map =
        scheduler_utils::maybeReorderAsAllocationMap(reference_tv);
    if (!reorder_map.empty()) {
      reference_tv->reorder(reorder_map);
    }
    reorderDIDToFront(reference_tv);

    // Merge right side of break point
    for (int64_t i = reference_tv->nDims(); i > device_aware_break_point; i--) {
      auto axis_i = i - 1;
      if (rhs_i == -1) {
        rhs_i = axis_i;
      } else {
        reference_tv->merge(axis_i, rhs_i);
        rhs_i = axis_i;
      }
    }
    if (rhs_i >= 0) {
      // If there's an rhs
      reference_tv->reorder({{rhs_i, -1}});
    }

    // Merge left side of break point
    for (int64_t i = device_aware_break_point; i > num_device_dims; i--) {
      auto axis_i = i - 1;
      if (lhs_i == -1) {
        lhs_i = axis_i;
      } else {
        reference_tv->merge(axis_i, lhs_i);
        lhs_i = axis_i;
      }
    }
  }

  int64_t unswitch_pos = 0;
  IterDomain* vectorize_id = nullptr;
  if (pparams->break_point) {
    // 2D parallelization scheme
    NVF_ERROR(rhs_i >= 0 && lhs_i >= 0);

    // Right (inner merged) dimension is at inner most position, left (outer
    // merged) dimension is at lhs_i. Order as [lhs_i, rhs_i, unmerged...]
    reference_tv->reorder({{lhs_i, 0}, {-1, 1}});

    // vectorization without unroll
    if (pparams->unroll_factor_outer == 1 &&
        pparams->unroll_factor_inner == 1 &&
        pparams->vectorization_factor > 1) {
      reference_tv->split(1, pparams->vectorization_factor);
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      reference_tv->split(0, 1);
      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      // Here and in the following comments:
      // prefix [i] represent inner dimension
      // prefix [o] represent inner dimension
      // [|] separates the outer and inner dimensions
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      reference_tv->axis(3)->parallelize(ParallelType::TIDx);
      // Vectorization are propagated separately
      vectorize_id = reference_tv->axis(4);

      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 2}, {2, 1}, {3, 4}, {4, 3}});
      //[outer | i-remainder, Unswitch, Vectorization, TIDx]
    } else {
      // [outer | inner]
      if (pparams->vectorization_factor > 1) {
        reference_tv->split(1, pparams->vectorization_factor);
      }
      // [outer | i-remainder, Vect]
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      // [outer | i-remainder, TIDx, Vect]

      if (pparams->unroll_factor_inner > 1) {
        reference_tv->split(1, pparams->unroll_factor_inner);
      }
      // [outer| i-remainder, i-Unroll, TIDx, Vect]

      if (pparams->unroll_factor_outer > 1) {
        reference_tv->split(0, pparams->unroll_factor_outer);
      }
      // [o-remainder, o-Unroll, | i-remainder, i-Unroll, TIDx, Vect]

      reference_tv->split(0, 1);
      // [o-remainder, Unswitch, o-Unroll | i-remainder, i-Unroll, TIDx, Vect]

      int i_remainder_pos = pparams->unroll_factor_outer > 1 ? 3 : 2;
      reference_tv->reorder({{i_remainder_pos, 1}});
      // [o-remainder, i-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]

      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      // Here we do not set axis(3)->parallelize(Unroll) because we do not want
      // it to be propagated. We manually unroll by splitting the inline
      // propagation process into two steps:
      // step 1: inline at the unswitch position for cached inputs and outputs
      // step 2: inline at the inner most dim for the rest of the graph
      int tidx_pos = 3;
      if (pparams->unroll_factor_inner > 1) {
        tidx_pos++;
      }
      if (pparams->unroll_factor_outer > 1) {
        tidx_pos++;
      }
      reference_tv->axis(tidx_pos)->parallelize(ParallelType::TIDx);
      if (pparams->vectorization_factor > 1) {
        // can't use {-1}, there may be deviceId
        vectorize_id = reference_tv->axis(tidx_pos + 1);
      }
      // [o-remainder, i-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
    }

    // Move out of the way to furthest left point
    reference_tv->reorder({{1, 0}});
    // [i-remainder, o-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
    if (pparams->split_block) {
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
      // [i-remainder, o-remainder, TIDy, Unswitch, o-Unroll, i-Unroll, TIDx,
      // Vect]
      if (pparams->flip_grid_binding) {
        // [BIDy | BIDx, TIDy | Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (pparams->split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx, TIDy | Unswitch, o-Unroll,
          // i-Unroll, TIDx, Vect]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 5;
        } else {
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        }
      } else {
        // [BIDx | BIDy TIDy | Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (pparams->split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535}, TIDy | Unswitch, o-Unroll,
          // i-Unroll, TIDx, Vect]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
          unswitch_pos = 5;
        } else {
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        }
      }
    } else {
      // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
      if (pparams->flip_grid_binding) {
        // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        if (pparams->split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx | Unswitch, Unroll, TIDx]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        } else {
          // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
          unswitch_pos = 3;
        }
      } else {
        // [BIDx | BIDy | Unswitch, Unroll, TIDx, Vect]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        if (pparams->split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535} | Unswitch, Unroll, TIDx, Vect]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        } else {
          // [BIDx | BIDy | Unswitch, Unroll, TIDx, Vect]
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 3;
        }
      }
    }
  } else {
    // 1D Scheduler
    NVF_ERROR(rhs_i >= 0 && lhs_i == -1);

    // right hand side exists and is the only axis we care to schedule, move
    // it from the inner most position to left most. Order as [rhs_i,
    // unmerged...]
    reference_tv->reorder({{-1, 0}});

    if (pparams->unroll_factor_inner == 1 &&
        pparams->vectorization_factor > 1) {
      // Vectorize
      reference_tv->split(0, pparams->vectorization_factor);
      // Unswitch
      reference_tv->split(0, 1);
      // Threads
      reference_tv->split(0, kThreadX);

      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::TIDx);
      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      // Vectorization are propagated separately
      vectorize_id = reference_tv->axis(3);

      //[BIDx, TIDx, Unswitch, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 3}, {2, 1}, {3, 2}});
      //[BIDx, Unswitch, Vectorization, TIDx]
    } else {
      // Vectorize
      if (pparams->vectorization_factor > 1) {
        reference_tv->split(0, pparams->vectorization_factor);
      }
      // Threads
      reference_tv->split(0, kThreadX);
      // Unroll
      if (pparams->unroll_factor_inner > 1) {
        reference_tv->split(0, pparams->unroll_factor_inner);
      }
      // Unswitch
      reference_tv->split(0, 1);

      // [BIDx, Unswitch, Unroll, TIDx, Vect]
      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      // Here we do not set axis(2)->parallelize(Unroll) because we do not want
      // it to be propagated. We manually unroll by splitting the inline
      // propagation process into two steps:
      // step 1: inline at the unswitch position for cached inputs and outputs
      // step 2: inline at the inner most dim for the rest of the graph
      int tidx_pos = pparams->unroll_factor_inner > 1 ? 3 : 2;
      reference_tv->axis(tidx_pos)->parallelize(ParallelType::TIDx);
      if (pparams->vectorization_factor > 1) {
        vectorize_id = reference_tv->axis(tidx_pos + 1);
      }
    }
    unswitch_pos = 2;
  }

  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree spanning_tree(reference_tv);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference_tv);

  if (pparams->vectorization_factor > 1) {
    // Grab all tensor views that should be vectorized
    auto inputs_outputs =
        scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true, true);
    std::vector<TensorView*> vectorized_tvs;
    bool should_vectorize_reference_tv = false;
    for (auto tv : inputs_outputs) {
      if (tv == reference_tv) {
        should_vectorize_reference_tv = true;
      }
      if (!tv->isFusionInput()) {
        vectorized_tvs.emplace_back(tv);
        continue;
      }
      // move inputs to consumers of inputs
      auto consumer_tvs = ir_utils::consumerTvsOf(tv);
      vectorized_tvs.insert(
          vectorized_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
    }
    if (!vectorized_tvs.empty()) {
      // Aggressively mark with vectorized and cleanup later. That way we
      // don't have to manually specify parallelization outside the reference.
      vectorize_id->parallelize(ParallelType::Vectorize);
      scheduler_utils::parallelizeAllLike(
          reference_tv, vectorized_tvs, {ParallelType::Vectorize});
      if (!should_vectorize_reference_tv) {
        vectorize_id->parallelize(ParallelType::Serial);
      }
    }
  }

  // Begin by inlining at the unswitch position for the entire DAG. The cached
  // inputs, and outputs will keep this inline position, but other tensors will
  // get a higher position in later inline propagation. We need this separate
  // step because we were not using ParallelType::Unroll, so we have to do
  // unrolling manually.
  inlineAllAt(reference_tv, unswitch_pos, true);

  auto all_tvs = fusion->allTvs();

  // Inline at the inner most position. The CA position of all tensors except
  // inputs, cached inputs and outputs will be updated.
  std::unordered_set<TensorView*> inner_most_tensors(
      all_tvs.begin(), all_tvs.end());
  for (auto cached_input : cached_inputs) {
    inner_most_tensors.erase(cached_input);
  }
  for (auto entry : cached_outputs) {
    auto output = entry.second;
    inner_most_tensors.erase(output);
  }
  // IndexSelectOp reads lookup tv without cache. Because pointwise scheduler
  // doesn't use ParallelType::Unroll, we need to exclude consumer of fusion
  // inputs to be inlineMost. This allows us to aggregate the allocation of
  // manual unroll ID and its inner ID.
  for (auto idx_sel : ir_utils::getOpsOfType<IndexSelectOp>(fusion)) {
    inner_most_tensors.erase(idx_sel->output(0)->as<TensorView>());
  }

  inlineMost(inner_most_tensors);

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  // TODO(#1401): We could let segmentation split a partially alias-producing
  // fusion into an alias-only segment and the rest. This way, the rest of the
  // fusion (which has fewer expressions) can potentially find a better
  // scheduler and we need to call markAliases only in NoOpScheduler.
  markAliases(fusion);
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
