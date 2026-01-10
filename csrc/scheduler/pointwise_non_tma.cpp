// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <scheduler/pointwise_non_tma.h>

#include <ATen/cuda/CUDAContext.h>
#include <debug.h>
#include <ir/printer.h>
#include <multidevice/utils.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>
#include "exceptions.h"

namespace nvfuser {
namespace pointwise {
namespace non_tma {
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
    int64_t vectorization_bits,
    std::vector<TensorView*> vectorizable_inputs) {
  // no need to unroll if no vectorizable inputs
  if (vectorizable_inputs.empty()) {
    return 1;
  }
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // calculate the required bytes in flight to cover the latency.
  // assuming 100% occupancy.
  int64_t required_bits_per_thread =
      scheduler_utils::getRequiredBitsInFlight() /
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  int64_t unroll_factor =
      std::max(1L, required_bits_per_thread / vectorization_bits);
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
    int64_t vectorization_bits,
    bool divisible_split,
    std::vector<TensorView*> vectorizable_io_tvs,
    HeuristicDataCache* data_cache) {
  // Check if fusion has BlockQuantizationOp(s)
  // Limit unroll factor for fusions with BlockQuantizationOp(s). The runtime
  // function which implements quantization assumes no unrolling
  auto has_block_quantization_ops =
      HeuristicDataCacheEntry<HeuristicCompileTime::HasBlockQuantizationOps>(
          data_cache,
          [fusion]() {
            return std::make_unique<bool>(
                !ir_utils::getOpsOfType<BlockQuantizationOp>(fusion).empty() ||
                !ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion)
                     .empty());
          })
          .get();

  if (has_block_quantization_ops) {
    // Runtime function implementing Block Quantization Op requires unroll
    // factor to be 1
    return 1;
  }

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
      fusion, break_point, vectorization_bits, vectorizable_inputs);

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
    HeuristicDataCache* data_cache,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  FusionGuard fg(fusion);
  auto params = std::make_unique<PointwiseParams>();
  params->tag = "Pointwise heuristics";
  // Incase any buffer is of type DataType::Index
  const auto index_type = prop.index_type;
  params->cparams.index_type = index_type;

  int64_t max_dtype_size_bit_for_vectorization =
      prop.max_dtype_size_bit_for_vectorization;
  const auto& vectorizable_inputs_outputs = prop.vectorizable_inputs_outputs;
  int64_t n_elems = prop.n_elems;
  const auto device_multiprocessor_count = prop.device_multiprocessor_count;
  constexpr int64_t max_vectorization_size_in_bit = 128;

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

  // Only calculate break point if there's enough parallelism for 2D scheduling
  pointwise_utils::BreakPointInfo bp_info;
  if (n_elems * 2 > device_multiprocessor_count * kThreadX) {
    bp_info = pointwise_utils::getBreakPoint(
        fusion,
        prop,
        data_cache,
        /*is_tma =*/false,
        max_vectorization_size_in_bit /
            prop.max_dtype_size_bit_for_vectorization,
        kThreadX);
  } else {
    // Use default 1D scheduling (break_point = 0)
    bp_info.break_point = 0;
    bp_info.flip_grid_binding = false;
    bp_info.right_elem_count = 0;
    bp_info.is_outer_broadcast_dominated = false;
  }

  // Use unified function that handles all constraints internally
  const int64_t vectorization_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      prop.largest_out,
      data_cache,
      bp_info.break_point,
      max_vectorization_size_in_bit,
      prop.min_dtype_size_bit_for_vectorization,
      prop.max_dtype_size_bit_for_vectorization,
      /*n_vectorizable_tensors=*/(int64_t)vectorizable_inputs_outputs.size(),
      // should use the actual wave count, here just to keep consistency with
      // the old code.
      // TODO: check how should we limit the vectorization factor based on the
      // wave count.
      /*n_waves=*/n_elems > device_multiprocessor_count * kThreadX ? -1 : 1,
      /*logical_reorder_map=*/
      pointwise_utils::getLogicalReorderMap(
          prop.largest_out, prop.has_reshapes, data_cache));
  params->vectorization_factor = vectorization_factor;

  // Calculate block and grid configuration based on break point
  auto config = pointwise_utils::getBlockGridConfig(
      prop, bp_info, vectorization_factor, kThreadX);

  // Extract results
  break_point = config.break_point;
  flip_grid_binding = config.flip_grid_binding;
  right_elem_count = config.right_elem_count;
  bdimx = config.bdimx;
  bdimy = config.bdimy;
  gdim_left = config.gdim_left;
  gdim_right = config.gdim_right;
  bool is_outer_broadcast_dominated = config.is_outer_broadcast_dominated;
  const auto& elem_counts = prop.elem_counts;

  // get unroll factor:
  int64_t total_blocks = break_point > 0
      ? gdim_left * gdim_right
      : ceilDiv(n_elems / vectorization_factor, kThreadX);
  bool divisible_split = break_point > 0
      ? (right_elem_count % (params->vectorization_factor * bdimx) == 0)
      : (n_elems % (params->vectorization_factor * kThreadX) == 0);
  int64_t unroll_factor = getUnrollFactor(
      fusion,
      break_point,
      total_blocks,
      params->vectorization_factor * max_dtype_size_bit_for_vectorization,
      divisible_split,
      vectorizable_inputs_outputs,
      data_cache);

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
    // Get broadcast info for debug output
    TensorView* largest_out = prop.largest_out;
    auto broadcast_info_for_debug =
        scheduler_utils::getBroadcastMultiples(largest_out, index_type);

    debug() << "\n===== Pointwise Stats ========\n"
            << "num_elems: " << n_elems << "\n"
            << "elem_counts: " << elem_counts << "\n"
            << "max_dtype_size_bit_for_vectorization: "
            << max_dtype_size_bit_for_vectorization << "\n"
            << "unroll_factor_inner: " << params->unroll_factor_inner
            << std::endl
            << "unroll_factor_outer: " << params->unroll_factor_outer
            << std::endl
            << "vectorize_factor: " << params->vectorization_factor << std::endl
            << "\n"
            << "logical_reorder_map: ";
    for (auto [i, j] : pointwise_utils::getLogicalReorderMap(
             prop.largest_out, prop.has_reshapes, data_cache)) {
      debug() << "(" << i << ", " << j << "), ";
    }
    debug() << "\nbroadcast_byte_multiples: ";
    for (auto multiple : broadcast_info_for_debug.broadcast_multiples) {
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

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  auto schedule_info_opt =
      pointwise_utils::commonPointwiseSchedule(fusion, pparams->break_point);
  if (!schedule_info_opt.has_value()) {
    // Zero-dimensional tensors, nothing to schedule
    return;
  }
  auto& schedule_info = schedule_info_opt.value();

  auto& cached_inputs = schedule_info.cached_inputs;
  TensorView* reference_tv = schedule_info.reference_tv;

  int64_t unswitch_pos = 0;
  IterDomain* vectorize_id = nullptr;
  if (pparams->break_point) {
    // 2D parallelization scheme
    int64_t lhs_i = schedule_info.lhs_i;
    int64_t rhs_i = schedule_info.rhs_i;
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
    // 1D Scheduler (break_point == 0)
    int64_t lhs_i = schedule_info.lhs_i;
    int64_t rhs_i = schedule_info.rhs_i;
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

  // We first vectorize the quantized outputs of the block quantization ops.
  // We then convert the vectorized ID to group ID.
  // We do so as the runtime function for block quantization expects 2/4/8
  // elements per thread.
  auto bq_ops = ir_utils::getOpsOfType<BlockQuantizationOp>(fusion);
  auto gbq_ops = ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion);
  std::vector<TensorView*> nvfp4_quantized_outputs = {};
  for (auto bq_op : bq_ops) {
    nvfp4_quantized_outputs.push_back(
        bq_op->quantizedOutput()->as<TensorView>());
  }
  for (auto gbq_op : gbq_ops) {
    nvfp4_quantized_outputs.push_back(
        gbq_op->quantizedOutput()->as<TensorView>());
  }

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
    // Vectorize all casts
    if (pparams->vectorize_casts) {
      for (auto tv : fusion->allTvs()) {
        if (auto uop = dynamic_cast<UnaryOp*>(tv->definition())) {
          if (uop->getUnaryOpType() == UnaryOpType::Cast &&
              (dataTypeSizeBit(tv->dtype()) < 8 ||
               dataTypeSizeBit(uop->in()->dtype()) < 8)) {
            vectorized_tvs.emplace_back(tv);
          }
        }
      }
    }

    // Vectorize nvfp4 quantized outputs.
    // We will later change the vectorized ID to group ID
    for (auto quantized_output : nvfp4_quantized_outputs) {
      vectorized_tvs.emplace_back(quantized_output);
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

      // Change vectorized IDs to group IDs for quantized outputs
      for (auto quantized_output : nvfp4_quantized_outputs) {
        for (auto id : quantized_output->getLoopDomain()) {
          if (id->getParallelType() == ParallelType::Vectorize) {
            id->parallelize(ParallelType::Group);
          }
        }
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
  for (const auto& [cached_input, input_idx] : cached_inputs) {
    inner_most_tensors.erase(cached_input);
  }
  for (const auto& [cached_output, output_idx] : schedule_info.cached_outputs) {
    auto output = fusion->outputs()[output_idx]->as<TensorView>();
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

  markAliases(fusion);
}
} // namespace non_tma
} // namespace pointwise
} // namespace nvfuser
