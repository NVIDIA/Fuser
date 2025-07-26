// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_evaluator.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/arith.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/maxinfo_propagator.h>
#include <scheduler/utils.h>
#include <transform_replay.h>
#include <utils.h>

namespace nvfuser {

namespace reduction_scheduler_utils {

TensorView* scheduleReductionTV(
    const ReductionParams* rparams,
    TensorView* reduction_tv,
    bool has_iter_axis) {
  // Outer and inner reduction axis is relative. Outer reduce axis is only valid
  // in 3D scheduling. Otherwise inner_reduce_axis is the only reduction axis.
  // Inner here though is only relative to the other axis. When
  // rparams->fastest_dim == false, the reduction axis is logically outside the
  // iteration axis.
  //
  // Multidevice scheduling: we assume only the outermost domain can be
  // parallelized with DIDx at this point and in that case this reduction
  // scheduler only schedules the remaining domains while leaving the DIDx
  // domain unchanged.
  int64_t sharded_axis = getShardedLoopAxis(reduction_tv, ParallelType::DIDx);
  if (sharded_axis >= 0) {
    NVF_ERROR(
        sharded_axis == 0,
        "Expect 1D mesh and DIDx only appear outermost in loop, but found: ",
        reduction_tv->getLoopDomain());
  }
  NVF_ERROR(
      sharded_axis == -1 || !rparams->schedule_3D,
      "Mixing interdevice and 3D schedule is not supported");
  const int iter_axis = (sharded_axis >= 0) ? 1 : 0;
  const int outer_reduce_axis = rparams->schedule_3D ? 1 : 0;
  const int inner_reduce_axis =
      rparams->schedule_3D ? 2 : (sharded_axis >= 0) + has_iter_axis;

  const bool is_outer_grid_persistence = rparams->persistent_kernel &&
      rparams->cross_grid_inner_reduction && !rparams->fastest_dim;

  NVF_ERROR(
      reduction_tv->nDims() >
          std::max(iter_axis, std::max(outer_reduce_axis, inner_reduce_axis)),
      "Issue in scheduling reduction tv, expecting >",
      std::max(iter_axis, std::max(outer_reduce_axis, inner_reduce_axis)),
      " dimensions, but found ",
      reduction_tv->nDims());

  NVF_ERROR(
      !(rparams->fastest_dim && rparams->vectorize_iter_dom),
      "Cannot vectorize iteration domain on inner reductions.");

  NVF_ERROR(
      !(!rparams->fastest_dim && rparams->vectorize_inner_reduction),
      "Cannot vectorize reduction domain on outer reductions.");

  NVF_ERROR(
      !(rparams->multiple_reds_per_blk && !has_iter_axis),
      "Multiple reductions requires an iter domain, but one wasn't found.");

  NVF_ERROR(
      !(rparams->unroll_factor_iter_dom > 1 && !has_iter_axis),
      "Unrolling on iter domain requires an iter domain.");

  auto vectorize = [&reduction_tv](int64_t axis, int64_t factor) {
    reduction_tv->split(axis, factor);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Vectorize);
  };

  auto inner_parallel = [&reduction_tv](int64_t axis, ParallelType ptype) {
    reduction_tv->split(axis, NamedScalar::getParallelDim(ptype));
    reduction_tv->axis(axis + 1)->parallelize(ptype);
  };

  auto inner_parallel_static =
      [&reduction_tv](int64_t axis, ParallelType ptype, int64_t factor) {
        reduction_tv->split(axis, factor);
        reduction_tv->axis(axis + 1)->parallelize(ptype);
      };

  auto inner_unswitch = [&reduction_tv](int64_t axis) {
    reduction_tv->split(axis, 1);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Unswitch);
  };

  auto inner_unroll = [&reduction_tv](int64_t axis, int64_t factor) {
    reduction_tv->split(axis, factor);
    reduction_tv->axis(axis + 1)->parallelize(ParallelType::Unroll);
  };

  auto outer_parallel = [&reduction_tv](int64_t axis, ParallelType ptype) {
    reduction_tv->split(axis, NamedScalar::getParallelDim(ptype), false);
    reduction_tv->axis(axis)->parallelize(ptype);
  };

  auto outer_parallel_static =
      [&reduction_tv](int64_t axis, ParallelType ptype, int64_t factor) {
        reduction_tv->split(axis, factor, false);
        reduction_tv->axis(axis)->parallelize(ptype);
      };

  auto outer_unswitch = [&reduction_tv](int64_t axis) {
    reduction_tv->split(axis, 1, false);
    reduction_tv->axis(axis)->parallelize(ParallelType::Unswitch);
  };

  auto outer_unroll = [&reduction_tv](int64_t axis, int64_t factor) {
    reduction_tv->split(axis, factor, false);
    reduction_tv->axis(axis)->parallelize(ParallelType::Unroll);
  };
  if (rparams->tma_warp_specialized) {
    auto option = rparams->circular_buffer_options;
    auto ws_pt = std::get<WarpSpecialized>(option.type).on;
    // Reduction: [Persistent, TIDx, Vect]
    vectorize(inner_reduce_axis, rparams->unroll_factor_inner_reduction);

    // static bdimx is required for TMA warp specialization
    int64_t compute_bdimx = getComputeBdimx(option, rparams->lparams.bdimx());
    inner_parallel_static(inner_reduce_axis, ParallelType::TIDx, compute_bdimx);

    // Iteration: [I/Unroll/BIDy, BIDy, Unroll]
    if (rparams->unroll_factor_iter_dom > 1) {
      inner_unroll(iter_axis, rparams->unroll_factor_iter_dom);
    }
    inner_parallel_static(
        iter_axis, rparams->grid_dim_iter_dom, rparams->lparams.gdimy());
    if (rparams->computation_warp_groups > 1) {
      NVF_ERROR(
          ws_pt == ParallelType::TIDy,
          "Warp specialization only supports TIDy, got ",
          ws_pt);
      inner_parallel_static(iter_axis, ws_pt, rparams->computation_warp_groups);
    }
  } else if (is_outer_grid_persistence) {
    const auto reduction_axis = inner_reduce_axis;
    NVF_ERROR(rparams->static_bdimy, "blockDim.y must be static");
    inner_parallel_static(
        reduction_axis,
        rparams->block_dim_inner_reduction,
        rparams->lparams.bdimy());
    reduction_tv->split(
        reduction_axis, rparams->batches_per_block_inner_reduction);
    reduction_tv->axis(reduction_axis)
        ->parallelize(rparams->grid_dim_inner_reduction);
    // Unswitch the persistent buffer by a factor of
    // unroll_factor_inner_reduction. If that is equal to the
    // persistent buffer size, unswitch the whole buffer by
    // outer-unswith by 1. Otherwise, split the persistent buffer by
    // the unsiwtch factor and just unswitch the inner domain
    if (rparams->batches_per_block_inner_reduction ==
        rparams->unroll_factor_inner_reduction) {
      outer_unswitch(reduction_axis + 1);
    } else {
      reduction_tv->split(
          reduction_axis + 1, rparams->unroll_factor_inner_reduction);
      outer_unswitch(reduction_axis + 2);
    }
  } else if (rparams->persistent_kernel) {
    // Persistent Format:
    // [Grid Split, persistent buffer, unswitch, unroll, thread dim,
    // vectorize]
    if (rparams->vectorize_inner_reduction) {
      vectorize(inner_reduce_axis, rparams->unroll_factor_inner_reduction);
    }
    if (rparams->combined_inner_outer && !rparams->multiple_reds_per_blk) {
      NVF_ERROR(
          rparams->static_bdimx,
          "blockDim.x must be static for combined_inner_outer");
      inner_parallel_static(
          inner_reduce_axis,
          rparams->block_dim_inner_reduction,
          rparams->lparams.bdimx());

      NVF_ERROR(
          rparams->static_bdimy,
          "blockDim.y must be static for combined_inner_outer");
      inner_parallel_static(
          inner_reduce_axis,
          rparams->block_dim_inner_reduction_extra,
          rparams->lparams.bdimy());
    }
    auto outer_i = inner_reduce_axis;
    if (rparams->cross_grid_inner_reduction) {
      // outer_parallel(outer_i++, rparams->grid_dim_inner_reduction);
      outer_parallel_static(outer_i++, rparams->grid_dim_inner_reduction, rparams->lparams.gdimx());
    }

    reduction_tv->split(
        outer_i++, rparams->batches_per_block_inner_reduction, false);

    outer_unswitch(outer_i++);

    if (!rparams->vectorize_inner_reduction &&
        rparams->unroll_factor_inner_reduction > 1) {
      outer_unroll(outer_i++, rparams->unroll_factor_inner_reduction);
    }

    if (rparams->combined_inner_outer && !rparams->multiple_reds_per_blk) {
      reduction_tv->axis(outer_i)->parallelize(ParallelType::TIDz);
    } else {
      reduction_tv->axis(outer_i)->parallelize(
          rparams->block_dim_inner_reduction);
    }
    if (rparams->pad_inner_reduction_to_warp) {
      reduction_tv->axis(outer_i)->padToMultipleOfWarp();
    }
  } else {
    // Non-persistent format:
    // [Grid Split, Remainder, unswitch, unroll, thread dim, vectorize]
    if (rparams->vectorize_inner_reduction) {
      vectorize(inner_reduce_axis, rparams->unroll_factor_inner_reduction);
    }

    if (rparams->cross_block_inner_reduction) {
      inner_parallel(inner_reduce_axis, rparams->block_dim_inner_reduction);
      if (rparams->pad_inner_reduction_to_warp) {
        reduction_tv->axis(inner_reduce_axis + 1)->padToMultipleOfWarp();
      }
    }

    if (!rparams->vectorize_inner_reduction &&
        rparams->unroll_factor_inner_reduction > 1) {
      inner_unroll(inner_reduce_axis, rparams->unroll_factor_inner_reduction);
    } else if (rparams->unroll_factor_top_of_vectorization > 1) {
      inner_unroll(
          inner_reduce_axis, rparams->unroll_factor_top_of_vectorization);
    }

    inner_unswitch(inner_reduce_axis);
    if (rparams->cross_grid_inner_reduction) {
      if (rparams->split_grid_dim_inner_reduction) {
        outer_parallel(inner_reduce_axis, rparams->grid_dim_inner_reduction);
      } else {
        reduction_tv->axis(inner_reduce_axis)
            ->parallelize(rparams->grid_dim_inner_reduction);
      }
    }
  }
  // Outer reduction axis
  if (!rparams->tma_warp_specialized && rparams->schedule_3D) {
    if (rparams->persistent_kernel) {
      // Persistent Format:
      // [Grid Split, persistent buffer, unroll, thread dim]
      auto outer_i = outer_reduce_axis;
      if (rparams->cross_grid_outer_reduction) {
        outer_parallel(outer_i++, rparams->grid_dim_outer_reduction);
      }

      reduction_tv->split(
          outer_i++, rparams->batches_per_block_outer_reduction, false);

      if (rparams->unroll_factor_outer_reduction > 1) {
        outer_unroll(outer_i++, rparams->unroll_factor_outer_reduction);
      }

      reduction_tv->axis(outer_i)->parallelize(
          rparams->block_dim_outer_reduction);
    } else {
      // Non-persistent format:
      // [Grid Split, Remainder, unroll, thread dim]
      if (rparams->cross_block_outer_reduction) {
        inner_parallel(outer_reduce_axis, rparams->block_dim_outer_reduction);
      }

      if (rparams->unroll_factor_outer_reduction > 1) {
        inner_unroll(outer_reduce_axis, rparams->unroll_factor_outer_reduction);
      }

      if (rparams->cross_grid_outer_reduction) {
        outer_parallel(outer_reduce_axis, rparams->grid_dim_outer_reduction);
      }
    }
  }

  // Iteration domain
  if (!rparams->tma_warp_specialized && has_iter_axis) {
    // [Grid Split, unswitch, unroll, thread dim, vectorize]

    if (rparams->vectorize_iter_dom) {
      vectorize(iter_axis, rparams->unroll_factor_iter_dom);
    }

    if (isParallelTypeThread(rparams->block_dim_iter_dom)) {
      if (is_outer_grid_persistence) {
        NVF_ERROR(rparams->static_bdimx, "blockDim.x must be static");
        inner_parallel_static(
            iter_axis, rparams->block_dim_iter_dom, rparams->lparams.bdimx());
      } else {
        inner_parallel(iter_axis, rparams->block_dim_iter_dom);
      }
    }

    if (!rparams->vectorize_iter_dom && rparams->unroll_factor_iter_dom > 1) {
      inner_unroll(iter_axis, rparams->unroll_factor_iter_dom);
    }

    // Do not unswitch interation domain in the case of outer grid
    // persistence as it's unclear if it's beneficial.
    if (rparams->unroll_factor_iter_dom > 1 && !is_outer_grid_persistence) {
      inner_unswitch(iter_axis);
    }

    if (isParallelTypeThread(rparams->grid_dim_iter_dom)) {
      if (rparams->split_grid_dim_iter_dom_outer) {
        if (rparams->combined_inner_outer && !rparams->multiple_reds_per_blk) {
          inner_parallel_static(
              iter_axis, rparams->grid_dim_iter_dom, rparams->lparams.gdimy());
        } else {
          outer_parallel(iter_axis, rparams->grid_dim_iter_dom);
        }
      } else if (rparams->split_grid_dim_iter_dom_inner) {
        if(rparams->lparams.gdimy() != LaunchParams::UNINITIALIZED_VAL){
          inner_parallel_static(
            iter_axis, rparams->grid_dim_iter_dom, rparams->lparams.gdimy());
        }else{
          inner_parallel(iter_axis, rparams->grid_dim_iter_dom);
        }
      } else {
        reduction_tv->axis(iter_axis)->parallelize(rparams->grid_dim_iter_dom);
      }
    }
  }
  std::cout << "reduction_tv: " << reduction_tv->toString() << std::endl;
  reduction_tv->printTransforms();

  const bool is_non_persistent_outer_reduction =
      !rparams->persistent_kernel && !rparams->fastest_dim;
  auto reduction_rf_tv =
      sortAndRFactor(reduction_tv, is_non_persistent_outer_reduction);

  // In the case of outer grid persistence, make sure the vectorized
  // domain placed at the innermost position.
  // TODO: Why isn't this the case by default?
  if (is_outer_grid_persistence) {
    int64_t vec_id_cur_pos = -1;
    std::unordered_map<int64_t, int64_t> vec_reorder_map;
    for (const auto i : arange(reduction_rf_tv->nDims())) {
      auto id = reduction_rf_tv->axis(i);
      if (id->getParallelType() == ParallelType::Vectorize) {
        vec_id_cur_pos = i;
        vec_reorder_map[i] = -1;
      } else if (vec_id_cur_pos >= 0) {
        vec_reorder_map[i] = i - 1;
      }
    }
    NVF_ERROR(vec_id_cur_pos != -1, "Vectorized ID not found");
    reduction_rf_tv->reorder(vec_reorder_map);
  }
  // [BIDy, TIDy, CircularLoop, Unroll, ...] to
  // [BIDy, CircularLoop, TIDy, Unroll, ...]
  // TIDy represents different computation warp groups and
  // will be changed to serial for TMA loads.
  if (rparams->computation_warp_groups > 1) {
    reduction_rf_tv->reorder({{1, 2}});
  }
  return reduction_rf_tv;
}

// Input: a vector of axes in the given tensor ignoring broadcasts. For example,
//        if you have a tensor T1[b, rS1, rS2, rS3], and you want to specify
//        axis rS2 and rS3, then your `non_broadcast_axes` should be {1, 2}.
// Output: the raw positions (counting broadcasts). In the above example, the
//         output should be {2, 3}.
std::vector<int64_t> addBackBroadcasts(
    TensorView* tv,
    const std::unordered_set<int64_t>& non_broadcast_axes) {
  // convert non-broadcast positions to raw positions
  std::vector<int64_t> axes;
  int64_t non_broadcast_pos = 0;
  for (const auto i : arange(tv->nDims())) {
    if (tv->axis(i)->isBroadcast()) {
      continue;
    }
    if (non_broadcast_axes.count(non_broadcast_pos)) {
      axes.emplace_back(i);
    }
    non_broadcast_pos++;
  }
  return axes;
}

void propagateTransformation(
    TensorView* reference_tv,
    const std::unordered_set<TensorView*>& boundaryNodesSet) {
  InternalBoundarySelector ibSelector(boundaryNodesSet);
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree(reference_tv, &ibSelector)
      .traverse(&propagator);
}

void propagateRFactor(
    TensorView* reference_tv,
    TensorView* reduction_tv,
    const std::vector<TensorView*>& reduction_tvs) {
  // We use axes ignoring broadcasts because in checkPatternEquivalence,
  // broadcast is ignored, we might end up having multiple reductions with
  // pattern equivalence but have different number of broadcasts, so the
  // position in the reference tensor is not necessary the same as the
  // position in other reduction TVs.
  std::unordered_set<int64_t> non_broadcast_rfactor_axes_ir;
  int64_t non_broadcast_pos_ir = 0;
  for (const auto i : arange(reference_tv->nDims())) {
    if (reference_tv->axis(i)->isBroadcast()) {
      continue;
    }
    if (reference_tv->axis(i)->isReduction() &&
        reference_tv->axis(i)->isRFactorProduct()) {
      non_broadcast_rfactor_axes_ir.insert(non_broadcast_pos_ir);
    }
    non_broadcast_pos_ir++;
  }

  for (auto reduction_tv_ : reduction_tvs) {
    if (reduction_tv_ == reduction_tv ||
        reduction_tv_->definition()->isA<GroupedReductionOp>()) {
      // This should come in already rfactored
      continue;
    } else {
      ir_utils::rFactorHelper(
          reduction_tv_,
          reduction_scheduler_utils::addBackBroadcasts(
              reduction_tv_, non_broadcast_rfactor_axes_ir));
    }
  }
}

std::unordered_set<TensorView*> getCachedTvsToUnrollOrVectorize(
    TensorView* reference_tv,
    bool vectorize,
    const std::vector<TensorView*>& cached_inputs,
    const std::vector<std::pair<TensorView*, TensorView*>>& cached_outputs) {
  auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
  // Grab all tensor views that should be vectorized
  auto vectorizable_inputs_outputs =
      scheduler_utils::getInputsOutputsWithInnerDim(reduced_tv, true, true);

  auto vectorizable_expr = [](Expr* e) { return e->isA<LoadStoreOp>(); };

  std::unordered_set<TensorView*> unroll_vectorizable_tvs;
  for (auto cached_input : cached_inputs) {
    if (vectorize) {
      auto producer_tvs = ir_utils::producerTvsOf(cached_input);
      if (producer_tvs.size() == 1 &&
          vectorizable_expr(cached_input->definition()) &&
          std::find(
              vectorizable_inputs_outputs.begin(),
              vectorizable_inputs_outputs.end(),
              producer_tvs[0]) != vectorizable_inputs_outputs.end()) {
        unroll_vectorizable_tvs.emplace(cached_input);
      }
    } else {
      unroll_vectorizable_tvs.emplace(cached_input);
    }
  }

  for (auto cached_output_pair : cached_outputs) {
    auto output = cached_output_pair.second;
    if (vectorize) {
      if (vectorizable_expr(output->definition()) &&
          std::find(
              vectorizable_inputs_outputs.begin(),
              vectorizable_inputs_outputs.end(),
              output) != vectorizable_inputs_outputs.end()) {
        unroll_vectorizable_tvs.emplace(output);
      }
    } else {
      unroll_vectorizable_tvs.emplace(output);
    }
  }

  return unroll_vectorizable_tvs;
}

namespace {

// Clear unroll or vectorization parallelization for reduction_tv and
// reference_tv if they shouldn't be unrolled or vectorized.
// When group reduction is used, convert vectorization to group parallelization
// and propagate group parallelization to other reduction tvs.

// Parameters:
//   reduction_tv: The reduction TensorView being scheduled and parallelized.
//                 Needs to clear its vectorization or convert to grouped
//                 reduction.

//   reference_tv: The reference TensorView being scheduled and parallelized,
//                 Needs to clear its vectorization.

//   use_grouped_reduction: Indicates if group reduction is used in the
//                          scheduler.

//   reduction_tvs: All reduction TensorViews in the fusion. May add grouped
//                  parallelization.
//
//   unroll_vectorizable_cached_tvs: Cached TensorViews that are unrollable
//                                   or vectorizable.
void clearUnrollVectorizationAddGroupReduction(
    TensorView* reduction_tv,
    TensorView* reference_tv,
    const bool use_grouped_reduction,
    const std::vector<TensorView*>& reduction_tvs,
    const std::unordered_set<TensorView*>& unroll_vectorizable_cached_tvs) {
  std::vector<TensorView*> rfactor_and_reduction_tvs = {
      reference_tv, reduction_tv};
  bool is_inner_reduction =
      scheduler_utils::isFastestDimReduction(reduction_tv);
  auto convertParallelToGrouped = [&is_inner_reduction](IterDomain* id) {
    auto pt = id->getParallelType();
    // For inner reduction, convert outer dim unroll to group.
    // For outer reduction, convert inner dim vectorization to group.
    if (is_inner_reduction) {
      return pt == ParallelType::Unroll && !id->isReduction();
    } else {
      return pt == ParallelType::Vectorize;
    }
  };
  for (auto tv : rfactor_and_reduction_tvs) {
    if (unroll_vectorizable_cached_tvs.count(tv) != 0) {
      continue;
    }
    for (const auto i : arange(tv->nDims())) {
      auto id = tv->axis(i);
      if (use_grouped_reduction &&
          std::find(reduction_tvs.begin(), reduction_tvs.end(), tv) !=
              reduction_tvs.end() &&
          convertParallelToGrouped(id)) {
        tv->axis(i)->parallelize(ParallelType::Group);
        for (auto sibling : ir_utils::siblingTvsOf(tv)) {
          sibling->axis(i)->parallelize(ParallelType::Group);
        }
      } else if (
          id->getParallelType() == ParallelType::Unroll ||
          id->getParallelType() == ParallelType::Vectorize) {
        tv->axis(i)->parallelize(ParallelType::Serial);
        for (auto sibling : ir_utils::siblingTvsOf(tv)) {
          sibling->axis(i)->parallelize(ParallelType::Serial);
        }
      }
    }
  }

  // Propagate group to other reduction tvs
  if (use_grouped_reduction && reduction_tvs.size() > 1) {
    std::vector<TensorView*> other_reduction_tvs;
    std::copy_if(
        reduction_tvs.begin(),
        reduction_tvs.end(),
        std::back_inserter(other_reduction_tvs),
        [&](auto tv) { return reduction_tv != tv; });
    scheduler_utils::parallelizeAllLike(
        reduction_tv, -1, other_reduction_tvs, {ParallelType::Group});
  }
}
} // namespace

void propagateParallelization(
    TensorView* reduction_tv,
    TensorView* reference_tv,
    const bool is_unroll_or_vectorization,
    const bool use_grouped_reduction,
    const std::vector<TensorView*>& reduction_tvs,
    const std::unordered_set<TensorView*>& unroll_vectorizable_cached_tvs,
    const std::vector<TensorView*>& selected_tvs,
    const bool skip_input_output_unroll) {
  // Propagate parallelization except vectorization and unrolling
  scheduler_utils::parallelizeAllLike(
      reference_tv,
      -1,
      selected_tvs,
      allParallelTypesExcept({ParallelType::Unroll, ParallelType::Vectorize}));

  if (is_unroll_or_vectorization) {
    if (!unroll_vectorizable_cached_tvs.empty()) {
      std::unordered_set<ParallelType> selected_pts{ParallelType::Vectorize};
      if (!skip_input_output_unroll) {
        selected_pts.insert(ParallelType::Unroll);
      }
      // Propagate vectorization/unrolling to those tensors that need it
      scheduler_utils::parallelizeAllLike(
          reference_tv,
          -1,
          {unroll_vectorizable_cached_tvs.begin(),
           unroll_vectorizable_cached_tvs.end()},
          selected_pts);
    }
    // If reference shouldn't be unrolled, clear that parallel type.
    // In the case of outer grid persistence, replace Vector with Group.
    clearUnrollVectorizationAddGroupReduction(
        reduction_tv,
        reference_tv,
        use_grouped_reduction,
        reduction_tvs,
        unroll_vectorizable_cached_tvs);
  }
}

namespace {

// Convert properties of an ID to a numeric value
int idPos(const IterDomain* id) {
  int inner_most = std::numeric_limits<int>::max();
  int outer_most = std::numeric_limits<int>::min();

  // Reduction and unrolled
  if (id->isReduction() &&
      (id->getParallelType() == ParallelType::Unroll ||
       id->getParallelType() == ParallelType::Vectorize)) {
    return inner_most;
  }
  inner_most--;

  // Reduction and constant
  if (id->isReduction() && id->extent()->isConstScalar()) {
    return inner_most;
  }
  inner_most--;

  // Reduction and unswitched
  if (id->isReduction() && id->getParallelType() == ParallelType::Unswitch) {
    return inner_most;
  }
  inner_most--;

  // Reduction and thread
  if (id->isReduction() && id->isThread()) {
    return inner_most;
  }
  inner_most--;

  // Broadcast
  if (id->isBroadcast() || id->isImplicitBroadcast()) {
    return inner_most;
  }
  inner_most--;

  // Iter and unrolled
  if (!id->isReduction() &&
      (id->getParallelType() == ParallelType::Unroll ||
       id->getParallelType() == ParallelType::Vectorize)) {
    return inner_most;
  }
  inner_most--;

  // Iter and unswitched
  if (!id->isReduction() && id->getParallelType() == ParallelType::Unswitch) {
    return inner_most;
  }
  inner_most--;

  // Reduction and non-constant
  if (id->isReduction() && !id->extent()->isConstScalar()) {
    return inner_most;
  }
  inner_most--;

  // Iter and device (outer)
  if (!id->isReduction() && id->isDeviceDim()) {
    return outer_most;
  }
  outer_most++;

  // Iter and block (outer)
  if (!id->isReduction() && id->isBlockDim()) {
    return outer_most;
  }
  outer_most++;

  // Iter and thread (outer)
  if (!id->isReduction() && id->isThreadDim()) {
    return outer_most;
  }
  outer_most++;

  // Iter and constant
  if (!id->isReduction() && id->extent()->isConstScalar()) {
    return outer_most;
  }
  outer_most++;

  // Iter and non-constant
  if (!id->isReduction() && !id->extent()->isConstScalar()) {
    return outer_most;
  }
  outer_most++;

  return 0;
}

// Return if id0 should be before id1
bool placedBefore(const IterDomain* id0, const IterDomain* id1) {
  return idPos(id0) < idPos(id1);
}
} // namespace

TensorView* sortAndRFactor(
    TensorView* reference_tv,
    bool is_non_persistent_outer_reduction) {
  auto domain = reference_tv->getLoopDomain();
  std::sort(domain.begin(), domain.end(), placedBefore);
  std::unordered_map<int64_t, int64_t> reorder_map;
  std::unordered_map<IterDomain*, int64_t> domain_pos;
  for (auto axis_i : arange(static_cast<int64_t>(domain.size()))) {
    domain_pos[domain[axis_i]] = axis_i;
  }
  for (int64_t old_i : arange(reference_tv->nDims())) {
    reorder_map[old_i] = domain_pos.at(reference_tv->axis(old_i));
  }
  reference_tv->reorder(reorder_map);
  // For outer reduction, if an Id after vectorization Id is a constant
  // serial Id, swap it with the vectorization Id to reduce register usage.
  // For example, in a thread-local outer reduction, we want to transform:
  //   [..., iV{8}, rS{7}, rUS{1}, rUR{4}]
  // to:
  //   [..., rS{7}, iV{8}, rUS{1}, rUR{4}]
  // After change, each thread only needs to cache 8 × 4 elements instead of
  // 8 × 7 × 4 elements.
  // See https://github.com/NVIDIA/Fuser/issues/4172 for real examples.
  if (is_non_persistent_outer_reduction) {
    auto vect_iter =
        std::find_if(domain.begin(), domain.end(), [](IterDomain* id) {
          return id->getParallelType() == ParallelType::Vectorize;
        });
    if (vect_iter != domain.end()) {
      int64_t vect_id_pos = vect_iter - domain.begin();
      std::unordered_map<int64_t, int64_t> reorder_map;
      for (auto iter = vect_iter + 1; iter != domain.end(); iter++) {
        if ((*iter)->getParallelType() == ParallelType::Serial &&
            (*iter)->extent()->isConstScalar()) {
          int64_t id_pos = iter - domain.begin();
          reorder_map[id_pos] = vect_id_pos++;
        }
      }
      // Although we support reordering multiple constant serial IDs after the
      // vectorization ID, the current scheduler only emits one. It may be worth
      // exploring performance implications if multiple such IDs are introduced
      // in the future.
      NVF_ERROR(
          reorder_map.size() <= 1,
          "Expect one constant serial Id after vectorization Id, but found ",
          reorder_map.size());
      reference_tv->reorder(reorder_map);
    }
  }

  std::vector<int64_t> rfactor_axes;
  std::vector<int64_t> rfactor_axes_no_unswitch;
  size_t reduction_dims = 0;
  for (int64_t axis_i : arange(reference_tv->nDims())) {
    auto id = reference_tv->axis(axis_i);
    if (!id->isReduction()) {
      continue;
    }

    reduction_dims++;
    if (id->isThread()) {
      continue;
    }

    // We always want an rfactor axis because our inlining logic expects it. If
    // there's no parallelization to split out, just rfactor everything but the
    // unswitch dim.
    if (!(id->getParallelType() == ParallelType::Unswitch &&
          id->extent()->isOneInt())) {
      rfactor_axes_no_unswitch.push_back(axis_i);
    }
    rfactor_axes.push_back(axis_i);
  }

  if (reduction_dims == rfactor_axes.size()) {
    return ir_utils::rFactorHelper(reference_tv, rfactor_axes_no_unswitch);
  }

  return ir_utils::rFactorHelper(reference_tv, rfactor_axes);
}

namespace {
// If project_to_inputs is true, take all projectable persistent buffers,
// and move them to the inputs. Otherwise, try to project to their immediate
// producers if these producers are persistent buffers.
// This function create dummy outputs which should be used in later stages of
// the scheduling.
class PersistentBufferProjector {
 public:
  PersistentBufferProjector(
      Fusion* fusion,
      scheduler_utils::PersistentBufferInfo persistent_info,
      const bool project_to_inputs)
      : fusion_(fusion),
        persistent_info_(std::move(persistent_info)),
        persistent_buffers(persistent_info_.persistent_buffers),
        persistent_buffer_resolution_points(
            persistent_info_.persistent_buffer_resolution_points),
        projectable_persistent_buffers(
            persistent_info_.projectable_persistent_buffers),
        project_to_inputs_(project_to_inputs) {}

  const std::vector<TensorView*>& project() {
    if (project_to_inputs_) {
      projectToInputs();
    } else {
      projectToProducers();
    }
    return dummy_outputs_;
  }

 private:
  Fusion* fusion_;
  const scheduler_utils::PersistentBufferInfo persistent_info_;
  const std::vector<TensorView*>& persistent_buffers;
  const std::vector<std::vector<TensorView*>>&
      persistent_buffer_resolution_points;
  const std::vector<TensorView*>& projectable_persistent_buffers;
  std::vector<TensorView*> dummy_outputs_;
  const bool project_to_inputs_;

  void projectToInputs() {
    // Iterate through projected buffers, tracking which index it corresponds
    // too since there's a resolution point entry for every buffer.
    const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion_);
    for (auto buffer_i : arange(persistent_buffers.size())) {
      auto buffer = persistent_buffers[buffer_i];
      if (std::find(
              projectable_persistent_buffers.begin(),
              projectable_persistent_buffers.end(),
              buffer) == projectable_persistent_buffers.end()) {
        continue;
      }
      // when project to inputs, if the buffer depends on reduction tvs,
      // additional reduction is required to re-calculate the buffer.
      // Consider the following fusion where f() is a trivial op.
      // t1 = f(t0); t2 = sum(t1); t3 = broadcast(t2); t4 = add(t1, t3);
      // t5 = f(t4); t6 = sum(t5); t7 = broadcast(t6); t8 = add(t5, t7);
      // In this case t0 is input, t1 and t5 are persistent buffers.
      // Re-calculation of t1 from t0 is trivial, just f(t0).
      // Re-calculation of t5 from t0 needs t0->t1->t2->t3->t4->t5 where
      // t1->t2 is a reduction, which is considered very expensive and should
      // be avoided. Since t3 is a broadcast tv, all the persitent batches are
      // sharing the same value. It can be considered as a `free` persistent
      // buffer. So, t5 can be re-calculated directly from t3, this skips the
      // reduciton and broadcast from input t0 to t3. The broadcast here is not
      // just a local register copy but involves an inter-thread communication.
      std::vector<Val*> vals_project_to = fusion_->inputs();
      const auto& [can_project, broadcast_tvs] =
          scheduler_utils::canProjectToInputsWithoutReduction(
              reduction_tvs, buffer);
      if (can_project) {
        vals_project_to.insert(
            vals_project_to.end(), broadcast_tvs.begin(), broadcast_tvs.end());
      }

      projectToInputOrImmediatePersistentProducer(
          (int)buffer_i, vals_project_to);
    }
  }

  void projectToProducers() {
    // visit consumer before producer. e.g.
    // T1 = f(T0); Tx = add(T1, broadcast(sum(T1)));
    // T2 = f(T1); Ty = add(T2, broadcast(sum(T2)));
    // T3 = f(T2); Tz = add(T3, broadcast(sum(T3)));
    // T1, T2, T3 are persistent buffers.
    // The visiting order should be [T3, T2, T1].
    // After project T3 to its producers, we have:
    // Tz = add(f(T2),broadcast(sum(T3)));
    // After project T2 to its producers, we have:
    // Tz = add(f(f(T1)),broadcast(sum(T3)));
    // Ty = add(f(T1), broadcast(sum(T2)));
    // At last, the only persistent buffer is T1.
    // For a solid case, see NVFuserTest.ChainProjectionToPersistentProducer.
    std::vector<int> visiting_order(persistent_buffers.size());
    std::iota(visiting_order.begin(), visiting_order.end(), 0);
    std::stable_sort(
        visiting_order.begin(), visiting_order.end(), [this](int a, int b) {
          return !DependencyCheck::isDependencyOf(
              persistent_buffers[a], persistent_buffers[b]);
        });

    // try to project buffer to its producers when
    // (1) all producers are persistent buffers
    // (2) or, the buffer is the input to an upcast op
    std::unordered_set<TensorView*> persistent_buffer_set(
        persistent_buffers.begin(), persistent_buffers.end());
    for (auto buffer_i : visiting_order) {
      auto buffer = persistent_buffers[buffer_i];
      const auto& producers = ir_utils::producerTvsOf(buffer);
      if (scheduler_utils::canProjectToPersistentProducer(
              buffer, producers, persistent_buffer_set)) {
        projectToInputOrImmediatePersistentProducer(
            (int)buffer_i,
            std::vector<Val*>(producers.begin(), producers.end()));
      } else if (
          auto upcast_input = scheduler_utils::getUpCastInputOf(buffer)) {
        // Similar to projecting to inputs and persistent producers, this logic
        // projects the buffer to its producer when the buffer is the output of
        // an upcast op. This optimization reduces buffer size and can be
        // extended to project to low-precision intermediate tensors, even if
        // the recomputation involves non-cast ops. However, this should be
        // avoided when the recomputation cost outweighs the benefits of reduced
        // register usage.
        // TODO: extend to allow non-cast ops in the recomputation.
        auto consumers = ir_utils::consumerTvsOf(buffer);
        for (auto i : arange(1, consumers.size())) {
          ir_utils::replaceValInExprInputs(
              consumers.at(i)->definition(),
              buffer,
              RecomputeTv::recompute(buffer, {upcast_input}));
        }
      }
    }
  }

  // get all uses of the persistent buffer
  std::vector<Val*> getPersistentUseOfBuffer(int buffer_i) {
    std::vector<Val*> persistent_use_of_buffer;
    // Go through the resolution points one by one. Resolution points are points
    // in which the reduction branch meets the residual branch. These are points
    // where the persitent buffer may no longer be needed (one point could be
    // after another, and the buffer would be needed until the last resolution
    // points)
    auto buffer = persistent_buffers[buffer_i];
    auto resolution_points = persistent_buffer_resolution_points[buffer_i];
    for (auto resolution_point : resolution_points) {
      // Need to go through all paths from the persistent buffer to the
      // resolution point
      auto chains_to_resolution =
          DependencyCheck::getAllDependencyChains(buffer, resolution_point);
      for (auto chain : chains_to_resolution) {
        auto tv_chain = ir_utils::filterByType<TensorView>(chain);

        // To move the persistent buffers to the inputs, we need to recompute
        // the persistent buffer for all branches that don't go through a
        // reduction. If there's a reduction on the current path between the
        // persistent buffer and resolution, continue, there's no need to
        // replicate this use.
        if (std::any_of(tv_chain.begin(), tv_chain.end(), [](TensorView* tv) {
              return tv->hasReduction();
            })) {
          continue;
        }

        // Grab use of the buffer, chain[0] is the persistent buffer, chain[1]
        // is its first use.
        auto use = chain[1];

        // Only grab unique uses, a persistent buffer could be used multiple
        // times in the same expression.
        if (std::find(
                persistent_use_of_buffer.begin(),
                persistent_use_of_buffer.end(),
                use) != persistent_use_of_buffer.end()) {
          continue;
        }
        persistent_use_of_buffer.emplace_back(use);
      }
    }
    return persistent_use_of_buffer;
  }

  void projectToInputOrImmediatePersistentProducer(
      int buffer_i,
      const std::vector<Val*>& producers) {
    // For all uses that do not go towards the reduction operations in the
    // persistent section of the graph, recompute the persistent buffer.
    auto buffer = persistent_buffers[buffer_i];
    for (auto use : getPersistentUseOfBuffer(buffer_i)) {
      NVF_ERROR(use->definition() != nullptr);
      auto buffer_replicate = RecomputeTv::recompute(buffer, producers);
      // Create a shortcut buffer <--> buffer_replicate for propagation.
      // Why is this needed?
      // Consider that we have a fusion
      //
      //   T0[I]
      //   T1[b b I] = broadcast(T0)
      //   T2[b b r] = reduction(T1)
      //   T3[b b b] = broadcast(T2)
      //   T4[b, b, I] = T1 + T3
      //   T5[b, b, r] = reduction(T4)
      //
      // After projection, it becomes
      //
      //   T0[I]
      //   T1[b b I] = broadcast(T0)
      //   T2[b b r] = reduction(T1)
      //   T3[b b b] = broadcast(T2)
      //   T6[b b I] = broadcast(T0)
      //   T4[b, b, I] = T6 + T3
      //   T5[b, b, r] = reduction(T4)
      //
      // During schedule, we need to propagate from T2 to T5. However, in the
      // resulting DAG, neither the propagation path T2->T3->T4->T5 nor
      // T2->T1->T0->T6->T4->T5 works because they both have missing root
      // domain. But adding `T7 = T1 + T6` creates a new propagation path
      // `T2->T1->T7->T6->T4->T5` which has all root domain information.
      // See FusionBroadcastPersistentReduction_CUDA for an example
      // avoid replacing the use with itself, see
      // https://github.com/NVIDIA/Fuser/issues/1533
      if (buffer != buffer_replicate) {
        dummy_outputs_.emplace_back(add(buffer_replicate, buffer));
        ir_utils::replaceValInExprInputs(
            use->definition(), buffer, buffer_replicate);
      }
    }
  }
};
} // namespace
std::vector<TensorView*> projectPersistentBuffers(
    Fusion* fusion,
    const scheduler_utils::PersistentBufferInfo& persistent_info,
    const bool project_to_inputs) {
  PersistentBufferProjector pb_projector(
      fusion, persistent_info, project_to_inputs);
  return pb_projector.project();
}

ReductionType getReductionType(const std::vector<TensorView*>& reduction_tvs) {
  bool is_inner_reduction = false;
  bool is_outer_reduction = false;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      is_inner_reduction = true;
    } else {
      is_outer_reduction = true;
    }
  }
  if (is_inner_reduction && is_outer_reduction) {
    return ReductionType::InnerOuter;
  } else if (is_inner_reduction) {
    return ReductionType::Inner;
  } else if (is_outer_reduction) {
    return ReductionType::Outer;
  } else {
    return ReductionType::None;
  }
}

ReductionType getReductionType(Fusion* fusion) {
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  return getReductionType(reduction_tvs);
}

std::string toString(ReductionType reduction_type) {
  switch (reduction_type) {
    case ReductionType::Inner:
      return "InnerReduction";
    case ReductionType::Outer:
      return "OuterReduction";
    case ReductionType::InnerOuter:
      return "InnerOuterReduction";
    case ReductionType::None:
      return "NoneReduction";
    default:
      NVF_THROW("undefined ReductionType");
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, ReductionType reduction_type) {
  os << toString(reduction_type);
  return os;
}

void sharedMemoryConsumerVectorization(
    std::vector<TensorView*>& smem_consumers,
    int64_t io_vectorization_factor) {
  for (auto tv : smem_consumers) {
    // they were creatd with cacheAfter.
    NVF_ERROR(
        tv->definition()->isA<LoadStoreOp>(),
        "smem consumers should be LoadStoreOp. Got: ",
        tv->definition()->toString());

    // non-concretized broadcast domains are moved to the innermost before
    // transform propagation, should skip these axes.
    int64_t vect_axis_pos = -1;
    while (tv->axis(vect_axis_pos)->isBroadcast()) {
      vect_axis_pos--;
      NVF_ERROR(
          vect_axis_pos + tv->nDims() >= 0,
          "Out of bound access when visiting dim ",
          vect_axis_pos,
          " in Tv: ",
          tv->toString());
    }
    // they were transformed with innermost axis has extent equal to
    // vectorization factor set for io tvs.
    NVF_ERROR(
        tv->axis(vect_axis_pos)->extent()->isConst(),
        "Extent of the innermost axis of smem consumers should be constant. "
        "Got: ",
        tv->toString());
    auto innermost_extent =
        tv->axis(vect_axis_pos)->extent()->evaluate().as<int64_t>();
    NVF_ERROR(
        innermost_extent == io_vectorization_factor,
        "Extent of the innermost axis of smem consumers should be equal to the "
        "vectorization factor of fuion inputs and outputs. Got: ",
        innermost_extent,
        ", expected: ",
        io_vectorization_factor);
    auto dtype_bits = dataTypeSizeBit(tv->getDataType().value());
    auto max_vect_factor = getMaxVectorizationSizeInBit() / dtype_bits;
    // additional split is added if the innermost extent is greater than max
    // vectorization factor.
    if (innermost_extent > max_vect_factor) {
      tv->split(vect_axis_pos, max_vect_factor);
    }
    tv->axis(vect_axis_pos)->parallelize(ParallelType::Vectorize);
  }
}

int64_t getComputeBdimx(ParallelType warp_specialized_on, int64_t bdimx) {
  return warp_specialized_on == ParallelType::TIDx
      ? bdimx - kWarpSpecializationPaddedThreads
      : bdimx;
}

int64_t getComputeBdimx(
    const CircularBufferOptions& circular_buffer_opt,
    int64_t bdimx) {
  return (circular_buffer_opt.isEnable() &&
          std::holds_alternative<WarpSpecialized>(circular_buffer_opt.type) &&
          std::get<WarpSpecialized>(circular_buffer_opt.type).on ==
              ParallelType::TIDx)
      ? bdimx - kWarpSpecializationPaddedThreads
      : bdimx;
}
} // namespace reduction_scheduler_utils
} // namespace nvfuser
