// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/normalization_utils.h>
#include <scheduler/registry.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>

#include <bfs.h>
#include <contiguity.h>
#include <expr_evaluator.h>
#include <id_model/id_model.h>
#include <id_model/schedule.h>
#include <instrumentation.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>
#include <scheduler/runtime_info.h>
#include <transform_iter.h>
#include <transform_replay.h>
#include <val_graph_visitor.h>

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <queue>
#include "scheduler/tools/loop_domain_scheduler.h"
#include "type.h"

namespace nvfuser {
namespace scheduler_utils {

// Returns number of "valid" dimensions. e.g. if tv has
// [I1, R2, I3, I4, R3{1}]
// where R3{1} is in dont_merge, resulting domain should be:
// [I1, I3*I4, R2, R3{1}] with return value 3
//
// if tv has
// [R1, I2, R3, I4, R4, R5{1}, R6{1}]
//  where R5{1} and R6{1} are in dont_merge, resulting domain should be:
// [I2*I4, R1*R3, R4, R5{1}, R6{1}]
// with return value 3
size_t merge_3d(TensorView* tv) {
  bool active_is_reduction = false;
  bool first_dim = true;
  int prev_i = -1;

  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (first_dim) {
      active_is_reduction = tv->axis(i)->isReduction();
      prev_i = i;
      first_dim = false;
    } else {
      if (tv->axis(i)->isReduction() != active_is_reduction) {
        break;
      }
      tv->merge(i, prev_i);
      prev_i = i;
    }
  }

  if (prev_i == -1) {
    // Zero dimensional
    return 0;
  }

  // put inner most dimension as last dimension
  tv->reorder({{prev_i, -1}});
  active_is_reduction = false;
  first_dim = true;
  prev_i = -1;

  for (int i = static_cast<int>(tv->nDims()) - 2; i >= 0; i--) {
    auto id = tv->axis(i);
    if (first_dim) {
      active_is_reduction = id->isReduction();
      prev_i = i;
      first_dim = false;
    } else if (id->isReduction() == active_is_reduction) {
      tv->merge(i, prev_i);
      prev_i = i;
    }
  }

  // put second dimension as second to last dimension
  if (prev_i == -1) {
    // One dimensional, put merged dimension as first
    tv->reorder({{-1, 0}});
    return 1;
  } else {
    // put new dimension as second to last
    tv->reorder({{prev_i, -2}});
  }

  active_is_reduction = false;
  first_dim = true;
  prev_i = -1;

  for (int i = static_cast<int>(tv->nDims()) - 3; i >= 0; i--) {
    if (first_dim) {
      active_is_reduction = tv->axis(i)->isReduction();
      prev_i = i;
      first_dim = false;
    } else if (tv->axis(i)->isReduction() == active_is_reduction) {
      tv->merge(i, prev_i);
      prev_i = i;
    }
  }

  // put third dimension as second to last dimension
  if (prev_i == -1) {
    // Two dimensional, put merged dimensions first
    tv->reorder({{-1, 0}, {-2, 1}});
    // [outer, inner, dont_merge...]
    if (tv->axis(0)->isReduction()) {
      // put reductions as second axis
      tv->reorder({{0, 1}, {1, 0}});
    }
    return 2;
  } else {
    // put new dimension as third to last
    tv->reorder({{prev_i, -3}});
    // Stable sort to have iteration domains first, then reduction
    if (tv->axis(0)->isReduction() && !tv->axis(1)->isReduction()) {
      tv->reorder({{0, 1}, {1, 0}});
    }
    if (tv->axis(1)->isReduction() && !tv->axis(2)->isReduction()) {
      tv->reorder({{1, 2}, {2, 1}});
    }
    if (tv->axis(0)->isReduction() && !tv->axis(1)->isReduction()) {
      tv->reorder({{0, 1}, {1, 0}});
    }
    return 3;
  }
}

void splitDims(
    TensorView* tv,
    std::vector<std::pair<int64_t, int64_t>> to_split, // (dim, size)
    std::vector<int64_t>& to_update) {
  std::stable_sort(
      to_split.begin(),
      to_split.end(),
      [](const std::pair<int64_t, int64_t>& p1,
         const std::pair<int64_t, int64_t>& p2) {
        return p1.first < p2.first;
      });
  int64_t dim_offset = 0;
  int64_t pending_dim_offset = 0;
  int64_t prev_dim = 0;
  for (auto entry : to_split) {
    int64_t dim = entry.first;
    int64_t size = entry.second;
    if (dim != prev_dim) {
      dim_offset += pending_dim_offset;
      pending_dim_offset = 0;
    }
    int64_t actual_dim = dim_offset + dim;
    tv->split(actual_dim, size);
    pending_dim_offset++;
    for (auto& i : to_update) {
      if (i > actual_dim) {
        i++;
      }
    }
    prev_dim = dim;
  }
}

std::optional<int64_t> mergeDims(
    TensorView* tv,
    std::vector<int64_t> to_merge,
    std::vector<int64_t>& to_update) {
  if (to_merge.empty()) {
    return std::nullopt;
  }
  if (to_merge.size() == 1) {
    return to_merge[0];
  }
  auto inner = to_merge[0];

  // NOTE: The merge is done in the order of `to_merge`, assuming going from
  // inner to outer dimensions. We want the merged IterDomain to be like:
  //
  // tv->axis(to_merge[i-1])*tv->axis(to_merge[i-2])*...*tv->axis(to_merge[0])
  //
  // Otherwise this could results in misaligned memory access due to indexing.
  // This is because we compute vectorization width before applying scheduling
  // transformations.
  for (int64_t i = 1; i < (int64_t)to_merge.size(); i++) {
    auto outer = to_merge[i];
    // If outer > inner, the merge order conflicts with their order in loop
    // domain
    if (outer > inner) {
      // NOTE: reorder here is necessary to work around the automatic swap in
      // `TensorDomain::merge`, if the first axis position is larger than the
      // second. we want to have the merge dimension be like
      // (tv->axis(to_merge[i]) * tv->axis(to_merge[i-1])), reorder allows us to
      // compensate the automatic swap in `TensorDomain::merge`.
      tv->reorder({{inner, outer}, {outer, inner}});
      // swapping inner with outer since we also need to keep track of the
      // actual outer position for the remaining merge operations as well as for
      // return value.
      std::swap(inner, outer);
    }
    // from
    //   (i..., tv->axis(outer), j..., tv->axis(inner), k...)
    // to
    //   (i..., tv->axis(outer) * tv->axis(inner), j..., k...)
    tv->merge(static_cast<int>(outer), static_cast<int>(inner));

    // compensate future indices for the diminishing inner.
    for (int64_t j = i + 1; j < (int64_t)to_merge.size(); j++) {
      if (to_merge[j] > inner) {
        to_merge[j]--;
      }
    }
    for (auto& val : to_update) {
      if (val == inner) {
        val = outer;
      } else if (val > inner) {
        val--;
      }
    }
    inner = outer;
  }
  return inner;
}

int64_t mergeReduction(TensorView* tv) {
  int prev_i = -1;
  int64_t num_merged = 0;
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (!tv->axis(i)->isReduction()) {
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i != 0) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

int64_t mergeNonReduction(TensorView* tv) {
  bool has_device_dim = false;
  int prev_i = -1;
  int64_t num_merged = 0;
  if (tv->nDims() == 0) {
    return 0;
  }
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (tv->axis(i)->isReduction()) {
      continue;
    }
    if (tv->axis(i)->isDeviceDim()) {
      has_device_dim = true;
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i != -1) {
    tv->reorder({{prev_i, 0}});
  }
  if (has_device_dim) {
    // in this case the layout at this point is [i, r , d]
    // we want to put the device dim back to outmost
    tv->reorder({{prev_i != -1 ? 2 : 1, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

void parallelizeAllLike(
    TensorView* reference_tv,
    int64_t pos,
    std::vector<TensorView*> selected_tvs,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    bool propagate_padding,
    bool parallelize_inputs_on_did) {
  FusionGuard fg(reference_tv->fusion());

  if (pos < 0) {
    pos += (int64_t)reference_tv->nDims() + 1;
  }
  NVF_CHECK(
      pos >= 0 && pos <= (int64_t)reference_tv->nDims(),
      "parallelizeAllLike called on an position outside valid range.");

  std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;

  auto ca_map = ComputeAtMap(FusionGuard::getCurFusion());

  const auto& reference_dom = reference_tv->getLoopDomain();
  for (auto it = reference_dom.begin(); it != reference_dom.begin() + pos;
       it++) {
    auto ca_id =
        ca_map.getConcreteMappedID(*it, IdMappingMode::PERMISSIVE_RESIZE);
    concrete_to_reference_map[ca_id] = *it;
  }

  if (selected_tvs.empty()) {
    selected_tvs = reference_tv->fusion()->allTvs();
  }
  for (auto tv : selected_tvs) {
    if (tv->isFusionInput() && !parallelize_inputs_on_did) {
      continue;
    }
    bool is_fusion_input = tv->isFusionInput();
    for (const auto i : arange((int64_t)tv->getLoopDomain().size())) {
      auto ca_id = ca_map.getConcreteMappedID(
          tv->axis(i), IdMappingMode::PERMISSIVE_RESIZE);
      if (concrete_to_reference_map.count(ca_id) == 0) {
        continue;
      }
      auto reference_id = concrete_to_reference_map.at(ca_id);
      auto reference_parallel_type = reference_id->getParallelType();
      if (is_fusion_input &&
          !isParallelTypeDeviceDim(reference_parallel_type)) {
        continue;
      }
      if (selected_parallel_types.empty() ||
          selected_parallel_types.count(reference_parallel_type)) {
        tv->axis(i)->parallelize(reference_parallel_type);
      }
      if (propagate_padding) {
        if (reference_id->hasPaddingToMultipleOfWarp()) {
          tv->axis(i)->padToMultipleOfWarp(
              reference_id->getMaybeSizeAfterPadding());
        }
      }
    }
  }
}

namespace {

// Find the resolution points of the persistent buffers in the provided
// persistent_buffer_info. Resolution points are identified by tracking if a
// tensor view is dependent on a reduction, or a persistent buffer. When an
// expression has inputs that are on both a reduction and persistent buffer
// path, that's a point where we may be resolving the persistent buffer. In
// other words, we know the persistent buffer has to be live at that point, but
// don't know if it has to be live after it.
//
// For example if we have:
//
// t0 = makeSymbolicTensor(2)
// t1 = set(t0)
// t2 = sum(t1, 1)
// t3 = broadcast(t2, {false, true})
// t4 = set(t1)
// t5 = add(t4, t3)
//
// In this case, t1 is the persistent buffer, that buffer is resolved at t5, so
// it needs to exist in full until t5 is "resolved". This class assumes all
// reduction patterns in the fusion are matching.
class PersistentBufferResolution : public IterVisitor {
 public:
  static std::vector<TensorView*> getResolutionPointsOf(
      Fusion* fusion,
      TensorView* persistent_buffer) {
    PersistentBufferResolution resolution(fusion, persistent_buffer);
    return resolution.resolution_points_;
  }

  PersistentBufferResolution() = delete;

 private:
  PersistentBufferResolution(Fusion* fusion, TensorView* persistent_buffer)
      : persistent_buffer_(persistent_buffer) {
    traverse(fusion);
  }

 private:
  void dispatch(Val* val) final {
    if (!val->isA<TensorView>()) {
      return;
    }
    auto tv = val->as<TensorView>();
    if (tv == persistent_buffer_) {
      persistent_buffer_hit = true;
      on_persitent_buffer_path_.emplace(tv);
      return;
    }

    if (!persistent_buffer_hit) {
      return;
    }

    if (tv->hasReduction()) {
      if (std::any_of(
              resolution_points_.begin(),
              resolution_points_.end(),
              [&tv](TensorView* resolution_point) {
                return DependencyCheck::isDependencyOf(resolution_point, tv);
              })) {
        // If already resolved, don't start a new reduction path.
        return;
      }
      on_reduction_path_.emplace(tv);
    }
  }

  void dispatch(Expr* expr) final {
    if (!persistent_buffer_hit) {
      return;
    }

    bool output_is_reduction =
        std::any_of(expr->outputs().begin(), expr->outputs().end(), [](Val* v) {
          if (!v->isA<TensorView>()) {
            return false;
          }
          return v->as<TensorView>()->hasReduction();
        });

    // Persistent buffers cannot be resolved on a reduction expression
    if (output_is_reduction) {
      return;
    }

    bool input_on_reduction_path = std::any_of(
        expr->inputs().begin(), expr->inputs().end(), [&](Val* inp) {
          return on_reduction_path_.count(inp);
        });

    auto input_on_persitent_buffer_path_it = std::find_if(
        expr->inputs().begin(), expr->inputs().end(), [&](Val* inp) {
          return on_persitent_buffer_path_.count(inp);
        });

    bool input_on_persistent_buffer_path =
        input_on_persitent_buffer_path_it != expr->inputs().end();

    if (input_on_reduction_path && input_on_persistent_buffer_path) {
      // Expression has inputs on both a reduction and persistent buffer path,
      // this is a resolution.
      auto out_tvs = ir_utils::filterByType<TensorView>(expr->outputs());

      // Add resolution point
      resolution_points_.insert(
          resolution_points_.end(), out_tvs.begin(), out_tvs.end());

      // Outputs are still on a persistent path
      for (auto out : expr->outputs()) {
        on_persitent_buffer_path_.emplace(out);
      }
    } else if (input_on_reduction_path) {
      // Propagate forward the reduction path
      on_reduction_path_.insert(expr->outputs().begin(), expr->outputs().end());
    } else if (input_on_persistent_buffer_path) {
      // Propagate forward the persistent path
      for (auto out : expr->outputs()) {
        on_persitent_buffer_path_.emplace(out);
      }
    }
  }

  // Don't do processing until we see the buffer we're looking for
  bool persistent_buffer_hit = false;

  // Track if key is dependent on a persistent reduction, resolves if
  // encountering a persistent buffer. For this analysis doesn't matter which
  // reduction the path is based on.
  std::unordered_set<Val*> on_reduction_path_;

  // Track if key is dependent on a persistent buffer, resolves if encountering
  // a persistent reduction or changes path if encountering another persistent
  // buffer
  std::unordered_set<Val*> on_persitent_buffer_path_;

  // Tracks where the persistent buffer (key) is resolved (values)
  std::vector<TensorView*> resolution_points_;

  const TensorView* persistent_buffer_;
};

} // namespace

namespace {
// This function checks if there is a broadcast tv in the dependencies between
// the reduction_tv and the persistent_buffer. A tv is considered projectable
// if its definition is a broadcast, has the same number of dimensions as the
// reduction_tv, and each reduction dimension in reduction_tv corresponds to
// a broadcast dimension in the broadcast tv.
// Return the broadcast tv if there is one, otherwise return nullptr.
TensorView* getBufferProjectableBroadcastsTv(
    TensorView* reduction_tv,
    TensorView* persistent_buffer) {
  const auto& dep_vals =
      DependencyCheck::getAllValsBetween({reduction_tv}, {persistent_buffer});
  for (auto val : dep_vals) {
    if (auto tv = dynamic_cast<TensorView*>(val)) {
      if (!tv->definition()->isA<BroadcastOp>()) {
        continue;
      }
      if (reduction_tv->nDims() != tv->nDims()) {
        continue;
      }
      // Each reduction dimension the producer, must be mapped to a broadcast
      // dimension in the consumer, otherwise it is not a valid broadcast after
      // reduction.
      bool is_broadcast_after_reduction = true;
      for (auto i : arange(reduction_tv->nDims())) {
        if (reduction_tv->axis(i)->isReduction() &&
            !tv->axis(i)->isBroadcast()) {
          is_broadcast_after_reduction = false;
          break;
        }
      }
      if (is_broadcast_after_reduction) {
        return tv;
      }
    }
  }
  return nullptr;
}
} // namespace

std::pair<bool, std::vector<TensorView*>> canProjectToInputsWithoutReduction(
    const std::vector<TensorView*> reduction_tvs,
    TensorView* persistent_buffer) {
  std::vector<TensorView*> dep_reduction_tvs, target_broadcast_tvs;
  dep_reduction_tvs.reserve(reduction_tvs.size());
  for (auto tv : reduction_tvs) {
    if (DependencyCheck::isDependencyOf(tv, persistent_buffer)) {
      dep_reduction_tvs.push_back(tv);
    }
  }
  // (1) The persistent buffer doesn't depend on any reduction tv
  if (dep_reduction_tvs.empty()) {
    return std::make_pair(true, target_broadcast_tvs);
  }
  // (2) It depends on reduction tv(s), but after each reduction tv, there is a
  // broadcasted tv can be projected to.
  target_broadcast_tvs.reserve(dep_reduction_tvs.size());
  for (auto reduction_tv : dep_reduction_tvs) {
    auto broadcast_tv =
        getBufferProjectableBroadcastsTv(reduction_tv, persistent_buffer);
    if (!broadcast_tv) {
      return std::make_pair(false, target_broadcast_tvs);
    }
    target_broadcast_tvs.push_back(broadcast_tv);
  }
  return std::make_pair(true, target_broadcast_tvs);
}

TensorView* getUpCastInputOf(const TensorView* tv) {
  // skip if definition is not a unary op
  if (auto uop = dynamic_cast<UnaryOp*>(tv->definition())) {
    // skip if the input is a fusion input or the op is not a cast
    if (uop->input(0)->isFusionInput() ||
        uop->getUnaryOpType() != UnaryOpType::Cast) {
      return nullptr;
    }
    // skip if the cast is not upcast
    auto precisions = ir_utils::getPrecisionOfProducerConsumerTensorsBit(uop);
    if (!precisions.has_value() || precisions->first >= precisions->second) {
      return nullptr;
    }
    return uop->input(0)->as<TensorView>();
  }
  return nullptr;
}

PersistentBufferInfo persistentBuffers(Fusion* fusion) {
  FusionGuard fg(fusion);
  PersistentBufferInfo persistent_buffer_info;

  ComputeAtLogicalDomainMap logical_map;
  logical_map.build();

  auto all_tvs = fusion->allTvs();

  // Find projectable persistent buffers
  auto reduction_tvs = getReductionTvs(fusion);

  // TODO: Reuse this id_model in getResolutionPointsOf
  IdModel id_model(fusion);
  std::vector<TensorView*> persistent_buffer_candidates;

  for (auto producer : all_tvs) {
    // Are all producer ids mappable to all consumers
    bool mappable = true;
    auto consumers = ir_utils::consumerTvsOf(producer);
    if (consumers.empty()) {
      continue;
    }

    for (auto consumer : consumers) {
      if (dynamic_cast<SelectOp*>(consumer->definition()) ||
          dynamic_cast<IndexSelectOp*>(consumer->definition()) ||
          dynamic_cast<GatherOp*>(consumer->definition())) {
        continue;
      }
      auto mappable_roots =
          logical_map.getMappableDims(producer->domain(), consumer->domain());

      auto p_logical = producer->getLogicalDomain();

      for (auto p_logical_id : p_logical) {
        if (p_logical_id->isReduction() || p_logical_id->isBroadcast()) {
          continue;
        }
        if (!mappable_roots.count(p_logical_id)) {
          mappable = false;
          persistent_buffer_info.unmappable_dims.emplace(p_logical_id);
        }
      }
    }

    if (mappable) {
      continue;
    }

    // If there's unmappable dims from producer to consumer, producer is a
    // persistent buffer. However, if it may not be possible to be
    // persistent due to broadcast inlining
    if (normalization_scheduler_utils::isCacheableUnmappableTv(
            producer,
            reduction_tvs,
            id_model.maybeBuildGraph(IdMappingMode::ALMOSTEXACT))) {
      persistent_buffer_candidates.emplace_back(producer);
    } else {
      persistent_buffer_info.non_persistent_buffers.emplace_back(producer);
    }
  }

  // Set the persistent buffer resolution points
  persistent_buffer_info.persistent_buffer_resolution_points = {};

  // PersistentBufferResolution::getResolutionPointsOf does not work
  // with non-straightline dependencies. See for example issue
  // #1123. normalization_scheduler_utils::getResolutionPointsOf
  // addresses the limitation, but those two functions currently do
  // not produce the same results even for fusions that
  // PersistentBufferResolution::getResolutionPointsOf can analyze. To
  // unblock #1123, the old analysis remains to be used for now, and
  // only when it fails to find resolution points, the new analysis is
  // used as a fallback option.
  // TODO: Completely replace the old analysis
  for (auto buffer : persistent_buffer_candidates) {
    auto resolution_points =
        PersistentBufferResolution::getResolutionPointsOf(fusion, buffer);
    if (resolution_points.empty()) {
      resolution_points = normalization_scheduler_utils::getResolutionPointsOf(
          buffer, id_model);
    }
    if (resolution_points.empty()) {
      continue;
    }
    persistent_buffer_info.persistent_buffers.emplace_back(buffer);
    persistent_buffer_info.persistent_buffer_resolution_points.emplace_back(
        resolution_points);
  }

  // don't project if there are view ops and no buffer can be projected
  persistent_buffer_info.has_view_ops = !ir_utils::getViewOps(fusion).empty();
  if (persistent_buffer_info.has_view_ops) {
    return persistent_buffer_info;
  }

  for (auto persistent_buffer : persistent_buffer_info.persistent_buffers) {
    // Inputs marked as persistent buffers can't be projected any further back
    if (persistent_buffer->isFusionInput()) {
      continue;
    }

    // can't project to input if the recomputation needs reduction
    if (!canProjectToInputsWithoutReduction(reduction_tvs, persistent_buffer)
             .first) {
      continue;
    }

    //  All inputs of the persistent buffer should be cacheable.
    auto all_inputs = ir_utils::inputTvsOf(persistent_buffer);
    if (std::all_of(
            all_inputs.begin(),
            all_inputs.end(),
            [&reduction_tvs, &id_model](TensorView* input) {
              return normalization_scheduler_utils::isCacheableUnmappableTv(
                  input,
                  reduction_tvs,
                  id_model.maybeBuildGraph(IdMappingMode::ALMOSTEXACT));
            })) {
      persistent_buffer_info.projectable_persistent_buffers.push_back(
          persistent_buffer);
    }
  }

  // Projection analysis below
  if (persistent_buffer_info.projectable_persistent_buffers.empty()) {
    return persistent_buffer_info;
  }

  // Get a list of inputs of the projectable buffers
  auto all_inputs = ir_utils::inputTvsOf(
      persistent_buffer_info.projectable_persistent_buffers);

  // Map unmappable dims to inputs, doesn't matter which compute at map used
  auto ca_map = ComputeAtMap(fusion);

  std::unordered_set<IterDomain*> unmappable_concrete_ids;
  for (auto id : persistent_buffer_info.unmappable_dims) {
    unmappable_concrete_ids.emplace(
        ca_map.getConcreteMappedID(id, IdMappingMode::EXACT));
  }

  for (auto input : all_inputs) {
    bool has_unmappable_dim = false;
    for (auto input_id : input->getLogicalDomain()) {
      auto concrete_input_id =
          ca_map.getConcreteMappedID(input_id, IdMappingMode::EXACT);
      if (unmappable_concrete_ids.find(concrete_input_id) !=
          unmappable_concrete_ids.end()) {
        persistent_buffer_info.unamppable_dims_projected_to_inputs.emplace(
            input_id);
        has_unmappable_dim = true;
      }
    }
    if (has_unmappable_dim) {
      persistent_buffer_info.projectable_buffer_inputs.emplace_back(input);
    }
  }

  // check ops between persistent buffer and inputs.
  // TODO: check more ops
  const auto all_exprs = StmtSort::getExprsBetween(
      {all_inputs.begin(), all_inputs.end()},
      {persistent_buffer_info.projectable_persistent_buffers.begin(),
       persistent_buffer_info.projectable_persistent_buffers.end()});
  for (auto expr : all_exprs) {
    if (expr->isA<UnaryOp>() &&
        expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Exp) {
      persistent_buffer_info.projection_with_exp_op = true;
    }

    if (expr->isA<RNGOp>()) {
      persistent_buffer_info.projection_with_rng_op = true;
    }
  }

  return persistent_buffer_info;
}

ReductionTvProperties getReductionProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* tv) {
  FusionGuard fg(fusion);

  NVF_ERROR(tv != nullptr);

  bool fastest_dim_reduction = isFastestDimReduction(tv);

  // Tracks the dimensionality of the problem starts on inner most dim and works
  // outward
  int64_t dimensionality = 1;
  // Initialize for dimensionality analysis
  bool cur_dim_is_reduction = fastest_dim_reduction;
  // Compute the size of the inner most dimension
  int64_t inner_most_dimension_numel = 1;
  int64_t inner_most_dimension_ndims = 0;

  // Start from the inner most dimension, and work outwards. If this is a 3D
  // pattern, i.e. theres a pattern like [r0, r1, i2, r3] or [i0, r1, r2, i3,
  // i4] then compute the inner most dimension to compute separately.
  const auto& root_dom = tv->getMaybeRootDomain();
  for (size_t i = root_dom.size(); i > 0; i--) {
    auto id = root_dom[i - 1];
    if (id->isBroadcast()) {
      continue;
    }
    if (id->isReduction() != cur_dim_is_reduction) {
      dimensionality++;
      cur_dim_is_reduction = !cur_dim_is_reduction;
    } else if (dimensionality == 1) {
      auto inferred_val =
          runtime_info.expressionEvaluator().evaluate(id->extent());
      NVF_ERROR(inferred_val.hasValue(), "Error inferring reduction size.");
      inner_most_dimension_numel =
          inner_most_dimension_numel * inferred_val.as<int64_t>();
      inner_most_dimension_ndims++;
    }
  }

  // Non reduction element count
  int64_t total_iteration_numel = 1;
  // Reduction element count
  int64_t total_reduction_numel = 1;

  for (auto id : root_dom) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(id->extent());
    NVF_ERROR(
        inferred_val.hasValue(),
        "Error inferring dimensions of reduction fusion.");
    if (id->isReduction()) {
      total_reduction_numel *= inferred_val.as<int64_t>();
    } else {
      total_iteration_numel *= inferred_val.as<int64_t>();
    }
  }

  ReductionTvProperties properties;
  properties.total_reduction_numel = total_reduction_numel;
  properties.total_iteration_numel = total_iteration_numel;
  properties.fastest_dim_reduction = fastest_dim_reduction;
  properties.inner_most_dimension_numel = inner_most_dimension_numel;
  properties.inner_most_dimension_ndims = inner_most_dimension_ndims;
  properties.dimensionality = dimensionality;

  return properties;
}

namespace {

// Figure out which persistent buffers are active at the generation of values in
// the fusion. This will be used at runtime to compute the size and max size of
// the persistent buffers.
std::unique_ptr<HeuristicCompileTime::ScopedPersistenceBufferMap>
getScopePersistenceFactors(
    Fusion* fusion,
    const PersistentBufferInfo& persistent_buffer_info) {
  auto new_persistent_factor_map_ptr =
      std::make_unique<HeuristicCompileTime::ScopedPersistenceBufferMap>();
  auto& new_persistent_factor_map = *new_persistent_factor_map_ptr;

  // Convenience accessors
  const auto& persistent_buffers = persistent_buffer_info.persistent_buffers;
  const auto& projectable_buffer_inputs =
      persistent_buffer_info.projectable_buffer_inputs;
  const auto& projectable_persistent_buffers =
      persistent_buffer_info.projectable_persistent_buffers;
  const auto& persistent_buffer_resolution_points =
      persistent_buffer_info.persistent_buffer_resolution_points;

  // Append projectable buffer inputs, going to compute size of those as well.
  auto persistent_buffers_and_inputs = persistent_buffers;
  persistent_buffers_and_inputs.insert(
      persistent_buffers_and_inputs.end(),
      projectable_buffer_inputs.begin(),
      projectable_buffer_inputs.end());

  for (auto persistent_buffer_i : arange(persistent_buffers.size())) {
    auto persistent_buffer = persistent_buffers[persistent_buffer_i];
    // All expressions between tv and its resolution points must have tv's
    // persistent buffer allocated. This is an optimistic view on how many
    // registers we need allocated in the kernel, since if we ordered two
    // persistent buffers that are completely independent to somehow overlap
    // with eachothers loop nests both persistent buffers would have to be
    // allocated at the same time even though this function would assume they
    // don't.
    //
    // Unfortunately this limitation is hard to work around as we would have
    // to actually generate the kernel before we know if it would fit
    // persistently in registers. In practice, though, this should not happen
    // as inlining loop structures where the persistent buffer is used should
    // prevent muiltiple persistent buffers from being merged togther if not
    // necessary.
    auto resolution_points =
        persistent_buffer_resolution_points[persistent_buffer_i];
    for (auto val : DependencyCheck::getAllValsBetween(
             {persistent_buffer},
             {resolution_points.begin(), resolution_points.end()})) {
      // Persistent normalization kernels imply that all persistent buffers
      // have the same dimensionality. Assume if a persistent buffer is
      // consumed by another we can alias and reuse the memory.
      if (val == persistent_buffer) {
        continue;
      }

      // All vals between resolution point and the corresponding buffer have
      // that buffer live during their generation.
      if (new_persistent_factor_map.find(val) ==
          new_persistent_factor_map.end()) {
        new_persistent_factor_map[val] =
            std::vector<bool>(persistent_buffers_and_inputs.size(), false);
      }
      new_persistent_factor_map.at(val)[persistent_buffer_i] = true;
    }
  }

  // Processing projectable persistent buffers is a little more complex, simply
  // because we have to line up inputs with their persistent buffers.

  // Offset into the bool vector
  size_t bool_vector_offset = persistent_buffers.size();
  for (auto projectable_persistent_buffer_i :
       arange(projectable_persistent_buffers.size())) {
    auto projectable_persistent_buffer =
        projectable_persistent_buffers[projectable_persistent_buffer_i];
    auto inputs = ir_utils::inputTvsOf(projectable_persistent_buffer);

    for (auto input : inputs) {
      auto input_it = std::find(
          projectable_buffer_inputs.begin(),
          projectable_buffer_inputs.end(),
          input);
      // If input wasn't recorded as a projectable buffer input, then it doesn't
      // have any persistent dims, so ignore it.
      if (input_it == projectable_buffer_inputs.end()) {
        continue;
      }

      // get inuput index entry in the buffer inputs vector
      auto input_i = std::distance(projectable_buffer_inputs.begin(), input_it);

      // Get the offset in the bool vector for this input
      input_i += (int64_t)bool_vector_offset;

      // If we project persistence from the persistent buffers to the inputs,
      // then it would have to be active from the resolution points of the
      // persistent buffer all the way back to the projected inputs.
      auto resolution_points =
          persistent_buffer_resolution_points[projectable_persistent_buffer_i];

      for (auto val : DependencyCheck::getAllValsBetween(
               {input}, {resolution_points.begin(), resolution_points.end()})) {
        // Persistent normalization kernels imply that all persistent buffers
        // have the same dimensionality. Assume if a persistent buffer is
        // consumed by another we can alias and reuse the memory.
        if (val == input) {
          continue;
        }

        if (new_persistent_factor_map.find(val) ==
            new_persistent_factor_map.end()) {
          new_persistent_factor_map[val] =
              std::vector<bool>(persistent_buffers_and_inputs.size(), false);
        }
        new_persistent_factor_map.at(val)[input_i] = true;
      }
    }
  }
  return new_persistent_factor_map_ptr;
}

} // namespace

// Returns true if a persistent tv can be projected to its persistent producers.
bool canProjectToPersistentProducer(
    TensorView* buffer,
    const std::vector<TensorView*>& producers,
    const std::unordered_set<TensorView*>& persistent_buffer_set) {
  if (buffer->hasReduction() || producers.empty()) {
    return false;
  }
  if (std::all_of(producers.begin(), producers.end(), [&](auto producer) {
        return persistent_buffer_set.count(producer) > 0;
      })) {
    return true;
  } else {
    return false;
  }
}

int64_t getPersistentBufferSizeBitOfTensor(
    const TensorView* buffer,
    SchedulerRuntimeInfo& runtime_info,
    const PersistentBufferInfo& persistent_buffer_info) {
  int64_t buffer_bits = -1;
  bool is_input =
      std::find(
          persistent_buffer_info.projectable_buffer_inputs.begin(),
          persistent_buffer_info.projectable_buffer_inputs.end(),
          buffer) != persistent_buffer_info.projectable_buffer_inputs.end();

  for (auto id : buffer->getLogicalDomain()) {
    if (id->isReduction() || id->isBroadcast()) {
      continue;
    }
    // Unmappable dimensions are those that we cannot inline into other
    // tensor views. So they're the ones that need to be persistent.
    if (!is_input && !persistent_buffer_info.unmappable_dims.count(id)) {
      continue;
    }

    if (is_input &&
        !persistent_buffer_info.unamppable_dims_projected_to_inputs.count(id)) {
      continue;
    }

    auto id_size = runtime_info.expressionEvaluator().evaluate(id->extent());
    NVF_ERROR(id_size.hasValue(), "Could not infer persistent buffer size.");
    if (buffer_bits == -1) {
      buffer_bits = id_size.as<int64_t>();
    } else {
      buffer_bits *= id_size.as<int64_t>();
    }
  }
  // If the persistent buffer is the output of an upcast op, scheduler will
  // project it back to the input to save register usage. This is similar to
  // project to inputs, not abosutely necessary but we always do it to
  // save register usage. So, need to compute the buffer size using the data
  // type before upcast.
  int64_t dtype_size_bit = 1;
  if (auto upcast_input = getUpCastInputOf(buffer)) {
    dtype_size_bit = dataTypeSizeBit(
        upcast_input->getDataType().value(), runtime_info.getIndexType());
  } else {
    dtype_size_bit = dataTypeSizeBit(
        buffer->getDataType().value(), runtime_info.getIndexType());
  }

  buffer_bits = buffer_bits == -1 ? 0 : buffer_bits * dtype_size_bit;
  return buffer_bits;
}

PersistentBufferSizeReturn persistentBufferSizeBit(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const PersistentBufferInfo& persistent_buffer_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("scheduler_utils::persistentBufferSizeBit");

  if (persistent_buffer_info.persistent_buffers.empty()) {
    PersistentBufferSizeReturn empty_sizes;
    return empty_sizes;
  }

  // Compute size of all the buffers
  const auto& persistent_buffers = persistent_buffer_info.persistent_buffers;
  const auto& projectable_buffers =
      persistent_buffer_info.projectable_persistent_buffers;
  const auto& projectable_buffers_inputs =
      persistent_buffer_info.projectable_buffer_inputs;

  std::vector<TensorView*> all_buffers = persistent_buffers;
  all_buffers.insert(
      all_buffers.end(),
      projectable_buffers_inputs.begin(),
      projectable_buffers_inputs.end());

  std::vector<int64_t> persistent_buffer_sizes_bit(all_buffers.size(), -1);

  for (auto buffer_i : arange(all_buffers.size())) {
    auto buffer = all_buffers[buffer_i];
    persistent_buffer_sizes_bit[buffer_i] = getPersistentBufferSizeBitOfTensor(
        buffer, runtime_info, persistent_buffer_info);
  }

  // Buffers involved in normal persistence
  std::vector<bool> persistent_mask(all_buffers.size(), false);
  std::unordered_set<TensorView*> persistent_buffer_set(
      persistent_buffers.begin(), persistent_buffers.end());
  for (auto buffer_i : arange(persistent_buffers.size())) {
    auto buffer = persistent_buffers[buffer_i];
    const auto& producers = ir_utils::producerTvsOf(buffer);
    if (!canProjectToPersistentProducer(
            buffer, producers, persistent_buffer_set)) {
      persistent_mask[buffer_i] = true;
    }
  }

  // Buffers involved in projected to inputs
  std::vector<bool> projected_mask(all_buffers.size(), true);
  for (auto buffer_i : arange(persistent_buffers.size())) {
    auto buffer = persistent_buffers[buffer_i];
    // Not a projectable buffer, or an input of a projectable buffer
    if (std::find(
            projectable_buffers.begin(), projectable_buffers.end(), buffer) !=
        projectable_buffers.end()) {
      projected_mask[buffer_i] = false;
    }
  }

  // Function to take the mask of active buffers at a val, the mask (for if this
  // is a normal persistent calculation, or a calculation projected on to the
  // input buffers), and sizes, and returns total persistent buffer size.
  auto masked_dot_product = [](const std::vector<bool>& mask0,
                               const std::vector<bool>& mask1,
                               const std::vector<int64_t>& sizes,
                               const std::vector<TensorView*>& all_buffers) {
    int64_t buffer_size_bit = 0;
    NVF_ERROR(
        mask0.size() == mask1.size() && mask0.size() == sizes.size() &&
        mask0.size() == all_buffers.size());
    // Keep track of which buffer is counted as there can be tensors
    // that are both a persistent buffer and an input to a projectable
    // buffer
    std::unordered_set<TensorView*> active_buffers;
    for (auto buffer_i : arange(sizes.size())) {
      if (mask0[buffer_i] && mask1[buffer_i] &&
          active_buffers.count(all_buffers[buffer_i]) == 0) {
        buffer_size_bit += sizes[buffer_i];
        active_buffers.insert(all_buffers[buffer_i]);
      }
    }
    return buffer_size_bit;
  };

  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ScopePersistentFactorInfo>(
          data_cache, [&fusion, &persistent_buffer_info]() {
            return getScopePersistenceFactors(fusion, persistent_buffer_info);
          });

  auto& scoped_persistence_factor = persistent_buffer_info_entry.get();

  // Go through all values, compute the size of the active persistent buffers,
  // do both without and with projection
  int64_t max_persistence_size_bit = 0;
  int64_t max_proj_persistence_size_bit = 0;
  for (const auto& entry : scoped_persistence_factor) {
    auto active_buffers = entry.second;
    auto persistent_buffer_size_bit = masked_dot_product(
        persistent_mask,
        active_buffers,
        persistent_buffer_sizes_bit,
        all_buffers);
    max_persistence_size_bit =
        std::max(max_persistence_size_bit, persistent_buffer_size_bit);

    auto projected_buffer_size_bit = masked_dot_product(
        projected_mask,
        active_buffers,
        persistent_buffer_sizes_bit,
        all_buffers);
    max_proj_persistence_size_bit =
        std::max(max_proj_persistence_size_bit, projected_buffer_size_bit);
  }

  PersistentBufferSizeReturn persistent_buffer_size_bit;
  persistent_buffer_size_bit.persistent_buffer_size_bit =
      max_persistence_size_bit;
  persistent_buffer_size_bit.projected_persistent_buffer_size_bit =
      max_proj_persistence_size_bit;
  return persistent_buffer_size_bit;
}

std::pair<bool, bool> canonicalDimReduction(
    Fusion* fusion,
    TensorView* tv,
    bool schedule_3D) {
  NVF_ERROR(tv != nullptr);

  if (!schedule_3D) {
    // We coalesce all reduction axes to the right;
    bool has_red_axis = mergeReduction(tv) > 0;

    bool has_iter_axis = mergeNonReduction(tv) > 0;
    return {has_iter_axis, has_red_axis};
  } else {
    NVF_ERROR(merge_3d(tv) == 3, "Tried 3D merge, but result is not 3D.");
    if (tv->axis(1)->isBroadcast()) {
      NVF_ERROR(
          !tv->axis(0)->isBroadcast(),
          "3D reduction with first two merged axes broadcast should be 2D "
          "reduction.");
      tv->reorder({{0, 1}});
    }
    return {true, true};
  }
}

std::vector<TensorView*> getReductionTvs(Fusion* fusion) {
  auto all_tvs = fusion->allTvs();
  std::vector<TensorView*> reduction_tvs;
  for (auto tv : all_tvs) {
    if (!tv->isFusionInput() &&
        std::any_of(
            tv->getLoopDomain().begin(),
            tv->getLoopDomain().end(),
            [](IterDomain* id) { return id->isReduction(); }) &&
        !isResharding(tv->definition())) {
      reduction_tvs.emplace_back(tv);
    }
  }

  // Remove multi outputs from reduction tensor views
  std::unordered_set<Expr*> seen_reduction_exprs;
  reduction_tvs.erase(
      std::remove_if(
          reduction_tvs.begin(),
          reduction_tvs.end(),
          [&seen_reduction_exprs](TensorView* tv) {
            NVF_ERROR(
                tv->definition() != nullptr,
                "Somehow a tensor view without a definition but a reduction "
                "snuck into the scheduler reduction list.");
            if (!seen_reduction_exprs.emplace(tv->definition()).second) {
              return true;
            }
            return false;
          }),
      reduction_tvs.end());
  return reduction_tvs;
}

std::vector<TensorView*> getViewTVs(Fusion* fusion) {
  std::vector<TensorView*> view_tvs;
  auto fusion_vals = fusion->usedMathVals();
  for (auto producer_tv : ir_utils::filterByType<TensorView>(fusion_vals)) {
    auto consumer_tvs = ir_utils::consumerTvsOf(producer_tv);
    for (auto consumer_tv : consumer_tvs) {
      if (consumer_tv->isDefinitionType<ViewOp>()) {
        view_tvs.push_back(consumer_tv);
      }
    }
  }
  return view_tvs;
}

std::vector<TensorView*> getTVsWithNonReductionRFactor(Fusion* fusion) {
  std::vector<TensorView*> tvs_with_rfactor;
  auto fusion_vals = fusion->usedMathVals();
  std::copy_if(
      ir_utils::filterByType<TensorView>(fusion_vals).begin(),
      ir_utils::filterByType<TensorView>(fusion_vals).end(),
      std::back_inserter(tvs_with_rfactor),
      [](TensorView* tv) {
        return tv->hasRoot() &&
            std::none_of(
                   tv->getLogicalDomain().begin(),
                   tv->getLogicalDomain().end(),
                   [](auto id) {
                     return id->isReduction() && id->isRFactorProduct();
                   });
      });
  return tvs_with_rfactor;
}

// Reset inputs and outputs to global memory, everything else to local.
void clearMemorySpace(Fusion* fusion) {
  for (auto tv : fusion->allTvs()) {
    if (tv->isFusionInput() || tv->isFusionOutput()) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      tv->setMemoryType(MemoryType::Local);
    }
  }
}

// Returns cached after tensors of the fusion inputs if unrolled. Otherwise
// return empty vector.
std::vector<TensorView*> cacheInputs(Fusion* fusion, bool unroll) {
  if (!unroll) {
    return {};
  }

  std::vector<TensorView*> cached_inputs;
  // If we're going to unroll, make a cache of the inputs
  auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  for (auto tv : in_tvs) {
    if (tv->uses().empty() || ir_utils::isGatherLookupTv(tv) ||
        ir_utils::isIndexSelectLookupTv(tv) ||
        ir_utils::isTvUsedByOpsOfType<SelectOp>(tv)) {
      // Right now, tensors that are input to the select, gather and
      // index_select ops can't be cached as they must be in global memory.
      continue;
    }

    // TODO: might need to reverse this when scheduler handles pad directly
    // Do not insert a cache for pad as vectorization needs to be
    // done directly.
    //
    // Note that this means that if an input is padded and also is
    // used without padding, it will be read twice, once for pad and
    // once more for caching load. It would make sense to use the PTX
    // caching load instructions.
    std::vector<Expr*> cached_uses;
    for (auto use : tv->uses()) {
      if (!use->isOneOf<PadOp, SliceOp>()) {
        cached_uses.push_back(use);
      }
    }

    if (cached_uses.empty()) {
      continue;
    }

    auto cached_tv = tv->cacheAfter(
        /*op_type=*/LoadStoreOpType::Set,
        /*cache_op=*/CacheOp::Unspecified,
        /*propagate_allocation_domain=*/true,
        /*cached_uses=*/cached_uses);
    cached_inputs.emplace_back(cached_tv);
  }
  return cached_inputs;
}

// Returns the pairs of <cache of each fusion output, corresponding output> for
// all outputs.
std::vector<std::pair<TensorView*, TensorView*>> cacheAndForkOutputs(
    Fusion* fusion,
    bool unroll) {
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  // For intermediate outputs, apply cacheFork
  for (auto output : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (output->definition() == nullptr ||
        // the output of ScatterOp must on the global memory due to the random
        // or atomic access.
        output->definition()->isA<ScatterOp>()) {
      continue;
    }
    if (!output->uses().empty()) {
      output = output->cacheFork();
    }
    // We shouldn't necessarily need to fork and cache for unrolling, but
    // compute at best effort replay doesn't look at multiple outputs to limit
    // itself by, so to make sure vectorization is done correctly we fork and
    // cache. This is partially a compute at issue, but even with that fixed,
    // we'd likely want to cache a forked output to make sure our inlining
    // strategy is optimal.
    if (unroll) {
      auto cached_output = output->cacheBefore();
      cached_outputs.emplace_back(cached_output, output);
    }
  }
  return cached_outputs;
}

namespace {

// Take the inner most logical id from innerMostAllocDim and project it to the
// root domain if the provided domain is on the logical domain. If vectorize,
// will not project if not following the inner most path.
IterDomain* projectIdToRoot(
    TensorView* tv,
    IterDomain* reference_id,
    bool inner_only,
    bool vectorize_pass) {
  if (reference_id == nullptr) {
    return nullptr;
  }

  auto replay_exprs = StmtSort::getExprsTo({reference_id});
  if (replay_exprs.empty()) {
    return reference_id;
  }

  IterDomain* projected_id = reference_id;
  for (auto expr_it = replay_exprs.rbegin(); expr_it != replay_exprs.rend();
       ++expr_it) {
    auto expr = *expr_it;
    if (expr->isA<Merge>()) {
      auto merge = expr->as<Merge>();
      if (merge->out() == projected_id) {
        if (!merge->inner()->isBroadcast()) {
          projected_id = merge->inner();
        } else {
          projected_id = merge->outer();
        }
      }
    } else if (expr->isA<Split>()) {
      auto split = expr->as<Split>();
      if (split->inner() == projected_id) {
        projected_id = split->in();
      } else if (split->outer() == projected_id) {
        if (inner_only) {
          projected_id = nullptr;
        } else {
          projected_id = split->in();
        }
      }
    } else if (expr->isA<Resize>()) {
      auto resize = expr->as<Resize>();
      if (resize->out() == projected_id) {
        projected_id = resize->in();
      }
    } else {
      NVF_THROW("Didn't recognize the iterdomain expression: ", expr);
    }
    if (projected_id == nullptr) {
      break;
    }
  }
  return projected_id;
}

// Take the inner most root id from innerMostAllocDim and project it to the
// allocation domain if the provided reference_id is on the allocation domain.
// If vectorize, will not project if not following the inner most path.
IterDomain* projectIdToAllocation(
    TensorView* tv,
    IterDomain* reference_id,
    bool inner_only,
    bool vectorize_pass) {
  if (reference_id == nullptr) {
    return nullptr;
  }

  auto replay_exprs = StmtSort::getExprsTo(
      {tv->getMaybeAllocationDomain().begin(),
       tv->getMaybeAllocationDomain().end()},
      false);
  if (replay_exprs.empty()) {
    return reference_id;
  }

  IterDomain* projected_id = reference_id;
  for (auto expr : replay_exprs) {
    if (expr->isA<Merge>()) {
      auto merge = expr->as<Merge>();
      if (merge->inner() == projected_id) {
        projected_id = merge->out();
      } else if (merge->outer() == projected_id) {
        if (merge->inner()->isBroadcast() || !inner_only) {
          projected_id = merge->out();
        } else {
          projected_id = nullptr;
        }
      }
    } else if (expr->isA<Split>()) {
      auto split = expr->as<Split>();
      if (split->in() == projected_id) {
        projected_id = split->inner();
      }
    } else if (expr->isA<Resize>()) {
      auto resize = expr->as<Resize>();
      if (resize->in() == projected_id) {
        projected_id = resize->out();
      }
    } else {
      NVF_THROW("Didn't recognize the iterdomain expression: ", expr);
    }
    if (projected_id == nullptr) {
      break;
    }
  }
  return projected_id;
}

} // namespace

IterDomain* innerMostAllocDim(TensorView* tv) {
  const auto& alloc_domain = tv->getMaybeAllocationDomain();

  if (tv->nDims() == 0) {
    return nullptr;
  }

  IterDomain* inner_most_id = nullptr;

  for (auto it = alloc_domain.rbegin(); it != alloc_domain.rend(); it++) {
    if ((*it)->isReduction() || (*it)->isBroadcast()) {
      continue;
    }
    inner_most_id = *it;
    break;
  }

  return inner_most_id;
}

FindAllMappedDims::FindAllMappedDims(
    TensorView* from,
    IterDomain* id,
    bool inner_only,
    bool vectorize_pass)
    : starting_tv_(from),
      starting_id_(id),
      inner_only_(inner_only),
      vectorize_pass_(vectorize_pass) {}

void FindAllMappedDims::setUp() {
  mapped_root_ids_[starting_tv_] =
      projectIdToRoot(starting_tv_, starting_id_, inner_only_, vectorize_pass_);
  mapped_logical_ids_[starting_tv_] = projectIdToAllocation(
      starting_tv_, starting_id_, inner_only_, vectorize_pass_);
}

void FindAllMappedDims::propagateC2P(TensorView* from, TensorView* to) {
  auto from_id = mapped_root_ids_.at(from);
  PairwiseLogicalDomainMap logical_map(to, from);
  auto c2p_map = logical_map.mapConsumerToProducer();
  auto p_it = c2p_map.find(from_id);
  if (p_it != c2p_map.end()) {
    mapped_root_ids_[to] =
        projectIdToRoot(to, p_it->second, inner_only_, vectorize_pass_);
    // Note, we want to project to allocation, since we could have
    // transformation from logical to allocation. e.g. for multi-device, we
    // could have DID related split between logical to allocation.
    mapped_logical_ids_[to] =
        projectIdToAllocation(to, p_it->second, inner_only_, vectorize_pass_);
  } else {
    mapped_root_ids_[to] = nullptr;
    mapped_logical_ids_[to] = nullptr;
  }
}

void FindAllMappedDims::propagateP2C(TensorView* from, TensorView* to) {
  auto from_id = mapped_logical_ids_.at(from);
  PairwiseLogicalDomainMap logical_map(from, to);
  auto p2c_map = logical_map.mapProducerToConsumer();
  auto c_it = p2c_map.find(from_id);
  if (c_it != p2c_map.end()) {
    mapped_root_ids_[to] = c_it->second;
    mapped_logical_ids_[to] =
        projectIdToAllocation(to, c_it->second, inner_only_, vectorize_pass_);
  } else {
    mapped_root_ids_[to] = nullptr;
    mapped_logical_ids_[to] = nullptr;
  }
}

void FindAllMappedDims::propagateSibling(TensorView* from, TensorView* to) {
  auto from_id = mapped_root_ids_.at(from);
  if (from_id == nullptr) {
    mapped_root_ids_[to] = nullptr;
  } else {
    for (auto i : arange(from->getMaybeRootDomain().size())) {
      if (from_id == from->getMaybeRootDomain()[i]) {
        mapped_root_ids_[to] = to->getMaybeRootDomain()[i];
        break;
      }
    }
  }
  from_id = mapped_logical_ids_.at(from);
  if (from_id == nullptr) {
    mapped_root_ids_[to] = nullptr;
  } else {
    for (auto i : arange(from->getLogicalDomain().size())) {
      if (from_id == from->getLogicalDomain()[i]) {
        mapped_logical_ids_[to] = to->getLogicalDomain()[i];
        return;
      }
    }
  }
  NVF_THROW("Unable to find mapped root/logical domain");
}

std::unordered_set<IterDomain*> FindAllMappedDims::get() const {
  std::unordered_set<IterDomain*> mapped_id_set;
  for (auto entry : mapped_root_ids_) {
    if (entry.second != nullptr) {
      mapped_id_set.emplace(entry.second);
    }
  }
  for (auto entry : mapped_logical_ids_) {
    if (entry.second != nullptr) {
      mapped_id_set.emplace(entry.second);
    }
  }
  return mapped_id_set;
}

bool hasInnerDim(
    TensorView* tv,
    std::unordered_set<IterDomain*> inner_dims,
    bool should_vectorize) {
  const auto& inner_most_dim = innerMostAllocDim(tv);
  if (inner_most_dim == nullptr) {
    return false;
  }

  // Make sure inner most dimension is in the inner_dims set
  if (inner_dims.count(inner_most_dim) == 0) {
    return false;
  }

  if (!should_vectorize) {
    return true;
  }

  auto alloc_dom = tv->getMaybeAllocationDomain();

  auto root_pos_it = std::find_if(
      alloc_dom.begin(), alloc_dom.end(), [&inner_most_dim](IterDomain* id) {
        return inner_most_dim == id;
      });

  if (root_pos_it == alloc_dom.end()) {
    return false;
  }

  auto inner_most_dim_pos = std::distance(alloc_dom.begin(), root_pos_it);

  const auto& contiguity = tv->domain()->contiguity();

  NVF_ERROR(contiguity.size() == alloc_dom.size());

  // Don't vectorize if inner most dimension is not contiguous
  auto contiguity_opt = contiguity.at(inner_most_dim_pos);
  NVF_ERROR(contiguity_opt.has_value())
  if (!*contiguity_opt) {
    return false;
  }

  return true;
}

std::vector<TensorView*> getInputsOutputsWithInnerDim(
    TensorView* reference_tv,
    bool inner_only,
    bool vectorize_pass) {
  if (vectorize_pass) {
    NVF_ERROR(inner_only, "Can only vectorize inner-most dimensions");
  }

  auto inner_most_id = innerMostAllocDim(reference_tv);

  if (inner_most_id == nullptr) {
    return {};
  }

  FindAllMappedDims all_mapped_root_dims(
      reference_tv, inner_most_id, inner_only, vectorize_pass);
  MaxLogicalDomainInfoSpanningTree tree(reference_tv);
  tree.traverse(&all_mapped_root_dims);

  auto vectorizable_dims = all_mapped_root_dims.get();

  std::vector<TensorView*> vectorizable_tensors;

  // We put outputs in front of inputs because this would make the transpose
  // scheduler prefer to use output instead of input as reference tensor.
  for (auto output_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->outputs())) {
    if (hasInnerDim(output_tv, vectorizable_dims, vectorize_pass)) {
      vectorizable_tensors.push_back(output_tv);
    }
  }

  for (auto input_tv :
       ir_utils::filterByType<TensorView>(reference_tv->fusion()->inputs())) {
    // for indexSelect(lookup_tv, dim, index_tv) op
    // ignore it's lookup_tv.
    if (ir_utils::isGatherLookupTv(input_tv)) {
      continue;
    }

    if (hasInnerDim(input_tv, vectorizable_dims, vectorize_pass)) {
      vectorizable_tensors.push_back(input_tv);
    }
  }

  return vectorizable_tensors;
}

DisjointLogicalSetInfo getDisjointLogicalSetsOf(
    Fusion* fusion,
    TensorView* of,
    DisjointSets<IterDomain*>& disjoint_logical_set,
    const std::unordered_map<int64_t, int64_t>& logical_reorder_map) {
  auto logical_dom = of->getLogicalDomain();
  if (logical_dom.empty()) {
    return {};
  }

  DisjointLogicalSetInfo info;
  if (!logical_reorder_map.empty()) {
    logical_dom = TensorDomain::orderedAs(logical_dom, logical_reorder_map);
  }

  // Start naming id's based on 0 so the inner most dimension will always be
  // 0, then as groups are discovered marching to the left their id will
  // increase. i.e. we could have something like [0, 3, 1, 2, 1, 0] as a
  // result.
  std::vector<int64_t> disjoint_group_ids(logical_dom.size(), -1);
  std::vector<const VectorOfUniqueEntries<IterDomain*>*> disjoint_set_of_id(
      logical_dom.size(), nullptr);
  int64_t current_group_id = 0;
  int64_t ref_dim_i = (int64_t)logical_dom.size() - 1;

  while (ref_dim_i >= 0) {
    if (disjoint_group_ids[ref_dim_i] != -1) {
      // Already put in a group, continue
      ref_dim_i--;
      continue;
    }

    const auto& ref_group =
        disjoint_logical_set.getDisjointSetOf(logical_dom[ref_dim_i]);

    int64_t other_dim_i = ref_dim_i;
    while (other_dim_i >= 0) {
      const auto& other_group =
          disjoint_logical_set.getDisjointSetOf(logical_dom[other_dim_i]);
      if (&ref_group == &other_group) {
        disjoint_group_ids[other_dim_i] = current_group_id;
        disjoint_set_of_id[other_dim_i] = &ref_group;
      }
      other_dim_i--;
    }

    ref_dim_i--;
    current_group_id++;
  }

  NVF_ERROR(
      std::none_of(
          disjoint_group_ids.begin(),
          disjoint_group_ids.end(),
          [](int i) { return i == -1; }),
      "Failed to generate the rfactor disjoint groups of the reference ",
      of->toString());

  NVF_ERROR(
      std::none_of(
          disjoint_set_of_id.begin(),
          disjoint_set_of_id.end(),
          [](const VectorOfUniqueEntries<IterDomain*>* ptr) {
            return ptr == nullptr;
          }),
      "Failed to generate the rfactor disjoint groups of the reference ",
      of->toString());

  info.disjoint_sets_of_ref = disjoint_set_of_id;
  info.disjoint_set_ids = disjoint_group_ids;
  info.ref = of;

  return info;
}

BroadcastMultipleInformation getBroadcastMultiples(
    TensorView* reference_tv,
    DataType index_type,
    const std::unordered_map<int64_t, int64_t>& logical_reorder_map) {
  auto fusion = reference_tv->fusion();
  FusionGuard fg(fusion);

  // We always cacheBefore output at the beginning of the scheduling. And after
  // cacheBefore, the reference tensor will have all reduction IDs removed.
  auto ref_root_domain = TensorDomain::noDevices(
      TensorDomain::noReductions(reference_tv->getLogicalDomain()));

  if (!logical_reorder_map.empty()) {
    ref_root_domain =
        TensorDomain::orderedAs(ref_root_domain, logical_reorder_map);
  }

  std::vector<BroadcastMultiple> multiples(ref_root_domain.size());

  auto disjoint_logical_sets = disjointLogicalSets(fusion);
  auto disjoint_set_information = scheduler_utils::getDisjointLogicalSetsOf(
      fusion, reference_tv, disjoint_logical_sets, logical_reorder_map);

  auto ref_disjoint_sets = disjoint_set_information.disjoint_sets_of_ref;
  auto ref_disjoint_set_ids = disjoint_set_information.disjoint_set_ids;

  // All input or output tensor views
  std::vector<TensorView*> in_out_tvs;
  {
    auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    in_out_tvs.insert(in_out_tvs.end(), inp_tvs.begin(), inp_tvs.end());
    auto out_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());
    in_out_tvs.insert(in_out_tvs.end(), out_tvs.begin(), out_tvs.end());
  }

  // Shouldn't matter if we use EXACT or PERMISSIVE mapping mode for compute
  // at map as we're just looking at the root mappings.
  auto ca_map = ComputeAtMap(fusion);

  // Map all inputs and output domains to reference tv domains
  for (auto in_out_tv : in_out_tvs) {
    std::vector<bool> mapped_axes(ref_root_domain.size(), false);

    auto in_out_tv_domain =
        TensorDomain::noDevices(in_out_tv->getMaybeRootDomain());
    auto in_out_tv_domain_list = std::list<IterDomain*>(
        in_out_tv_domain.begin(), in_out_tv_domain.end());

    for (const auto ref_i : arange(ref_root_domain.size())) {
      auto ref_id = ref_root_domain[ref_i];

      if (ref_id->isBroadcast()) {
        continue;
      }

      bool ref_id_has_view_transforms = std::count(
                                            ref_disjoint_set_ids.begin(),
                                            ref_disjoint_set_ids.end(),
                                            ref_disjoint_set_ids[ref_i]) > 1;

      // Could have multiple mappings if there's view transforms
      std::vector<IterDomain*> mapped_ids;
      if (!ref_id_has_view_transforms) {
        auto mapped_it = std::find_if(
            in_out_tv_domain_list.begin(),
            in_out_tv_domain_list.end(),
            [&ref_id, &ca_map](IterDomain* in_out_tv_id) {
              return ca_map.areMapped(
                  in_out_tv_id, ref_id, IdMappingMode::EXACT);
            });
        if (mapped_it != in_out_tv_domain_list.end()) {
          mapped_ids.push_back(*mapped_it);
        }
      } else {
        for (auto in_out_id : in_out_tv_domain) {
          if (ref_disjoint_sets[ref_i]->has(in_out_id)) {
            mapped_ids.push_back(in_out_id);
          }
        }
      }

      // Nothing maps to reference, no contribution to multiples for this dim
      if (mapped_ids.empty()) {
        continue;
      }

      if (std::all_of(mapped_ids.begin(), mapped_ids.end(), [](IterDomain* id) {
            return id->isBroadcast();
          })) {
        continue;
      }

      // If any iteration domain in the input or output that's mapped through
      // the view disjoint set is not a reduction or broadcast, assume it's a
      // full dimension for the sake of the pointwise scheduler.
      mapped_axes[ref_i] = true;
    }

    // For each break point position if there an lhs or rhs multiple based on
    // this tensor add it to the global multiplier. The only time we consider
    // we can benefit from broadcast is if the entire left or right side the
    // break point is all broadcasts.
    {
      bool rhs = false;
      bool lhs = false;
      auto dtype_size_bit =
          dataTypeSizeBit(in_out_tv->getDataType().value(), index_type);
      for (auto mapped_axes_i : arange(mapped_axes.size())) {
        auto lhs_i = mapped_axes_i;
        auto rhs_i = mapped_axes.size() - 1 - mapped_axes_i;

        if (lhs) {
          multiples[lhs_i].lhs_multiple += (int64_t)dtype_size_bit;
        } else if (mapped_axes[lhs_i]) {
          lhs = true;
        }

        if (rhs || mapped_axes[rhs_i]) {
          multiples[rhs_i].rhs_multiple += (int64_t)dtype_size_bit;
          rhs = true;
        }
      }
    }
  }
  BroadcastMultipleInformation bcast_info;
  bcast_info.view_disjoint_set_ids = ref_disjoint_set_ids;
  bcast_info.broadcast_multiples = multiples;
  return bcast_info;
}

//! Propagate current transformations on from_tv to all graphs
void transformPropagateToAllFrom(TensorView* from_tv, int64_t pos) {
  TransformPropagator propagator(from_tv, pos);
  MaxLogicalDomainInfoSpanningTree(from_tv, nullptr).traverse(&propagator);
}

namespace {

//! Returns true if the given tensorview is a fake boundary
//!  TensorView, see Note [Fake Boundary Tensorview].
//! This function assumes and would not check that tv is a boundary
//!  of the select_tv set.
bool isFakeBoundaryTensorview(
    TensorView* tv,
    const std::unordered_set<TensorView*>& selected_tv_set,
    PropagateDirection direction) {
  if (direction == PropagateDirection::kForward) {
    // In the case of forward propagation,
    //  a boundary tv is a fake boundary if
    //  it has any consumer tv that's in the selected
    //  set.
    for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
      if (selected_tv_set.count(consumer_tv)) {
        // Found a consumer that's in selected tv set.
        return true;
      }
    }

  } else {
    // In the case of backward propagation,
    //  a boundary tv is a fake boundary if it has any producer
    //  that is within the selected set.
    for (auto producer_tv : ir_utils::producerTvsOf(tv)) {
      if (selected_tv_set.count(producer_tv)) {
        // Found a producer that's in selected tv set.
        return true;
      }
    }
  }

  // Didn't find any producer/consumer in the selected tv set.
  //  The given tv is not a fake boundary tv.
  return false;
}

//! Utility function to generate the set of tensorviews to propagate
//!  transform to by BoundedDirectionalTransformPropagator.
std::unordered_set<TensorView*> getDirectionalPropagatePathSet(
    TensorView* from_tv,
    const std::vector<TensorView*>& boundary_tvs,
    BoundedDirectionalTransformPropagator::Options options,
    PropagateDirection direction) {
  // Prepare to collect all candidate tensorviews
  //  within the specified boundary.
  std::vector<Val*> propagate_candidate;

  // Collect boundary tvs in a set.
  std::unordered_set<TensorView*> boundary_tv_set(
      boundary_tvs.begin(), boundary_tvs.end());

  if (direction == PropagateDirection::kForward) {
    // In the case of forward propagation, collect all tvs
    //  that are consumers of `from_tv` and producers of
    //  boundary tvs.
    propagate_candidate = DependencyCheck::getAllValsBetween(
        {from_tv}, {boundary_tvs.begin(), boundary_tvs.end()});
  } else {
    // In the case of backward propagation, collect all tvs
    //  that are producers of `from_tv` and consumers of
    //  boundary tvs.
    propagate_candidate = DependencyCheck::getAllValsBetween(
        {boundary_tvs.begin(), boundary_tvs.end()}, {from_tv});
  }

  // Populate initial selected tensorviews in a set.
  auto propagate_candidate_tv_view =
      ir_utils::filterByType<TensorView>(propagate_candidate);
  // Prepare to filter out un-wanted tensorviews according
  //  to the option parameters.
  std::unordered_set<TensorView*> propagate_path_set{
      propagate_candidate_tv_view.begin(), propagate_candidate_tv_view.end()};

  // Remove boundary tensorviews if we don't want to transform
  //  tensorviews on the boundary.
  if (!options.transform_boundary) {
    // Additional refining step to identify "fake boundary" tensorviews.
    //  We don't want to erase fake boundary tensorviews from the selected
    //  set when we are erasing boundary tvs.
    //
    // Note [Fake Boundary Tensorview]
    // A tensorview, tv0, is defined as fake boundary tv if
    //  1. Tv0 is on the given boundary set.
    //  2. There is a path from another boundary tv, Tv1 to from_tv that
    // goes through Tv0.
    //
    // In this case the propagation behavior is not precisely defined.
    // Our current decision is to treat such tensorview as non-boundary
    //  tv to make sure the propagation paths are not blocked. E.g.:
    //
    //  T1 = T0
    //  T2 = T1
    //  T3 = T2 + T1
    // if we propagate with from_tv = {T3}, boundary_tv = {T0, T2},
    // transform_boundary=false
    //
    // Here T2 is a fake boundary and we will still transform T2 as it is
    //  on the path between T3 and T0.

    // Initialize set of fake boundary tvs.
    std::unordered_set<TensorView*> fake_boundary_set;

    // Populate the set of fake boundary tvs.
    std::copy_if(
        boundary_tvs.begin(),
        boundary_tvs.end(),
        std::inserter(fake_boundary_set, fake_boundary_set.end()),
        [&propagate_path_set, direction](TensorView* tv) {
          return isFakeBoundaryTensorview(tv, propagate_path_set, direction);
        });

    // Remove boundary tvs from the selected set, keeping fake boundary tvs.
    for (auto boundary_tv : boundary_tvs) {
      if (!fake_boundary_set.count(boundary_tv)) {
        propagate_path_set.erase(boundary_tv);
      }
    }
  }

  return propagate_path_set;
}

} // namespace

void BoundedDirectionalTransformPropagator::propagate(
    TensorView* from_tv,
    int64_t pos,
    std::unordered_set<TensorView*> included_tvs,
    Options options) {
  // Run transform propagation using the custom selector.
  SetSelector selector(included_tvs);
  TransformPropagator propagator(from_tv, pos);
  MaxLogicalDomainInfoSpanningTree(from_tv, &selector).traverse(&propagator);

  // Propagate parallel type if requested by option parameters.
  if (options.propagate_parallel_type) {
    scheduler_utils::parallelizeAllLike(
        from_tv,
        options.parallel_propagation_pos,
        {included_tvs.begin(), included_tvs.end()},
        allParallelTypesExcept({ParallelType::Vectorize, ParallelType::Mma}));
  }
}

void BoundedDirectionalTransformPropagator::backward(
    TensorView* from,
    int64_t pos,
    std::vector<TensorView*> to,
    std::optional<Options> options) {
  if (!options.has_value()) {
    options = Options();
  }
  if (to.empty()) {
    to = ir_utils::inputTvsOf(from);
  }

  // Collect all tvs to included on the backward path as specified
  //  by boundary and options.
  auto included_tvs = getDirectionalPropagatePathSet(
      from, to, *options, PropagateDirection::kBackward);
  // Actually run the propagation.
  propagate(from, pos, included_tvs, *options);
}

void BoundedDirectionalTransformPropagator::forward(
    TensorView* from,
    int64_t pos,
    std::vector<TensorView*> to,
    std::optional<Options> options) {
  if (!options.has_value()) {
    options = Options();
  }
  NVF_ERROR(
      !to.empty(),
      "Propagation needs to be bounded, so no support for empty boundary.")

  // Collect all tvs to included on the forward path as specified
  //  by boundary and options.
  auto included_tvs = getDirectionalPropagatePathSet(
      from, to, *options, PropagateDirection::kForward);

  // Actually run the propagation.
  propagate(from, pos, included_tvs, *options);
}

void BoundedDirectionalTransformPropagator::bothWays(
    TensorView* from,
    int64_t pos,
    std::vector<TensorView*> backward_to,
    std::vector<TensorView*> forward_to,
    std::optional<Options> options) {
  if (!options.has_value()) {
    options = Options();
  }
  NVF_ERROR(
      !backward_to.empty() && !forward_to.empty(),
      "Propagation needs to be bounded, so no support for empty boundary.")

  // Collect all tvs to included on the backward and forward path as specified
  //  by boundary and options.
  auto backward_included_tvs = getDirectionalPropagatePathSet(
      from, backward_to, *options, PropagateDirection::kBackward);
  auto forward_included_tvs = getDirectionalPropagatePathSet(
      from, forward_to, *options, PropagateDirection::kForward);

  // Combined the included tvs on both paths.
  auto included_tvs = backward_included_tvs;
  included_tvs.insert(forward_included_tvs.begin(), forward_included_tvs.end());

  // Run the propagation on the combined set of tvs.
  propagate(from, pos, included_tvs, *options);
}

DisjointSets<IterDomain*> disjointLogicalSets(Fusion* fusion) {
  // Start from the exact iter domain graph of the fusion
  IterDomainGraph id_graph(fusion);
  auto disjoint_logical_ids = id_graph.exactNodes();

  // If iter domains are involved in any transformation from root domains to
  // logical domains they should be considered "contaminated".
  for (auto tv : fusion->allTvs()) {
    for (auto expr : StmtSort::getExprsTo(
             {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()})) {
      if (expr->isA<Merge>()) {
        auto merge = expr->as<Merge>();
        disjoint_logical_ids.mapEntries(merge->inner(), merge->out());
        disjoint_logical_ids.mapEntries(merge->outer(), merge->out());
      } else if (expr->isA<Split>()) {
        auto split = expr->as<Split>();
        disjoint_logical_ids.mapEntries(split->in(), split->inner());
        disjoint_logical_ids.mapEntries(split->in(), split->outer());
      } else if (expr->isA<Resize>()) {
        auto resize = expr->as<Resize>();
        disjoint_logical_ids.mapEntries(resize->in(), resize->out());
      } else {
        NVF_THROW("Expression type: ", expr->toString(), " not supported.");
      }
    }
  }
  return disjoint_logical_ids;
}

bool breakIsDisjoint(std::vector<int64_t> group_ids, int64_t pos) {
  if (pos < 0) {
    pos += (int64_t)group_ids.size();
  }
  NVF_ERROR(
      pos >= 0 && pos <= (int64_t)group_ids.size(),
      "Invalid position, size of vec is ",
      group_ids.size(),
      " but position is ",
      pos);

  if (pos == 0 || pos == (int64_t)group_ids.size()) {
    return true;
  }

  std::unordered_set<int64_t> left_ints(
      group_ids.begin(), group_ids.begin() + pos);

  for (auto i = pos; i < (int64_t)group_ids.size(); i++) {
    if (left_ints.count(group_ids[i]) > 0) {
      return false;
    }
  }
  return true;
}

namespace {

void applySplitTransform(Split* split, std::vector<IterDomain*>& ids) {
  auto find_it = std::find(ids.begin(), ids.end(), split->in());
  NVF_ERROR(
      find_it != ids.end(),
      "Split input ",
      split->in()->toString(),
      " not found in given ids: ",
      ids);
  auto pos = std::distance(ids.begin(), find_it);
  ids[pos] = split->inner();
  ids.insert(ids.begin() + pos, split->outer());
}

void applyMergeTransform(Merge* merge, std::vector<IterDomain*>& ids) {
  auto find_it_0 = std::find(ids.begin(), ids.end(), merge->outer());
  auto find_it_1 = std::find(ids.begin(), ids.end(), merge->inner());
  NVF_ERROR(
      find_it_0 != ids.end(),
      "Merge outer ",
      merge->outer()->toString(),
      " not found in given ids: ",
      ids);
  NVF_ERROR(
      find_it_1 != ids.end(),
      "Merge inner ",
      merge->inner()->toString(),
      " not found in given ids: ",
      ids);
  auto pos0 = std::distance(ids.begin(), find_it_0);
  auto pos1 = std::distance(ids.begin(), find_it_1);
  if (pos0 > pos1) {
    std::swap(pos0, pos1);
  }
  // Should be impossible.
  NVF_ERROR(
      pos0 != pos1,
      "Didn't expect merge inputs to be the same iteration domain:\n",
      merge->toString());

  ids.erase(ids.begin() + pos0);
  ids[--pos1] = merge->out();
}

void applyResizeTransform(Resize* resize, std::vector<IterDomain*>& ids) {
  auto find_it = std::find(ids.begin(), ids.end(), resize->in());
  NVF_ERROR(
      find_it != ids.end(),
      "Resize input ",
      resize->in()->toString(),
      " not found in given ids: ",
      ids);
  *find_it = resize->out();
}

} // namespace

void applyTransforms(
    std::vector<IterDomain*>& ids_to_transform,
    const std::vector<Expr*>& transform_exprs) {
  for (auto* expr : transform_exprs) {
    if (Split* split = dynamic_cast<Split*>(expr)) {
      applySplitTransform(split, ids_to_transform);
    } else if (Merge* merge = dynamic_cast<Merge*>(expr)) {
      applyMergeTransform(merge, ids_to_transform);
    } else if (Resize* resize = dynamic_cast<Resize*>(expr)) {
      applyResizeTransform(resize, ids_to_transform);
    } else {
      NVF_ERROR(expr != nullptr);
      NVF_THROW("Unexpected expression: ", expr->toString());
    }
  }
}

// Returns a permutation reordering the loop domain of the tensor view as the
// logical domain
std::vector<int64_t> domainReorderAsLogicalMap(TensorView* tv) {
  FusionGuard fg(tv->fusion());
  auto transform_exprs = DependencyCheck::getAllExprsBetween(
      {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()},
      {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
  std::vector<IterDomain*> ids_to_transform = tv->getLogicalDomain();
  applyTransforms(ids_to_transform, transform_exprs);
  std::optional<std::vector<int64_t>> permutation =
      ir_utils::computePermutation(ids_to_transform, tv->getLoopDomain());
  NVF_ERROR(
      permutation.has_value(),
      "Failed to find a valid permutation for reordering",
      tv->getLoopDomain(),
      " as ",
      ids_to_transform);
  return *permutation;
}

std::unordered_map<int64_t, int64_t> maybeReorderAsAllocationMap(
    TensorView* tv) {
  std::unordered_map<int64_t, int64_t> ret;
  if (!tv->hasAllocation()) {
    return ret;
  }
  const auto& alloc_dom = tv->getAllocationDomain();
  const auto& loop_dom = tv->getLoopDomain();
  if (alloc_dom == loop_dom) {
    return ret;
  }
  if (!std::is_permutation(
          alloc_dom.begin(), alloc_dom.end(), loop_dom.begin())) {
    return ret;
  }
  std::unordered_map<IterDomain*, int64_t> alloc_index;
  std::unordered_map<IterDomain*, int64_t> rfactor_index;
  for (auto i : arange((int64_t)alloc_dom.size())) {
    alloc_index[alloc_dom[i]] = i;
    rfactor_index[loop_dom[i]] = i;
  }
  for (auto iter_dom : alloc_dom) {
    ret[rfactor_index[iter_dom]] = alloc_index[iter_dom];
  }
  return ret;
}

void propagateReshapeTransforms(Fusion* fusion, const ComputeAtMap& ca_map) {
  std::unordered_set<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
      transformed_disjoint_sets;

  // If iter domains are involved in any transformation from root domains to
  // logical domains they should be considered "contaminated".
  for (auto tv : fusion->allTvs()) {
    for (auto expr : StmtSort::getExprsBetween(
             {tv->getMaybeRootDomain().begin(), tv->getMaybeRootDomain().end()},
             {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()})) {
      for (auto id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
        transformed_disjoint_sets.emplace(
            ca_map.disjointSetOf(id, IdMappingMode::EXACT));
      }
    }
  }

  std::unordered_set<IterDomain*> terminating_reshape_dims;
  for (const auto& disjoint_set_shared_ptr :
       ca_map.idGraph().exactNodes().disjointSets()) {
    // Find a disjoint set that is produced by a reshape
    // operation. Ignore resize as it isn't reshape
    if (std::none_of(
            disjoint_set_shared_ptr->vector().begin(),
            disjoint_set_shared_ptr->vector().end(),
            [](IterDomain* id) {
              return id->isRFactorProduct() && id->definition() &&
                  !id->definition()->isA<Resize>();
            })) {
      continue;
    }
    if (transformed_disjoint_sets.find(disjoint_set_shared_ptr) !=
        transformed_disjoint_sets.end()) {
      // Disjoint set was transformed for view, ignore it
      continue;
    }
    for (auto id : disjoint_set_shared_ptr->vector()) {
      terminating_reshape_dims.emplace(id);
    }
  }

  // If iter domains are involved in any transformation from root domains to
  // logical domains they should be considered "contaminated".
  for (auto tv : fusion->allTvs()) {
    if (!tv->hasRoot()) {
      continue;
    }

    std::unordered_map<int64_t, int64_t> old2new;
    // Make sure rfactor dims we need are in domain, and reorder them in domain
    // so they're consecutive starting from the left of domain.
    // The reordering is to limit the propagation to only the view
    // transformations.
    for (auto logical_id : tv->getLogicalDomain()) {
      if (terminating_reshape_dims.find(logical_id) !=
          terminating_reshape_dims.end()) {
        // The rfactor dims are not in the loop domain directly if they are
        // sharded. For example, Consider the split reshape: `[h]->[a, h/a]` `h`
        // and `a` are both sharded by `d`. The loop domain of the consumer is
        // `[DIDx(d), a/d, h/a]`. Hence, we cannot directly find logical ID `a`
        // in the loop domain. Similarly, for merge reshape: `[a, h/a]->[h]`, we
        // cannot directly find `h` in the loop domain when `h` is sharded by
        // `d`.

        // Find all reachable ids between the logical id and the loop domain.
        // If the ids are in the loop domain, reorder them to the front.
        auto transforms = DependencyCheck::getAllExprsBetween(
            {logical_id},
            {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
        std::unordered_set<IterDomain*> reachable_ids;
        // Add the logical id for the case where it is directly in the loop
        // domain.
        reachable_ids.insert(logical_id);

        for (auto expr : transforms) {
          auto outputs = ir_utils::filterByType<IterDomain>(expr->outputs());
          reachable_ids.insert(outputs.begin(), outputs.end());
        }

        bool has_reachable_loop_id = false;
        for (auto loop_idx :
             arange(static_cast<int64_t>(tv->getLoopDomain().size()))) {
          if (reachable_ids.count(tv->axis(loop_idx)) == 0) {
            continue;
          }
          has_reachable_loop_id = true;
          // Reorder the reshape dimensions to the front of the domain
          old2new[loop_idx] = (int64_t)old2new.size();
        }

        NVF_ERROR(
            has_reachable_loop_id,
            "Require ",
            logical_id,
            " is in the active domain of ",
            tv->toString(),
            " for view propagation.");
      }
    }

    if (old2new.empty()) {
      continue;
    }

    // Propagate the view transformations
    tv->reorder(old2new);
    //! Propagate current transformations on from_tv to all graphs
    transformPropagateToAllFrom(tv, (int64_t)old2new.size());

    // Propgating the transforms will not replay the DIDx parallelization, so we
    // need to do it manually here.
    parallelizeAllLike(
        tv,
        /*pos=*/(int64_t)old2new.size(),
        /*selected_tvs=*/{},
        /*selected_parallel_types=*/{ParallelType::DIDx},
        /*propagate_padding=*/false,
        /*parallelize_inputs_on_did=*/true);
  }
}

bool isFastestDimReduction(TensorView* tv) {
  for (auto it = tv->getMaybeAllocationDomain().rbegin();
       it != tv->getMaybeAllocationDomain().rend();
       ++it) {
    auto root_id = *it;
    if (root_id->isBroadcast()) {
      continue;
    } else if (root_id->isReduction()) {
      return true;
    } else {
      return false;
    }
  }

  return false;
}

namespace {

// Grab producer and consumer pairs that have non-pointwise
// access patterns. Those pairs would have inter-thread data
// dependencies if parallelized.
std::vector<std::pair<TensorView*, TensorView*>>
getNonPointwiseProducerConsumerPairs(Fusion* fusion) {
  std::vector<std::pair<TensorView*, TensorView*>> tvs;

  for (auto consumer : fusion->allTvs()) {
    if (consumer->isFusionInput()) {
      continue;
    }
    if (auto gather = dynamic_cast<GatherOp*>(consumer->definition())) {
      tvs.emplace_back(gather->lookupTv(), consumer);
    } else if (
        auto index_select =
            dynamic_cast<IndexSelectOp*>(consumer->definition())) {
      tvs.emplace_back(index_select->lookupTv(), consumer);
    } else if (auto select = dynamic_cast<SelectOp*>(consumer->definition())) {
      tvs.emplace_back(select->lookupTv(), consumer);
    } else if (ir_utils::hasResizedRfactor(consumer)) {
      // Exprs based on ResizeOp, e.g., slice
      auto producers = ir_utils::producerTvsOf(consumer);
      NVF_ERROR(
          producers.size() == 1,
          "Unexpected number of inputs of the defining expression: ",
          consumer->definition()->toString());
      tvs.emplace_back(producers.at(0), consumer);
    }
  }

  return tvs;
}

// If an input cache is promoted to global memory, just do not cache
// the input but directly read from the global memory input. It
// doesn't make any sense to cache a global memory input in global
// memory. Note that a copy of an input cache is inserted by
// prepareForMemoryTypePromotion, so grab the producer of the
// producer and see if it's included in input_caches.
bool revertUseOfInputCache(
    TensorView* consumer,
    TensorView* promoted_producer,
    MemoryType promoted_memory_type,
    const std::vector<TensorView*>& input_caches) {
  auto get_copy_src = [](TensorView* tv) -> TensorView* {
    if (auto uop = dynamic_cast<LoadStoreOp*>(tv->definition())) {
      return uop->in()->as<TensorView>();
    }
    return nullptr;
  };

  // Only applies if the promoted new type is Global
  if (promoted_memory_type != MemoryType::Global) {
    return false;
  }

  // To see if the producer is a cache of an input, need to look at
  // its producer as a copy is inserted
  auto producer_of_producer = get_copy_src(promoted_producer);
  if (producer_of_producer == nullptr) {
    // No copy is detected. This must mean the producer is not a copy
    // of any input cache
    return false;
  }

  auto cache_it =
      std::find(input_caches.begin(), input_caches.end(), producer_of_producer);
  if (cache_it == input_caches.end()) {
    return false;
  }

  auto fusion_input = get_copy_src(producer_of_producer);
  NVF_ERROR(
      fusion_input != nullptr,
      "Unexpected input cache: ",
      producer_of_producer->toString());

  // Currently, the ops look like:
  // tv0: fusion input
  // tv1 = tv0 // cache of the input
  // tv2 = tv1 // copy of the tv1. Placed on Global
  // tv3 = resizeOp(tv2) // some op using resize

  // Translate it to:
  // tv0: fusion input
  // tv3 = resizeOp(tv0) // some op using resize

  ir_utils::replaceValInExprInputs(
      consumer->definition(), promoted_producer, fusion_input);

  return true;
}

} // namespace

void prepareForMemoryTypePromotion(Fusion* fusion) {
  auto non_pwise_pairs = getNonPointwiseProducerConsumerPairs(fusion);

  // Inserting a copy of each producer. If a tensor shows up as a
  // producer for multiple consumers, only insert one
  // copy and share it with all the consumers.

  // Map to keep track producer and its copy
  std::unordered_map<TensorView*, TensorView*> producer_copy_map;

  for (auto& [producer, consumer] : non_pwise_pairs) {
    // At this point, all tensors should be either on Global or Local
    NVF_ERROR(
        producer->getMemoryType() == MemoryType::Local ||
            producer->getMemoryType() == MemoryType::Global,
        "Unexpected memory type: ",
        producer->getMemoryType());

    // If already placed on Global, nothing to worry about
    if (producer->getMemoryType() == MemoryType::Global) {
      continue;
    }

    auto producer_copy_map_it = producer_copy_map.find(producer);
    if (producer_copy_map_it == producer_copy_map.end()) {
      // Create a copy of the producer that is to be inserted between
      // consumer and producer
      auto copy_of_producer = set(producer);
      producer_copy_map_it =
          producer_copy_map.emplace(producer, copy_of_producer).first;
    }

    // Insert a copy between consumer and producer
    ir_utils::replaceValInExprInputs(
        consumer->definition(), producer, producer_copy_map_it->second);
  }
}

void promoteProducerMemoryTypes(
    Fusion* fusion,
    const std::vector<TensorView*>& input_caches) {
  auto non_pwise_pairs = getNonPointwiseProducerConsumerPairs(fusion);

  // Just make it simpler to promote memory types. Minimum is
  // preferred. Increased as required.
  auto memoryTypeToInt = [](MemoryType m_type) -> int {
    switch (m_type) {
      case MemoryType::Local:
        return 1;
      case MemoryType::Shared:
        return 2;
      case MemoryType::Global:
        return 3;
      default:
        NVF_THROW("Unexpected memory type: ", m_type);
    }
  };

  std::unordered_map<TensorView*, MemoryType> tvs_to_promote;

  auto setPromotion = [&](TensorView* tv, MemoryType m_type) {
    // Initialize the memory type with the current type
    tvs_to_promote.emplace(tv, tv->getMemoryType());

    if (memoryTypeToInt(m_type) > memoryTypeToInt(tvs_to_promote.at(tv))) {
      tvs_to_promote[tv] = m_type;
    }
  };

  // Analyze each produce and consumer if there's inter-thread data
  // dependencies
  // TODO: Clean up once the index map refactor is done
  for (auto& [producer, consumer] : non_pwise_pairs) {
    auto c2p_exact_map = BestEffortReplay(
                             producer->getLoopDomain(),
                             consumer->getLoopDomain(),
                             PairwiseLogicalDomainMap(producer, consumer)
                                 .mapBroadcast(false)
                                 .mapConsumerToProducer())
                             .getReplay();

    for (const auto i :
         arange(producer->nDims() - producer->getComputeAtPosition())) {
      auto producer_non_ca_id =
          producer->axis((i + producer->getComputeAtPosition()));
      auto producer_non_ca_id_ptype = producer_non_ca_id->getParallelType();
      if (!isParallelTypeThread(producer_non_ca_id_ptype)) {
        continue;
      }

      auto consumer_exact_map_id_it = std::find_if(
          consumer->getLoopDomain().begin(),
          consumer->getLoopDomain().end(),
          [&](IterDomain* consumer_loop_id) {
            auto it = c2p_exact_map.find(consumer_loop_id);
            return it != c2p_exact_map.end() &&
                it->second == producer_non_ca_id;
          });
      if (consumer_exact_map_id_it != consumer->getLoopDomain().end() &&
          (*consumer_exact_map_id_it)->getParallelType() ==
              producer_non_ca_id_ptype) {
        continue;
      }

      // Promotion required
      if (isParallelTypeThreadDim(producer_non_ca_id_ptype)) {
        setPromotion(producer, MemoryType::Shared);
      } else if (isParallelTypeBlockDim(producer_non_ca_id_ptype)) {
        setPromotion(producer, MemoryType::Global);
      } else {
        NVF_THROW("Unexpected parallel type: ", producer_non_ca_id_ptype);
      }
    }
  }

  // Iterate over non_pwise_pairs so that promotion is done in a
  // deterministic order
  for (auto& [producer, consumer] : non_pwise_pairs) {
    auto it = tvs_to_promote.find(producer);
    if (it == tvs_to_promote.end() || it->second == producer->getMemoryType()) {
      continue;
    }

    // Required memory type of the producer
    const auto new_mem_type = it->second;

    if (revertUseOfInputCache(consumer, producer, new_mem_type, input_caches)) {
      continue;
    }

    producer->setMemoryType(new_mem_type);
  }
}

std::unordered_set<TensorView*> getAllTvsFrom(
    const std::vector<TensorView*>& from_tvs,
    const std::unordered_set<TensorView*>& cutoff_tv_set) {
  std::unordered_set<TensorView*> tv_group;
  std::queue<TensorView*> tensors_to_visit;
  auto addIfNotVisited = [&](TensorView* tv) {
    if (tv_group.find(tv) == tv_group.end() &&
        cutoff_tv_set.find(tv) == cutoff_tv_set.end()) {
      tv_group.emplace(tv);
      tensors_to_visit.push(tv);
    }
  };

  for (auto tv : from_tvs) {
    tensors_to_visit.push(tv);
  }
  while (!tensors_to_visit.empty()) {
    auto next_tv = tensors_to_visit.front();
    tensors_to_visit.pop();
    // visit consumers
    for (auto tv : ir_utils::consumerTvsOf(next_tv)) {
      addIfNotVisited(tv);
    }
    // visit siblings
    for (auto tv : ir_utils::siblingTvsOf(next_tv)) {
      addIfNotVisited(tv);
    }
    // visit producer
    for (auto tv : ir_utils::producerTvsOf(next_tv)) {
      addIfNotVisited(tv);
    }
  }
  return tv_group;
}

int64_t getReductionSmemWorkspaceBit(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    int64_t threads_per_block) {
  const auto& dev_prop = at::cuda::getCurrentDeviceProperties();
  // use device max threads per block if threads_per_block is not provided
  threads_per_block =
      threads_per_block > 0 ? threads_per_block : dev_prop->maxThreadsPerBlock;
  // (1) part-1, space for the reduction broadcast.
  int64_t dtype_size_bit = 1;
  for (auto tv : reduction_tvs) {
    dtype_size_bit =
        std::max(dtype_size_bit, dataTypeSizeBit(tv->getDataType().value()));
  }
  // for welford, three arrays of type nvfuser_index_t are used to store var,
  // avg, and n. see KernelExecutor::computeLaunchParams. Here index type is
  // assumed as int64_t
  int64_t welford_factor = ir_utils::hasOpsOfType<WelfordOp>(fusion) ? 3l : 1l;
  if (welford_factor == 3l) {
    dtype_size_bit = std::max(dtype_size_bit, (int64_t)sizeof(int64_t) * 8);
  }
  int64_t reduction_broadcast_workspace_bit =
      threads_per_block * dtype_size_bit * welford_factor;

  return reduction_broadcast_workspace_bit;
}

bool isResharding(Fusion* fusion) {
  const std::vector<Expr*>& exprs = fusion->exprs();
  return std::any_of(
      exprs.begin(), exprs.end(), [](Expr* e) { return isResharding(e); });
}

void moveNonConcretizedBroadcastInnermost(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& ignored_tvs) {
  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.maybeBuildGraph(IdMappingMode::EXACT);
  const auto& permissive_graph =
      id_model.maybeBuildGraph(IdMappingMode::PERMISSIVE);

  // This function is meant to be used as a preprocessing step of each
  // segment scheduling. The goal is to find unmapped non-concretized
  // broadcast domains. It is not meant to find all unmapped dangling
  // domains. Any non-broadcast or concretized broadcast domains
  // should be guaranteed to be mapped with reference tensors, i.e.,
  // ignored_tvs. They should be taken care by the respective
  // scheduler.
  //
  // As such, all non-broadcast domains are skipped. Furthermore, any
  // domains that can be reachable from (i.e., mapped with)
  // ignored_tvs can be skipped. Since only broadcast domains are
  // considered, the exact mapping is enough to find such
  // domains.
  ValGroups ignored_groups;
  for (auto ignored_tv : ignored_tvs) {
    for (auto id : ignored_tv->domain()->allIDs()) {
      if (id->isBroadcast()) {
        ignored_groups.pushBack(exact_graph.toGroup(id));
      }
    }
  }

  for (auto tv : fusion->allTvs()) {
    std::vector<int64_t> broadcast_to_move;
    for (const auto i : arange(tv->getLoopDomain().size())) {
      auto loop_id = tv->getLoopDomain().at(i);
      if (!loop_id->isBroadcast()) {
        continue;
      }

      if (ignored_groups.has(exact_graph.toGroup(loop_id))) {
        continue;
      }

      // If the permissive group has a non-broadcast domain, it means it's
      // concretized.
      // TODO: Add a separate analysis for detecting concretized
      // broadcast domains using the Exact graph and replace the use
      // of the Permissive graph.
      const auto& permissive_group = permissive_graph.toGroup(loop_id);
      if (std::any_of(
              permissive_group->begin(),
              permissive_group->end(),
              [](Val* id) -> bool {
                return !id->as<IterDomain>()->isBroadcast();
              })) {
        continue;
      }

      broadcast_to_move.push_back((int64_t)i);
    }

    if (broadcast_to_move.empty()) {
      continue;
    }

    std::unordered_map<int64_t, int64_t> old2new;
    int64_t move_to_pos =
        (int64_t)(tv->getLoopDomain().size() - broadcast_to_move.size());
    for (const auto i : broadcast_to_move) {
      old2new[i] = move_to_pos;
      ++move_to_pos;
    }

    tv->reorder(old2new);
  }
}

int64_t reorderDevicesToOuter(TensorView* tv) {
  int64_t reorder_pos = 0;
  std::unordered_map<int64_t, int64_t> old2new;
  for (const auto i : arange(tv->getLoopDomain().size())) {
    if (tv->axis((int64_t)i)->isDeviceDim()) {
      old2new.emplace((int64_t)i, reorder_pos);
      ++reorder_pos;
    }
  }
  tv->reorder(old2new);
  return (int64_t)old2new.size();
}

std::vector<int64_t> reorderDomainLike(
    const std::vector<IterDomain*>& domain_to_reorder,
    const std::vector<IterDomain*>& ref) {
  if (domain_to_reorder.empty()) {
    return {};
  }

  Fusion* fusion = domain_to_reorder.at(0)->fusion();
  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto& graph = id_model.buildBroadcastGraph();

  ValGroups target_groups = graph.toGroups(domain_to_reorder);

  ValGroups ref_groups = graph.toGroups(ref);

  // Traverse from the reference to the target tv. The reference is
  // not guaranteed to cover all loop IDs of target, so
  // require_all_to_visited needs to be false
  auto path = ValGraphBFS::getExprGroupsBetween(
                  graph,
                  ref_groups,
                  target_groups,
                  /*require_all_to_visited=*/false)
                  .first;

  // Traverse the expr path to create an ordered ID groups
  std::deque<ValGroup> ordered_domain{
      ref_groups.vector().begin(), ref_groups.vector().end()};

  for (const auto& [expr_g, dir] : path) {
    auto inputs = getInputsOfExpr(
        expr_g, dir, ValGraphInputs(graph), ValGraphOutputs(graph));
    auto outputs = getOutputsOfExpr(
        expr_g, dir, ValGraphInputs(graph), ValGraphOutputs(graph));

    // Inserts the outputs at the innermost position
    auto innermost_it =
        std::find(ordered_domain.begin(), ordered_domain.end(), inputs.back());
    NVF_ERROR(innermost_it != ordered_domain.end());
    ordered_domain.insert(innermost_it, outputs.begin(), outputs.end());

    // Removes the inputs
    for (const auto& inp : inputs) {
      ordered_domain.erase(
          std::remove(ordered_domain.begin(), ordered_domain.end(), inp),
          ordered_domain.end());
    }
  }

  std::vector<int64_t> permutation(domain_to_reorder.size(), -1);

  // Place IDs that do not appear in ref at the outer position
  int64_t new_id_pos = 0;
  for (const auto i : arange(domain_to_reorder.size())) {
    const auto& loop_id_group = graph.toGroup(domain_to_reorder.at(i));
    auto it =
        std::find(ordered_domain.begin(), ordered_domain.end(), loop_id_group);
    if (it == ordered_domain.end()) {
      permutation.at(i) = new_id_pos;
      ++new_id_pos;
    }
  }
  for (const auto i : arange(domain_to_reorder.size())) {
    const auto& loop_id_group = graph.toGroup(domain_to_reorder.at(i));
    auto it =
        std::find(ordered_domain.begin(), ordered_domain.end(), loop_id_group);
    if (it != ordered_domain.end()) {
      int64_t new_pos =
          (int64_t)std::distance(ordered_domain.begin(), it) + new_id_pos;
      permutation.at(i) = new_pos;
    }
  }

  // domain_to_reorder can be partial with respect to ref, that is,
  // ordered_domain may contain IDs that do not appear in
  // domain_to_reorder. In that case, at this point, the permutation
  // vector may be sparse, e.g., {2, 0, 3}, which needs to be packed
  // to {1, 0, 2}.
  if (std::ranges::max(permutation) >= (int64_t)permutation.size()) {
    auto permutation_copy = permutation;
    std::ranges::sort(permutation_copy);
    for (auto& pos : permutation) {
      auto it = std::ranges::find(permutation_copy, pos);
      NVF_ERROR(it != permutation_copy.end());
      pos = static_cast<int64_t>(std::distance(permutation_copy.begin(), it));
    }
  }

  NVF_ERROR(
      std::ranges::is_permutation(
          permutation, std::ranges::iota_view(0, (int64_t)permutation.size())),
      "Invalid permutation: ",
      toDelimitedString(permutation));

  return permutation;
}

namespace {
// Class to handle expensive operations information and calculation of unroll
// factors
class ExpensiveOpInfo {
 public:
  ExpensiveOpInfo() : n_tanh_(0), n_exp_(0), n_reciprocal_(0) {}
  void analyzeFusion(Fusion* fusion) {
    for (auto expr : fusion->exprs()) {
      if (auto unary = dynamic_cast<UnaryOp*>(expr)) {
        switch (unary->getUnaryOpType()) {
          case UnaryOpType::Tanh:
            n_tanh_++;
            break;
          case UnaryOpType::Exp:
            n_exp_++;
            break;
          case UnaryOpType::Reciprocal:
            n_reciprocal_++;
            break;
          default:
            break;
        }
      }
    }
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "ExpensiveOpInfo: {";
    ss << "n_tanh: " << n_tanh_ << ", ";
    ss << "n_exp: " << n_exp_ << ", ";
    ss << "n_reciprocal: " << n_reciprocal_ << "}";
    return ss.str();
  }

  int64_t getComputationCostFactor() const {
    auto factor =
        n_tanh_ * f_tanh_ + n_exp_ * f_exp_ + n_reciprocal_ * f_reciprocal_;
    factor = std::max(factor, 1);

    // capped at 4 to avoid excessive unrolling which may lead to high register
    // usage and low occupancy.
    factor = std::min(factor, 4);
    return factor;
  }

 private:
  // Number of each expensive operation in the fusion
  int n_tanh_;
  int n_exp_;
  int n_reciprocal_;

  // Empirical factors to consider the cost of each operation
  static constexpr int f_tanh_ = 4;
  static constexpr int f_exp_ = 1;
  static constexpr int f_reciprocal_ = 1;
};
} // namespace

int64_t getComputationCostFactor(Fusion* fusion) {
  ExpensiveOpInfo info;
  info.analyzeFusion(fusion);
  return info.getComputationCostFactor();
}

// Calculate hardware bandwidth and required bytes in flight based on
// little's law. bytes_in_flight = bandwidth * latency
int64_t getRequiredBitsInFlight() {
  // H100, 32KB in flight @ 3352 GB/s = 9.5e-9 seconds
  constexpr float empirical_gmem_latency = 9.5e-9;
  const auto dev_idx = at::cuda::current_device();
  int gpu_mem_clock_khz;
  cudaDeviceGetAttribute(
      &gpu_mem_clock_khz, cudaDevAttrMemoryClockRate, dev_idx);
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  float hardware_bandwidth =
      2.f * (float)dev_prop->memoryBusWidth * (float)gpu_mem_clock_khz * 1000.f;
  return (int64_t)(empirical_gmem_latency * hardware_bandwidth);
}

namespace {
// Function to get the number of CUDA cores per SM.
// convert {major, minor} to hex number and check the map.
int getCoresPerSM(int major, int minor) {
  int sm_version = (major << 4) + minor;
  std::unordered_map<int, int> cores_per_sm_map = {
      {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
      {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
      {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
      {0x87, 128}, {0x89, 128}, {0x90, 128}, {0xa0, 128}, {0xc0, 128}};
  auto it = cores_per_sm_map.find(sm_version);
  if (it != cores_per_sm_map.end()) {
    return it->second;
  }
  // Use the default value of 128 for any architecture not listed,
  // applicable to all current Blackwell GPUs.
  return 128;
}
} // namespace
// Compute bandwidth flops ratio, return true if it's higher than
// the reference value of 0.07. It returns true for B100/200 and A100.
// Returns false for H100. The reference value is based on test of softmax,
// layer norm, and rms norm. Treating A100 as high bandwidth to flops ratio
// leads to better performance for softmax and dropout fused with layer norm
// or rms norm, but caused minor regressions for layer norm or rms norm alone.
bool isHighBandwidthFlopsRatio() {
  // A100-PCIe-80GB, 1.935e12 B/s, 1.95e13 flops, ratio = 0.0993
  // A100-SXM4-40GB, 1.555e12 B/s, 1.95e13 flops, ratio = 0.0798
  // H100-HBM3-80GB, 3.352e12 B/s, 6.69e13 flops, ratio = 0.0501
  constexpr float reference_ratio = 0.07f;
  const auto dev_idx = at::cuda::current_device();
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // bandwidth
  int gpu_mem_clock_khz;
  cudaDeviceGetAttribute(
      &gpu_mem_clock_khz, cudaDevAttrMemoryClockRate, dev_idx);
  float hardware_bandwidth = 2.f * (float)dev_prop->memoryBusWidth / 8.f *
      (float)gpu_mem_clock_khz * 1000.f;
  // fp32 cuda core flops
  const int cuda_core_per_sm = getCoresPerSM(dev_prop->major, dev_prop->minor);
  const int flops_per_cycle = 2;
  int gpu_clock_khz;
  cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev_idx);
  float flops = (float)gpu_clock_khz * 1000.f *
      (float)dev_prop->multiProcessorCount * (float)cuda_core_per_sm *
      (float)flops_per_cycle;

  float bandwidth_flops_ratio = hardware_bandwidth / flops;
  return bandwidth_flops_ratio > reference_ratio;
}

bool hasExpensiveMUFUops(Fusion* fusion) {
  const std::unordered_set<UnaryOpType> expensive_unary_ops{
      UnaryOpType::Exp,
      UnaryOpType::Tanh,
      UnaryOpType::Reciprocal,
      UnaryOpType::Rsqrt,
      UnaryOpType::Log,
      UnaryOpType::Log10,
      UnaryOpType::Log2,
      UnaryOpType::Sin,
      UnaryOpType::Cos};

  for (auto expr : fusion->exprs()) {
    if (expr->isA<UnaryOp>()) {
      if (auto unary = expr->as<UnaryOp>()) {
        if (expensive_unary_ops.count(unary->getUnaryOpType())) {
          return true;
        }
      }
    }
  }
  return false;
}

TensorView* scheduleInputToSkipIntermediates(TensorView* tv) {
  // First check that tv is fully contiguous. If not, then we can't currently
  // skip it.
  for (std::optional<bool> c : tv->getContiguity()) {
    if (c.has_value() && c.value() == false) {
      return tv;
    }
  }

  while (tv != nullptr) {
    if (tv->uses().size() != 1) {
      break;
    }
    Expr* use = tv->uses().front();

    // TODO: support ViewOp here too
    if (!use->isOneOf<BroadcastOp, SqueezeOp, LoadStoreOp>()) {
      break;
    }
    TensorView* consumer = ir_utils::getTvOutput(use);
    if (consumer == nullptr) {
      break;
    }

    // Setting memory type to Global and allocation to be exact mapped with
    // that of tv is enough to guarantee that it will be skipped during
    // lowering as a tensor producer alias.
    consumer->setMemoryType(MemoryType::Global);

    // reorder consumer's allocation domain to match the original input
    const std::vector<IterDomain*> old_loop = tv->getLoopDomain();

    // TODO: Ideally we would use a tool like the following, but this does not
    // preserve broadcasts that are missing in the target allocation domain.
    //
    //   scheduler_tools::scheduleLoopDomainsLike({consumer}, target_alloc);
    //   consumer->setAllocationDomain(consumer->getLoopDomain(), true);
    //   consumer->setLoopDomain(old_loop);
    //
    // Instead, we currently restrict to permutations and place new broadcasts
    // on the outside

    // Since we traverse in a p2c direction, we can use a pairwise map to
    // propagate allocation domain from the producer tv to the consumer.
    std::unordered_map<IterDomain*, IterDomain*> p2c =
        PairwiseLogicalDomainMap(tv, consumer).mapProducerToConsumer();
    std::vector<IterDomain*> new_consumer_alloc;
    new_consumer_alloc.reserve(tv->getMaybeAllocationDomain().size());
    for (IterDomain* p_id : tv->getMaybeAllocationDomain()) {
      // NOTE: This simple approach assumes that the allocation domains of the
      // producer are also logical domains. We can then map those to producer to
      // get IDs to use in the consumer's allocation domain to get IDs to use in
      // the consumer's allocation domain. This fails for ViewOp, which is why
      // it is currently disabled. In the future, we should propagate through
      // transforms as well using something similar to
      // scheduler_tools::scheduleLoopDomainsLike();
      auto it = p2c.find(p_id);
      NVF_ERROR(it != p2c.end());
      new_consumer_alloc.push_back(it->second);
    }
    consumer->setAllocationDomain(new_consumer_alloc, /*contiguity=*/true);

    tv = consumer;
  }
  return tv;
}

bool isSymbolicTensor(const TensorView* tv) {
  return std::any_of(
      tv->getLogicalDomain().begin(),
      tv->getLogicalDomain().end(),
      [](IterDomain* id) { return !id->extent()->isConst(); });
}

// This function requires the allocation domain to be a permutation of the
// logical domain.
// For each allocation domain ID (which is a logical domain ID),
// replace it with all the loop domain IDs that were derived from it.
void buildAllocationDomainFromLoopIds(TensorView* tv) {
  const auto& logical = tv->getLogicalDomain();
  const auto& alloc = tv->getMaybeAllocationDomain();
  NVF_ERROR(
      std::is_permutation(
          logical.begin(), logical.end(), alloc.begin(), alloc.end()),
      "buildAllocationDomainFromLoopIds expects the allocation domain to be a "
      "permutation of the logical domain");
  const auto& loop = tv->getLoopDomain();

  // Get all transformation expressions from allocation domain to loop domain
  // Equvalent to logical to loop domain since the allocation domain is a
  // permutation of the logical domain
  auto transform_exprs = DependencyCheck::getAllExprsBetween(
      {alloc.begin(), alloc.end()}, {loop.begin(), loop.end()});

  // Follow transformations, build map from ID to the original allocation IDs
  std::unordered_map<IterDomain*, std::vector<IterDomain*>> id_to_alloc_ids;
  for (auto alloc_id : alloc) {
    id_to_alloc_ids[alloc_id] = {alloc_id};
  }
  for (auto expr : transform_exprs) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      NVF_ERROR(
          id_to_alloc_ids.contains(split->in()),
          "Split input ",
          split->in()->toString(),
          " not found in id_to_alloc_ids");
      id_to_alloc_ids[split->outer()] = id_to_alloc_ids[split->in()];
      id_to_alloc_ids[split->inner()] = id_to_alloc_ids[split->in()];
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      NVF_ERROR(
          id_to_alloc_ids.contains(merge->outer()),
          "Merge input ",
          merge->outer()->toString(),
          " not found in id_to_alloc_ids");
      NVF_ERROR(
          id_to_alloc_ids.contains(merge->inner()),
          "Merge input ",
          merge->inner()->toString(),
          " not found in id_to_alloc_ids");
      // merge alloc ids of two merge inputs
      id_to_alloc_ids[merge->out()] = id_to_alloc_ids[merge->outer()];
      id_to_alloc_ids[merge->out()].insert(
          id_to_alloc_ids[merge->out()].end(),
          id_to_alloc_ids[merge->inner()].begin(),
          id_to_alloc_ids[merge->inner()].end());
    } else {
      NVF_ERROR(false, "Unsupported expression type: ", expr->toString());
    }
  }

  // Build map from allocation ID to its derived loop IDs
  std::unordered_map<IterDomain*, std::vector<IterDomain*>>
      alloc_to_derived_loop_ids;
  for (const auto& [id, alloc_list] : id_to_alloc_ids) {
    if (std::find(loop.begin(), loop.end(), id) == loop.end()) {
      continue;
    }
    for (auto alloc_id : alloc_list) {
      alloc_to_derived_loop_ids[alloc_id].push_back(id);
    }
  }

  // Build the new allocation domain
  // For each allocation ID, add its derived loop IDs.
  // Each allocation ID may have multiple derived loop IDs, but we only add
  // each loop ID once.
  std::vector<IterDomain*> new_alloc_domain;
  for (auto alloc_id : alloc) {
    const auto& derived_loop_ids = alloc_to_derived_loop_ids.at(alloc_id);
    for (auto loop_id : loop) {
      // skip if the loop ID has already been used
      if (std::find(
              new_alloc_domain.begin(), new_alloc_domain.end(), loop_id) !=
          new_alloc_domain.end()) {
        continue;
      }
      // skip if the loop ID is not derived from the allocation ID
      if (std::find(
              derived_loop_ids.begin(), derived_loop_ids.end(), loop_id) ==
          derived_loop_ids.end()) {
        continue;
      }
      // use the loop ID to build the new allocation domain
      new_alloc_domain.push_back(loop_id);
    }
  }
  tv->setAllocationDomain(new_alloc_domain, true);
}

void buildAllocationDomainForSharedMemoryTvs(Fusion* fusion) {
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() != MemoryType::Shared) {
      continue;
    }
    if (!tv->hasAllocation()) {
      continue;
    }
    buildAllocationDomainFromLoopIds(tv);
  }
}
} // namespace scheduler_utils
} // namespace nvfuser
