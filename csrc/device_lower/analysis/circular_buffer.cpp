// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <ir/utils.h>
#include <kernel_ir.h>

#include <device_lower/pass/circular_buffer.h>

#include <algorithm>
#include <iterator>
#include <vector>

namespace nvfuser {

namespace {

// Returns the leftmost position between the first unroll axis and computeAt
// position. Throws an error if the position is invalid for circular buffering.
int64_t getUnrollOrComputeAtPosition(const TensorView* tv) {
  NVF_ERROR(
      tv->getComputeAtPosition() > 0,
      "Expected computeAt for circular buffered TensorView");

  // Unroll must not exist outside of circular-buffer axis
  auto first_unroll_it = std::find_if(
      tv->getLoopDomain().begin(),
      tv->getLoopDomain().end(),
      [](const auto axis) {
        return axis->getParallelType() == ParallelType::Unroll;
      });

  const int64_t first_unroll_pos =
      (int64_t)std::distance(tv->getLoopDomain().begin(), first_unroll_it);

  const int64_t unroll_or_ca_pos =
      std::min(tv->getComputeAtPosition(), first_unroll_pos);

  NVF_ERROR(
      unroll_or_ca_pos > 0,
      "Invalid tensor to circular-buffer. ",
      "Valid circular buffer axis not found due to Unroll. ",
      tv->toString());

  return unroll_or_ca_pos;
}

// Returns the position of the circular buffer axis for non-warp-specialized
// tensors. Returns the size of the loop domain if no valid position is found.
int64_t getInnerMostCircularBufferPosition(const TensorView* tv) {
  const int64_t unroll_or_ca_pos = getUnrollOrComputeAtPosition(tv);
  int64_t valid_pos = (int64_t)tv->getLoopDomain().size();
  // Skip parallelized or broadcast axes
  for (int64_t i = unroll_or_ca_pos - 1; i >= 0; --i) {
    auto pt = tv->axis(i)->getParallelType();
    if (!isParallelTypeThread(pt) && !tv->axis(i)->isBroadcast()) {
      valid_pos = i;
      break;
    }
  }
  return valid_pos;
}

// Returns the position of the circular buffer axis for warp-specialized tensors
// with register sharing enabled. Returns the size of the loop domain if no
// valid position is found.
int64_t getOuterMostCircularBufferPosition(const TensorView* tv) {
  const int64_t unroll_or_ca_pos = getUnrollOrComputeAtPosition(tv);
  // Skip parallelized or broadcast axes
  for (int64_t i = 0; i < unroll_or_ca_pos; ++i) {
    auto pt = tv->axis(i)->getParallelType();
    if (!isParallelTypeThread(pt) && !tv->axis(i)->isBroadcast()) {
      return i;
    }
  }
  return (int64_t)tv->getLoopDomain().size();
}

// Circular-buffering prefetches the future subregions of the tensor.
// The subregion is defined by the axes inside of the CA position.
// There must be at least one axis that is outside (left) of the CA position,
// which defines the loop where prefetching is applied. Therefore,
// the CA position must be larger than 0.
int64_t getCircularBufferAxisPosition(const TensorView* tv) {
  // For warp-specialized tensors, the outer-most loop is the circular buffer
  // loop.
  if (std::holds_alternative<WarpSpecialized>(
          tv->circularBufferOptions().type)) {
    return getOuterMostCircularBufferPosition(tv);
  }

  // For pipelined tensors, the inner-most serial loop is the circular buffer
  // loop.
  return getInnerMostCircularBufferPosition(tv);
}

// Initial inspection of a fusion to find and validate circular buffered tensors
class CircularBufferFusionInspector : private IterVisitor {
 public:
  CircularBufferFusionInspector(Fusion* fusion, CircularBufferInfo& db_info)
      : db_info_(db_info) {
    traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void handle(TensorView* tv) final {
    if (!tv->isCircularBuffered()) {
      return;
    }

    NVF_ERROR(
        tv->definition(), "Fusion input shouldn't be circular buffered.", tv);

    db_info_.setCircularBufferTv(tv);
  }

 private:
  CircularBufferInfo& db_info_;
};

} // namespace

void validateCircularBufferedTensor(const TensorView* tv) {
  int64_t circular_buffer_pos = getCircularBufferAxisPosition(tv);
  NVF_ERROR(
      circular_buffer_pos >= 0,
      "Invalid tensor to circular-buffer. ",
      "Valid circular buffer axis not found. ",
      tv->toString());

  // Like vectorization, only LoadStoreOp with another TensorView is
  // considered.
  Expr* def = tv->definition();
  NVF_ERROR(
      def->isA<LoadStoreOp>(),
      "Invalid tensor to circular-buffer. ",
      "Only tensor defined by LoadStoreOp is supported: ",
      def->toString());

  NVF_ERROR(
      def->input(0)->isA<TensorView>(),
      "Invalid tensor to circular-buffer. ",
      "Only tensor defined by LoadStoreOp with TensorView is supported: ",
      def->toString());

  NVF_ERROR(
      !tv->hasComputeWith(),
      "computeWith is not supported with circular buffering: ",
      tv->toString());

  // Require the producer tensor to have been computed entirely for
  // the circular-buffering loop. Otherwise, the producer itself would
  // also need to be circular-bufferred.
  auto producer = def->input(0)->as<TensorView>();
  NVF_ERROR(
      producer->getComputePosition(tv) <= circular_buffer_pos,
      "Invalid tensor to circular-buffer. ",
      "The computeAt position of the producer tensor must be moved left: ",
      producer->toString());

  // Not strictly necessary, but only gmem -> smem or local and smem -> local
  // are allowed.
  const auto p_mem_type = producer->getMemoryType();
  const auto c_mem_type = tv->getMemoryType();
  NVF_ERROR(
      (p_mem_type == MemoryType::Global &&
       (c_mem_type == MemoryType::Shared || c_mem_type == MemoryType::Local)) ||
          (p_mem_type == MemoryType::Shared && c_mem_type == MemoryType::Local),
      "Invalid tensor to circular-buffer: ",
      tv->toString(),
      ". Producer memory type: ",
      p_mem_type,
      ". Consumer memory type: ",
      c_mem_type);

  // Due to limitation in predicate, 1D TMA load can only be safely used with
  // WarpSpecialized
  if (tv->circularBufferOptions().isEnable() &&
      !std::holds_alternative<WarpSpecialized>(
          tv->circularBufferOptions().type)) {
    NVF_ERROR(
        tv->definition() && !ir_utils::isCpAsyncBulk1D(tv->definition()),
        "1D TMA load can only be used with WarpSpecialized circular buffer: ",
        tv->definition()->toString());
  }
  // Ensure that the warp-specialized circular buffer loop is the outer-most
  // for-loop if register sharing is enabled.
  if (std::holds_alternative<WarpSpecialized>(
          tv->circularBufferOptions().type) &&
      std::get<WarpSpecialized>(tv->circularBufferOptions().type)
          .num_registers.has_value()) {
    for (int64_t axis : arange((int64_t)tv->getLoopDomain().size())) {
      // short-circuit: only check IterDomains to the left of the circular
      // buffer position
      if (axis >= circular_buffer_pos) {
        break;
      }
      NVF_ERROR(
          tv->getLoopDomain().at(axis)->isThread() ||
              tv->getLoopDomain().at(axis)->isDeviceDim() ||
              tv->getLoopDomain().at(axis)->isBroadcast() ||
              tv->getLoopDomain().at(axis)->isOneInt(),
          "When using register sharing with warp-specialized circular "
          "buffering, the circular buffer loop must be the outer-most "
          "for-loop.");
    }
  }

  return;
}

void CircularBufferInfo::build(Fusion* fusion) {
  CircularBufferFusionInspector inspector(fusion, *this);

  // Build circular buffered loop id's
  for (auto& info : map_) {
    auto circular_buffer_axis = info.second.circular_buffer_axis;
    // Keeps track of which loop disjoint set has been
    //  circular buffered. In index allocation, one index
    //  variable would need to be allocated in each
    //  circular buffer stage.
    concrete_circular_buffered_loop_id_.insert(
        lower_utils::getConcreteLoopID(circular_buffer_axis));
  }
}

bool CircularBufferInfo::isCircularBufferedIterDomain(IterDomain* id) {
  auto concrete_loop_id = lower_utils::getConcreteLoopID(id);
  return concrete_circular_buffered_loop_id_.count(concrete_loop_id);
}

CircularBufferInfo::TvInfo& CircularBufferInfo::getTvInfo(
    const TensorView* tv) {
  NVF_ERROR(
      tv->isCircularBuffered(),
      "Not a circular-buffered tensor: ",
      tv->toString());
  return map_[tv];
}

const CircularBufferInfo::TvInfo& CircularBufferInfo::getTvInfo(
    const TensorView* tv) const {
  NVF_ERROR(
      tv->isCircularBuffered(),
      "Not a circular-buffered tensor: ",
      tv->toString());
  return map_.at(tv);
}

namespace {

bool hasHopperMatmulConsumer(const TensorView* tv) {
  NVF_ERROR(tv != nullptr);
  for (Expr* use : tv->uses()) {
    if (auto mma = dynamic_cast<const MmaOp*>(use)) {
      return mma->isHopper();
    }
  }
  return false;
}

// NvFuser permits launching multiple TMA loads with a thread parallel axis.
// However, we cannot use the warp specialized axis for this. In the AsyncWarp,
// the warp specialized axis is the padded size. If you try to use that axis to
// launch TMA loads, you get incorrect results.
//
// For example, take a CTA with (TIDx = 128, TIDy = 2, TIDz = 1) and use
// TIDy as the warp specialized axis. A user can unintentionally thread
// parallelize the TMA load with `inlineMost`. See PingPongCircularBuffering.
// ProducerWarpSpecializedError.
//
// if (TIDy >= 2) {
//   Issue TMA-Load with ParallelType::TIDy.
//   It expects ParallelType::TIDy to be [0, 1], but it runs with [2].
// } else {
//   Run a computation with ParallelType::TIDy.
//   ParallelType::TDimY is in range [0, 1].
// }
void checkWarpSpecializedAxis(const TensorView* tv) {
  if (!std::holds_alternative<WarpSpecialized>(
          tv->circularBufferOptions().type)) {
    return;
  }

  const auto& warp_specialized =
      std::get<WarpSpecialized>(tv->circularBufferOptions().type);
  const std::vector<IterDomain*>& producer_loop = tv->domain()->loop();
  // Broadcast axis are not materialized in a TensorView, so it can be thread
  // parallelized.
  auto ws_id_producer_iter = std::find_if(
      producer_loop.begin(),
      producer_loop.end(),
      [&warp_specialized](IterDomain* id) {
        return !id->isBroadcast() &&
            id->getParallelType() == warp_specialized.on;
      });
  NVF_ERROR(
      ws_id_producer_iter == producer_loop.end(),
      "The warp specialized thread axis cannot appear in the AsyncWarp ",
      "TensorView: ",
      tv->toString());
}

IterDomain* findWarpSpecializedIterDomain(TensorView* tv, ParallelType ws_pt) {
  const std::vector<IterDomain*>& loop = tv->domain()->loop();
  auto ws_id_iter =
      std::find_if(loop.begin(), loop.end(), [ws_pt](IterDomain* id) {
        return id->getParallelType() == ws_pt;
      });
  NVF_ERROR(ws_id_iter != loop.end());
  return *ws_id_iter;
}

// If compute warp groups are independent, then the mbarrier waits for 128
// threads. Otherwise, it waits for all threads in ComputeWarp.
bool hasIndependentWarpGroups(const TensorView* tv) {
  if (!std::holds_alternative<WarpSpecialized>(
          tv->circularBufferOptions().type)) {
    return false;
  }

  NVF_ERROR(
      FusionInfoGuard::hasCurrent() &&
      FusionInfoGuard::current()->hasIdModel());
  const auto& id_graph =
      FusionInfoGuard::current()->idModel().idGraph(IdMappingMode::BROADCAST);

  const auto& warp_specialized =
      std::get<WarpSpecialized>(tv->circularBufferOptions().type);
  if (!warp_specialized.stage_slice_position.has_value()) {
    return false;
  }

  // Step 1: Get warp specialized iterDomain in consumer
  TensorView* consumer = ir_utils::consumerTvsOf(tv).at(0);
  IterDomain* ws_id =
      findWarpSpecializedIterDomain(consumer, warp_specialized.on);

  // ValGroup = std::shared_ptr<VectorOfUniqueEntries<Val*>>;
  // Step 2: Get ValGroup for warp specialized iterDomain
  const ValGroup& val_group = id_graph.toGroup(ws_id);

  // Step 3: Find corresponding producer iterDomain to warp specialized
  // iterDomain
  const std::vector<IterDomain*>& producer_loop = tv->domain()->loop();
  auto ws_id_producer_iter = std::find_if(
      producer_loop.begin(), producer_loop.end(), [val_group](IterDomain* id) {
        return val_group->has(id);
      });
  NVF_ERROR(ws_id_producer_iter != producer_loop.end());

  // Step 4: Map iterDomain to position in producer's loop domain
  int64_t ws_id_producer_pos =
      std::distance(producer_loop.begin(), ws_id_producer_iter);

  // Step 5: Use independent warp groups if warp specialized axis is to the
  // left of the stage_slice_position
  return ws_id_producer_pos < warp_specialized.stage_slice_position.value();
}

// All the iterDomains to the left of the slice position in the producer and
// consumer must belong to same iterDomain.
void checkTraversalIterDomains(const TensorView* tv, int64_t slice_position) {
  NVF_ERROR(
      FusionInfoGuard::hasCurrent() &&
      FusionInfoGuard::current()->hasIdModel());
  const auto& id_graph =
      FusionInfoGuard::current()->idModel().idGraph(IdMappingMode::BROADCAST);
  TensorView* consumer = ir_utils::consumerTvsOf(tv).at(0);
  const std::vector<IterDomain*>& consumer_loop = consumer->domain()->loop();
  const std::vector<IterDomain*>& producer_loop = tv->domain()->loop();
  for (int64_t idx : arange(slice_position)) {
    IterDomain* producer_id = producer_loop.at(idx);
    NVF_ERROR(
        idx < consumer->nDims(),
        "The corresponding consumer axis does not exist.");
    IterDomain* consumer_id = consumer_loop.at(idx);
    NVF_ERROR(
        id_graph.toGroup(producer_id) == id_graph.toGroup(consumer_id),
        "All iterDomains of the producer and consumer TensorViews to the left ",
        "of the stage_slice_position must be in the same Broadcast ValGroup.");
  }
}

void validateStageSlicePosition(const TensorView* tv) {
  if (!std::holds_alternative<WarpSpecialized>(
          tv->circularBufferOptions().type)) {
    return;
  }

  const auto& warp_specialized =
      std::get<WarpSpecialized>(tv->circularBufferOptions().type);
  if (!warp_specialized.stage_slice_position.has_value()) {
    return;
  }

  int64_t slice_position = warp_specialized.stage_slice_position.value();
  NVF_ERROR(
      slice_position >= 0, "Slice position must be non-negative integer.");
  NVF_ERROR(
      slice_position <= tv->nDims(),
      "Slice position must be inside TensorView nDims.");

  const std::vector<IterDomain*>& loop = tv->domain()->loop();
  bool is_slice_after_bulk = std::all_of(
      loop.begin(), loop.begin() + slice_position, [](IterDomain* id) {
        return id->getParallelType() != ParallelType::Bulk;
      });
  NVF_ERROR(
      is_slice_after_bulk,
      "Detected an iterDomain with ParallelType::Bulk to the left of stage ",
      "slice position.");

  checkTraversalIterDomains(tv, slice_position);
}

} // namespace

void CircularBufferInfo::setCircularBufferTv(const TensorView* tv) {
  IterDomain* cb_axis = nvfuser::getCircularBufferAxis(tv);
  NVF_ERROR(cb_axis != nullptr);
  auto concrete_loop_id = lower_utils::getConcreteLoopID(cb_axis);
  NVF_ERROR(concrete_loop_id != nullptr);

  validateCircularBufferedTensor(tv);

  getTvInfo(tv).circular_buffer_axis = cb_axis;
  circular_buffer_tvs_[concrete_loop_id].insert(tv);
  // Set and validate the new stage depth.
  setCircularBufferOptions(cb_axis, tv->circularBufferOptions());

  independent_compute_warp_groups_ = hasIndependentWarpGroups(tv);

  setCircularBufferInsertionPosition(tv, cb_axis);

  initializePingPongTracking(tv, concrete_loop_id);
}

void CircularBufferInfo::initializePingPongTracking(
    const TensorView* tv,
    IterDomain* cb_axis) {
  NVF_ERROR(tv != nullptr);
  NVF_ERROR(cb_axis != nullptr);

  // short-circuit: already tracking
  if (ping_pong_mbarriers_.contains(cb_axis)) {
    return;
  }

  // short-circuit: cooperative computation
  if (!hasIndependentWarpGroups(tv)) {
    return;
  }

  // short-circuit: only applied for hopper matmul
  if (!hasHopperMatmulConsumer(tv)) {
    return;
  }

  // short-circuit: only applied to persistent kernels
  if (getCircularBufferInsertionPosition(cb_axis) == 1) {
    return;
  }

  const auto& warp_specialized =
      std::get<WarpSpecialized>(tv->circularBufferOptions().type);
  TensorView* consumer = ir_utils::consumerTvsOf(tv).at(0);
  IterDomain* ws_id =
      findWarpSpecializedIterDomain(consumer, warp_specialized.on);
  NVF_ERROR(ws_id->extent()->isConst());
  int num_warp_groups = ws_id->extent()->value().as<int64_t>();
  ping_pong_mbarriers_.emplace(
      cb_axis,
      std::make_shared<HopperPingPongMbarriers>(
          num_warp_groups, warp_specialized.on));
}

void CircularBufferInfo::setCircularBufferOptions(
    IterDomain* id,
    const CircularBufferOptions& opt) {
  auto concrete_loop_id = lower_utils::getConcreteLoopID(id);
  NVF_ERROR(concrete_loop_id != nullptr);

  auto maybe_existing_depth_it =
      circular_buffer_options_.find(concrete_loop_id);
  if (maybe_existing_depth_it == circular_buffer_options_.end()) {
    circular_buffer_options_[concrete_loop_id] = opt;
    // Set the warp specialized dim and ensure there is only one
    if (std::holds_alternative<WarpSpecialized>(opt.type)) {
      auto ws_pt = std::get<WarpSpecialized>(opt.type).on;
      NVF_ERROR(
          warp_specialized_on_ == ParallelType::Serial ||
              warp_specialized_on_ == ws_pt,
          "Multiple warp specialization is not supported: ",
          warp_specialized_on_,
          " and ",
          ws_pt);
      warp_specialized_on_ = ws_pt;
    }
  } else {
    NVF_ERROR(
        opt == maybe_existing_depth_it->second,
        "Unsupported multiple options pipelining, was set to ",
        maybe_existing_depth_it->second,
        " by ",
        maybe_existing_depth_it->first->toString(),
        " and then set to ",
        opt,
        " by ",
        concrete_loop_id->toString());
  }
}

IterDomain* CircularBufferInfo::getCircularBufferAxis(
    const TensorView* tv) const {
  if (!tv->isCircularBuffered()) {
    return nullptr;
  }

  return getTvInfo(tv).circular_buffer_axis;
}

const CircularBufferOptions& CircularBufferInfo::getCircularBufferOptionsFor(
    IterDomain* circular_buffer_axis) const {
  if (GpuLower::hasCurrent()) {
    circular_buffer_axis = lower_utils::getConcreteLoopID(circular_buffer_axis);
  }

  auto maybe_it = circular_buffer_options_.find(circular_buffer_axis);

  NVF_ERROR(
      maybe_it != circular_buffer_options_.end(),
      "CircularBufferOptions is not found.");

  return maybe_it->second;
}

HopperPingPongMbarriers* CircularBufferInfo::getPingPongMbarriersFor(
    IterDomain* circular_buffer_axis) {
  if (GpuLower::hasCurrent()) {
    circular_buffer_axis = lower_utils::getConcreteLoopID(circular_buffer_axis);
  }

  auto maybe_it = ping_pong_mbarriers_.find(circular_buffer_axis);

  if (maybe_it == ping_pong_mbarriers_.end()) {
    return nullptr;
  }

  return maybe_it->second.get();
}

int64_t CircularBufferInfo::getCircularBufferInsertionPosition(
    IterDomain* circular_buffer_axis) const {
  if (GpuLower::hasCurrent()) {
    circular_buffer_axis = lower_utils::getConcreteLoopID(circular_buffer_axis);
  }

  auto maybe_depth_it =
      circular_buffer_insertion_position_.find(circular_buffer_axis);

  NVF_ERROR(
      maybe_depth_it != circular_buffer_insertion_position_.end(),
      "Circular buffer insertion position not found");

  return maybe_depth_it->second;
}

void CircularBufferInfo::setCircularBufferInsertionPosition(
    const TensorView* circular_buffer_tv,
    IterDomain* circular_buffer_axis) {
  if (GpuLower::hasCurrent()) {
    circular_buffer_axis = lower_utils::getConcreteLoopID(circular_buffer_axis);
  }
  checkWarpSpecializedAxis(circular_buffer_tv);
  validateStageSlicePosition(circular_buffer_tv);

  // short-circuit: insertion position is only for warp specialization.
  if (!std::holds_alternative<WarpSpecialized>(
          circular_buffer_tv->circularBufferOptions().type)) {
    circular_buffer_insertion_position_[circular_buffer_axis] = 1;
    return;
  }

  // short-circuit: stage_slice_position is specified in WarpSpecialized.
  const auto& warp_specialized = std::get<WarpSpecialized>(
      circular_buffer_tv->circularBufferOptions().type);

  // The outer_most position is the cloned for-loop for warp specialization.
  int64_t outer_most_circular_buffer_position =
      getOuterMostCircularBufferPosition(circular_buffer_tv);

  // The default inner_most position is the first serial iterDomain to the left
  // of the ComputeAt position. It can be specified in WarpSpecialized options.
  // stage_slice_position is an inclusive range from
  // [stage_slice_position, num_dimensions), so it corresponding for-loop is
  // stage_slice_position - 1.
  int64_t inner_most_circular_buffer_position =
      (warp_specialized.stage_slice_position.has_value())
      ? warp_specialized.stage_slice_position.value() - 1
      : getInnerMostCircularBufferPosition(circular_buffer_tv);

  NVF_ERROR(
      inner_most_circular_buffer_position < circular_buffer_tv->nDims(),
      "Expected inner_most_circular_buffer_position <= number of tensor "
      "dimensions ",
      "but got ",
      inner_most_circular_buffer_position,
      " and ",
      circular_buffer_tv->nDims());

  NVF_ERROR(
      outer_most_circular_buffer_position <=
          inner_most_circular_buffer_position,
      "Expected outer_most_circular_buffer_position <= "
      "inner_most_circular_buffer_position ",
      "but got ",
      outer_most_circular_buffer_position,
      " and ",
      inner_most_circular_buffer_position);

  // When inner_most position is used for cloning, the insertion point
  // for mbarrier synchronization is 1 or the cloned for-loop.
  // When outer_most != inner_most position, then the mbarrier synchronization
  // is placed at inner_most for-loop. The insertion_point is the number of
  // nested for-loops relative to the outer_most position.
  int64_t insertion_position = inner_most_circular_buffer_position -
      outer_most_circular_buffer_position + 1;
  circular_buffer_insertion_position_[circular_buffer_axis] =
      insertion_position;
}

namespace {

// Map iterDomain axis through IdModel loop map to get corresponding for-loop.
// Then, return the index of the for-loop in the stack of for-loops.
int64_t getForLoopIndex(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    bool is_inner_most_axis) {
  int64_t axis_position = is_inner_most_axis
      ? getInnerMostCircularBufferPosition(tv)
      : getOuterMostCircularBufferPosition(tv);
  IterDomain* axis = tv->axis(axis_position);
  kir::ForLoop* fl = CircularBufferInfo::getCircularBufferLoop(axis, loops);
  auto fl_it = std::find(loops.begin(), loops.end(), fl);
  return (int64_t)std::distance(loops.begin(), fl_it);
}

} // namespace

Val* CircularBufferInfo::getLinearIndex(
    TensorView* circular_buffer_tv,
    const std::vector<kir::ForLoop*>& loops) const {
  int64_t compute_at_loop_index =
      getForLoopIndex(circular_buffer_tv, loops, /*is_inner_most_axis=*/true);

  // short-circuit: return index for inner-most for-loop if not warp specialized
  // with register sharing
  const auto& circular_buffer_type =
      circular_buffer_tv->circularBufferOptions().type;
  bool is_warp_specialized =
      std::holds_alternative<WarpSpecialized>(circular_buffer_type);
  if (!is_warp_specialized) {
    return loops[compute_at_loop_index]->indexOrStartIfTrivial();
  }

  // The inner-most and outer-most for loops can be different.
  // Get outer-most for-loop index.
  int64_t outer_loop_index =
      getForLoopIndex(circular_buffer_tv, loops, /*is_inner_most_axis=*/false);

  // The inner_loop_index is the for-loop where the mbarrier arrive and wait
  // operations are inserted in circular buffering pass. Use
  // stage_slice_position if available. Otherwise, the default value is the
  // first serial for-loop to the left of compute_at_position.
  auto warp_specialized = std::get<WarpSpecialized>(circular_buffer_type);
  int64_t inner_loop_index = (warp_specialized.stage_slice_position.has_value())
      ? warp_specialized.stage_slice_position.value() - 1
      : compute_at_loop_index;

  // Calculate insertion position.
  int64_t insertion_position = inner_loop_index - outer_loop_index + 1;
  return getLinearIndexRelativeForLoopStack(
      loops, insertion_position, /*start=*/outer_loop_index);
}

Val* CircularBufferInfo::getLinearIndexRelativeForLoopStack(
    const std::vector<kir::ForLoop*>& loops,
    int64_t insertion_position,
    int64_t start_loop_index) const {
  NVF_ERROR(insertion_position > 0);
  NVF_ERROR(insertion_position <= (int64_t)loops.size());
  NVF_ERROR(start_loop_index >= 0);
  NVF_ERROR(start_loop_index < (int64_t)loops.size());

  // Insertion position is the number of for-loops in the nested for-loop
  // structure. end_loop_index is the last for-loop while start_loop_index is
  // the first for-loop.
  int64_t end_loop_index = insertion_position - 1 + start_loop_index;

  NVF_ERROR(end_loop_index < (int64_t)loops.size());
  NVF_ERROR(start_loop_index <= end_loop_index);

  Val* index = GpuLower::current()->kernel()->zeroVal();
  Val* extent = GpuLower::current()->kernel()->oneVal();
  for (int64_t i = end_loop_index; i >= start_loop_index; --i) {
    IterDomain* id = loops[i]->iter_domain();
    if (id->isBroadcast()) {
      continue;
    }
    // Skip parallelized axes except for warp specialized dim
    // when warp specialized dim is used in computation branch,
    // it represents index of computation warp groups.
    if (id->isThread() && (id->getParallelType() != warp_specialized_on_)) {
      continue;
    }
    index = SimplifyingIrBuilder::addExpr(
        index,
        SimplifyingIrBuilder::mulExpr(
            loops[i]->indexOrStartIfTrivial(), extent));
    extent = SimplifyingIrBuilder::mulExpr(
        extent, loops[i]->iter_domain()->extent());
  }
  return index;
}

kir::ForLoop* CircularBufferInfo::getCircularBufferLoop(
    IterDomain* axis,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return FusionInfoGuard::current()->caMap().areMapped(
               loop->iter_domain(), axis, IdMappingMode::LOOP) &&
        (!ignore_prologue ||
         loop->circularBufferLoopStage() != CircularBufferLoopStage::Prolog);
  });

  if (loop_it != loops.end()) {
    return *loop_it;
  } else {
    return nullptr;
  }
}

kir::ForLoop* CircularBufferInfo::getCircularBufferLoop(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) const {
  IterDomain* axis = getCircularBufferAxis(tv);

  if (axis == nullptr) {
    return nullptr;
  }

  return getCircularBufferLoop(axis, loops, ignore_prologue);
}

std::unordered_set<const TensorView*> CircularBufferInfo::getCircularBufferTvs(
    kir::ForLoop* axis) const {
  return getCircularBufferTvs(axis->iter_domain());
}

std::unordered_set<const TensorView*> CircularBufferInfo::getCircularBufferTvs(
    IterDomain* axis) const {
  auto concrete_id = lower_utils::getConcreteLoopID(axis);

  auto maybe_tvs_it = circular_buffer_tvs_.find(concrete_id);

  if (maybe_tvs_it == circular_buffer_tvs_.end()) {
    return {};
  }

  return maybe_tvs_it->second;
}

void CircularBufferInfo::setOriginalAllocSize(
    const TensorView* tv,
    Val* original_alloc_size) {
  getTvInfo(tv).original_alloc_size = original_alloc_size;
}

Val* CircularBufferInfo::getOriginalAllocSize(const TensorView* tv) {
  if (!tv->isCircularBuffered()) {
    return nullptr;
  }

  return getTvInfo(tv).original_alloc_size;
}

std::vector<const TensorView*> CircularBufferInfo::getCircularBufferTvs()
    const {
  std::vector<const TensorView*> keys;
  keys.reserve(map_.size());
  std::transform(
      map_.begin(), map_.end(), std::back_inserter(keys), [](auto pair) {
        return pair.first;
      });
  return keys;
}

std::string CircularBufferInfo::toString() const {
  std::stringstream ss;
  ss << "CircularBufferInfo: {" << std::endl;
  ss << "\tmap_:" << std::endl;
  for (const auto& pair : map_) {
    ss << "\t\t" << pair.first->toString() << " -> { circular_buffer_axis="
       << ir_utils::nullOrToString(pair.second.circular_buffer_axis)
       << ", original_alloc_size="
       << ir_utils::nullOrToInlineString(pair.second.original_alloc_size)
       << " }" << std::endl;
  }
  ss << "\tconcrete_circular_buffered_loop_id_:" << std::endl;
  ss << "\t\t" << ir_utils::toString(concrete_circular_buffered_loop_id_)
     << std::endl;
  ss << "\tcircular_buffer_options_:" << std::endl;
  for (const auto& pair : circular_buffer_options_) {
    ss << "\t\t" << pair.first->toString()
       << " -> { stage=" << pair.second.stage
       << ", prefetch=" << pair.second.prefetch << " }" << std::endl;
  }
  ss << "\tcircular_buffer_tvs_:" << std::endl;
  for (const auto& pair : circular_buffer_tvs_) {
    ss << "\t\t" << pair.first->toString() << " -> { ";
    for (const auto tv : pair.second) {
      ss << tv->toString() << ", ";
    }
    ss << " }" << std::endl;
  }
  ss << "}" << std::endl;
  return ss.str();
}

IterDomain* getCircularBufferAxis(const TensorView* tv) {
  int64_t cb_axis = getCircularBufferAxisPosition(tv);
  if (cb_axis == (int64_t)tv->getLoopDomain().size()) {
    return nullptr;
  }
  return tv->axis(cb_axis);
}

std::vector<AsyncWarp> createAsyncWarps(const std::vector<Expr*>& exprs) {
  std::vector<AsyncWarp> async_warps;

  // Gather all async operations.
  // TODO Add support for blackwell MmaOp
  std::vector<Expr*> async_warp_exprs;
  std::copy_if(
      exprs.begin(),
      exprs.end(),
      std::back_inserter(async_warp_exprs),
      [](Expr* e) {
        return ir_utils::isCpAsyncBulkLoad(e) &&
            e->output(0)->as<TensorView>()->isCircularBuffered();
      });

  // short-circuit: no async operations detected.
  if (async_warp_exprs.empty()) {
    return async_warps;
  }

  // TODO Divide into operations into separate AsyncWarps for multi-role
  // specialization. The current assumption is a single AsyncWarp with same
  // stage_slice_position.

  // Get TensorViews for async warps.
  std::vector<TensorView*> async_warp_tvs;
  std::transform(
      async_warp_exprs.begin(),
      async_warp_exprs.end(),
      std::back_inserter(async_warp_tvs),
      [](Expr* e) {
        auto output_tvs =
            ir_utils::filterByType<TensorView>(e->outputs()).vector();
        NVF_ERROR(output_tvs.size() == 1);
        return output_tvs.front();
      });
  NVF_ERROR(!async_warp_tvs.empty());

  // Check that all operations in the same warp have the same
  // stage_slice_position.
  std::vector<int64_t> stage_slice_positions;
  std::transform(
      async_warp_tvs.begin(),
      async_warp_tvs.end(),
      std::back_inserter(stage_slice_positions),
      [](TensorView* tv) {
        std::optional<int64_t> opt_stage_slice_position =
            ir_utils::getStageSlicePosition(tv);
        return opt_stage_slice_position.value_or(-1);
      });
  NVF_ERROR(
      stage_slice_positions.size() == 1 ||
      std::all_of(
          stage_slice_positions.begin() + 1,
          stage_slice_positions.end(),
          [&](int64_t v) { return v == stage_slice_positions.front(); }));

  TensorView* async_warp_tv = async_warp_tvs.front();
  NVF_ERROR(async_warp_tv != nullptr);
  int64_t stage_slice_position = stage_slice_positions.front();

  async_warps.emplace_back(
      async_warp_exprs, async_warp_tvs, stage_slice_position);
  return async_warps;
}

} // namespace nvfuser
