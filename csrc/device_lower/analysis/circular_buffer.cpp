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
          tv->circularBufferOptions().type) &&
      std::get<WarpSpecialized>(tv->circularBufferOptions().type)
          .num_registers.has_value()) {
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

  // short-circuit: insertion position is only for warp specialization with
  // register sharing
  if (!std::holds_alternative<WarpSpecialized>(
          tv->circularBufferOptions().type) ||
      !std::get<WarpSpecialized>(tv->circularBufferOptions().type)
           .num_registers.has_value()) {
    circular_buffer_insertion_position_[lower_utils::getConcreteLoopID(
        cb_axis)] = 1;
    return;
  }

  // The outer_most position is the cloned for-loop for warp specialization.
  // The inner_most position is the default cloned for-loop.
  // When inner_most position is used for cloning, the insertion point
  // for mbarrier synchronization is 1 or the cloned for-loop.
  // When outer_most != inner_most position, then the mbarrier synchronization
  // is still placed at inner_most for-loop. The insertion_point is the
  // number of nested for-loops relative to the outer_most position.
  int64_t outer_most_circular_buffer_position =
      getOuterMostCircularBufferPosition(tv);
  int64_t inner_most_circular_buffer_position =
      getInnerMostCircularBufferPosition(tv);
  NVF_ERROR(
      outer_most_circular_buffer_position <=
          inner_most_circular_buffer_position,
      "Expected outer_most_circular_buffer_position <= inner_most_circular_buffer_position",
      "but got ",
      outer_most_circular_buffer_position,
      " and ",
      inner_most_circular_buffer_position);
  int64_t insertion_position = (inner_most_circular_buffer_position -
                                outer_most_circular_buffer_position) +
      1;
  circular_buffer_insertion_position_[lower_utils::getConcreteLoopID(cb_axis)] =
      insertion_position;
}

void CircularBufferInfo::setCircularBufferOptions(
    IterDomain* id,
    const CircularBufferOptions& opt) {
  auto concrete_loop_id = lower_utils::getConcreteLoopID(id);

  auto maybe_existing_depth_it =
      circular_buffer_options_.find(concrete_loop_id);
  if (maybe_existing_depth_it == circular_buffer_options_.end()) {
    circular_buffer_options_[concrete_loop_id] = opt;
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

  auto maybe_depth_it = circular_buffer_options_.find(circular_buffer_axis);

  NVF_ERROR(
      maybe_depth_it != circular_buffer_options_.end(),
      "Prefetch distance not found");

  return maybe_depth_it->second;
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

Val* CircularBufferInfo::getLinearizeIndex(
    TensorView* circular_buffer_tv,
    const std::vector<ForLoop*>& loops) const {
  ForLoop* circular_buffer_loop =
      getCircularBufferLoop(circular_buffer_tv, loops);
  int64_t insertion_position =
      getCircularBufferInsertionPosition(circular_buffer_loop->iter_domain());
  bool is_warp_specialized_register_sharing =
      std::holds_alternative<WarpSpecialized>(
          circular_buffer_tv->circularBufferOptions().type) &&
      std::get<WarpSpecialized>(
          circular_buffer_tv->circularBufferOptions().type)
          .num_registers.has_value();
  int64_t offset = (is_warp_specialized_register_sharing)
      ? getOuterMostCircularBufferPosition(circular_buffer_tv)
      : getInnerMostCircularBufferPosition(circular_buffer_tv);
  return getLinearizeIndex(loops, insertion_position, offset);
}

Val* CircularBufferInfo::getLinearizeIndex(
    const std::vector<ForLoop*>& loops,
    int64_t insertion_position,
    int64_t start) const {
  int64_t end = insertion_position + start;
  NVF_ERROR((int64_t)loops.size() >= end);
  Val* index = GpuLower::current()->kernel()->zeroVal();
  Val* extent = GpuLower::current()->kernel()->oneVal();
  for (int64_t i = end; i >= start; --i) {
    if (loops[i]->iter_domain()->isParallelized() ||
        loops[i]->iter_domain()->isBroadcast()) {
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

ForLoop* CircularBufferInfo::getCircularBufferLoop(
    IterDomain* axis,
    const std::vector<ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return GpuLower::current()->caMap()->areMapped(
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

ForLoop* CircularBufferInfo::getCircularBufferLoop(
    const TensorView* tv,
    const std::vector<ForLoop*>& loops,
    bool ignore_prologue) const {
  IterDomain* axis = getCircularBufferAxis(tv);

  if (axis == nullptr) {
    return nullptr;
  }

  return getCircularBufferLoop(axis, loops, ignore_prologue);
}

std::unordered_set<const TensorView*> CircularBufferInfo::getCircularBufferTvs(
    ForLoop* axis) const {
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

} // namespace nvfuser
