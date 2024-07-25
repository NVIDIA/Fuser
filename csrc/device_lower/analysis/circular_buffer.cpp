// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <ir/utils.h>
#include <kernel_ir.h>

#include <device_lower/pass/circular_buffer.h>

#include <algorithm>
#include <iterator>
#include <vector>

namespace nvfuser {

namespace {

int64_t getCircularBufferAxisPosition(const TensorView* tv) {
  // Circular-buffering prefetches the next subregion of the tensor by
  // doubling the allocation. The subregion is defined by the axes
  // at the CA position till the inner-most position. There must be
  // at least one axis that is outside (left) of the CA position,
  // which defines the loop where prefetching is applied. Therefore,
  // the CA position must be larger than 0.

  NVF_ERROR(tv->getComputeAtPosition() > 0);

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

  int64_t valid_pos = -1;
  // Skip parallelized or broadcast axes
  for (int64_t i = unroll_or_ca_pos - 1; i >= 0; --i) {
    auto pt = tv->axis(i)->getParallelType();
    if (!isParallelTypeThread(pt) && !tv->axis(i)->isBroadcast()) {
      valid_pos = i;
      break;
    }
  }

  NVF_ERROR(
      valid_pos >= 0,
      "Invalid tensor to circular-buffer. ",
      "Valid circular buffer axis not found. ",
      tv->toString());

  return valid_pos;
}

IterDomain* getCircularBufferAxis(const TensorView* tv) {
  return tv->axis(getCircularBufferAxisPosition(tv));
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
    if (!(tv->isCircularBuffered() || tv->isCircularBuffered())) {
      return;
    }

    NVF_ERROR(
        tv->definition(), "Fusion input shouldn't be circular buffered.", tv);

    validateCircularBufferedTensor(tv);

    IterDomain* db_axis = getCircularBufferAxis(tv);

    db_info_.setCircularBufferAxis(tv, db_axis);
  }

 private:
  CircularBufferInfo& db_info_;
};

} // namespace
  
void validateCircularBufferedTensor(const TensorView* tv) {
  int64_t circular_buffer_pos = getCircularBufferAxisPosition(tv);

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
        GpuLower::current()->caMap()->getConcreteMappedID(
            circular_buffer_axis, IdMappingMode::LOOP));
  }
}

bool CircularBufferInfo::isCircularBufferedIterDomain(IterDomain* id) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
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

void CircularBufferInfo::setCircularBufferAxis(
    const TensorView* tv,
    IterDomain* axis) {
  getTvInfo(tv).circular_buffer_axis = axis;

  // Also validate the stage consistency with CA map.
  int64_t stage_depth = 0;
  if (tv->isCircularBuffered()) {
    stage_depth = tv->circularBufferDepth();
  } else {
    // Double buffer is a circular buffer with depth 2.
    stage_depth = 2;
  }

  // Set and validate the new stage depth.
  setStageDepth(axis, stage_depth);
}

void CircularBufferInfo::setStageDepth(IterDomain* id, int64_t stage_depth) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);

  auto maybe_exisiting_depth_it = stage_depth_.find(concrete_loop_id);
  if (maybe_exisiting_depth_it == stage_depth_.end()) {
    stage_depth_[concrete_loop_id] = stage_depth;
  } else {
    NVF_ERROR(
        stage_depth == maybe_exisiting_depth_it->second,
        "Unsupported multiple depth pipelining, was set to ",
        maybe_exisiting_depth_it->second,
        " by ",
        maybe_exisiting_depth_it->first->toString(),
        " and then set to ",
        stage_depth,
        " by ",
        concrete_loop_id->toString());
  }
}

IterDomain* CircularBufferInfo::getCircularBufferAxis(const TensorView* tv) {
  if (!tv->isCircularBuffered()) {
    return nullptr;
  }

  return getTvInfo(tv).circular_buffer_axis;
}

int64_t CircularBufferInfo::getStageDepthFor(
    IterDomain* circular_buffer_axis) const {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      circular_buffer_axis, IdMappingMode::LOOP);

  auto maybe_depth_it = stage_depth_.find(concrete_id);

  NVF_ERROR(maybe_depth_it != stage_depth_.end(), "Stage depth not found");

  return maybe_depth_it->second;
}

ForLoop* CircularBufferInfo::getCircularBufferLoop(
    IterDomain* axis,
    const std::vector<ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return GpuLower::current()->caMap()->areMapped(
               loop->iter_domain(), axis, IdMappingMode::EXACT) &&
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
    bool ignore_prologue) {
  IterDomain* axis = getCircularBufferAxis(tv);

  if (axis == nullptr) {
    return nullptr;
  }

  return getCircularBufferLoop(axis, loops, ignore_prologue);
}

void CircularBufferInfo::setOriginalAllocSize(
    const TensorView* tv,
    Val* original_alloc_size) {
  getTvInfo(tv).original_alloc_size = original_alloc_size;
}

Val* CircularBufferInfo::getOriginalAllocSize(const TensorView* tv) {
  if (!(tv->isCircularBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).original_alloc_size;
}

} // namespace
