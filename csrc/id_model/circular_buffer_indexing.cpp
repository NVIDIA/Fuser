// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/circular_buffer_indexing.h>
#include <id_model/indexing_utils.h>
#include <ir/utils.h>

namespace nvfuser {

Val* getLoopIndexOfCircularBufferLoop(
    IterDomain* loop_id,
    const std::vector<ForLoop*>& for_loops,
    const IdModel& id_model) {
  ForLoop* fl = indexing_utils::getForLoop(
      loop_id, for_loops, id_model.idGraph(IdMappingMode::LOOP));

  // It's possible that there's no corresponding ForLoop, i.e,
  // when this loop ID corresponds to a reduction
  // domain and we are building a map for the expression to
  // initialize the reduction buffer. For such a case, this WAR is
  // irrelevant.
  if (fl == nullptr) {
    return nullptr;
  }

  if (fl->circularBufferLoopStage() != CircularBufferLoopStage::NotApplicable) {
    // Always return index even if for-loop is trivial.
    // Simplify index with substitution.
    return fl->index();
  } else {
    return nullptr;
  }
}

Val* getLoopIndexOffsetForProducerOfCircularBuffer(
    const Expr* expr,
    const ForLoop* for_loop,
    const IdModel& id_model) {
  NVF_ERROR(for_loop != nullptr);

  if (for_loop->circularBufferLoopStage() ==
      CircularBufferLoopStage::NotApplicable) {
    return nullptr;
  }

  NVF_ERROR(
      GpuLower::hasCurrent(),
      "Circular buffering info of GpuLower is required but GpuLower is "
      "missing");

  CircularBufferInfo& info = GpuLower::current()->circularBufferInfo();

  auto consumer_tv = ir_utils::getTvOutput(expr);

  if (!consumer_tv->isCircularBuffered()) {
    return nullptr;
  }

  auto circular_buffer_axis = info.getCircularBufferAxis(consumer_tv);
  if (!id_model.idGraph(IdMappingMode::LOOP)
           .disjointValSets()
           .strictAreMapped(for_loop->iter_domain(), circular_buffer_axis)) {
    // This loop is not the circular buffer loop for this tensor
    return nullptr;
  }

  // This loop should be either prologue or main
  NVF_ERROR(
      hasCircularBufferLoad(for_loop->circularBufferLoopStage()),
      "Unexpected loop stage: ",
      for_loop->circularBufferLoopStage(),
      ". ",
      expr->toString());

  // This offsetting is only necessary in the main loop
  if (for_loop->circularBufferLoopStage() != CircularBufferLoopStage::Main) {
    return nullptr;
  }

  auto prefetch_distance =
      info.getCircularBufferOptionsFor(for_loop->iter_domain()).prefetch;

  return IrBuilder::create<Val>(prefetch_distance, DataType::Index);
}

Val* getOffsetForCircularBufferTensor(
    TensorView* circular_buffer_tv,
    bool as_consumer,
    const std::vector<ForLoop*>& for_loops) {
  NVF_ERROR(circular_buffer_tv->isCircularBuffered());

  const auto gpu_lower = GpuLower::current();
  NVF_ERROR(
      gpu_lower != nullptr,
      "Circular buffering info of GpuLower is required but GpuLower is "
      "missing");

  auto circular_buffer_loop =
      gpu_lower->circularBufferInfo().getCircularBufferLoop(
          circular_buffer_tv, for_loops);

  NVF_ERROR(circular_buffer_loop != nullptr);

  // Mostly just copied from getNonGlobalConsumerStridedIndices
  const CircularBufferLoopStage stage =
      circular_buffer_loop->circularBufferLoopStage();

  Val* loop_index = gpu_lower->circularBufferInfo().getLinearIndex(
      circular_buffer_tv, for_loops);

  const auto& opt =
      GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
          circular_buffer_loop->iter_domain());

  // If this appears as a consumer, it should be either prologue or
  // main. If it's producer, it should be main or epilogue
  NVF_ERROR(
      (as_consumer && hasCircularBufferLoad(stage)) ||
          (!as_consumer && hasCircularBufferConsume(stage)),
      "Unexpected circular buffer stage: ",
      stage,
      " for using ",
      circular_buffer_tv->toString(),
      " as ",
      (as_consumer ? "consumer" : "producer"));

  auto offset = loop_index;

  // If this is a consumer and in the main loop, advance the offset
  // for read-ahead
  if (as_consumer && stage == CircularBufferLoopStage::Main) {
    offset = SimplifyingIrBuilder::addExpr(
        offset,
        SimplifyingIrBuilder::create<Val>(opt.prefetch, DataType::Index));
  }

  // Add "offset % num_stages", except when it's in prologue
  if (stage != CircularBufferLoopStage::Prolog) {
    offset = SimplifyingIrBuilder::modExpr(
        offset, SimplifyingIrBuilder::create<Val>(opt.stage, DataType::Index));
  }

  auto original_alloc_size =
      gpu_lower->circularBufferInfo().getOriginalAllocSize(circular_buffer_tv);

  return SimplifyingIrBuilder::mulExpr(offset, original_alloc_size);
}

CircularBufferLoopStage getCircularBufferLoopStage(
    const TensorView* circular_buffer_tv,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph) {
  NVF_ERROR(
      GpuLower::hasCurrent(),
      "Circular buffering info of GpuLower is required but GpuLower is "
      "missing");

  auto circular_buffer_axis =
      GpuLower::current()->circularBufferInfo().getCircularBufferAxis(
          circular_buffer_tv);

  if (circular_buffer_axis == nullptr) {
    return CircularBufferLoopStage::NotApplicable;
  }

  for (const auto fl : for_loops) {
    if (loop_graph.disjointValSets().strictAreMapped(
            fl->iter_domain(), circular_buffer_axis)) {
      return fl->circularBufferLoopStage();
    }
  }

  NVF_THROW(
      "Circular buffer loop not found for ", circular_buffer_tv->toString());
}

} // namespace nvfuser
