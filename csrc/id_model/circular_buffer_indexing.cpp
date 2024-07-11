// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/circular_buffer_indexing.h>

namespace nvfuser {

// If the for-loop is double-buffered and not prologue, the loop
// index should be advanced by one except for the double-buffered
// tensor itself
Val* adjustProducerLoopIndexForCircularBuffering(
    const Expr* expr,
    const ForLoop* for_loop,
    const IdModel& id_model,
    Val* loop_index) {
  NVF_ERROR(for_loop != nullptr);

  auto consumer_tv = ir_utils::getTvOutput(expr);

  if (!consumer_tv->isCircularBuffered()) {
    return loop_index;
  }

  NVF_ERROR(expr->inputs().size() == 1);

  auto producer_tv = expr->input(0)->as<TensorView>();

  // Double-buffered tensor itself does not need this adjustment
  if (producer_tv->isCircularBuffered() &&
      id_model.idGraph(IdMappingMode::LOOP)
          .disjointValSets()
          .strictAreMapped(
              getCircularBufferAxis(producer_tv), for_loop->iter_domain())) {
    return loop_index;
  }

  if (for_loop->circularBufferLoopStage() != CircularBufferLoopStage::Main &&
      for_loop->circularBufferLoopStage() != CircularBufferLoopStage::Epilog) {
    return loop_index;
  }

  const auto gpu_lower = GpuLower::current();
  NVF_ERROR(
      gpu_lower != nullptr,
      "Double buffering info of GpuLower is required but GpuLower is missing");

  auto stage_depth =
      (int64_t)GpuLower::current()->circularBufferInfo().getStageDepthFor(
          for_loop->iter_domain());

  auto adjusted_loop_index = SimplifyingIrBuilder::addExpr(
      loop_index,
      SimplifyingIrBuilder::create<Val>(stage_depth - 1L, DataType::Index));

  VERBOSE() << "Adjusted initial producer index: "
            << adjusted_loop_index->toInlineString() << std::endl;
  VERBOSE() << expr->toString();

  return adjusted_loop_index;
}

Val* adjustIndexToSwitchBuffer(
    TensorView* tv,
    bool as_consumer,
    const std::vector<ForLoop*>& for_loops,
    Val* idx) {
  if (!tv->isCircularBuffered()) {
    return idx;
  }

  const auto gpu_lower = GpuLower::current();
  NVF_ERROR(
      gpu_lower != nullptr,
      "Double buffering info of GpuLower is required but GpuLower is missing");

  auto db_loop =
      gpu_lower->circularBufferInfo().getCircularBufferLoop(tv, for_loops);

  NVF_ERROR(db_loop != nullptr);

  // Mostly just copied from getNonGlobalConsumerStridedIndices

  bool is_prolog =
      db_loop->circularBufferLoopStage() == CircularBufferLoopStage::Prolog;

  auto loop_index = db_loop->indexOrStartIfTrivial();

  const auto stage_depth =
      (int64_t)gpu_lower->circularBufferInfo().getStageDepthFor(
          db_loop->iter_domain());

  auto db_index_offset = loop_index;
  if (as_consumer && !is_prolog) {
    // Read-ahead offset for consumer indexing
    db_index_offset = SimplifyingIrBuilder::addExpr(
        db_index_offset,
        SimplifyingIrBuilder::create<Val>(stage_depth - 1, DataType::Index));
  }

  // % `num_stages` not necessary in prologue
  if (!is_prolog) {
    db_index_offset = SimplifyingIrBuilder::modExpr(
        db_index_offset,
        SimplifyingIrBuilder::create<Val>(stage_depth, DataType::Index));
  }

  auto original_alloc_size =
      gpu_lower->circularBufferInfo().getOriginalAllocSize(tv);

  auto db_strided_index =
      SimplifyingIrBuilder::mulExpr(db_index_offset, original_alloc_size);

  auto updated_idx = SimplifyingIrBuilder::addExpr(idx, db_strided_index);
  return updated_idx;
}

std::optional<CircularBufferLoopStage> getCircularBufferLoopStage(
    TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const ValGraph& loop_graph) {
  auto db_axis =
      GpuLower::current()->circularBufferInfo().getCircularBufferAxis(tv);
  if (db_axis == nullptr) {
    return std::nullopt;
  }

  for (const auto fl : for_loops) {
    if (loop_graph.disjointValSets().strictAreMapped(
            fl->iter_domain(), db_axis)) {
      return fl->circularBufferLoopStage();
    }
  }

  NVF_ERROR(false, "Double-buffered loop not found for ", tv->toString());
}

} // namespace nvfuser
