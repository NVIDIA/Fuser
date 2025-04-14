// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/lower.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <preseg_passes/stream_parallel_type.h>

namespace nvfuser::preseg_passes {

// Helper function to find the first stream-parallelized axis in a domain.
// This function throws if multiple stream-parallelized axes are found (only one
// is allowed)
IterDomain* getStreamAxis(const std::vector<IterDomain*>& domain) {
  IterDomain* ret = nullptr;
  for (auto id : domain) {
    if (id->getParallelType() == ParallelType::Stream) {
      NVF_CHECK(
          ret == nullptr,
          "Expected at most one stream axis in the domain, but found ",
          id,
          " and ",
          ret);
      ret = id;
    }
  }
  return ret;
}

// StreamParallelType pass implementation.
// This pass handles stream parallelization of operations in a fusion.
// It works by:
// 1. Identifying stream-parallelized axes in tensor operations
// 2. Grouping compatible operations into stream-parallel for-loops
// 3. Setting up proper stream synchronization and management
//
// The pass ensures that:
// - Input tensors don't have stream axes
// - Only one stream axis exists per tensor
// - Stream axes are properly synchronized
// - Operations are correctly grouped into stream-parallel regions
// - The resulting HostIrContainer's top level expression is valid for execution
// and does not contain any stream axes
//
// TODO: Here, we assume that the fusion input is a HostIrContainer and use the
// linear structure of the HostIrContainer::topLevelExpr to greedily merge the
// adjacent compatible stream for-loop bodies. Ideally we should look at the dag
// and use the segmenter.
void StreamParallelType::runPass(Fusion* fusion) {
  // Verify that input tensors don't have stream axes
  NVF_CHECK(
      std::all_of(
          fusion->inputs().begin(),
          fusion->inputs().end(),
          [](Val* input) {
            auto input_tv = dynamic_cast<TensorView*>(input);
            return input_tv == nullptr ||
                getStreamAxis(input_tv->getLoopDomain()) == nullptr;
          }),
      "Expected no stream axis in the TensorView inputs.");

  // Set up the fusion environment and build the ID model
  FusionGuard fg(fusion);
  hir::HostIrContainer* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");

  IdModel id_model(fusion);
  id_model.buildAlmostExactGraph();

  std::vector<Expr*> new_top_level_exprs;

  // Step 1: Group expressions into stream-parallel regions
  // This step identifies which expressions can be merged into single stream
  // for-loops
  //
  // After this step, new_top_level_exprs contains a
  // list of expressions including newly created for-loops representing
  // the stream parallelization containing and the relevant expressions
  for (auto expr : hic->topLevelExprs()) {
    // Skip expressions with no outputs
    if (expr->outputs().size() == 0) {
      new_top_level_exprs.push_back(expr);
      continue;
    }

    // Verify single output constraint
    NVF_CHECK(
        expr->outputs().size() == 1,
        "Each expr should have at most one output.");

    // Get the output tensor and check for stream parallelization
    TensorView* output = expr->output(0)->as<TensorView>();
    IterDomain* stream_axis = getStreamAxis(output->getLoopDomain());

    // If no stream axis, keep expression as is
    if (stream_axis == nullptr) {
      new_top_level_exprs.push_back(expr);
      continue;
    }

    // Verify expression can be handled as a standalone host operation
    NVF_ERROR(
        HostIrLower::isLoweredAsStandaloneHostOp(expr),
        "Stream parallel type not supported for expr ",
        expr);

    // Find the stream axis in the logical (and not loop) domain
    auto it_logical_stream_axis = std::find(
        output->getLogicalDomain().begin(),
        output->getLogicalDomain().end(),
        stream_axis);

    // Verify stream axis is not split/merged
    NVF_ERROR(
        it_logical_stream_axis != output->getLogicalDomain().end(),
        "Cannot stream parallelize on a split/merge axis ",
        stream_axis);

    // Verify stream axis is an iteration axis (not reduction/broadcast)
    NVF_CHECK(
        stream_axis->getIterType() == IterType::Iteration,
        "Stream axis ",
        stream_axis,
        " should be an iteration axis.");

    // Check if expression can be merged with previous stream for-loop
    if (!new_top_level_exprs.empty() &&
        new_top_level_exprs.back()->isA<ForLoop>() &&
        id_model.idGraph(IdMappingMode::ALMOSTEXACT)
            .disjointValSets()
            .strictAreMapped(
                stream_axis,
                new_top_level_exprs.back()->as<ForLoop>()->iterDomain())) {
      // Merge with existing for-loop
      new_top_level_exprs.back()->as<ForLoop>()->body().push_back(expr);
    } else {
      // Create new for-loop for stream parallelization
      auto* for_loop = IrBuilder::create<ForLoop>(
          stream_axis,
          /*index=*/IrBuilder::create<Val>(DataType::Index),
          /*start=*/hic->zeroVal(),
          /*stop=*/stream_axis->extent(),
          /*step=*/hic->oneVal(),
          /*vectorize=*/false,
          /*vectorize_shift=*/nullptr,
          /*unroll_required=*/false,
          CircularBufferLoopStage::NotApplicable,
          /*circular_buffer_loop_stage_depth=*/0);
      for_loop->body().push_back(expr);
      // replace the current expr by the for-loop containing it
      new_top_level_exprs.push_back(for_loop);
    }
  }

  // Step 2: Process each for-loop's body by slicing tensors
  // This step handles the actual tensor slicing for stream parallelization
  std::vector<Expr*> top_level_exprs = std::move(new_top_level_exprs);
  new_top_level_exprs.clear();

  for (auto top_level_expr : top_level_exprs) {
    if (!top_level_expr->isA<ForLoop>()) {
      new_top_level_exprs.push_back(top_level_expr);
      continue;
    }

    auto* for_loop = top_level_expr->as<ForLoop>();
    std::vector<Expr*> new_loop_body;

    // Process each expression in the loop body
    std::vector<Expr*> current_loop_body = for_loop->body().exprs();
    for (auto it_expr = current_loop_body.begin();
         it_expr != current_loop_body.end();
         ++it_expr) {
      Expr* expr = *it_expr;

      // Process input tensors
      for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
        // Find stream axis index in input tensor
        int64_t input_stream_id_logical_index = -1;
        for (auto id : input->getLoopDomain()) {
          if (id_model.idGraph(IdMappingMode::ALMOSTEXACT)
                  .disjointValSets()
                  .strictAreMapped(for_loop->iterDomain(), id)) {
            // Verify only one stream axis exists
            NVF_CHECK(
                input_stream_id_logical_index == -1,
                "Expected at most one axis mapping to the stream axis ",
                for_loop->iterDomain(),
                " in the tensor ",
                input,
                " loop's domain ",
                input->getLoopDomain());

            // Find stream axis in logical domain
            auto it_input_stream_id_logical = std::find(
                input->getLogicalDomain().begin(),
                input->getLogicalDomain().end(),
                id);
            NVF_CHECK(
                it_input_stream_id_logical != input->getLogicalDomain().end(),
                "Expected to find ",
                id,
                " in ",
                input,
                "'s logical domain ",
                input->getLogicalDomain());
            input_stream_id_logical_index = std::distance(
                input->getLogicalDomain().begin(), it_input_stream_id_logical);
          }
        }

        // Skip if no stream axis found
        if (input_stream_id_logical_index == -1) {
          continue;
        }

        // Create sliced tensor for current stream iteration
        TensorView* input_j = select(
            input,
            input_stream_id_logical_index,
            for_loop->index(),
            /*keep_reduction_axis=*/true);
        new_loop_body.push_back(input_j->definition());

        // Update all expressions using this input
        for (auto it_running_expr = current_loop_body.begin();
             it_running_expr != current_loop_body.end();
             ++it_running_expr) {
          Expr* running_expr = *it_running_expr;
          for (auto* running_input :
               ir_utils::filterByType<TensorView>(running_expr->inputs())) {
            if (running_input == input) {
              *it_running_expr = ir_utils::replaceValInExprInputs(
                  running_expr, input, input_j);
            }
          }
        }
      }

      // Process output tensors
      for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
        // Find stream axis index in output tensor
        int64_t output_stream_id_logical_index = -1;
        for (auto id : output->getLoopDomain()) {
          if (id_model.idGraph(IdMappingMode::ALMOSTEXACT)
                  .disjointValSets()
                  .strictAreMapped(for_loop->iterDomain(), id)) {
            // Verify only one stream axis exists
            NVF_CHECK(
                output_stream_id_logical_index == -1,
                "Expected at most one axis mapping to the stream axis ",
                for_loop->iterDomain(),
                " in the tensor ",
                output,
                " loop's domain ",
                output->getLoopDomain());

            // Find stream axis in logical domain
            auto it_output_stream_id_logical = std::find(
                output->getLogicalDomain().begin(),
                output->getLogicalDomain().end(),
                id);
            NVF_CHECK(
                it_output_stream_id_logical != output->getLogicalDomain().end(),
                "Expected to find ",
                id,
                " in ",
                output,
                "'s logical domain ",
                output->getLogicalDomain());
            output_stream_id_logical_index = std::distance(
                output->getLogicalDomain().begin(),
                it_output_stream_id_logical);
          }
        }

        // Skip if no stream axis found
        if (output_stream_id_logical_index == -1) {
          continue;
        }

        // Create sliced tensor for current stream iteration
        TensorView* output_j = select(
            output,
            output_stream_id_logical_index,
            for_loop->index(),
            /*keep_reduction_axis=*/true);

        // Allocate memory for the output tensor
        new_top_level_exprs.push_back(
            IrBuilder::create<kir::Allocate>(output, MemoryType::Global));
        new_loop_body.push_back(output_j->definition());

        // Update all expressions using this output
        for (auto it_running_expr = current_loop_body.begin();
             it_running_expr != current_loop_body.end();
             ++it_running_expr) {
          Expr* running_expr = *it_running_expr;
          for (auto* running_output :
               ir_utils::filterByType<TensorView>(running_expr->outputs())) {
            if (running_output == output) {
              *it_running_expr = ir_utils::transferDefinitionToNewOutputs(
                  running_expr, {output_j});
            }
          }
        }
      }
      new_loop_body.push_back(*it_expr);
    }

    // Update for-loop body with processed expressions
    for_loop->body().clear();
    for (auto* expr : new_loop_body) {
      for_loop->body().push_back(expr);
    }
    new_top_level_exprs.push_back(top_level_expr);
  }

  // Step 3: Add stream management and synchronization
  for (auto* top_level_expr : new_top_level_exprs) {
    if (!top_level_expr->isA<ForLoop>()) {
      continue;
    }
    auto* for_loop = top_level_expr->as<ForLoop>();
    std::vector<Expr*> new_loop_body;

    // Get current stream for later synchronization
    auto* get_current_stream = IrBuilder::create<hir::GetCurrentStream>();
    hir::Stream* original_stream = get_current_stream->stream();
    new_loop_body.push_back(get_current_stream);

    // Set up stream for current iteration
    auto* number_of_streams =
        IrBuilder::create<NamedScalar>("numberOfStreams", DataType::Int);
    auto* stream_index = mod(for_loop->index(), number_of_streams);
    auto* stream = IrBuilder::create<hir::Stream>(stream_index);
    auto* set_stream = IrBuilder::create<hir::SetCurrentStream>(stream);
    new_loop_body.push_back(set_stream);

    // Synchronize with original stream
    auto* initial_sync_stream =
        IrBuilder::create<hir::Synchronize>(original_stream);
    new_loop_body.push_back(initial_sync_stream);

    // Add the actual computation expressions
    for (auto* expr : for_loop->body().exprs()) {
      new_loop_body.push_back(expr);
    }

    // Restore original stream and synchronize
    auto* set_back_original_stream =
        IrBuilder::create<hir::SetCurrentStream>(original_stream);
    new_loop_body.push_back(set_back_original_stream);
    auto* sync_stream = IrBuilder::create<hir::Synchronize>(stream);
    new_loop_body.push_back(sync_stream);

    // Update for-loop body with stream management
    for_loop->body().clear();
    for (auto* expr : new_loop_body) {
      for_loop->body().push_back(expr);
    }
  }

  // Update the container's top-level expressions
  hic->resetTopLevelExprs(new_top_level_exprs);
}

} // namespace nvfuser::preseg_passes
