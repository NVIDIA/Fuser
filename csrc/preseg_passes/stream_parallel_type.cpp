// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <preseg_passes/stream_parallel_type.h>


namespace nvfuser::preseg_passes {

// returns the first stream axis in the domain, or nullptr if there is none. Throws if two axis are stream parallelized
IterDomain* getStreamAxis(const std::vector<IterDomain*>& domain) {
  IterDomain* ret = nullptr;
  for (auto id : domain) {
    if (id->getParallelType() == ParallelType::Stream) {
      NVF_CHECK(ret == nullptr, "Expected at most one stream axis in the domain, but found ", id, " and ", ret);
      ret = id;
    }
  }
  return ret;
}


// TODO: ideally we should look at the dag and use the segmenter. Here we take advantage of the linear structure of HostIrContainer::topLevelExprs to greedily merge the adjacent compatible stream for-loop bodies
void StreamParallelType::runPass(Fusion* fusion) {
  // check that there are no stream axes in the inputs
  NVF_CHECK(std::all_of(fusion->inputs().begin(), fusion->inputs().end(), [](Val* input) {
    auto input_tv = dynamic_cast<TensorView*>(input);
    return input_tv == nullptr || getStreamAxis(input_tv->getLoopDomain()) == nullptr;
  }), "Expected no stream axis in the TensorView inputs.");

  FusionGuard fg(fusion); // set as current container to register the newly created for-loops
  hir::HostIrContainer* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");
  // needed ?
  IdModel id_model(fusion);
  id_model.buildAlmostExactGraph();

  std::vector<Expr*> new_top_level_exprs;
  // Step 1. Find the segments of expressions that can be merged into a single stream for-loop
  // At the end of this step, new_top_level_exprs contains a list of expressions including newly created for-loops that will represent the stream parallelization, and the relevant expressions grouped inside the for-loops bodies.
  for (auto expr : hic->topLevelExprs()) {
    // we only support exprs having 0 or 1 output for now
    if (expr->outputs().size() == 0) {
      // If the expr has no output, we try to merge it with the previous for loop
      Expr* previous_expr = new_top_level_exprs.back();
      if (previous_expr->isA<ForLoop>()) {
        previous_expr->as<ForLoop>()->body().push_back(expr);
      } else {
        new_top_level_exprs.push_back(expr);
      }
    }
    NVF_CHECK(expr->outputs().size() == 1, "Each expr should have at most one output.");
    TensorView* output = expr->output(0)->as<TensorView>();
    // retrieves the Loop IterDomain that is stream parallelized, if any
    IterDomain* stream_axis = getStreamAxis(output->getLoopDomain());
    if (stream_axis == nullptr) {
      // if the consumer is not stream parallelized, it means the expr need not be inside a stream for-loop
      new_top_level_exprs.push_back(expr);
      continue;
    }
    // find the corresponding stream axis but in the Logical (and not Loop Domain)
    auto it_logical_stream_axis = std::find(output->getLogicalDomain().begin(), output->getLogicalDomain().end(), stream_axis);
    // for now we do not support split/merge stream axis
    NVF_ERROR(
      it_logical_stream_axis !=
          output->getLogicalDomain().end(),
      "Cannot stream parallelize on a split/merge axis ",
      stream_axis);
    // we don't support reducing or broadcasting a stream axis
    NVF_CHECK(stream_axis->getIterType() == IterType::Iteration, "Stream axis ", stream_axis, " should be an iteration axis.");
    // we don't support stream axis in the inputs nor the first expression
    NVF_CHECK(new_top_level_exprs.empty() == false, "Expected the first expr to not have a stream axis.");
    // We consider the previous expression to check whether the expr should create a new stream for-loop or be integrated into the previous one
    auto prev_expr = new_top_level_exprs.back();
    // check if the current expr can be merged with the previous stream for-loop
    if (auto prev_for_loop = dynamic_cast<ForLoop*>(prev_expr); prev_for_loop && id_model.idGraph(IdMappingMode::ALMOSTEXACT).disjointValSets().strictAreMapped(stream_axis, prev_for_loop->iterDomain())) {
      // merge with previous for-loop
      prev_for_loop->body().push_back(expr);
    } else {
      // create a new for-loop
      auto* j =
          IrBuilder::create<Val>(DataType::Index); // running index of the for-loop
      auto* start = hic->zeroVal();
      auto* stop = stream_axis->extent();
      auto* step = hic->oneVal();
      auto* for_loop = IrBuilder::create<ForLoop>(
          stream_axis,
          /*index=*/j,
          start,
          stop,
          step,
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

  // Step 2. Setup each for loop's body by Slicing the tensors.
  for (auto* top_level_expr : new_top_level_exprs) {
    // TODO: change in place? consr issue
    if (!top_level_expr->isA<ForLoop>()) {
      continue;
    }
    auto* for_loop = top_level_expr->as<ForLoop>();
    // this will contain the new body of the current for-loop
    std::vector<Expr*> new_loop_body;


    // std::vector<TensorView*> tvs;
    // for (auto expr: for_loop->body()) {
    //   for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    //     tvs.push_back(input);
    //   }
    //   for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
    //     tvs.push_back(output);
    //   }
    // }

    std::vector<Expr*> current_loop_body = for_loop->body().exprs();
    for (auto it_expr = current_loop_body.begin(); it_expr != current_loop_body.end(); ++it_expr) {
      Expr* expr = *it_expr;
      for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
        int64_t input_stream_id_logical_index = -1;
        for (auto id : input->getLoopDomain()) {
          if (id_model.idGraph(IdMappingMode::ALMOSTEXACT).disjointValSets().strictAreMapped(for_loop->iterDomain(), id)) {
            NVF_CHECK(input_stream_id_logical_index == -1, "Expected at most one axis mapping to the stream axis ", for_loop->iterDomain(), " in the tensor ", input, " loop's domain ", input->getLoopDomain());
            auto it_input_stream_id_logical = std::find(input->getLogicalDomain().begin(), input->getLogicalDomain().end(), id);
            NVF_CHECK(it_input_stream_id_logical != input->getLogicalDomain().end(), "Expected to find ", id, " in ", input, "'s logical domain ", input->getLogicalDomain());
            input_stream_id_logical_index = std::distance(input->getLogicalDomain().begin(), it_input_stream_id_logical);
          }
        }
        if (input_stream_id_logical_index == -1) {
          continue;
        }
        TensorView* input_j = select(input, input_stream_id_logical_index, for_loop->index());
        new_loop_body.push_back(input_j->definition());
        for (auto it_running_expr = current_loop_body.begin(); it_running_expr != current_loop_body.end(); ++it_running_expr) {
          Expr* running_expr = *it_running_expr;
          for (auto* running_input : ir_utils::filterByType<TensorView>(running_expr->inputs())) {
            if (running_input == input) {
              *it_running_expr = ir_utils::replaceValInExprInputs(running_expr, input, input_j);
            }
          }
        }
      }

      for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
        int64_t output_stream_id_logical_index = -1;
        for (auto id : output->getLoopDomain()) {
          if (id_model.idGraph(IdMappingMode::ALMOSTEXACT).disjointValSets().strictAreMapped(for_loop->iterDomain(), id)) {
            NVF_CHECK(output_stream_id_logical_index == -1, "Expected at most one axis mapping to the stream axis ", for_loop->iterDomain(), " in the tensor ", output, " loop's domain ", output->getLoopDomain());
            auto it_output_stream_id_logical = std::find(output->getLogicalDomain().begin(), output->getLogicalDomain().end(), id);
            NVF_CHECK(it_output_stream_id_logical != output->getLogicalDomain().end(), "Expected to find ", id, " in ", output, "'s logical domain ", output->getLogicalDomain());
            output_stream_id_logical_index = std::distance(output->getLogicalDomain().begin(), it_output_stream_id_logical);
          }
        }
        if (output_stream_id_logical_index == -1) {
          continue;
        }
        TensorView* output_j = select(output, output_stream_id_logical_index, for_loop->index());
        new_loop_body.push_back(output_j->definition());
        for (auto it_running_expr = current_loop_body.begin(); it_running_expr != current_loop_body.end(); ++it_running_expr) {
          Expr* running_expr = *it_running_expr;
          for (auto* running_output : ir_utils::filterByType<TensorView>(running_expr->outputs())) {
            if (running_output == output) {
              *it_running_expr = ir_utils::transferDefinitionToNewOutputs(running_expr, {output_j});
            }
          }
        }
      }
    }
    // reseting the for-loop body
    for_loop->body().clear();
    for (auto* expr : new_loop_body) {
      for_loop->body().push_back(expr);
    }
  }


  // Step 3. Finalize the for-loop bodies by adding the stream setup and synchronization
  for (auto* top_level_expr : new_top_level_exprs) {
    if (!top_level_expr->isA<ForLoop>()) {
      continue;
    }
    auto* for_loop = top_level_expr->as<ForLoop>();
    std::vector<Expr*> new_loop_body;

    // Get the current stream to later synchronize subsequent new streams
    auto* get_current_stream = IrBuilder::create<hir::GetCurrentStream>();
    hir::Stream* original_stream = get_current_stream->stream();
    new_loop_body.push_back(get_current_stream);

    // set the stream to the one corresponding to the current for-loop index
    auto* j = for_loop->index();
    auto* number_of_streams =
        IrBuilder::create<NamedScalar>("numberOfStreams", DataType::Int);
    auto* stream_index = mod(j, number_of_streams);
    auto* stream = IrBuilder::create<hir::Stream>(stream_index);
    auto* set_stream = IrBuilder::create<hir::SetCurrentStream>(stream);
    new_loop_body.push_back(set_stream);

    // sync the new stream with the original stream
    auto* initial_sync_stream =
        IrBuilder::create<hir::Synchronize>(original_stream);
    new_loop_body.push_back(initial_sync_stream);

    // add the actual exprs to the for-loop body
    for (auto* expr : for_loop->body().exprs()) {
      new_loop_body.push_back(expr);
    }

    // set back the original stream
    auto* set_back_original_stream =
        IrBuilder::create<hir::SetCurrentStream>(original_stream);
    new_loop_body.push_back(set_back_original_stream);
    // synchronize original stream with the for-loop's streams
    auto* sync_stream = IrBuilder::create<hir::Synchronize>(stream);
    new_loop_body.push_back(sync_stream);

    // reset the for-loop's body to the one we constructed.
    for_loop->body().clear();
    for (auto* expr : new_loop_body) {
      for_loop->body().push_back(expr);
    }
  }

  // reset hic topLevelExprs to new_top_level_exprs

}

} // namespace nvfuser::preseg_passes
