// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/lower.h>
#include <host_ir/pass/stream_parallel_type.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>
#include <ops/utils.h>

namespace nvfuser::hir_pass {

namespace {

// Finds the stream axis in a tensor's domain. There should be at most one
// stream axis.
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

// Validates that a stream axis is valid in a tensor
void validateStreamAxis(IterDomain* stream_axis, const TensorView* tv) {
  // Find the stream axis in the logical domain
  auto it_logical_stream_axis = std::find(
      tv->getLogicalDomain().begin(),
      tv->getLogicalDomain().end(),
      stream_axis);

  // Verify stream axis is not split/merged
  NVF_ERROR(
      it_logical_stream_axis != tv->getLogicalDomain().end(),
      "Cannot stream parallelize on a split/merge axis ",
      stream_axis);

  // Verify stream axis is an iteration or broadcast axis
  NVF_CHECK(
      stream_axis->getIterType() == IterType::Iteration ||
          stream_axis->getIterType() == IterType::Broadcast,
      "Stream axis ",
      stream_axis,
      " should be an iteration or broadcast axis.");
}

// Checks if two iteration domains are mapped in the ID model
bool areIdsMapped(const IdModel& id_model, IterDomain* id1, IterDomain* id2) {
  return id_model.idGraph(IdMappingMode::BROADCAST)
      .disjointValSets()
      .strictAreMapped(id1, id2);
}

// Determines if a stream-parallel for-loop can be merged with the previous one
bool canMergeWithPreviousForLoop(
    const std::vector<Expr*>& new_top_level_exprs,
    IterDomain* stream_axis,
    const IdModel& id_model) {
  return !new_top_level_exprs.empty() &&
      new_top_level_exprs.back()->isA<ForLoop>() &&
      areIdsMapped(
          id_model,
          stream_axis,
          new_top_level_exprs.back()->as<ForLoop>()->iterDomain());
}

// Finds where a stream axis appears in a tensor's logical domain
int64_t findStreamAxisIndex(
    const TensorView* tv,
    IterDomain* stream_axis,
    const IdModel& id_model) {
  int64_t stream_id_logical_index = -1;
  for (auto id : tv->getLoopDomain()) {
    if (areIdsMapped(id_model, stream_axis, id)) {
      // Verify only one stream axis exists
      NVF_CHECK(
          stream_id_logical_index == -1,
          "Expected at most one axis mapping to the stream axis ",
          stream_axis,
          " in the tensor ",
          tv,
          " loop's domain ",
          tv->getLoopDomain());

      // Find stream axis in logical domain
      auto it_stream_id_logical = std::find(
          tv->getLogicalDomain().begin(), tv->getLogicalDomain().end(), id);
      NVF_CHECK(
          it_stream_id_logical != tv->getLogicalDomain().end(),
          "Expected to find ",
          id,
          " in ",
          tv,
          "'s logical domain ",
          tv->getLogicalDomain());
      stream_id_logical_index =
          std::distance(tv->getLogicalDomain().begin(), it_stream_id_logical);
    }
  }
  return stream_id_logical_index;
}

// Cache for tensor slicing operations in stream parallelization.
// This cache stores previously created sliced versions of tensors to avoid
// redundant slicing operations. A sliced tensor is created by removing a
// specific axis (stream axis) from the tensor's domain and creating a new
// tensor that represents a slice of the original tensor at a given index.
// The cache key is a tuple of (original tensor, axis index to remove, slice
// index).
struct TensorSlicingCache {
  // Type aliases
  using Key = std::tuple<TensorView*, int64_t, Val*>;

  // Custom hash function for the tuple used as cache key
  struct Hash {
    size_t operator()(const Key& t) const {
      auto [tv, idx, val] = t;
      return std::hash<TensorView*>{}(tv) ^ std::hash<int64_t>{}(idx) ^
          std::hash<Val*>{}(val);
    }
  };

  // Map type for storing cached sliced tensors
  using Map = std::unordered_map<Key, hir::HirAliasSelect*, Hash>;

  // Get the expr producing the indexed version of a tensor. If the expr already
  // exists in the cache, returns the cached version. Otherwise, creates a new
  // expr, producing a tensor "selected" on its dimension `stream_axis_index` at
  // index `index`. Returns a pair of (expr, is_new) where is_new indicates
  // whether the expr was newly created.
  std::pair<hir::HirAliasSelect*, bool> get(
      TensorView* tensor,
      int64_t stream_axis_index,
      Val* index) {
    auto key = std::make_tuple(tensor, stream_axis_index, index);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return {it->second, false};
    }

    auto dom = tensor->getLogicalDomain();
    std::vector<IterDomain*> new_root;
    new_root.reserve(dom.size() - 1);

    for (auto i : arange((int64_t)dom.size())) {
      if (i != stream_axis_index) {
        new_root.emplace_back(dom[i]->cloneWithoutRFactor());
      }
    }

    auto td = IrBuilder::create<TensorDomain>(
        new_root, TensorDomain::getContiguityFilledWith(new_root, true));
    auto out = IrBuilder::create<TensorView>(td, *tensor->getDataType());
    auto result = IrBuilder::create<hir::HirAliasSelect>(
        tensor, out, stream_axis_index, index);

    cache_[key] = result;
    return {result, true};
  }

 private:
  Map cache_; // Storage for cached sliced tensors
};

// Step 1: Group expressions into stream-parallel regions
std::vector<Expr*> groupStreamParallelRegions(
    const std::vector<Expr*>& top_level_exprs,
    const IdModel& id_model) {
  std::vector<Expr*> new_top_level_exprs;

  for (auto* expr : top_level_exprs) {
    // Skip expressions with no outputs
    if (expr->outputs().size() == 0) {
      new_top_level_exprs.push_back(expr);
      continue;
    }

    // Each expression should have exactly one output
    NVF_CHECK(
        expr->outputs().size() == 1,
        "Each expr should have at most one output.");

    // Get the output tensor and check for stream parallelization
    TensorView* output = expr->output(0)->as<TensorView>();
    IterDomain* stream_axis = getStreamAxis(output->getLoopDomain());

    // If no stream axis found, keep the expression as is
    if (stream_axis == nullptr) {
      new_top_level_exprs.push_back(expr);
      continue;
    }

    // Verify that the expression can be handled as a standalone host operation
    NVF_ERROR(
        HostIrLower::isLowerableAsStandaloneHostOp(expr),
        "Stream parallel type not supported for expr ",
        expr);

    // Validate stream axis
    validateStreamAxis(stream_axis, output);

    // Check if we can merge this expression with the previous for-loop
    if (canMergeWithPreviousForLoop(
            new_top_level_exprs, stream_axis, id_model)) {
      // Merge with existing for-loop by adding the expression to its body
      new_top_level_exprs.back()->as<ForLoop>()->body().push_back(expr);
    } else {
      // Create a new for-loop for stream parallelization
      auto* for_loop = IrBuilder::create<ForLoop>(
          stream_axis,
          /*index=*/NamedScalar::getParallelIndex(ParallelType::Stream),
          /*start=*/FusionGuard::getCurFusion()->zeroVal(),
          /*stop=*/stream_axis->extent(),
          /*step=*/FusionGuard::getCurFusion()->oneVal(),
          /*vectorize=*/false,
          /*vectorize_shift=*/nullptr,
          /*unroll_required=*/false,
          CircularBufferLoopStage::NotApplicable,
          /*circular_buffer_loop_stage_depth=*/0);
      // Add the expression to the new for-loop's body
      for_loop->body().push_back(expr);
      new_top_level_exprs.push_back(for_loop);
    }
  }

  return new_top_level_exprs;
}

// Helper function to add allocations for tensors that need them
std::vector<Expr*> addTensorAllocations(
    std::vector<Expr*> top_level_exprs,
    const IdModel& id_model) {
  std::vector<Expr*> new_top_level_exprs;

  for (auto* expr : top_level_exprs) {
    if (expr->isA<ForLoop>()) {
      // add allocations for tensors produced in the loop that have a stream
      // axes
      auto* for_loop = expr->as<ForLoop>();
      for (auto* body_expr : for_loop->body().exprs()) {
        for (auto* output :
             ir_utils::filterByType<TensorView>(body_expr->outputs())) {
          if (findStreamAxisIndex(output, for_loop->iterDomain(), id_model) !=
              -1) {
            new_top_level_exprs.push_back(
                IrBuilder::create<kir::Allocate>(output, MemoryType::Global));
          }
        }
      }
    }
    new_top_level_exprs.push_back(expr);
  }

  return new_top_level_exprs;
}

// Step 3: Process for-loop bodies by slicing tensors
std::vector<Expr*> processForLoopBodies(
    std::vector<Expr*> top_level_exprs,
    const IdModel& id_model) {
  TensorSlicingCache tensor_slicing_cache;

  for (auto* expr : top_level_exprs) {
    if (!expr->isA<ForLoop>()) {
      continue;
    }

    auto* for_loop = expr->as<ForLoop>();
    std::vector<Expr*> new_loop_body;

    // Lambda to process a tensor in a for-loop body
    auto processTensor = [&](Expr*& expr, TensorView* tensor) {
      if (auto stream_idx =
              findStreamAxisIndex(tensor, for_loop->iterDomain(), id_model);
          stream_idx != -1) {
        auto [slicing, is_new] =
            tensor_slicing_cache.get(tensor, stream_idx, for_loop->index());
        if (is_new) {
          new_loop_body.push_back(slicing);
        }
        expr = ir_utils::replaceValInExprInputs(expr, tensor, slicing->out());
        if (expr->outputs().size() > 0 && expr->outputs()[0] == tensor) {
          expr =
              ir_utils::transferDefinitionToNewOutputs(expr, {slicing->out()});
        }
      }
    };

    for (auto* body_expr : for_loop->body().exprs()) {
      for (auto* input :
           ir_utils::filterByType<TensorView>(body_expr->inputs())) {
        processTensor(body_expr, input);
      }
      for (auto* output :
           ir_utils::filterByType<TensorView>(body_expr->outputs())) {
        processTensor(body_expr, output);
      }
      new_loop_body.push_back(body_expr);
    }

    for_loop->body().clear();
    for (auto* expr : new_loop_body) {
      for_loop->body().push_back(expr);
    }
  }

  return top_level_exprs;
}

// Step 4: Add stream management and synchronization
std::vector<Expr*> addStreamManagement(std::vector<Expr*> top_level_exprs) {
  std::vector<Expr*> new_top_level_exprs;

  // Process each top-level expression
  for (auto* top_level_expr : top_level_exprs) {
    // Skip non-for-loop expressions
    if (!top_level_expr->isA<ForLoop>()) {
      new_top_level_exprs.push_back(top_level_expr);
      continue;
    }

    auto* for_loop = top_level_expr->as<ForLoop>();

    // Get the current stream before entering the loop
    auto* get_current_stream = IrBuilder::create<hir::GetCurrentStream>();
    hir::Stream* original_stream = get_current_stream->stream();
    new_top_level_exprs.push_back(get_current_stream);

    // Create a new for-loop for getting the current stream
    auto* for_loop_initial_sync = IrBuilder::create<ForLoop>(
        for_loop->iterDomain(),
        for_loop->index(),
        for_loop->start(),
        for_loop->stop(),
        for_loop->step(),
        /*vectorize=*/false,
        /*vectorize_shift=*/nullptr,
        /*unroll_required=*/false,
        CircularBufferLoopStage::NotApplicable,
        /*circular_buffer_loop_stage_depth=*/0);
    new_top_level_exprs.push_back(for_loop_initial_sync);

    // Set up a new stream for this iteration based on the loop index
    auto* number_of_streams =
        IrBuilder::create<NamedScalar>("numberOfStreams", DataType::Int);
    auto* stream_index = mod(for_loop->index(), number_of_streams);
    auto* stream = IrBuilder::create<hir::Stream>(stream_index);
    auto* set_stream = IrBuilder::create<hir::SetCurrentStream>(stream);
    // Synchronize with the original stream before starting computation
    auto* initial_sync_stream =
        IrBuilder::create<hir::Synchronize>(original_stream);

    for_loop_initial_sync->body().push_back(set_stream);
    for_loop_initial_sync->body().push_back(initial_sync_stream);

    // create the new body of the current for-loop
    std::vector<Expr*> new_loop_body;
    // When entering the loop, set the stream
    new_loop_body.push_back(set_stream);

    // Add all the current for-loop body expressions to the new loop body
    for (auto* expr : for_loop->body().exprs()) {
      new_loop_body.push_back(expr);
    }

    // Restore the original stream and synchronize with the iteration's stream
    auto* set_back_original_stream =
        IrBuilder::create<hir::SetCurrentStream>(original_stream);
    new_loop_body.push_back(set_back_original_stream);
    auto* sync_stream = IrBuilder::create<hir::Synchronize>(stream);
    new_loop_body.push_back(sync_stream);

    // Update the for-loop body with the new expressions
    for_loop->body().clear();
    for (auto* expr : new_loop_body) {
      for_loop->body().push_back(expr);
    }
    new_top_level_exprs.push_back(for_loop);
  }

  return new_top_level_exprs;
}

} // anonymous namespace

// StreamParallelType pass implementation.
// This pass handles stream parallelization of operations in a fusion.
// It works by:
// 1. Identifying stream-parallelized axes in tensor operations
// 2. Grouping compatible operations into stream-parallel for-loops
// 3. Setting up proper stream synchronization and management
// 4. Adding allocations for tensors that need them
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
void StreamParallelType::passImplementation(Fusion* fusion) {
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
  id_model.buildBroadcastGraph();

  // Step 1: Group expressions into stream-parallel regions
  std::vector<Expr*> top_level_exprs =
      groupStreamParallelRegions(hic->topLevelExprs(), id_model);

  // Step 2: Add allocations for tensors that need them
  top_level_exprs = addTensorAllocations(std::move(top_level_exprs), id_model);

  // Step 3: Process for-loop bodies by slicing tensors
  top_level_exprs = processForLoopBodies(std::move(top_level_exprs), id_model);

  // Step 4: Add stream management and synchronization
  top_level_exprs = addStreamManagement(std::move(top_level_exprs));

  // Update the container's top-level expressions
  hic->resetTopLevelExprs(top_level_exprs);
}

} // namespace nvfuser::hir_pass
