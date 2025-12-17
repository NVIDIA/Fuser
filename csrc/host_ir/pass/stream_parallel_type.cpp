// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "host_ir/pass/stream_parallel_type.h"

#include <list>

#include "host_ir/container.h"
#include "host_ir/lower.h"
#include "id_model/id_model.h"
#include "ir/all_nodes.h"
#include "ir/builder.h"
#include "ir/internal_base_nodes.h"
#include "ir/utils.h"
#include "kernel_ir.h"
#include "multidevice/cuda_p2p.h"
#include "multidevice/resharding.h"
#include "multidevice/utils.h"
#include "ops/all_ops.h"
#include "ops/utils.h"

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

// Determines if a stream-parallel for-loop can be merged with the previous one.
// If the stream axis of the expr doesn't map to the for loop, normally we would
// not merge it because we need to synchronize streams before continuing,
// but if the expr is resharding, we can merge it anyways because the resulting
// communication will have an implicit synchronization.
bool canMergeWithPreviousForLoop(
    const std::list<Expr*>& new_top_level_exprs,
    IterDomain* stream_axis,
    const IdModel& id_model,
    bool is_resharding) {
  return !new_top_level_exprs.empty() &&
      new_top_level_exprs.back()->isA<kir::ForLoop>() &&
      (is_resharding ||
       areIdsMapped(
           id_model,
           stream_axis,
           new_top_level_exprs.back()->as<kir::ForLoop>()->iterDomain()));
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
    out->setDeviceMesh(tensor->getDeviceMesh());
    auto result = IrBuilder::create<hir::HirAliasSelect>(
        tensor, out, stream_axis_index, index);

    cache_[key] = result;
    return {result, true};
  }

 private:
  Map cache_; // Storage for cached sliced tensors
};

// Step 1: Group expressions into stream-parallel regions
std::list<Expr*> groupStreamParallelRegions(
    const std::list<Expr*>& top_level_exprs,
    const IdModel& id_model) {
  std::list<Expr*> new_top_level_exprs;

  for (Expr* expr : top_level_exprs) {
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
    auto* output = expr->output(0)->as<TensorView>();
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
            new_top_level_exprs, stream_axis, id_model, isResharding(expr))) {
      // Merge with existing for-loop by adding the expression to its body
      new_top_level_exprs.back()->as<kir::ForLoop>()->body().push_back(expr);
    } else {
      // Create a new for-loop for stream parallelization
      auto* for_loop = IrBuilder::create<kir::ForLoop>(
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
std::list<Expr*> addTensorAllocations(
    std::list<Expr*> top_level_exprs,
    const IdModel& id_model) {
  std::list<Expr*> new_top_level_exprs;

  for (auto* expr : top_level_exprs) {
    if (expr->isA<kir::ForLoop>()) {
      // add allocations for tensors produced in the loop that have a stream
      // axes
      auto* for_loop = expr->as<kir::ForLoop>();
      for (auto* body_expr : for_loop->body().exprs()) {
        for (auto* output :
             ir_utils::filterByType<TensorView>(body_expr->outputs())) {
          if (findStreamAxisIndex(output, for_loop->iterDomain(), id_model) !=
              -1) {
            new_top_level_exprs.push_back(IrBuilder::create<kir::Allocate>(
                output, output->getMemoryType()));
          }
        }
      }
    }
    new_top_level_exprs.push_back(expr);
  }

  return new_top_level_exprs;
}

// Step 3: Process for-loop bodies by slicing tensors
std::list<Expr*> processForLoopBodies(
    std::list<Expr*> top_level_exprs,
    const IdModel& id_model,
    const CommunicatorBackend& communicator_backend) {
  TensorSlicingCache tensor_slicing_cache;

  for (auto* expr : top_level_exprs) {
    if (!expr->isA<kir::ForLoop>()) {
      continue;
    }

    auto* for_loop = expr->as<kir::ForLoop>();
    std::vector<Expr*> new_loop_body;
    std::vector<Expr*> new_loop_body_epilogue;

    // Lambda to process a tensor in a for-loop body
    auto processTensor = [&](Expr*& expr, TensorView* tensor, Val* index) {
      if (auto stream_idx =
              findStreamAxisIndex(tensor, for_loop->iterDomain(), id_model);
          stream_idx != -1) {
        auto [slicing, is_new] =
            tensor_slicing_cache.get(tensor, stream_idx, index);
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

    auto* my_device_id = IrBuilder::create<NamedScalar>("rank", DataType::Int);
    // We need to make indexing different for when the pipeline will result in
    // a p2p ring pipeline backed by cuda ipc, or will result in a collective
    // based pipeline. On the one hand, for the case of collective-based
    // pipeline, all ranks must index the tensors uniformly, because the
    // successive collective must be posted in a globally coherent order (this
    // can actually be relaxed by using different process groups, namely, one
    // process group per tile, using tags, but this unfortunately hurts
    // performance). On the other hand, the case with cuda ipc p2p needs a
    // ring pattern where each rank sends and receives to one and only one
    // peer, therefore, indexing must be offset by the rank. This is needed
    // for two reasons, 1) performance-wise, this is a more efficient way to
    // use the network than to have all ranks send or receive to/from one
    // device 2) our semantics of sharing the memory handles can only express
    // this type of scenario. P2p backend by ProcessGroup can relax condition
    // 2) because there is no explicit need to share the memhandle.
    auto tensor_index = communicator_backend == CommunicatorBackend::kCuda
        ? mod(add(my_device_id, for_loop->index()), for_loop->stop())
        : for_loop->index();
    auto recv_peer = communicator_backend == CommunicatorBackend::kCuda
        ? mod(add(for_loop->stop(), sub(my_device_id, for_loop->index())),
              for_loop->stop())
        : for_loop->index();

    for (auto* body_expr : for_loop->body().exprs()) {
      bool did_to_stream = false;
      bool stream_to_did = false;
      auto inputs = ir_utils::filterByType<TensorView>(body_expr->inputs());

      if (!inputs.empty()) {
        auto input = *inputs.begin();
        if (!input->getLogicalDomain().empty() &&
            input->getLogicalDomain()[0]->isDeviceDim()) {
          auto outputs =
              ir_utils::filterByType<TensorView>(body_expr->outputs());
          if (!outputs.empty() &&
              (*outputs.begin())->axis(0)->getParallelType() ==
                  ParallelType::Stream) {
            // First axis went from DID to Stream
            did_to_stream = true;
          }
        }
      }
      for (auto* output :
           ir_utils::filterByType<TensorView>(body_expr->outputs())) {
        auto stream_idx =
            findStreamAxisIndex(output, for_loop->iterDomain(), id_model);
        if (stream_idx != -1 &&
            output->getLogicalDomain()[stream_idx]->isDeviceDim()) {
          // Any axis went from Stream to DID
          stream_to_did = true;
          break;
        }
      }

      // Lower to MM + RS algorithm
      if (did_to_stream && stream_to_did) {
        NVF_ERROR(
            body_expr->isA<LoadStoreOp>() &&
                body_expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set,
            "expected a set operation but got ",
            body_expr);
        NVF_ERROR(
            body_expr->isA<LoadStoreOp>(),
            "expected a Tv operation but got ",
            body_expr);
        auto* set_op = body_expr->as<LoadStoreOp>();
        auto* input_tv = set_op->in()->as<TensorView>();
        auto* output_tv = set_op->out()->as<TensorView>();
        NVF_ERROR(
            input_tv->axis(0)->isDeviceDim(),
            "expected a sharded first axis on the input but got ",
            input_tv);
        NVF_ERROR(
            output_tv->axis(0)->getParallelType() == ParallelType::Stream,
            "expected a stream parallelized first axis on the output but got ",
            output_tv);
        NVF_ERROR(
            input_tv->axis(1)->getParallelType() == ParallelType::Stream,
            "expected a stream parallelized second axis on the input but got ",
            input_tv);
        NVF_ERROR(
            output_tv->axis(1)->isDeviceDim(),
            "expected a sharded second axis on the output but got ",
            output_tv);
        auto* is_sending_to_self =
            IrBuilder::create<kir::Predicate>(eq(tensor_index, my_device_id));
        auto if_sending_to_self =
            IrBuilder::create<kir::IfThenElse>(is_sending_to_self);
        auto [slicing_input, is_new] = tensor_slicing_cache.get(
            input_tv,
            /*dim*/
            findStreamAxisIndex(input_tv, for_loop->iterDomain(), id_model),
            /*index=*/tensor_index);
        auto [slicing_output, is_new_] =
            tensor_slicing_cache.get(output_tv, /*dim*/ 0, /*index=*/recv_peer);
        auto* local_copy = IrBuilder::create<LoadStoreOp>(
            LoadStoreOpType::Set, slicing_output->out(), slicing_input->out());
        if_sending_to_self->thenBody().push_back(local_copy);
        auto recv = IrBuilder::create<P2PCommunication>(
            P2PCommunicationType::RECV,
            slicing_output->out(),
            recv_peer,
            CommunicatorBackend::kNccl);
        auto send = IrBuilder::create<P2PCommunication>(
            P2PCommunicationType::SEND,
            slicing_input->out(),
            tensor_index,
            CommunicatorBackend::kNccl);
        auto start_coalescing = IrBuilder::create<hir::StartCoalescing>();
        auto end_coalescing = IrBuilder::create<hir::EndCoalescing>();
        auto wait = IrBuilder::create<hir::Wait>(end_coalescing);
        if_sending_to_self->elseBody().push_back(start_coalescing);
        if_sending_to_self->elseBody().push_back(recv);
        if_sending_to_self->elseBody().push_back(send);
        if_sending_to_self->elseBody().push_back(end_coalescing);
        if_sending_to_self->elseBody().push_back(wait);
        new_loop_body.push_back(slicing_input);
        new_loop_body.push_back(slicing_output);
        new_loop_body.push_back(if_sending_to_self);
      } else if (did_to_stream) {
        // Lower to AG+MM algorithm if did_to_stream=true && stream_to_did=false
        //
        // We have a special handling for when an axis pass from DIDx to Stream
        // parallel type in one expression. This case should be lowered to a P2P
        // Communication. Here, we lower the "Linear Allgather" case,
        // where tv0 [DIDx(i0), ...] and tv1=set(tv0) [Stream(i0), ...]. In this
        // case, the set should be lowered to something like
        //
        // FOR StreamIdx in range(i0):
        //   [...]
        //   SetCurrentStream to Stream ( StreamIdx % numberOfStreams )
        //   IF StreamIdx == rank: // This is the local copy
        //     Tv1[StreamIdx, ...].copy_(Tv0[0, ...]) // the index 0 because Tv0
        //     is sharded
        //   ELSE:
        //     Recv (buffer=Tv1[StreamIdx, ...], peer=StreamIdx)
        //     Send (buffer=Tv0[0, ...], peer=StreamIdx)
        //   [...]
        NVF_ERROR(
            body_expr->isA<LoadStoreOp>() &&
                body_expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set,
            "expected a set operation but got ",
            body_expr);
        NVF_ERROR(
            body_expr->isA<LoadStoreOp>(),
            "expected a Tv operation but got ",
            body_expr);
        auto* set_op = body_expr->as<LoadStoreOp>();
        auto* input_tv = set_op->in()->as<TensorView>();
        auto* output_tv = set_op->out()->as<TensorView>();
        NVF_ERROR(
            input_tv->axis(0)->isDeviceDim(),
            "expected a sharded first axis on the input but got ",
            input_tv);
        NVF_ERROR(
            output_tv->axis(0)->getParallelType() == ParallelType::Stream,
            "expected a stream parallelized first axis on the output but got ",
            output_tv);

        auto send_peer = (communicator_backend == CommunicatorBackend::kCuda)
            ? mod(add(for_loop->stop(), sub(my_device_id, for_loop->index())),
                  for_loop->stop())
            : for_loop->index();
        auto recv_peer = tensor_index;
        auto* is_sending_to_self =
            IrBuilder::create<kir::Predicate>(eq(send_peer, my_device_id));
        auto if_sending_to_self =
            IrBuilder::create<kir::IfThenElse>(is_sending_to_self);

        auto [slicing_input, is_new] = tensor_slicing_cache.get(
            input_tv,
            /*dim=*/0,
            /*index=*/FusionGuard::getCurFusion()->zeroVal());
        auto [slicing_output, is_new_] =
            tensor_slicing_cache.get(output_tv, /*dim=*/0, /*index=*/recv_peer);

        auto* local_copy = IrBuilder::create<LoadStoreOp>(
            LoadStoreOpType::Set, slicing_output->out(), slicing_input->out());

        if_sending_to_self->thenBody().push_back(slicing_input);
        if_sending_to_self->thenBody().push_back(local_copy);

        auto recv = IrBuilder::create<P2PCommunication>(
            P2PCommunicationType::RECV,
            slicing_output->out(),
            recv_peer,
            communicator_backend);
        auto send = IrBuilder::create<P2PCommunication>(
            P2PCommunicationType::SEND,
            input_tv,
            send_peer,
            communicator_backend);
        if (communicator_backend == CommunicatorBackend::kNccl) {
          // Using Start/EndCoalescing here is important to 1) avoid hangs
          // because of a wrong global order of send/recv and 2) enjoy full
          // bi-directional bandwith.
          auto start_coalescing = IrBuilder::create<hir::StartCoalescing>();
          auto end_coalescing = IrBuilder::create<hir::EndCoalescing>();
          auto wait = IrBuilder::create<hir::Wait>(end_coalescing);

          if_sending_to_self->elseBody().push_back(start_coalescing);
          if_sending_to_self->elseBody().push_back(recv);
          if_sending_to_self->elseBody().push_back(send);
          if_sending_to_self->elseBody().push_back(end_coalescing);
          if_sending_to_self->elseBody().push_back(wait);
        } else if (communicator_backend == CommunicatorBackend::kCuda) {
          auto share_mem_handles = IrBuilder::create<hir::ShareMemHandles>(
              std::vector<P2PCommunication*>({recv, send}));
          auto wait_send = IrBuilder::create<hir::Wait>(send);
          auto wait_recv = IrBuilder::create<hir::Wait>(recv);

          if_sending_to_self->elseBody().push_back(share_mem_handles);
          if (getP2pProtocol() == P2pProtocol::Put) {
            if_sending_to_self->elseBody().push_back(recv);
            if_sending_to_self->elseBody().push_back(send);
          } else if (getP2pProtocol() == P2pProtocol::Get) {
            if_sending_to_self->elseBody().push_back(send);
            if_sending_to_self->elseBody().push_back(recv);
          } else {
            NVF_ERROR("Invalid P2P protocol: ", getP2pProtocol());
          }
          if_sending_to_self->elseBody().push_back(wait_recv);
          // Defer the wait on send to the loop epilogue under the same
          // predicate
          auto* deferred_wait_if = IrBuilder::create<kir::IfThenElse>(
              if_sending_to_self->input(0)->as<kir::Predicate>());
          deferred_wait_if->elseBody().push_back(wait_send);
          new_loop_body_epilogue.push_back(deferred_wait_if);
        } else {
          NVF_THROW(
              "Unsupported communicator backend for lowering stream parallel "
              "type into p2p: ",
              communicator_backend);
        }

        new_loop_body.push_back(slicing_output);
        new_loop_body.push_back(if_sending_to_self);
      } else {
        // Process inputs and outputs normally
        for (auto* input :
             ir_utils::filterByType<TensorView>(body_expr->inputs())) {
          processTensor(body_expr, input, tensor_index);
        }
        for (auto* output :
             ir_utils::filterByType<TensorView>(body_expr->outputs())) {
          processTensor(body_expr, output, tensor_index);
        }
        new_loop_body.push_back(body_expr);
      }
    }

    for (auto* expr : new_loop_body_epilogue) {
      new_loop_body.push_back(expr);
    }

    for_loop->body().clear();
    for (auto* expr : new_loop_body) {
      for_loop->body().push_back(expr);
    }
  }

  return top_level_exprs;
}

// Step 4: Add stream management and synchronization
std::list<Expr*> addStreamManagement(std::list<Expr*> top_level_exprs) {
  std::list<Expr*> new_top_level_exprs;

  // Process each top-level expression
  for (auto* top_level_expr : top_level_exprs) {
    // Skip non-for-loop expressions
    if (!top_level_expr->isA<kir::ForLoop>()) {
      new_top_level_exprs.push_back(top_level_expr);
      continue;
    }

    auto* for_loop = top_level_expr->as<kir::ForLoop>();

    // Get the current stream before entering the loop
    auto* get_current_stream = IrBuilder::create<hir::GetCurrentStream>();
    hir::Stream* original_stream = get_current_stream->stream();
    new_top_level_exprs.push_back(get_current_stream);

    // Create a new for-loop for getting the current stream
    auto* for_loop_initial_sync = IrBuilder::create<kir::ForLoop>(
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
  // Set up the fusion environment and build the ID model
  FusionGuard fg(fusion);
  auto* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");

  IdModel id_model(fusion);
  id_model.buildBroadcastGraph();

  // Step 1: Group expressions into stream-parallel regions
  std::list<Expr*> top_level_exprs =
      groupStreamParallelRegions(hic->topLevelExprs(), id_model);

  // Step 2: Add allocations for tensors that need them
  top_level_exprs = addTensorAllocations(std::move(top_level_exprs), id_model);

  // Step 3: Process for-loop bodies by slicing tensors
  top_level_exprs = processForLoopBodies(
      std::move(top_level_exprs), id_model, params_.communicator_backend);

  // Step 4: Add stream management and synchronization
  top_level_exprs = addStreamManagement(std::move(top_level_exprs));

  // Update the container's top-level expressions
  hic->resetTopLevelExprs(top_level_exprs);
}

} // namespace nvfuser::hir_pass
