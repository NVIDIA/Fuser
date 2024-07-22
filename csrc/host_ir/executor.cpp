// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <dynamic_transform.h>
#include <host_ir/executor.h>
#include <ir/utils.h>

namespace nvfuser {

namespace hir {

namespace {

at::Tensor getKnownTensorOrUndefined(
    Val* val,
    const ExpressionEvaluator& expr_evaluator) {
  return expr_evaluator.isKnown(val)
      ? expr_evaluator.evaluate(val).as<at::Tensor>()
      : at::Tensor();
}

std::vector<at::Tensor> getKnownTensorOrUndefined(
    const std::vector<Val*>& vals,
    const ExpressionEvaluator& expr_evaluator) {
  std::vector<at::Tensor> tensors(vals.size());
  std::transform(
      vals.begin(),
      vals.end(),
      tensors.begin(),
      [&expr_evaluator](Val* val) -> at::Tensor {
        return getKnownTensorOrUndefined(val, expr_evaluator);
      });
  return tensors;
}

} // namespace

HostIrExecutor::HostIrExecutor(
    std::unique_ptr<HostIrContainer> container,
    Communicator* communicator,
    HostIrExecutorParams params)
    : container_(std::move(container)),
      communicator_(communicator),
      params_(params) {
  const DeviceIdxType device_index =
      (communicator_ != nullptr && communicator_->is_available())
      ? communicator_->deviceId()
      : 0;
  if (isDebugDumpEnabled(DebugDumpOption::HostIr) && device_index == 0) {
    container_->print(debug());
  }
  streams_.insert(
      {container_->getDefaultStream(),
       c10::cuda::getDefaultCUDAStream(
           static_cast<c10::DeviceIndex>(device_index))});
}

std::vector<at::Tensor> HostIrExecutor::runWithInput(
    std::unordered_map<Val*, c10::IValue> val_to_IValue) {
  // process input values
  for (const auto& [val, ivalue] : val_to_IValue) {
    expr_evaluator_.bind(val, ivalue.toTensor());
  }

  // Interpret each instruction in an "eager" way by iterate over the Host Ir
  // Container's top level expression list
  for (auto expr : container_->topLevelExprs()) {
    dispatch(expr);
  }

  // Collect global outputs
  return getKnownTensorOrUndefined(container_->outputs(), expr_evaluator_);
}

void HostIrExecutor::handle(SetCurrentStream* set_current_stream) {
  Stream* stream = set_current_stream->stream();
  if (streams_.find(stream) == streams_.end()) {
    auto i = (communicator_ != nullptr && communicator_->is_available())
        ? communicator_->deviceId()
        : 0;
    streams_.insert(
        {stream,
         c10::cuda::getStreamFromPool(
             /*isHighPriority=*/false, static_cast<c10::DeviceIndex>(i))});
  }
  setCurrentCUDAStream(streams_.at(stream));
}

void HostIrExecutor::handle(PostOnStream* post_ir) {
  std::vector<c10::IValue> input_IValues;
  for (auto& input : post_ir->inputs()) {
    NVF_ERROR(
        expr_evaluator_.isKnown(input),
        "No buffer associated with Val ",
        input,
        " for handling ",
        post_ir->toString());
    PolymorphicValue input_evaluation = expr_evaluator_.evaluate(input);
    c10::IValue value;
    if (input_evaluation.is<at::Tensor>()) {
      value = input_evaluation.as<at::Tensor>();
    } else if (input_evaluation.is<int64_t>()) {
      value = at::Scalar(input_evaluation.as<int64_t>());
    } else {
      NVF_ERROR(
          "Wrong type ",
          input_evaluation.type().name(),
          " for the PolymorphicValue ",
          input_evaluation,
          ", must be at::Tensor or int64_t");
    }
    input_IValues.push_back(value);
  }

  // placeholder for storing the outputs
  std::vector<at::Tensor> outputs;

  NVF_ERROR(
      post_ir->hostOpToPost()->isA<HostUnit>(),
      "op must be a HostUnit: ",
      post_ir->hostOpToPost());
  auto hu = post_ir->hostOpToPost()->as<HostUnit>();
  // Compile the fusion and execute it with FusionExecutor(Cache)
  // Check if the executor has been cached. If not, create and cache it
  if (params_.use_fusion_executor_cache) {
    if (!fec_.count(hu)) {
      fec_.try_emplace(
          hu,
          std::make_unique<Fusion>(*hu->fusion_to_execute()),
          /*fusion_id=*/0,
          !params_.skip_auto_scheduling);
    }
    outputs = fec_.at(hu).runFusionWithInputs(input_IValues);
  } else {
    FusionExecutor& fe = fe_[hu];
    if (!fe.isCompiled()) {
      Fusion* fusion = hu->fusion_to_execute();
      DynamicTransform::concretizeFusion(fusion, input_IValues);
      fe.compileFusion(fusion, input_IValues);
    }
    outputs = fe.runFusion(input_IValues);
    if (!params_.cache_fusion_executor) {
      fe_.erase(hu);
    }
  }

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    expr_evaluator_.bind(
        post_ir->outputs().at(output_idx), outputs.at(output_idx));
  }
}

void HostIrExecutor::handle(Communication* communication) {
  NVF_ERROR(
      communicator_ != nullptr && communicator_->is_available(),
      "A valid communicator must be provided");

  at::Tensor input_tensor =
      getKnownTensorOrUndefined(communication->input(0), expr_evaluator_);
  at::Tensor output_tensor =
      getKnownTensorOrUndefined(communication->output(0), expr_evaluator_);

  c10d::Backend* backend =
      communicator_->getBackendForTeam(communication->team(), std::nullopt);
  works_[communication] = postSingleCommunication(
      communication,
      communicator_->deviceId(),
      backend,
      input_tensor,
      output_tensor);
}

void HostIrExecutor::handle(Wait* wait) {
  Communication* communication = wait->communication();
  NVF_ERROR(works_.find(communication) != works_.end(), "no wait req");
  auto& work = works_.at(communication);
  if (work != nullptr) {
    work->wait();
  }
  works_.erase(communication);
}

namespace {

void allConsumerValsOfHelper(
    Val* val,
    std::unordered_set<Val*>& visisted_vals) {
  if (visisted_vals.find(val) != visisted_vals.end()) {
    return;
  }
  for (Val* consumer : ir_utils::consumerValsOf(val)) {
    visisted_vals.insert(consumer);
    allConsumerValsOfHelper(consumer, visisted_vals);
  }
}

// Return all (not only direct) consumers of vals, this function can be used on
// any vals and will return consumers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::unordered_set<Val*> allConsumerValsOf(Val* val) {
  std::unordered_set<Val*> consumer_vals;
  allConsumerValsOfHelper(val, consumer_vals);
  return consumer_vals;
}

} // namespace

void HostIrExecutor::handle(ForLoop* for_loop) {
  NVF_ERROR(for_loop->start()->isConstInt());
  NVF_ERROR(for_loop->step()->isConstInt());
  NVF_ERROR(for_loop->stop()->isConstInt());
  auto start = for_loop->start()->value().as<int64_t>();
  auto step = for_loop->step()->value().as<int64_t>();
  auto stop = for_loop->stop()->value().as<int64_t>();

  for (auto i = start; i < stop; i += step) {
    // invalidate i and its consumers before binding
    expr_evaluator_.invalidate(for_loop->index());
    for (auto consumer : allConsumerValsOf(for_loop->index())) {
      expr_evaluator_.invalidate(consumer);
    }
    expr_evaluator_.bind(for_loop->index(), i);
    for (Expr* expr : for_loop->body().exprs()) {
      dispatch(expr);
    }
  }
}

namespace {

void handleWithExpressionEvaluator(
    Expr* expr,
    ExpressionEvaluator& expr_evaluator) {
  for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    NVF_ERROR(
        expr_evaluator.isKnown(input),
        "input ",
        input->toString(),
        " of the expression ",
        expr->toString(),
        "must be precomputed before being retrieved");
  }
  for (auto output : expr->outputs()) {
    expr_evaluator.bind(
        output, expr_evaluator.evaluate(output), /*evaluate_validate=*/true);
  }
}

} // namespace

void HostIrExecutor::handle(SliceOp* slice_op) {
  return handleWithExpressionEvaluator(slice_op, expr_evaluator_);
}

void HostIrExecutor::handle(MatmulOp* matmul_op) {
  return handleWithExpressionEvaluator(matmul_op, expr_evaluator_);
}

} // namespace hir

} // namespace nvfuser
