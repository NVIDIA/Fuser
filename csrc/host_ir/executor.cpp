// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <dynamic_transform.h>
#include <fusion_profiler.h>
#include <host_ir/executor.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <multidevice/communication.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <options.h>
#include <runtime/allocations.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

HostIrExecutor::HostIrExecutor(
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id)
    : ExecutorAbstract(fusion_id, concrete_id, runtime_id, group_id),
      communicator_(&Communicator::getInstance()) {}

bool HostIrExecutor::supported(Fusion* fusion) {
  FUSER_PERF_SCOPE("HostIrExecutor::supported");
  std::vector<Expr*> exprs = fusion->exprs();
  if (std::any_of(exprs.begin(), exprs.end(), [](Expr* e) {
        return isResharding(e) && isLowerableToCommunication(e);
      })) {
    NVF_ERROR(
        std::all_of(
            exprs.begin(),
            exprs.end(),
            [](Expr* e) {
              return isResharding(e) && isLowerableToCommunication(e);
            }),
        "Could not execute fusion as all expressions in a host IR container must be communication based at this point.");
    return true;
  }
  return false;
}

void HostIrExecutor::compile(Fusion* fusion) {
  FUSER_PERF_SCOPE("HostIrExecutor::compile");
  NVF_ERROR(
      supported(fusion),
      "HostIrExecutor does not support the Fusion provided.");
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).startCompile();
  }
  std::vector<Expr*> exprs = fusion->exprs();
  host_ir_container_ = std::make_unique<hir::HostIrContainer>();
  IrCloner cloner = Fusion::copy(fusion, host_ir_container_.get());
  for (Expr* e : exprs) {
    std::vector<Communication*> communications =
        lowerCommunication(cloner.clone(e));
    for (auto* communication : communications) {
      host_ir_container_->pushBackTopLevelExprs(communication);
    }
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopCompile();
  }
}

bool HostIrExecutor::isCompiled() const {
  return host_ir_container_ != nullptr;
}

namespace {
// Host IR specific function, returns the at:Tensor (ordered list) associated
// with the provdied Fusion output tv
at::Tensor findBufferForFusionOutput(
    const std::vector<at::Tensor>& out_tensors,
    const Val* fusion_out,
    const Fusion* fusion) {
  auto i =
      std::find(fusion->outputs().begin(), fusion->outputs().end(), fusion_out);
  NVF_ERROR(i != fusion->outputs().end());
  auto index = std::distance(fusion->outputs().begin(), i);
  return out_tensors[index];
}
} // namespace

std::vector<at::Tensor> HostIrExecutor::run(
    KernelArgumentHolder& args,
    std::vector<at::Tensor> outputs) {
  FUSER_PERF_SCOPE("HostIrExecutor::run");
  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id_ >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id_);
    SegmentProfiler& sprof = FusionProfiler::segment(group_id_);
    sprof.inputBytesAccessed(inputBytesProcessed(args));
    sprof.scheduler(toString(SchedulerType::ExprEval));
    sprof.startKernel();
  }
  NVF_ERROR(host_ir_container_, "Need to compile before you can run.");
  // Bind fusion inputs
  auto expr_eval = executor_utils::bindInputs(args, host_ir_container_.get());

  if (outputs.empty()) {
    std::vector<GlobalBufferInfo> output_info = getBufferInfos(
        expr_eval, PrimDataType::Int, host_ir_container_->outputs());
    outputs = allocateOutputs(
        host_ir_container_.get(),
        output_info,
        c10::Device(c10::DeviceType::CUDA, args.getDeviceIndex()),
        expr_eval);
  }

  // TODO: If outputs are provided validate they're the correct size
  for (Expr* e : host_ir_container_->topLevelExprs()) {
    NVF_ERROR(e->isA<Communication>());
    auto* communication = e->as<Communication>();
    c10d::Backend* backend =
        communicator_->getBackendForTeam(communication->team(), std::nullopt);
    auto in_tensor = expr_eval.evaluate(communication->in()).as<at::Tensor>();
    at::Tensor out_tensor = findBufferForFusionOutput(
        outputs, communication->out(), host_ir_container_.get());
    c10::intrusive_ptr<c10d::Work> work = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend,
        in_tensor,
        out_tensor);
    if (work != nullptr) {
      work->wait();
    }
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).setDevice(args.getDeviceIndex());
    FusionProfiler::segment(group_id_).stopKernel();
  }
  return outputs;
}

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

HostIrEvaluator::HostIrEvaluator(
    std::unique_ptr<HostIrContainer> container,
    Communicator* communicator,
    HostIrEvaluatorParams params)
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

std::vector<at::Tensor> HostIrEvaluator::runWithInput(
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

c10::cuda::CUDAStream HostIrEvaluator::getCUDAStream(Stream* stream) {
  StreamKey stream_key = stream;
  // if stream points to an index, it represents the dynamic value of that index
  if (Val* index = stream->index(); index != nullptr) {
    auto value = expr_evaluator_.evaluate(index);
    NVF_ERROR(value.hasValue() && value.is<int64_t>());
    stream_key = value.as<int64_t>();
  }
  if (streams_.find(stream_key) == streams_.end()) {
    auto i = (communicator_ != nullptr && communicator_->is_available())
        ? communicator_->deviceId()
        : 0;
    streams_.insert(
        {stream_key,
         c10::cuda::getStreamFromPool(
             /*isHighPriority=*/false, static_cast<c10::DeviceIndex>(i))});
  }
  return streams_.at(stream_key);
}

void HostIrEvaluator::handle(SetCurrentStream* set_current_stream) {
  setCurrentCUDAStream(getCUDAStream(set_current_stream->stream()));
}

void HostIrEvaluator::handle(Synchronize* synchronize) {
  getCUDAStream(synchronize->stream()).synchronize();
}

void HostIrEvaluator::handle(PostOnStream* post_ir) {
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
  // Compile the fusion and execute it with HostIrExecutor
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
    HostIrExecutor& hire = hire_[hu];
    if (!hire.isCompiled()) {
      Fusion* fusion = hu->fusion_to_execute();
      DynamicTransform::concretizeFusion(fusion, input_IValues);
      hire.compile(fusion);
    }
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(input_IValues);
    outputs = hire.run(args);
    if (!params_.cache_fusion_executor) {
      hire_.erase(hu);
    }
  }

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    expr_evaluator_.bind(
        post_ir->outputs().at(output_idx), outputs.at(output_idx));
  }
}

void HostIrEvaluator::handle(Communication* communication) {
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

void HostIrEvaluator::handle(P2PCommunication* communication) {
  NVF_ERROR(
      communicator_ != nullptr && communicator_->is_available(),
      "A valid communicator must be provided");

  at::Tensor buffer =
      getKnownTensorOrUndefined(communication->buffer(), expr_evaluator_);

  works_[communication] = postSingleCommunication(
      communication,
      communicator_->deviceId(),
      expr_evaluator_.evaluate(communication->peer()).as<int64_t>(),
      communicator_->getWorld(),
      buffer);
}

void HostIrEvaluator::handle(Wait* wait) {
  Expr* communication = wait->communication();
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
  visisted_vals.insert(val);
  for (Val* consumer : ir_utils::consumerValsOf(val)) {
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

void HostIrEvaluator::handle(ForLoop* for_loop) {
  auto start = expr_evaluator_.evaluate(for_loop->start()).as<int64_t>();
  auto step = expr_evaluator_.evaluate(for_loop->step()).as<int64_t>();
  auto stop = expr_evaluator_.evaluate(for_loop->stop()).as<int64_t>();

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

void HostIrEvaluator::handle(StartCoalescing* start_coalescing) {
  auto backend = communicator_->getWorld();
  NVF_ERROR(
      backend->getBackendName() == "nccl",
      "ProcessGroupUCC does not implement coalescence");
  backend->startCoalescing();
}

void HostIrEvaluator::handle(EndCoalescing* end_coalescing) {
  auto backend = communicator_->getWorld();
  NVF_ERROR(
      backend->getBackendName() == "nccl",
      "ProcessGroupUCC does not implement coalescence");
  works_[end_coalescing] = backend->endCoalescing();
}

void HostIrEvaluator::handle(MatmulOp* matmul) {
  TensorView* a = matmul->inA();
  TensorView* b = matmul->inB();
  TensorView* out = matmul->out();
  NVF_ERROR(
      expr_evaluator_.isKnown(a) && expr_evaluator_.isKnown(b),
      "Inputs of the matmul ",
      matmul->toString(),
      "must be precomputed before being retrieved");
  if (expr_evaluator_.isKnown(out)) {
    auto t_a = expr_evaluator_.evaluate(a).as<at::Tensor>();
    auto t_b = expr_evaluator_.evaluate(b).as<at::Tensor>();
    auto t_out = expr_evaluator_.evaluate(out).as<at::Tensor>();
    at::matmul_out(t_out, t_a, t_b);
  } else {
    unhandled(matmul);
  }
}

void HostIrEvaluator::unhandled(Statement* stmt) {
  NVF_ERROR(stmt->isA<Expr>(), stmt, " must be an Expr");
  auto* expr = stmt->as<Expr>();
  for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
    NVF_ERROR(
        expr_evaluator_.isKnown(input),
        "input ",
        input->toString(),
        " of the expression ",
        expr->toString(),
        "must be precomputed before being retrieved");
  }
  for (auto output : expr->outputs()) {
    expr_evaluator_.bind(
        output, expr_evaluator_.evaluate(output), /*evaluate_validate=*/true);
  }
}

} // namespace hir

} // namespace nvfuser
