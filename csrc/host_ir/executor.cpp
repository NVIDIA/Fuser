// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>

#include <dynamic_transform.h>
#include <fusion_profiler.h>
#include <host_ir/executor.h>
#include <host_ir/lower.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <multidevice/communication.h>
#include <multidevice/cuda_p2p.h>
#include <multidevice/utils.h>
#include <options.h>
#include <runtime/allocations.h>
#include <runtime/executor_dispatch.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_kernel_runtime.h>
#include <tensor_metadata.h>

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
        return isResharding(e) && HostIrLower::canLower(e);
      })) {
    NVF_ERROR(
        std::all_of(
            exprs.begin(),
            exprs.end(),
            [](Expr* e) {
              return isResharding(e) && HostIrLower::canLower(e);
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

  host_ir_container_ = std::make_unique<hir::HostIrContainer>();
  IrCloner cloner = Fusion::copy(fusion, host_ir_container_.get());
  if (fusion->isA<hir::HostIrContainer>()) {
    for (auto expr : fusion->as<hir::HostIrContainer>()->topLevelExprs()) {
      host_ir_container_->pushBackTopLevelExprs(cloner.clone(expr));
    }
  } else {
    std::vector<Expr*> exprs = fusion->exprs();
    DeviceIdxType my_device_idx = communicator_ ? communicator_->deviceId() : 0;
    for (Expr* e : exprs) {
      HostIrLower lower;
      std::vector<Expr*> communications =
          lower.lower(cloner.clone(e), my_device_idx);
      for (auto* communication : communications) {
        host_ir_container_->pushBackTopLevelExprs(communication);
      }
    }
  }

  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopCompile();
  }
}

bool HostIrExecutor::isCompiled() const {
  return (bool)host_ir_container_;
}

namespace {
// Validates the sizes and strides of the input and output tensors
// against the tensorviews
void validateTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<TensorView*>& tvs,
    const ExpressionEvaluator& expr_eval) {
  NVF_ERROR(tensors.size() == tvs.size());
  for (const auto& [tensor, tv] : zip(tensors, tvs)) {
    if (tensor.defined()) {
      inferAndValidateAllocationSizesAndStrides(tensor, tv, expr_eval);
    }
  }
}
} // namespace

KernelArgumentHolder HostIrExecutor::run(
    KernelArgumentHolder& args,
    KernelArgumentHolder output_args) {
  FUSER_PERF_SCOPE("HostIrExecutor::run");
  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id_ >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id_);
    SegmentProfiler& sprof = FusionProfiler::segment(group_id_);
    sprof.inputBytesAccessed(computeBytes(args));
    sprof.scheduler(toString(SchedulerType::ExprEval));
    sprof.startKernel();
  }
  NVF_ERROR(host_ir_container_, "Need to compile before you can run.");
  // Bind fusion inputs
  auto expr_eval = executor_utils::bindInputs(args, host_ir_container_.get());

  if (output_args.empty()) {
    std::vector<GlobalBufferInfo> output_infos = getBufferInfos(
        expr_eval, PrimDataType::Int, host_ir_container_->outputs());
    auto output_alias_to_input =
        executor_utils::getOutputAliasToInputMap(host_ir_container_.get());
    output_args = allocateOutputs(
        host_ir_container_.get(),
        output_infos,
        output_alias_to_input,
        c10::Device(c10::DeviceType::CUDA, args.getDeviceIndex()),
        args,
        true);
  }

  // TODO: If outputs are provided validate they're the correct size
  for (Expr* e : host_ir_container_->topLevelExprs()) {
    NVF_ERROR(e->isA<Communication>());
    auto* communication = e->as<Communication>();
    c10d::Backend* backend =
        communicator_->getBackendForTeam(communication->team(), std::nullopt);
    auto in_tensor = expr_eval.evaluate(communication->in()).as<at::Tensor>();
    auto out_idx = std::distance(
        host_ir_container_->outputs().begin(),
        std::find(
            host_ir_container_->outputs().begin(),
            host_ir_container_->outputs().end(),
            communication->out()));

    NVF_ERROR(
        out_idx < (int64_t)host_ir_container_->outputs().size(),
        "Output tensor not found in fusion outputs");
    auto out_tensor = output_args[out_idx].as<at::Tensor>();

    // Inputs are already validated in bindInputs.
    validateTensors({out_tensor}, {communication->out()}, expr_eval);
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

  // Evaluate outputs that are marked as Evaluate
  for (auto out_idx : arange(host_ir_container_->outputs().size())) {
    auto out = host_ir_container_->outputs()[out_idx];
    auto alias_info = host_ir_container_->getOutputAlias(out);
    if (alias_info.type == AllocationType::Evaluate) {
      NVF_ERROR(
          !output_args[out_idx].hasValue(),
          "Output tensor already has a value");
      output_args[out_idx] = expr_eval.evaluate(out);
    }
  }

  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).setDevice(args.getDeviceIndex());
    FusionProfiler::segment(group_id_).stopKernel();
  }
  return output_args;
}

namespace hir {

HostIrEvaluator::HostIrEvaluator(
    std::unique_ptr<HostIrContainer> container,
    Communicator* communicator,
    HostIrEvaluatorParams params)
    : container_(std::move(container)),
      communicator_(communicator),
      params_(params),
      expr_evaluator_(),
      my_local_device_index_(communicator_ ? communicator_->local_rank() : 0),
      ipc_handle_cache_(expr_evaluator_) {
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
  NVF_ERROR(
      std::all_of(
          container_->inputs().begin(),
          container_->inputs().end(),
          [this](Val* input) { return !container_->alias().count(input); }),
      "Inputs cannot be aliased");
}

KernelArgumentHolder HostIrEvaluator::runWithInput(
    const std::unordered_map<Val*, PolymorphicValue>& val_to_PValue) {
  expr_evaluator_ = ExpressionEvaluator();
  expr_evaluator_.bind("numberOfStreams", params_.number_of_streams);
  // process input values, converting IValue to PolymorphicValue
  for (const auto& [val, pvalue] : val_to_PValue) {
    bind(val, pvalue);
  }

  // Interpret each instruction in an "eager" way by iterate over the Host Ir
  // Container's top level expression list
  for (auto expr : container_->topLevelExprs()) {
    dispatch(expr);
  }

  // Collect global outputs
  std::vector<at::Tensor> outputs(container_->outputs().size());
  std::transform(
      container_->outputs().begin(),
      container_->outputs().end(),
      outputs.begin(),
      [this](Val* val) -> at::Tensor {
        return this->getKnownTensorOrUndefined(val);
      });
  return KernelArgumentHolder(outputs);
}

std::string HostIrEvaluator::canRun() const {
  const int64_t requested_n_gpus = requestedNumberOfDevices(container_.get());

  if (requested_n_gpus == 1) {
    return "";
  }

  if (communicator_ == nullptr) {
    return "A communicator must be provided";
  }

  if (!communicator_->is_available()) {
    return "distributed configuration required";
  }

  if (requested_n_gpus > communicator_->size()) {
    return "the fusion requests " + std::to_string(requested_n_gpus) +
        " GPUs to run, but there are only " +
        std::to_string(communicator_->size()) + " ranks in the communicator";
  }

  if (communicator_->local_size() > at::cuda::getNumGPUs()) {
    return std::to_string(communicator_->local_size()) +
        " processes are spawn on the node but only " +
        std::to_string(at::cuda::getNumGPUs()) + " GPUs are available";
  }

  return "";
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

void HostIrEvaluator::handle(GetCurrentStream* get_current_stream) {
  streams_.insert(
      {get_current_stream->stream(),
       c10::cuda::getCurrentCUDAStream(
           static_cast<c10::DeviceIndex>(my_local_device_index_))});
}

void HostIrEvaluator::handle(Synchronize* synchronize) {
  cudaStream_t current_stream =
      c10::cuda::getCurrentCUDAStream(
          static_cast<c10::DeviceIndex>(my_local_device_index_))
          .stream();
  cudaStream_t stream_to_sync = getCUDAStream(synchronize->stream()).stream();

  cudaEvent_t event = {};
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(event, stream_to_sync));
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaStreamWaitEvent(current_stream, event, cudaEventWaitDefault));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(event));
}

void HostIrEvaluator::handle(LaunchKernel* launch_kernel) {
  KernelArgumentHolder args;
  for (auto& input : launch_kernel->inputs()) {
    args.push(getKnownConcreteValue(input));
  }

  // If all output buffers are known already, pass them to the executor
  KernelArgumentHolder outputs;
  bool preallocated_outputs = false;
  for (Val* output : launch_kernel->outputs()) {
    if (isKnown(output)) {
      preallocated_outputs = true;
      outputs.push(getKnownConcreteValue(output));
    }
  }

  NVF_ERROR(
      outputs.empty() || outputs.size() == launch_kernel->outputs().size());

  args.setDeviceIndex();

  // run the compiled kernel
  outputs = container_->getKernelExecutor(launch_kernel->getIndex())
                ->run(
                    args,
                    outputs,
                    launch_kernel->launch_params(),
                    launch_kernel->compile_params());

  if (!preallocated_outputs) {
    // Store the outputs in the context
    for (auto output_idx : arange(outputs.size())) {
      bind(launch_kernel->outputs().at(output_idx), outputs[output_idx]);
    }
  }
}

void HostIrEvaluator::handle(PostOnStream* post_ir) {
  KernelArgumentHolder input_args;
  for (auto& input : post_ir->inputs()) {
    input_args.push(getKnownConcreteValue(input));
  }
  input_args.setDeviceIndex();
  // placeholder for storing the outputs
  KernelArgumentHolder outputs;
  bool use_preallocated_outputs = std::all_of(
      post_ir->outputs().begin(),
      post_ir->outputs().end(),
      [this](Val* output) { return this->isKnown(output); });
  NVF_ERROR(
      use_preallocated_outputs ||
          std::all_of(
              post_ir->outputs().begin(),
              post_ir->outputs().end(),
              [this](Val* output) { return !this->isKnown(output); }),
      "outputs must be all or none preallocated in expr ",
      post_ir);
  if (use_preallocated_outputs) {
    for (auto output : post_ir->outputs()) {
      outputs.push(getKnownConcreteValue(output));
    }
  }

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
    if (use_preallocated_outputs) {
      TORCH_WARN(
          "FusionExecutorCache does not support with preallocated outputs, so we are copying the outputs in expr ",
          post_ir);
      auto tmp_outputs = fec_.at(hu).runFusionWithInputs(input_args);
      for (auto output_idx : c10::irange(tmp_outputs.size())) {
        outputs[output_idx].as<at::Tensor>().copy_(
            tmp_outputs[output_idx].as<at::Tensor>());
      }
    } else {
      outputs = fec_.at(hu).runFusionWithInputs(input_args);
    }
  } else {
    // This path should generally be avoided as it will likely send the fusion
    // held in HostUnit directly to KernelExecutor which means it will try to
    // compile and run a device kernel with a single thread.
    if (auto it = executors_.find(hu); it == executors_.end()) {
      DynamicTransform::concretizeFusion(hu->fusion_to_execute(), input_args);
      auto it2 = executors_.insert(
          {hu,
           ExecutorDispatch::makeExecutor(
               hu->fusion_to_execute(), 1, 1, 1, 1)});
      ExecutorAbstract* ea = it2.first->second.get();
      if (ea->isA<KernelExecutor>()) {
        ExecutorDispatch::compile(
            ea,
            hu->fusion_to_execute(),
            input_args,
            LaunchParams(),
            CompileParams());
      } else {
        ExecutorDispatch::compile(ea, hu->fusion_to_execute());
      }
    }
    ExecutorAbstract* ea = executors_[hu].get();
    if (use_preallocated_outputs) {
      ExecutorDispatch::run(ea, input_args, outputs);
    } else {
      outputs = ExecutorDispatch::run(ea, input_args);
    }
  }

  if (!use_preallocated_outputs) {
    // Store the outputs in the context
    for (auto output_idx : arange(outputs.size())) {
      bind(post_ir->outputs().at(output_idx), outputs[output_idx]);
    }
  }
}

void HostIrEvaluator::handle(ShareMemHandles* share_mem_handles) {
  ipc_handle_cache_.exchangeHandles(share_mem_handles->communications());
}

void HostIrEvaluator::handle(Communication* communication) {
  NVF_ERROR(
      communicator_ != nullptr && communicator_->is_available(),
      "A valid communicator must be provided");

  at::Tensor input_tensor = getKnownTensorOrUndefined(communication->input(0));
  at::Tensor output_tensor =
      getKnownTensorOrUndefined(communication->output(0));

  CommunicatorBackend backend_type = communication->backend();
  c10d::Backend* backend =
      communicator_->getBackendForTeam(communication->team(), backend_type);

  validateTensors(
      {input_tensor, output_tensor},
      {communication->in(), communication->out()},
      expr_evaluator_);

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

  at::Tensor buffer = getKnownTensorOrUndefined(communication->buffer());

  CommunicatorBackend backend_type = communication->backend();
  if (backend_type == CommunicatorBackend::kCuda) {
    const P2pIpcHandle& p2p_ipc_handle = ipc_handle_cache_.get(communication);
    const auto current_stream = static_cast<CUstream>(
        c10::cuda::getCurrentCUDAStream(my_local_device_index_).stream());
    if (communication->type() == P2PCommunicationType::RECV) {
      get_zcopy::recvPost(
          p2p_ipc_handle,
          buffer.numel() * buffer.element_size(),
          current_stream);
    } else {
      get_zcopy::sendPost(p2p_ipc_handle, current_stream);
    }
  } else {
    validateTensors({buffer}, {communication->buffer()}, expr_evaluator_);
    works_[communication] = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        expr_evaluator_.evaluate(communication->peer()).as<int64_t>(),
        communicator_->getWorld(communication->backend()),
        buffer);
  }
}

void HostIrEvaluator::handle(Wait* wait) {
  Expr* communication = wait->communication();
  auto* p2p_comm = dynamic_cast<P2PCommunication*>(communication);
  if (p2p_comm && p2p_comm->backend() == CommunicatorBackend::kCuda) {
    if (p2p_comm->type() == P2PCommunicationType::SEND) {
      const auto current_stream = static_cast<CUstream>(
          c10::cuda::getCurrentCUDAStream(my_local_device_index_).stream());
      const P2pIpcHandle& ipc_handles = ipc_handle_cache_.get(p2p_comm);
      get_zcopy::sendWait(ipc_handles, current_stream);
    }
  } else {
    auto i = works_.find(communication);
    NVF_ERROR(i != works_.end(), "no wait req");

    auto work = i->second;
    if (work != nullptr) {
      work->wait();
    }

    works_.erase(communication);
  }
}

namespace {

void allConsumerValsOfHelper(Val* val, std::unordered_set<Val*>& visited_vals) {
  if (visited_vals.find(val) != visited_vals.end()) {
    return;
  }
  visited_vals.insert(val);
  for (Val* consumer : ir_utils::consumerValsOf(val)) {
    allConsumerValsOfHelper(consumer, visited_vals);
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
    invalidate(for_loop->index());
    for (auto consumer : allConsumerValsOf(for_loop->index())) {
      invalidate(consumer);
    }
    bind(for_loop->index(), i);
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

void HostIrEvaluator::handle(kir::IfThenElse* if_then_else) {
  auto predicate =
      expr_evaluator_.evaluate(if_then_else->predicate()->value()).as<bool>();
  const auto& scope =
      predicate ? if_then_else->thenBody() : if_then_else->elseBody();
  for (Expr* expr : scope.exprs()) {
    dispatch(expr);
  }
}

void HostIrEvaluator::handle(MatmulOp* matmul) {
  TensorView* a = matmul->inA();
  TensorView* b = matmul->inB();
  TensorView* out = matmul->out();

  if (isKnown(out)) {
    auto t_a = getKnownConcreteValue(a).as<at::Tensor>();
    auto t_b = getKnownConcreteValue(b).as<at::Tensor>();
    auto t_out = getKnownConcreteValue(out).as<at::Tensor>();
    at::matmul_out(t_out, t_a, t_b);
  } else {
    unhandled(matmul);
  }
}

void HostIrEvaluator::handle(LinearOp* linear) {
  TensorView* in = linear->inA()->as<TensorView>();
  TensorView* weight = linear->inB()->as<TensorView>();
  TensorView* bias = linear->bias()->as<TensorView>();
  TensorView* out = linear->out()->as<TensorView>();

  if (!isKnown(out)) {
    unhandled(linear);
    return;
  }

  auto in_at = getKnownConcreteValue(in).as<at::Tensor>();
  auto weight_at = getKnownConcreteValue(weight).as<at::Tensor>();
  auto out_at = getKnownConcreteValue(out).as<at::Tensor>();

  if (linear->has_bias()) {
    auto bias_at = getKnownConcreteValue(bias).as<at::Tensor>();
    at::linear_out(out_at, in_at, weight_at.squeeze(), bias_at.squeeze());
  } else {
    at::linear_out(out_at, in_at, weight_at.squeeze());
  }
}

void HostIrEvaluator::handle(kir::Allocate* allocate) {
  NVF_ERROR(
      allocate->buffer()->isA<TensorView>(),
      "Allocation must be on a TensorView but got ",
      allocate->buffer());
  TensorView* tv = allocate->buffer()->as<TensorView>();
  if (isKnown(tv)) {
    return;
  }
  GlobalBufferInfo info =
      getBufferInfos(expr_evaluator_, PrimDataType::Int, {tv}).at(0);
  c10::Device device =
      communicator_ ? communicator_->device() : at::Device("cuda:0");
  auto tensor = at::native::empty_strided_cuda(
      info.shape_info.logical_sizes,
      info.shape_info.logical_strides,
      info.type,
      c10::nullopt,
      device,
      c10::nullopt);
  bind(tv, tensor);
}

void HostIrEvaluator::handle(Deallocate* deallocate) {
  auto* tv = deallocate->allocation()->buffer()->as<TensorView>();
  NVF_ERROR(
      isKnown(tv),
      "Tried to free buffer associated with unknown TensorView",
      tv);
  invalidate(tv);
}

void HostIrEvaluator::unhandled(Statement* stmt) {
  NVF_ERROR(stmt->isA<Expr>(), stmt, " must be an Expr");
  auto* expr = stmt->as<Expr>();
  std::vector<PolymorphicValue> inputs;
  for (auto input : expr->inputs()) {
    if (input->isA<TensorView>()) {
      // Tensor inputs must be already computed at this point
      inputs.push_back(getKnownConcreteValue(input));
    } else {
      inputs.push_back(expr_evaluator_.evaluate(input));
    }
  }

  // Check that there is no pre-allocated output
  NVF_ERROR(
      std::all_of(
          expr->outputs().begin(),
          expr->outputs().end(),
          [this](Val* output) {
            return !this->expr_evaluator_.isKnown(output);
          }),
      "Do not support pre-allocated outputs for the op ",
      expr);
  // using ExpressionEvaluator::evaluate to evaluate the output is not valid
  // here if the output or one of its producer is an alias
  auto concrete_outputs = expr->evaluate(expr_evaluator_, inputs);
  for (int64_t i : c10::irange(expr->outputs().size())) {
    bind(expr->output(i), concrete_outputs.at(i));
  }
}

} // namespace hir

} // namespace nvfuser
