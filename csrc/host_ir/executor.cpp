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
#include <multidevice/utils.h>
#include <options.h>
#include <runtime/allocations.h>
#include <runtime/executor_dispatch.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_kernel_runtime.h>
#include <cuda.h>

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
    for (Expr* e : exprs) {
      HostIrLower lower;
      std::vector<Expr*> communications = lower.lower(cloner.clone(e));
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
    sprof.inputBytesAccessed(computeBytes(args));
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
      params_(params),
      my_local_device_index_(communicator_ ? communicator_->local_rank() : 0) {
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
  expr_evaluator_.bind("numberOfStreams", params_.number_of_streams);
}

std::vector<at::Tensor> HostIrEvaluator::runWithInput(
    std::unordered_map<Val*, c10::IValue> val_to_IValue) {
  // process input values
  for (const auto& [val, ivalue] : val_to_IValue) {
    expr_evaluator_.bind(val, IValueToPolymorphicValue(ivalue));
  }

  // Interpret each instruction in an "eager" way by iterate over the Host Ir
  // Container's top level expression list
  for (auto expr : container_->topLevelExprs()) {
    dispatch(expr);
  }

  // Collect global outputs
  return getKnownTensorOrUndefined(container_->outputs(), expr_evaluator_);
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
  std::vector<c10::IValue> input_IValues;
  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder(input_IValues);
  for (auto& input : launch_kernel->inputs()) {
    NVF_ERROR(
        expr_evaluator_.isKnown(input),
        "No buffer associated with Val ",
        input,
        " for handling ",
        launch_kernel->toString());
    PolymorphicValue input_evaluation = expr_evaluator_.evaluate(input);
    args.push(input_evaluation);
  }

  // run the compiled kernel
  std::vector<at::Tensor> outputs =
      container_->getKernelExecutor(launch_kernel->getIndex())->run(args);

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    expr_evaluator_.bind(
        launch_kernel->outputs().at(output_idx), outputs.at(output_idx));
  }
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
    // This path should generally be avoided as it will likely send the fusion
    // held in HostUnit directly to KernelExecutor which means it will try to
    // compile and run a device kernel with a single thread.
    if (auto it = executors_.find(hu); it != executors_.end()) {
      ExecutorAbstract* ea = it->second.get();
      KernelArgumentHolder args =
          KernelArgumentHolder::createKernelArgumentHolder(input_IValues);
      outputs = ExecutorDispatch::run(ea, args, std::vector<at::Tensor>{});

    } else {
      DynamicTransform::concretizeFusion(
          hu->fusion_to_execute(), input_IValues);
      auto it2 = executors_.insert(
          {hu,
           ExecutorDispatch::makeExecutor(
               hu->fusion_to_execute(), 1, 1, 1, 1)});
      ExecutorAbstract* ea = it2.first->second.get();
      if (ea->isA<KernelExecutor>()) {
        KernelArgumentHolder args =
            KernelArgumentHolder::createKernelArgumentHolder(input_IValues);
        ExecutorDispatch::compile(
            ea, hu->fusion_to_execute(), args, LaunchParams(), CompileParams());
      } else {
        ExecutorDispatch::compile(ea, hu->fusion_to_execute());
      }
      KernelArgumentHolder args =
          KernelArgumentHolder::createKernelArgumentHolder(input_IValues);
      outputs = ExecutorDispatch::run(ea, args, std::vector<at::Tensor>{});
    }
  }

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    expr_evaluator_.bind(
        post_ir->outputs().at(output_idx), outputs.at(output_idx));
  }
}

RemoteBufferInfo::RemoteBufferInfo(at::Tensor tensor) : ptr_(tensor.data_ptr()), storage_offset_(tensor.storage_offset()), element_size_(tensor.element_size()), is_imported_(false) {

  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&ipc_handle_, tensor.data_ptr()));
  const auto number_of_semaphores = Communicator::getInstance().size();
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc((void**)&semaphores_, number_of_semaphores * sizeof(IpcSemaphore)));
  static_assert(sizeof(IpcSemaphore) == sizeof(int), "IpcSemaphore must be same size as int");
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemset((void*) semaphores_, (int)IpcSemaphore::kReady, number_of_semaphores * sizeof(IpcSemaphore)));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&semaphores_ipc_handle_, semaphores_));
}

RemoteBufferInfo::RemoteBufferInfo(std::vector<uint8_t> data) : is_imported_(true) {
  const RemoteBufferInfo& imported_buffer = fromBytes<RemoteBufferInfo>(data);

  storage_offset_ = imported_buffer.storage_offset_;
  element_size_ = imported_buffer.element_size_;
  ipc_handle_ = imported_buffer.ipc_handle_;
  semaphores_ipc_handle_ = imported_buffer.semaphores_ipc_handle_;

  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(&ptr_, ipc_handle_, cudaIpcMemLazyEnablePeerAccess));
  ptr_ = (void*)((uint8_t*)ptr_ + storage_offset_ * element_size_);

  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle((void**)&semaphores_, semaphores_ipc_handle_, cudaIpcMemLazyEnablePeerAccess));
}

RemoteBufferInfo::~RemoteBufferInfo() {
  if (is_imported_) {
    std::cout << "RANK " << Communicator::getInstance().deviceId() << " closes ipc handle" << std::endl;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(ptr_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle((void*)semaphores_));
  } else {
    std::cout << "RANK " << Communicator::getInstance().deviceId() << " frees semaphores" << std::endl;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaFree((void*)semaphores_));
  }
}

std::ostream& operator<<(std::ostream& os, const cudaIpcMemHandle_t& info) {
  uint8_t* ptr = (uint8_t*)&info;
  for (int i = 0; i < (int)sizeof(cudaIpcMemHandle_t); i++) {
    os << ptr[i];
  }
  return os;
}



std::ostream& operator<<(std::ostream& os, const RemoteBufferInfo& info) {
  os << "RemoteBufferInfo(ptr=" << info.ptr_
     << ", storage_offset=" << info.storage_offset_
     << ", element_size=" << info.element_size_
     << ", is_imported=" << info.is_imported_
     << ", semaphores_=" << info.semaphores_
     << ", ipc_handle_=" << info.ipc_handle_
     << ", semaphores_ipc_handle_=" << info.semaphores_ipc_handle_
     << ")";
  return os;
}

void HostIrEvaluator::handle(ShareMemHandles* share_mem_handles) {
  const int64_t my_rank = communicator_->deviceId();
  auto get_tensor = [this](P2PCommunication* communication) -> at::Tensor { 
    return this->expr_evaluator_.evaluate(communication->buffer()).as<at::Tensor>();
  };

  std::vector<P2PCommunication*> communications;
  for (auto expr: share_mem_handles->communications()) {
    auto communication = expr->as<P2PCommunication>();
    const auto dst = expr_evaluator_.evaluate(communication->dst()).as<int64_t>();
    const auto src = expr_evaluator_.evaluate(communication->src()).as<int64_t>();
    const bool is_sender = my_rank == src;
    const bool is_receiver = my_rank == dst;
    if (!(is_sender ^ is_receiver)) { // REMOVE or adapt exporting/opening the handle
      return;
    }
    if (remote_buffers_.find(get_tensor(communication)) != remote_buffers_.end()) {
      continue;
    }
    communications.push_back(communication);
  }

  // put memhandles to TCP store
  auto get_key = [this] (P2PCommunication* communication, int64_t rank) -> std::string { 
    return "nvfuser_remote_buffer_info_P2PComm_dst=" + std::to_string(this->expr_evaluator_.evaluate(communication->dst()).as<int64_t>()) + "_src=" + std::to_string(this->expr_evaluator_.evaluate(communication->src()).as<int64_t>()) + "_rank=" + std::to_string(rank);
  };
  std::unordered_map<P2PCommunication*, std::unique_ptr<RemoteBufferInfo>> buffer_handles;
  auto store = communicator_->getTcpStore();
  for (P2PCommunication* communication: communications) {
    auto buffer_handle = std::make_unique<RemoteBufferInfo>(get_tensor(communication));
    std::cout << "RANK " << my_rank << " registers at key " << get_key(communication, my_rank) << std::endl;
    store->set(get_key(communication, my_rank), toBytes(*buffer_handle));
    std::cout << "RANK "  << my_rank << " creates buffer_handle " << *buffer_handle << std::endl;
    buffer_handles.emplace(communication, std::move(buffer_handle));
  }

  // barrier to ensure all ranks have pushed their memhandles to the store
  // TODO: precisely select what ranks need to wait on that barrier.
  communicator_->barrier();

    // get memhandles to TCP store
  for (P2PCommunication* communication: communications) {
    std::vector<std::unique_ptr<RemoteBufferInfo>> remote_buffers;
    remote_buffers.reserve(communicator_->size());
    const auto dst = expr_evaluator_.evaluate(communication->dst()).as<int64_t>();
    const auto src = expr_evaluator_.evaluate(communication->src()).as<int64_t>();
    for (int64_t rank : c10::irange(communicator_->size())) {
      if (rank != src && rank != dst) {
        remote_buffers.push_back(nullptr);
        continue;
      }
      std::cout << "RANK " << my_rank << " after barrier for key " << get_key(communication, rank) << std::endl;
      if (rank == my_rank) {
        // opening an ipc handle on the exporter's device is not supported
        remote_buffers.push_back(std::move(buffer_handles.at(communication)));
      } else {
        std::string key = get_key(communication, rank);
        NVF_ERROR(store->check({key}), "key ", key, " not found in store at rank ", my_rank);
        auto imported_remote_buffer_info = std::make_unique<RemoteBufferInfo>(store->get(key));
        remote_buffers.push_back(std::move(imported_remote_buffer_info));
      }
      std::cout << "RANK "  << my_rank << " emplaces at rank " << rank << " remote buffer " << *remote_buffers.back() << std::endl;
    }
    remote_buffers_.emplace(get_tensor(communication), std::move(remote_buffers));
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

  CommunicatorBackend backend_type = communication->backend();

  if (backend_type != CommunicatorBackend::kCuda) {
    c10d::Backend* backend =
        communicator_->getBackendForTeam(communication->team(), backend_type);
    works_[communication] = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        backend,
        input_tensor,
        output_tensor);
    return;
  }

  NVF_ERROR(communication->type() == CommunicationType::Allgather);

  // std::vector<at::Tensor> output_tensors =
  //     at::tensor_split(output_tensor.squeeze(), communication->team_size(), 0);
  // const std::vector<void*>& input_ptrs = communicator_->getRemotePtrs(input_tensor);
  // cudaStream_t current_stream =
  //     c10::cuda::getCurrentCUDAStream(my_local_device_index_).stream();
  // // TODO: use multicast
  // for (auto i = 0; i < communicator_->size(); i++) {
  //   cudaStream_t stream = c10::cuda::getStreamFromPool(
  //                             /*isHighPriority=*/false, my_local_device_index_)
  //                             .stream();
  //   cudaEvent_t event = {};
  //   NVFUSER_CUDA_RT_SAFE_CALL(
  //       cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  //   NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(event, current_stream));
  //   NVFUSER_CUDA_RT_SAFE_CALL(
  //       cudaStreamWaitEvent(stream, event, cudaEventWaitDefault));
  //   NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(event));

  //   auto output = output_tensors.at(i);
  //   NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
  //       output.data_ptr(),
  //       input_ptrs.at(i),
  //       output.numel() * output.element_size(),
  //       cudaMemcpyDeviceToDevice,
  //       stream));

  //   // sync
  //   NVFUSER_CUDA_RT_SAFE_CALL(
  //       cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  //   NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(event, stream));
  //   NVFUSER_CUDA_RT_SAFE_CALL(
  //       cudaStreamWaitEvent(current_stream, event, cudaEventWaitDefault));
  //   NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(event));
  // }
}

void HostIrEvaluator::handle(P2PCommunication* communication) {
  NVF_ERROR(
      communicator_ != nullptr && communicator_->is_available(),
      "A valid communicator must be provided");

  const int64_t my_rank = communicator_->deviceId();
  const auto dst = expr_evaluator_.evaluate(communication->dst()).as<int64_t>();
  const auto src = expr_evaluator_.evaluate(communication->src()).as<int64_t>();
  const bool is_sender = my_rank == src;
  const bool is_receiver = my_rank == dst;
  if (!(is_sender ^ is_receiver)) {
    return;
  }

  CommunicatorBackend backend_type = communication->backend();
  at::Tensor buffer =
      getKnownTensorOrUndefined(communication->buffer(), expr_evaluator_);


  if (backend_type != CommunicatorBackend::kCuda) {
    works_[communication] = postSingleCommunication(
        communication,
        communicator_->deviceId(),
        expr_evaluator_.evaluate(communication->dst()).as<int64_t>(),
        expr_evaluator_.evaluate(communication->src()).as<int64_t>(),
        communicator_->getWorld(),
        buffer);
    return;
  }

  const auto it = remote_buffers_.find(buffer);
  NVF_ERROR(it != remote_buffers_.end(), "No remote buffer found for ", communication->toString(), " at rank ", my_rank);
  const std::vector<std::unique_ptr<RemoteBufferInfo>>& remote_buffers = it->second;
  const int64_t peer = is_sender ? dst : src;
  const RemoteBufferInfo& my_buffer = *remote_buffers.at(my_rank);
  const RemoteBufferInfo& peer_buffer = *remote_buffers.at(peer);
  const auto local_semaphore = reinterpret_cast<CUdeviceptr>(&my_buffer.semaphores()[peer]);
  const auto remote_semaphore = reinterpret_cast<CUdeviceptr>(&peer_buffer.semaphores()[my_rank]);
  static_assert(sizeof(IpcSemaphore) == sizeof(uint32_t), "IpcSemaphore must be 32 bits");

  const auto current_stream = reinterpret_cast<CUstream>(c10::cuda::getCurrentCUDAStream(my_local_device_index_).stream());

  if (is_receiver) {
    // wait for sender to be ready
    std::cout << "RANK " << my_rank << " RECV, peer=" << peer << ", local semaphore=" << local_semaphore << ", remote semaphore=" << remote_semaphore << std::endl;
    NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(current_stream, local_semaphore, (cuuint32_t)(IpcSemaphore::kTransferInProgress), CU_STREAM_WAIT_VALUE_EQ));
    std::cout << "RANK " << my_rank << " RECV after 1st WAIT" << ", buffer.data_ptr()=" << buffer.data_ptr() << ", my_buffer.ptr()=" << my_buffer.ptr() << ", sent tensor=" << buffer << ", buffer.numel()=" << buffer.numel() << ", buffer.element_size()=" << buffer.element_size() << std::endl;
    // RDMA get the data from the sender
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
        buffer.data_ptr(),
        peer_buffer.ptr(),
        buffer.numel() * buffer.element_size(),
        cudaMemcpyDeviceToDevice,
        current_stream));
    // Signals completion to self
    NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(current_stream, local_semaphore, (cuuint32_t)(IpcSemaphore::kReady), CU_STREAM_WRITE_VALUE_DEFAULT));
    // Signals completion to receiver
    NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(current_stream, remote_semaphore, (cuuint32_t)(IpcSemaphore::kReady), CU_STREAM_WRITE_VALUE_DEFAULT));
  } else /*sender*/ {
    std::cout << "RANK " << my_rank << " SEND, peer=" << peer << ", local semaphore=" << local_semaphore << ", remote semaphore=" << remote_semaphore << ", my_buffer.ptr()=" << my_buffer.ptr() << ", buffer.data_ptr()=" << buffer.data_ptr() << "recv tensor=" << buffer << std::endl;
    std::cout << "RANK " << my_rank << " SEND BEFORE signaling, peer=" << peer << ", local semaphore=" << local_semaphore << ", remote semaphore=" << remote_semaphore << ", my_buffer.ptr()=" << my_buffer.ptr() << ", buffer.data_ptr()=" << buffer.data_ptr() << "recv tensor=" << buffer << std::endl;
    // signal to self that transfer is in progress
    NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(current_stream, local_semaphore, (cuuint32_t)(IpcSemaphore::kTransferInProgress), CU_STREAM_WRITE_VALUE_DEFAULT));
    // signal to receiver that the buffer is ready
    NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(current_stream, remote_semaphore, (cuuint32_t)(IpcSemaphore::kTransferInProgress), CU_STREAM_WRITE_VALUE_DEFAULT)); // passing CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER gives an error
    std::cout << "RANK " << my_rank << " SEND AFTER signaling, peer=" << peer << ", local semaphore=" << local_semaphore << ", remote semaphore=" << remote_semaphore << ", my_buffer.ptr()=" << my_buffer.ptr() << ", buffer.data_ptr()=" << buffer.data_ptr() << "recv tensor=" << buffer << std::endl;
    // NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
    // NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
    //     peer_buffer.ptr(),
    //     buffer.data_ptr(),
    //     // my_buffer.ptr(),
    //     buffer.numel() * buffer.element_size(),
    //     cudaMemcpyDeviceToDevice
    //     // current_stream));
    //     ));
  }
}

void HostIrEvaluator::handle(Wait* wait) {
  Expr* communication = wait->communication();
  auto* p2p_comm = dynamic_cast<P2PCommunication*>(communication);
  if (p2p_comm && p2p_comm->backend() != CommunicatorBackend::kCuda) {
    auto it = works_.find(communication);
    if (it == works_.end()) {
      return;
    }
    auto& work = it->second;
    if (work != nullptr) {
      work->wait();
    }
    works_.erase(communication);
    return;
  }


  const auto dst = expr_evaluator_.evaluate(p2p_comm->dst()).as<int64_t>();
  const auto src = expr_evaluator_.evaluate(p2p_comm->src()).as<int64_t>();
  const int64_t my_rank = communicator_->deviceId();
  if (my_rank == src) {
    const auto current_stream = static_cast<CUstream>(c10::cuda::getCurrentCUDAStream(my_local_device_index_).stream());
    at::Tensor buffer = getKnownTensorOrUndefined(p2p_comm->buffer(), expr_evaluator_);
    const auto it = remote_buffers_.find(buffer);
    NVF_ERROR(it != remote_buffers_.end(), "No remote buffer found for ", p2p_comm->toString(), " at rank ", my_rank);
    const std::vector<std::unique_ptr<RemoteBufferInfo>>& remote_buffers = it->second;

    const RemoteBufferInfo& my_buffer = *remote_buffers.at(my_rank);
    const auto local_semaphore = reinterpret_cast<CUdeviceptr>(&my_buffer.semaphores()[dst]);

    std::cout << "RANK " << my_rank << " WAIT RECV BEFORE cuStreamWaitValue32 on local semaphore " << local_semaphore << std::endl;
    NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(current_stream, local_semaphore, (cuuint32_t)(IpcSemaphore::kReady), CU_STREAM_WAIT_VALUE_EQ));
    std::cout << "RANK " << my_rank << " FINISHED WAIT RECV AFTER cuStreamWaitValue32 on local semaphore " << local_semaphore << ", buffer.data_ptr()=" << buffer.data_ptr() << ", my_buffer.ptr()=" << my_buffer.ptr() << "recv tensor=" << buffer << std::endl;
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

void HostIrEvaluator::handle(LinearOp* linear) {
  TensorView* in = linear->inA()->as<TensorView>();
  TensorView* weight = linear->inB()->as<TensorView>();
  TensorView* bias = linear->bias()->as<TensorView>();
  TensorView* out = linear->out()->as<TensorView>();
  NVF_ERROR(
      expr_evaluator_.isKnown(in) && expr_evaluator_.isKnown(weight) &&
          (!linear->has_bias() || expr_evaluator_.isKnown(bias)),
      "Inputs of the Linear Op ",
      linear->toString(),
      "must be precomputed before being retrieved");

  if (!expr_evaluator_.isKnown(out)) {
    unhandled(linear);
    return;
  }

  auto in_at = expr_evaluator_.evaluate(in).as<at::Tensor>();
  auto weight_at = expr_evaluator_.evaluate(weight).as<at::Tensor>();
  auto out_at = expr_evaluator_.evaluate(out).as<at::Tensor>();

  if (linear->has_bias()) {
    auto bias_at = expr_evaluator_.evaluate(bias).as<at::Tensor>();
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
  GlobalBufferInfo info =
      getBufferInfos(expr_evaluator_, PrimDataType::Int, {tv}).at(0);
  AliasInfo alias_info = {
      .type = AllocationType::New, .aliased_io = nullptr, .hide_output = false};
  c10::Device device =
      communicator_ ? communicator_->device() : at::Device("cuda:0");
  at::Tensor tensor = allocateTensor(info, alias_info, device, expr_evaluator_);
  expr_evaluator_.bind(tv, tensor);
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
