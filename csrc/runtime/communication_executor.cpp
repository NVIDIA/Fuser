// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/communication_executor.h>

#include <fusion_profiler.h>
#include <host_ir/lower_to_communication.h>
#include <instrumentation.h>
#include <multidevice/utils.h>
#include <tensor_metadata.h>

namespace nvfuser {

CommunicationExecutor::CommunicationExecutor(
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id)
    : ExecutorAbstract(fusion_id, concrete_id, runtime_id, group_id),
      communicator_(&Communicator::getInstance()) {}

bool CommunicationExecutor::supported(Fusion* fusion) {
  FUSER_PERF_SCOPE("CommunicationExecutor::supported");
  std::vector<Expr*> exprs = fusion->exprs();
  if (std::any_of(exprs.begin(), exprs.end(), isResharding)) {
    NVF_ERROR(
        std::all_of(exprs.begin(), exprs.end(), isResharding),
        "Could not execute fusion as all expressions in a host IR container "
        "must be communication based at this point.");
    return true;
  }
  return false;
}

void CommunicationExecutor::compile(Fusion* fusion) {
  FUSER_PERF_SCOPE("CommunicationExecutor::compile");
  NVF_ERROR(
      supported(fusion),
      "CommunicationExecutor does not support the Fusion provided.");
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).startCompile();
  }

  host_ir_container_ = std::make_unique<hir::HostIrContainer>();
  IrCloner cloner = Fusion::copy(fusion, host_ir_container_.get());
  if (fusion->isA<hir::HostIrContainer>()) {
    for (Expr* e : fusion->as<hir::HostIrContainer>()->topLevelExprs()) {
      host_ir_container_->pushBackTopLevelExprs(cloner.clone(e));
    }
  } else {
    std::vector<Expr*> exprs = fusion->exprs();
    DeviceIdxType my_device_idx = communicator_ ? communicator_->deviceId() : 0;
    for (Expr* e : exprs) {
      std::vector<Expr*> communications =
          convertSingleOpToCommunication(cloner.clone(e), my_device_idx);
      for (auto* communication : communications) {
        host_ir_container_->pushBackTopLevelExprs(communication);
      }
    }
  }

  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopCompile();
  }
}

bool CommunicationExecutor::isCompiled() const {
  return host_ir_container_ != nullptr;
}

KernelArgumentHolder CommunicationExecutor::run(
    const KernelArgumentHolder& args,
    KernelArgumentHolder output_args) {
  FUSER_PERF_SCOPE("CommunicationExecutor::run");
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
        out_idx < std::ssize(host_ir_container_->outputs()),
        "Output tensor not found in fusion outputs");
    auto out_tensor = output_args[out_idx].as<at::Tensor>();

    // Inputs are already validated in bindInputs.
    validateSizesAndStrides({out_tensor}, {communication->out()}, expr_eval);
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

} // namespace nvfuser
