// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/cuda/CUDAStream.h>

#include "dispatch.h"
#include "expr_evaluator.h"
#include "host_ir/container.h"
#include "host_ir/ir.h"
#include "multidevice/communicator.h"
#include "multidevice/ipc_handle.h"
#include "runtime/executor_abstract.h"
#include "runtime/fusion_executor_cache.h"

namespace nvfuser {

namespace hir {

// Set of parameters that control the behavior of HostIrEvaluator
struct HostIrEvaluatorParams {
  // Experimental: whether to use FusionExecutorCache rather than
  // KernelExecutor.
  bool use_fusion_executor_cache = false;
  // Experimental: whether to apply auto-scheduling in FusionExecutorCache if
  // use_fusion_executor_cache=true. WAR: temporary hack mainly use for
  // development
  bool skip_auto_scheduling = false;
  // Experimental: whether to cache fusion executor. WAR: avoid recompilation
  // but implicitely assumes that the input shape don't change over iterations
  bool cache_fusion_executor = false;
  // number of additional cuda streams to use at runtime for comm+compute
  // pipelining
  int64_t number_of_streams = 4;
  // Whether to use allocation cache for tensor allocations
  bool use_allocation_cache = false;
};

// A HostIrEvaluator evaluates a host programs represented through a
// HostIrContainer It is instantiated with the desired HostIrContainer, and runs
// the Host program with concrete inputs by calling the method runWithInput.
//
// For now HostIrEvaluator is an interpreter; later we could rather compile host
// code.
//
// Note: most of the implementation is copy pasted for MultiDeviceExecutor. This
// duplication will be resolved in the future.
class NVF_API HostIrEvaluator final : public OptOutDispatch {
 public:
  HostIrEvaluator(
      std::unique_ptr<HostIrContainer> container,
      Communicator* communicator = &Communicator::getInstance(),
      HostIrEvaluatorParams = HostIrEvaluatorParams());

  // Used by FusionExecutorCache, the main stack.
  KernelArgumentHolder runWithInputs(const KernelArgumentHolder& args);

  // Only used by MultiDeviceExecutor.
  KernelArgumentHolder runWithInput(
      const std::unordered_map<Val*, PolymorphicValue>& val_to_PValue);

  const std::vector<Val*>& inputs() {
    return container_->inputs();
  }

  const std::vector<Val*>& outputs() {
    return container_->outputs();
  }

  const HostIrContainer& container() const {
    return *container_;
  }

  std::ostream& print(std::ostream& os) const {
    return container_->print(os);
  };

  // Only used by MultiDeviceExecutor.
  const auto& getFusionExecutorCaches() {
    return fec_;
  };

  const auto& getCudaStreams() {
    return streams_;
  }

 private:
  using OptOutDispatch::handle;
  void handle(SetCurrentStream*) override;
  void handle(GetCurrentStream*) override;
  void handle(Synchronize*) override;
  void handle(PostOnStream*) override;
  void handle(LaunchKernel*) override;
  void handle(Communication*) override;
  void handle(P2PCommunication*) override;
  void handle(Wait*) override;
  void handle(kir::ForLoop*) override;
  void handle(hir::ForLoop*) override;
  void handle(StartCoalescing*) override;
  void handle(EndCoalescing*) override;
  void handle(kir::IfThenElse*) override;
  void handle(MatmulOp*) override;
  void handle(LinearOp*) override;
  void handle(kir::Allocate*) override;
  void handle(LoadStoreOp*) override;
  void handle(BinaryOp*) override;
  void handle(ReductionOp*) override;
  void handle(ShareMemHandles*) override;
  void handle(HirAliasSelect*) override;
  void handle(Deallocate*) override;
  void handle(ShardByStream*) override;
  void handle(SymmetricContiguousView*) override;
  void unhandled(Statement*) override;

  c10::cuda::CUDAStream getCUDAStream(Stream* stream);

  PolymorphicValue getKnownConcreteValue(Val* val) const;

  at::Tensor getKnownTensorOrUndefined(Val* val) const;

  void validate() const;

  std::unique_ptr<HostIrContainer> container_;
  Communicator* communicator_;
  HostIrEvaluatorParams params_;
  // Stores concrete computed values
  ExpressionEvaluator expr_evaluator_;
  // Cache Fusions, KernelExecutors
  std::unordered_map<HostUnit*, std::unique_ptr<ExecutorAbstract>> executors_;
  std::unordered_map<HostUnit*, FusionExecutorCache> fec_;
  using StreamKey = std::variant<int64_t, Stream*>;
  std::unordered_map<StreamKey, c10::cuda::CUDAStream> streams_;
  std::unordered_map<Expr*, c10::intrusive_ptr<c10d::Work>> works_;
  const int64_t my_local_device_index_;
  IpcHandleCache ipc_handle_cache_;
  SymmetricMemoryHandleCache multicast_handle_cache_;
  // Allocation cache
  std::unordered_map<kir::Allocate*, at::Tensor> allocation_cache_;
};

} // namespace hir

} // namespace nvfuser
