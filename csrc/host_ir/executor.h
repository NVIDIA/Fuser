// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>
#include <expr_evaluator.h>
#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <multidevice/communicator.h>
#include <runtime/executor.h>
#include <runtime/executor_abstract.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>

#include <c10/cuda/CUDAStream.h>

namespace nvfuser {

class HostIrExecutor : public ExecutorAbstract {
 public:
  HostIrExecutor(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  static bool supported(Fusion* fusion);

  void compile(Fusion* fusion);

  bool isCompiled() const override;

  NVF_API std::vector<at::Tensor> run(
      KernelArgumentHolder& args,
      std::vector<at::Tensor> outputs = {});

  const std::unique_ptr<hir::HostIrContainer>& hostContainer() const {
    return host_ir_container_;
  }

 private:
  std::unique_ptr<hir::HostIrContainer> host_ir_container_;
  Communicator* communicator_;
};

namespace hir {

/*
a HostIrEvaluator evaluates a host programs represented through a
HostIrContainer It is instantiated with the desired HostIrContainer, and runs
the Host program with concrete inputs by calling the method runWithInput.

For now HostIrEvaluator is an interpreter; later we could rather compile host
code.

Note: most of the implementation is copy pasted for MultiDeviceExecutor. This
duplication will be resolved in the future.
*/

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
};

class HostIrEvaluator final : public OptOutDispatch {
 public:
  HostIrEvaluator(
      std::unique_ptr<HostIrContainer> container,
      Communicator* communicator = nullptr,
      HostIrEvaluatorParams = HostIrEvaluatorParams());
  std::vector<at::Tensor> runWithInput(
      std::unordered_map<Val*, c10::IValue> val_to_IValue);

  const std::vector<Val*>& inputs() {
    return container_->inputs();
  }

  std::ostream& print(std::ostream& os) const {
    return container_->print(os);
  };

  const auto& getFusionExecutorCaches() {
    return fec_;
  };

  const auto& getCudaStreams() {
    return streams_;
  }

 private:
  using OptOutDispatch::handle;
  void handle(SetCurrentStream* set_current_stream) override;
  void handle(Synchronize* synchronize) override;
  void handle(PostOnStream* post_ir) override;
  void handle(Communication* communication) override;
  void handle(P2PCommunication* communication) override;
  void handle(Wait* wait) override;
  void handle(ForLoop* for_loop) override;
  void handle(StartCoalescing* start_coalescing) override;
  void handle(EndCoalescing* end_coalescing) override;
  void handle(kir::IfThenElse* if_then_else) override;
  void handle(MatmulOp* matmul) override;
  void unhandled(Statement* stmt) override;

  c10::cuda::CUDAStream getCUDAStream(Stream* stream);

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
};

} // namespace hir

} // namespace nvfuser
