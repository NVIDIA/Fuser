// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/executor.h>
#include <ir/utils.h>

namespace nvfuser {

namespace hir {

class PostOnStreamExecutor final : public OptInDispatch {
  public:
    PostOnStreamExecutor() = default;
    std::vector<at::Tensor> post(Expr* op, std::vector<c10::IValue>& inputs, HostIrExecutorParams& params) {
      outputs_ = {};
      inputs_ = inputs;
      params_ = params;
      dispatch(op);
      return outputs_; //TODO: USE OUTPUT INSTEAD
    }

  private:
    std::vector<c10::IValue> inputs_;
    std::vector<at::Tensor> outputs_; //TODO: USE OUTPUT INSTEAD
    HostIrExecutorParams params_;
    // Cache Fusions, FusionExecutors
    std::unordered_map<HostUnit*, FusionExecutor> fe_;
    std::unordered_map<HostUnit*, FusionExecutorCache> fec_;

    using OptInDispatch::handle;
    void handle(HostUnit* hu) override {
      // Compile the fusion and execute it with FusionExecutor(Cache)
      // Check if the executor has been cached. If not, create and cache it
      if (params_.use_fusion_executor_cache) {
        fec_.try_emplace(
            hu,
            std::make_unique<Fusion>(*hu->fusion_to_execute()),
            0,
            !params_.skip_auto_scheduling);
        outputs_ = fec_.at(hu).runFusionWithInputs(inputs_); //TODO: USE OUTPUT INSTEAD
      } else {
        auto [it, has_emplaced] = fe_.try_emplace(hu);
        auto& fe = it->second;
        if (has_emplaced) {
          fe.compileFusion(
              hu->fusion_to_execute(), inputs_);
        }
        outputs_ = fe.runFusion(inputs_); //TODO: USE OUTPUT INSTEAD
        if (!params_.cache_fusion_executor) {
          fe_.erase(hu);
        }
      }

    }

    void handle(Communication* communication) override {
      std::cout << "POSTING A Communication" << std::endl;
    }
};

HostIrExecutor::HostIrExecutor(
    std::unique_ptr<HostIrContainer> container,
    HostIrExecutorParams params)
  : container_(std::move(container)), params_(std::move(params)) {};

std::vector<at::Tensor> HostIrExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // process input values
  NVF_ERROR(
      inputs.size() == container_->inputs().size(), "Wrong number of inputs");
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue_[container_->inputs().at(input_idx)] = inputs.at(input_idx);
  }

  // Interpret each instruction in an "eager" way by iterate over the Host Ir
  // Container's top level expression list
  for (auto expr : container_->topLevelExprs()) {
    dispatch(expr);
  }

  // Collect global outputs
  std::vector<at::Tensor> outputs;
  for (auto output_val : container_->outputs()) {
    auto output = val_to_IValue_.at(output_val).toTensor();
    outputs.push_back(output);
  }

  return outputs;
}

void HostIrExecutor::handle(PostOnStream* post) {
  std::vector<c10::IValue> input_IValues;
  for (auto& input : post->inputs()) {
    NVF_ERROR(
        val_to_IValue_.find(input) != val_to_IValue_.end(),
        "No buffer associated with Val ",
        input,
        " for handling ",
        post->toString());
    input_IValues.push_back(val_to_IValue_.at(input));
  }

  PostOnStreamExecutor post_executor;
  std::vector<at::Tensor> outputs = post_executor.post(post->hostOpToPost(), input_IValues, params_);

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    val_to_IValue_[post->outputs().at(output_idx)] = outputs.at(output_idx);
  }
}

} // namespace hir

} // namespace nvfuser
