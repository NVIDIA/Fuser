// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/utils.h>
#include <host_ir_executor.h>

namespace nvfuser {

namespace hir {

class PostOnStreamExecutor final : public OptInDispatch {
  public:
    PostOnStreamExecutor() = default;
    post(Expr* op, const std::vector<c10::IValue>& inputs, std::vector<at::Tensor>& outputs, const HostIrExecutorParams& params) {
      inputs_ = inputs;
      outputs_ = outputs;
      params_ = params;
      dispatch(op);
    }

  private:
    const std::vector<c10::IValue>& inputs_;
    std::vector<at::Tensor>& outputs_;
    const HostIrExecutorParams& params_;
    // Cache Fusions, FusionExecutors
    std::unordered_map<PostOnStream*, FusionExecutor> fe_;
    std::unordered_map<PostOnStream*, FusionExecutorCache> fec_;

    using OptInDispatch::handle;
    void handle(HostUnit* hu) override {
      // Compile the fusion and execute it with FusionExecutor(Cache)
      // Check if the executor has been cached. If not, create and cache it
      if (params_.use_fusion_executor_cache) {
        fec_.try_emplace(
            post,
            std::make_unique<Fusion>(*post->hostOpToPost()->as<hostUnit>()->fusion_to_execute()),
            0,
            !params_.skip_auto_scheduling);
        outputs = fec_.at(post).runFusionWithInputs(inputs_);
      } else {
        auto [it, has_emplaced] = fe_.try_emplace(post);
        auto& fe = it->second;
        if (has_emplaced) {
          fe.compileFusion(
              post->hostOpToPost()->as<hostUnit>()->fusion_to_execute(), inputs_);
        }
        outputs = fe.runFusion(inputs_);
        if (!params_.cache_fusion_executor) {
          fe_.erase(post);
        }
      }

    }

    void handle(Communication* communication) override {
      std::cout << "POSTING A Communication" << std::endl;
    }
};

HostIrExecutor::HostIrExecutor(std::unique_ptr<HostIrContainer> container, HostIrExecutorParams params)
  : container_(std::move(container)), params_(std::move(params)), post_executor_(std::make_unique<PostOnStreamExecutor>()) {};

std::vector<at::Tensor> HostIrExecutor::runWithInput(const std::vector<c10::IValue>& inputs) {
  // process input values:
  NVF_ERROR(
      inputs.size() == container_->inputs().size(),
      "Wrong number of inputs");
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue_[container_->inputs().at(input_idx)] =
        inputs.at(input_idx);
  }

  for (auto expr: container_->topLevelExprs()) {
    dispatch(expr);
  }

  // Collect global outputs from context
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
  std::vector<at::Tensor> outputs;

  post_executor_->post(post->hostOpToPost(), input_IValues, outputs);

  // placeholder for storing the outputs
    // Store the outputs in the context
    for (auto output_idx : c10::irange(outputs.size())) {
      val_to_IValue_[post->outputs().at(output_idx)] = outputs.at(output_idx);
    }

}

} // namespace hir

} // namespace nvfuser
