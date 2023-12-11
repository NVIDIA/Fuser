// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/utils.h>
#include <multidevice/allocator.h>
#include <multidevice/executor.h>
#include <multidevice/lower_communication.h>
#include <multidevice/pipeline.h>

namespace nvfuser {

bool PipelineExecutor::shouldRun(SegmentedGroup* group) {
  if (should_run_.find(group) == should_run_.end()) {
    NVF_ERROR(!group->exprs().empty()
              && !group->exprs().at(0)->outputs().empty()
              && group->exprs().at(0)->outputs().at(0)->isA<TensorView>()
              && group->exprs().at(0)->outputs().at(0)->as<TensorView>()->hasDeviceMesh());
    should_run_.emplace(
        group, group->exprs().at(0)->outputs().at(0)->as<TensorView>()->getDeviceMesh()->has(runtime_.comm_.deviceId()));
  }
  return should_run_[group];
}

void PipelineExecutor::executeKernel(SegmentedGroup* group) {
  if (!shouldRun(group)) {
    return;
  }
  // get the IValues corresponding to the group's input
  std::vector<c10::IValue> group_input_IValues;
  for (auto& input : group->inputs()) {
    NVF_ERROR(val_to_IValue_.find(input) != val_to_IValue_.end(), "Device ", runtime_.comm_.deviceId(), " has no buffer associated with Val ", input, " for handling group ", toString(group));
    NVF_ERROR(val_to_IValue_.at(input).isTensor());
    group_input_IValues.push_back(val_to_IValue_.at(input));
  }

  std::vector<at::Tensor> outputs;

  // Compile the group and execute it with FusionExecutor
  // Check if the executor has been cached. If not, create and cache it
  if (fe_.find(group) == fe_.end()) {
    fe_.emplace(group, std::make_unique<FusionExecutor>());
    fusions_.emplace(group, runtime_.pipeline_->sf_->makeFusion(group));
    fe_[group]->compileFusion(fusions_.at(group).get(), group_input_IValues);
  }
  // TODO: deal with aliases I/O. For example if the stage is empty, i.e., Inputs=Outputs, we need to handle them anyway
  outputs = fe_[group]->runFusion(group_input_IValues);

  // std::cout << "RANK " << runtime_.comm_.deviceId()
  //           << " handling KERNEL group " << toString(group)
  //           << "\n with inputs:{\n";
  // for (auto i: c10::irange(group_input_IValues.size())) {
  //   std::cout << " val: " << group->inputs()[i] << "\nIval: " << group_input_IValues[i] << "\n";
  // }
  // std::cout << "}\nAnd outputs:{\n";
  // for (auto i: c10::irange(outputs.size())) {
  //   std::cout << " val: " << group->outputs()[i] << "\nIval: " << outputs[i] << "\n";
  // }
  // std::cout << "}" << std::endl;

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    val_to_IValue_[group->outputs().at(output_idx)] = outputs.at(output_idx);
  }
}

void PipelineExecutor::executeCommunication(SegmentedGroup* group) {
  NVF_ERROR(group->exprs().size() == 1, "Communication segments must contain only one Expr");
  auto expr = group->exprs().at(0);
  NVF_ERROR(expr->inputs().size() == 1, "Communication must have exactly one input");
  NVF_ERROR(expr->outputs().size() == 1, "Communication must have exactly one output");
  auto input_val = expr->inputs().at(0);
  auto output_val = expr->outputs().at(0);
  at::Tensor input_tensor, output_tensor;
  if (val_to_IValue_.find(input_val) != val_to_IValue_.end()) {
    input_tensor = val_to_IValue_.at(input_val).toTensor();
    // std::cout << input_val << " FOUND!, value: " << input_tensor;
  } else {
    // std::cout << input_val << " NOT FOUND!";
  }
  if (val_to_IValue_.find(output_val) != val_to_IValue_.end()) {
    output_tensor = val_to_IValue_.at(output_val).toTensor();
    // std::cout << output_val << " FOUND!, value: " << output_tensor;
  } else {
    // std::cout << output_val << " NOT FOUND!";
  }

  // std::cout << "RANK " << runtime_.comm_.deviceId()
  //           << " handling COMMUNICATION group " << toString(group)
  //           << "\n with input:{\n"
  //           << " val: " << expr->inputs()[0] << "\nIval: " << input_tensor << "\n"
  //           << "}\nAnd outputs:{\n"
  //           << " val: " << expr->outputs()[0] << "\nIval: " << output_tensor << "\n"
  //           << std::endl;

  // Lower the Communication into a vector of Communications
  if (communications_.find(group) == communications_.end()) { // check if cached
    communications_.emplace(
        group,
        lowerCommunication(
            runtime_.comm_.deviceId(), expr, input_tensor, output_tensor));
  }
  auto& communications = communications_[group];

  // post and wait communications
  for (auto& communication : communications) {
    auto work = communication->post(runtime_.comm_);
    if (work) work->wait();
  }
}

std::vector<at::Tensor> PipelineExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // Make sure inputs align at global boundary.
  NVF_ERROR(
      inputs.size() == runtime_.pipeline_->sf_->inputs().size(),
      "Wrong number of inputs");

  val_to_IValue_ = allocatePipelineIntermediateBuffers(runtime_.pipeline_, runtime_.comm().deviceId(), inputs);

  // process input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue_[runtime_.pipeline_->sf_->inputs().at(input_idx)] =
        inputs.at(input_idx);
  }

  // Run through the stages to launch kernel
  // traverseTo(runtime_.pipeline_->outputs());
  prepareRuntimeOrder(runtime_.pipeline_->sf_.get(), workspace_);
  for (auto group: workspace_.group_run_order) {

    if (!runtime_.pipeline_->is_resharding.at(group)) {
      executeKernel(group);
    } else {
      executeCommunication(group);
    }
  }


  // Collect global outputs from context
  std::vector<at::Tensor> outputs;
  for (auto output_val : runtime_.pipeline_->sf_->outputs()) {
    outputs.push_back(val_to_IValue_.at(output_val).toTensor());
  }

  return outputs;
}

} // namespace nvfuser

#endif
