// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <host_ir/lower.h>
#include <instrumentation.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <multidevice/device_mesh.h>
#include <multidevice/executor.h>
#include <multidevice/utils.h>
#include <runtime/allocations.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

MultiDeviceExecutor::MultiDeviceExecutor(
    std::unique_ptr<Fusion> fusion,
    Communicator& comm,
    hir::HostIrEvaluatorParams params)
    : comm_(comm), number_of_outputs_(fusion->outputs().size()) {
  std::unique_ptr<hir::HostIrContainer> hic =
      HostIrLower::lower(std::move(fusion), comm.deviceId());
  // Create the HostIrEvaluator representing the host program
  host_ir_executor_ =
      std::make_unique<hir::HostIrEvaluator>(std::move(hic), &comm, params);
}

std::vector<at::Tensor> MultiDeviceExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // make sure the communicator can run the Fusion (e.g. there is enough GPUs,
  // etc)
  auto error_msg = validate();
  NVF_ERROR(error_msg.empty(), error_msg);

  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue;

  // Make sure inputs align at global boundary.
  NVF_ERROR(
      inputs.size() == host_ir_executor_->inputs().size(),
      "Wrong number of inputs");
  // process input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue[host_ir_executor_->inputs().at(input_idx)] =
        inputs.at(input_idx);
  }

  auto outputs = host_ir_executor_->runWithInput(val_to_IValue);
  return std::vector<at::Tensor>(
      outputs.end() - number_of_outputs_, outputs.end());
}

std::ostream& MultiDeviceExecutor::print(std::ostream& os) {
  return host_ir_executor_->print(os);
}

} // namespace nvfuser
