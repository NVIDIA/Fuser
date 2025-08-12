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
    MultiDeviceExecutorParams params)
    : comm_(comm) {
  HostIrLower lower(params.lower);
  std::unique_ptr<hir::HostIrContainer> hic =
      lower.lower(std::move(fusion), comm.deviceId());
  // Create the HostIrEvaluator representing the host program
  host_ir_executor_ = std::make_unique<hir::HostIrEvaluator>(
      std::move(hic), &comm, params.executor);
}

KernelArgumentHolder MultiDeviceExecutor::runWithInput(
    const KernelArgumentHolder& inputs) {
  // Stores concrete computed values,
  std::unordered_map<Val*, PolymorphicValue> val_to_PValue;

  // Make sure inputs align at global boundary.
  NVF_ERROR(
      inputs.size() == std::ssize(host_ir_executor_->inputs()),
      "Wrong number of inputs");
  // process input values:
  for (auto input_idx : arange(inputs.size())) {
    val_to_PValue[host_ir_executor_->inputs().at(input_idx)] =
        inputs[input_idx];
  }

  return host_ir_executor_->runWithInput(val_to_PValue);
}

std::ostream& MultiDeviceExecutor::print(std::ostream& os) {
  return host_ir_executor_->print(os);
}

} // namespace nvfuser
