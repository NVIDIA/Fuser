// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <direct_utils.h>
#include <algorithm>

namespace nvfuser::python {

KernelArgumentHolder from_pyiterable(
    const py::iterable& iter,
    std::optional<int64_t> device) {
  KernelArgumentHolder args;
  for (py::handle obj : iter) {
    // Allows for a Vector of Sizes to be inputed as a list/tuple
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
      for (py::handle item : obj) {
        args.push(torch::jit::toIValue(item, c10::AnyType::get()));
      }
    } else {
      args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
    }
  }

  // Transform int64_t device to int8_t
  std::optional<int8_t> selected_device = std::nullopt;
  if (device.has_value()) {
    NVF_CHECK(device.value() < 256, "Maximum device index is 255");
    selected_device = (int8_t)device.value();
  }
  args.setDeviceIndex(selected_device);
  return args;
}

std::vector<at::Tensor> to_tensor_vector(const KernelArgumentHolder& outputs) {
  // Convert outputs KernelArgumentHolder to std::vector<at::Tensor>
  std::vector<at::Tensor> out_tensors;
  out_tensors.reserve(outputs.size());
  std::transform(
      outputs.begin(),
      outputs.end(),
      std::back_inserter(out_tensors),
      [](const PolymorphicValue& out) { return out.as<at::Tensor>(); });
  return out_tensors;
}

} // namespace nvfuser::python
