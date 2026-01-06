// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>

#include <direct_utils.h>
#include <tensor_caster.h>
#include <algorithm>

namespace nvfuser::python {

namespace {

PolymorphicValue toPolymorphicValue(const nb::handle& obj) {
  if (nb::isinstance<nb::ndarray<nb::pytorch>>(obj)) {
    return PolymorphicValue(nb::cast<at::Tensor>(obj));
  } else if (nb::isinstance<nb::bool_>(obj)) {
    return PolymorphicValue(nb::cast<bool>(obj));
  } else if (nb::isinstance<nb::int_>(obj)) {
    return PolymorphicValue(nb::cast<int64_t>(obj));
  } else if (nb::isinstance<nb::float_>(obj)) {
    return PolymorphicValue(nb::cast<double>(obj));
  } else if (nb::isinstance<std::complex<double>>(obj)) {
    return PolymorphicValue(nb::cast<std::complex<double>>(obj));
  }
  NVF_THROW("Cannot convert provided nb::handle to a PolymorphicValue.");
}

} // namespace

KernelArgumentHolder from_pyiterable(
    const nb::iterable& iter,
    std::optional<int64_t> device) {
  KernelArgumentHolder args;
  for (nb::handle obj : iter) {
    // Allows for a Vector of Sizes to be inputed as a list/tuple
    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
      for (nb::handle item : obj) {
        args.push(toPolymorphicValue(item));
      }
    } else {
      args.push(toPolymorphicValue(obj));
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
