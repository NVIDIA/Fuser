// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <nanobind/nanobind.h>
#include <runtime/executor_kernel_arg.h>
#include <optional>
#include <typeinfo>
#include <vector>

namespace nb = nanobind;

namespace nvfuser::python {

// Convert a nb::sequence to a vector
template <typename T>
std::vector<T> from_pysequence(nb::sequence seq) {
  std::vector<T> result;
  result.reserve(nb::len(seq));
  std::transform(
      seq.begin(), seq.end(), std::back_inserter(result), [](nb::handle obj) {
        NVF_ERROR(nb::isinstance<T>(obj));
        return nb::cast<T>(obj);
      });
  return result;
}

// Convert a nb::iterable to a KernelArgumentHolder
nvfuser::KernelArgumentHolder from_pyiterable(
    const nb::iterable& iter,
    std::optional<int64_t> device = std::nullopt);

// Convert a KernelArgumentHolder to a std::vector<at::Tensor>
std::vector<at::Tensor> to_tensor_vector(
    const nvfuser::KernelArgumentHolder& outputs);

} // namespace nvfuser::python
