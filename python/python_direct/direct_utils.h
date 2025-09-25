// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <runtime/executor_kernel_arg.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <optional>
#include <vector>

namespace nvfuser::python {

// Convert a py::iterable to a KernelArgumentHolder
nvfuser::KernelArgumentHolder from_pyiterable(
    const py::iterable& iter,
    std::optional<int64_t> device = std::nullopt);

// Convert a KernelArgumentHolder to a std::vector<at::Tensor>
std::vector<at::Tensor> to_tensor_vector(
    const nvfuser::KernelArgumentHolder& outputs);

} // namespace nvfuser::python
