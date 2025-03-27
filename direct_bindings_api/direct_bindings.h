// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>
#include <visibility.h>

namespace direct_bindings {

void initNvFuserDirectBindings(PyObject* module);

// Add direct bindings for Fusion IR
void bindFusionIr(py::module& direct_bindings);

} // namespace direct_bindings
