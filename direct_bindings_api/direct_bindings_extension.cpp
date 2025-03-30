// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <direct_bindings.h>
#include <torch/extension.h>

PYBIND11_MODULE(DIRECT_BINDINGS_EXTENSION, m) {
  m.doc() = "Direct python bindings for NvFuser CPP API";
  direct_bindings::initNvFuserDirectBindings(m.ptr());
}
