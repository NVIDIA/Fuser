// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/Context.h>
#include <exceptions.h>
#include <torch/csrc/jit/ir/ir.h>

namespace nvfuser {

void TypePropagate(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace nvfuser
