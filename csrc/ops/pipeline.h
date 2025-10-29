// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <ir/interface_nodes.h>
#include <ops/utils.h>
#include <type.h>

namespace nvfuser {

NVF_API TensorView* launch_dependent_grid(std::vector<Val*> inputs);
NVF_API TensorView* wait_for_prior_grid(std::vector<Val*> inputs);

} // namespace nvfuser
