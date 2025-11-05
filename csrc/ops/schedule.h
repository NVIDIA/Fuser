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

// The launch dependent grid instruction allows the driver to start the
// dependent grid in programmatic dependent launch. This is commonly used after
// computation is finished but before storing results to global memory.
NVF_API TensorView* launch_dependent_grid(std::vector<Val*> inputs);

// The wait for prior grid instruction prevents the kernel from running before
// the prior grid is finished. This is used before any operations access global
// input variables modified by the prior kernel.
NVF_API TensorView* wait_for_prior_grid(std::vector<Val*> inputs);

} // namespace nvfuser
