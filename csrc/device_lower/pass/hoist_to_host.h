// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <kernel.h>

namespace nvfuser {

void hoistScalarComputationToHost(
    Fusion* kernel,
    std::vector<Val*>& all_known_vals);

} // namespace nvfuser
