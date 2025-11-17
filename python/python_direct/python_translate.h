// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>

namespace nvfuser::python {

// Translate a CPP Fusion to a bindings python function
std::string translateFusion(Fusion* f);

} // namespace nvfuser::python
