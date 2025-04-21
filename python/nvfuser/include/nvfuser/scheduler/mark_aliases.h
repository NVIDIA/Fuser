// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>

namespace nvfuser {

// Marks aliases between fusion inputs and outputs. This respects existing
// allocation domains **even when** they are empty (assuming default order). See
// [Note on overriding empty allocation domains].
void markAliases(Fusion* fusion);

} // namespace nvfuser
