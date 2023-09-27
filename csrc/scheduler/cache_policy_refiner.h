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

// Visits all global-to-local vector loads in `fusion` and refines their cache
// policies.
void refineCachePolicy(Fusion* fusion);

} // namespace nvfuser
