// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include "host_ir/container.h"

namespace nvfuser::hir {

// Run passes on the given HostIrContainer for functionality and/or performance.
void runPasses(HostIrContainer& hic);

} // namespace nvfuser::hir
