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
//
// This function is called by various schedulers (like pointwise and reduction)
// to handle a partially alias-producing fusion. We could instead improve
// MarkAliasesPreparePass to split a partially alias-producing fusion into an
// alias-only segment and the rest. This way, the rest of the fusion (which has
// fewer expressions) can potentially find a better scheduler or scheduling, and
// we can completely remove this function for simplicity.
//
// This function is typically called at the end of scheduling. This is because
// schedulers may add Exprs that affects aliasing:
// https://github.com/NVIDIA/Fuser/issues/1401#issuecomment-1840137527.
void markAliases(Fusion* fusion);

} // namespace nvfuser
