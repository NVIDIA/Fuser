// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/id_model.h>

namespace nvfuser {

// Get the domains to predicate for a given tensor used as a consumer
// of a given expr.
std::vector<IterDomain*> getPredicateDomains(
    TensorView* consumer_tv,
    const Expr* expr);

} // namespace nvfuser
