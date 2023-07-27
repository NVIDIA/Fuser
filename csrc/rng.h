// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>

#include <tuple>

namespace nvfuser {

std::tuple<Val*, Val*, GetRNGSeedAndOffsetFromHost*> getRNGSeedAndOffsetFromHost();

} // namespace nvfuser
