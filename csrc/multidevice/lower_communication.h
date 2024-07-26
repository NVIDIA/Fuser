// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/base_nodes.h>
#include <multidevice/communication.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

// Returns whether we support transforming a given expression into a series
// of communication.
bool isLowerableToCommunication(Expr* expr);

// Lower a PipelineCommunication into a series of Communication.
std::vector<Communication*> lowerCommunication(Expr* c);

} // namespace nvfuser
