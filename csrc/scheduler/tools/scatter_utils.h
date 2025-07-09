// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class ScatterOp;

namespace scheduler_tools {

void scheduleScatterLoopDomainAsIndexDomain(ScatterOp* sop);

} // namespace scheduler_tools
} // namespace nvfuser
