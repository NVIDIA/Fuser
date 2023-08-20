// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <c10/core/Device.h>

namespace nvfuser {
using RankType = int64_t;
using DeviceIdxType = RankType;
using DeviceType = c10::Device;
} // namespace nvfuser
