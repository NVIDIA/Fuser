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
using RankType = int;
using DeviceIdxType = RankType;
using DimensionType = int;
using DeviceType = c10::Device;
} // namespace nvfuser
