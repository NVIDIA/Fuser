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
using DeviceIdxType = int64_t;
using DimensionType = int;
using DeviceType = c10::Device;
using Team = std::vector<DeviceIdxType>;
} // namespace nvfuser
