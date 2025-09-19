// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/group_inputs.h>

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <dispatch.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>
#include <ops/arith.h>

namespace nvfuser {
