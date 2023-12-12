// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <multidevice/communication.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

bool isLowerableToCommunication(Expr* expr);

// Lower a resharding Expr into a series of Communication, given a
// device_index.
std::vector<std::shared_ptr<Communication>> lowerCommunication(
    DeviceIdxType device_index,
    Expr* c,
    at::Tensor input_tensor,
    at::Tensor output_tensor);
} // namespace nvfuser

#endif
