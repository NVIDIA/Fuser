// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda.h>

namespace nvfuser {

#define DECLARE_DRIVER_API_WRAPPER(funcName) \
  extern decltype(::funcName)* funcName;

DECLARE_DRIVER_API_WRAPPER(cuGetErrorName);
DECLARE_DRIVER_API_WRAPPER(cuGetErrorString);
DECLARE_DRIVER_API_WRAPPER(cuTensorMapEncodeTiled);

#undef DECLARE_DRIVER_API_WRAPPER

} // namespace nvfuser
