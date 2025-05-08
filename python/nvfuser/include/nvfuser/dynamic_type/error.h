// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <stdexcept>

// Users should provide their own implementation of DYNAMIC_TYPE_CHECK, so that
// the error message is more aligned with their codebase. However, if they don't
// provide one, we provide a default implementation here with minimal
// information.

#if !defined(DYNAMIC_TYPE_CHECK)

#define DYNAMIC_TYPE_CHECK(cond, msg1, ...) \
  if (!(cond)) [[unlikely]] {               \
    throw std::runtime_error(msg1);         \
  }

#endif
