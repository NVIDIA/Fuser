// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#if __has_include(<bits/c++config.h>)
#include <bits/c++config.h>
#endif

#if defined(__GLIBCXX__) && __GLIBCXX__ >= 20230000
#define STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE 1
#endif
