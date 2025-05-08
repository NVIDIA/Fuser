// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

/// Defines the NVF_API macro, which should be added on methods or classes
/// that are used outside of nvFuser. See doc/dev/visibility.md for details.

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#define NVF_API __declspec(dllexport)
#else
#define NVF_API __declspec(dllimport)
#endif
#else
#define NVF_API __attribute__((visibility("default")))
#endif
