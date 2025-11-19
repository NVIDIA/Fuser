// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cute/tensor.hpp>

template <class ElementType, class SmemLayout>
struct SharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayout>> smem;
  cute::uint64_t bulk_copy_mbar[1];
};
