
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <C++20/ranges>

namespace nvfuser {

template <typename T>
auto irange(T&& t) {
  using TT = std::decay_t<T>;
  return std::views::iota(TT(0), std::forward<T>(t));
}

} // namespace nvfuser
