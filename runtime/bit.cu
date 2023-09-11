// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef __NVCC__
#include <bit>
#else

namespace std {

template <class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From), To> bit_cast(
    const From& src) noexcept {
  return *reinterpret_cast<const To*>(&src);
}

} // namespace std

#endif
