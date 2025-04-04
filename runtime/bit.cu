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

// Intentionally not supporting signed integers to stay consistent with
// https://en.cppreference.com/w/cpp/numeric/bit_ceil
__device__ __forceinline__ unsigned int bit_ceil(unsigned int x) {
  if (x == 0) {
    return 1;
  }
  return 1u << (32 - __clz(x - 1));
}

__device__ __forceinline__ unsigned long long bit_ceil(unsigned long long x) {
  if (x == 0) {
    return 1;
  }
  return 1ull << (64 - __clzll(x - 1));
}

#endif
