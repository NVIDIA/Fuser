// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include "dynamic_type/dynamic_type.h"

#include <iostream>

using namespace dynamic_type;

template <typename T, int i>
T f(T x0) {
  auto x1 = x0 + i;
  auto x2 = x1 * x1;
  auto x3 = 1 / x2;
  auto x4 = x3 % x2;
  auto x5 = x4 + 1;
  auto x6 = 1 * x5;
  auto x7 = 1 / x6;
  auto x8 = x7 % x0;
  if (x8 > 10) {
    x8++;
  } else if (x8 >= 7) {
    x8--;
  } else if (x8 < 5) {
    x8 += 1;
  } else if (x8 <= 3) {
    x8 -= 1;
  }
  auto x9 = x8 && x8;
  auto x10 = x9 || x9;
  auto x11 = x10 == x10;
  auto x12 = x11 != x11;
  auto x13 = x12 < x12;
  auto x14 = x13 > x13;
  auto x15 = x14 <= x14;
  auto x16 = x15 >= x15;
  auto x17 = x16 + x16;
  auto x18 = x17 - x17;
  auto x19 = x18 * x18;
  auto x20 = x19 / x19;
  auto x21 = x20 % x20;
  auto x22 = x21 & x21;
  auto x23 = x22 | x22;
  auto x24 = x23 ^ x23;
  auto x25 = x24 << x24;
  auto x26 = x25 >> x25;
  return x26;
}

template <typename T, int i>
void g(T x) {
  std::cout << f<T, 0>(i + x) << std::endl;
  std::cout << f<T, 1>(i + x) << std::endl;
  std::cout << f<T, 2>(i + x) << std::endl;
  std::cout << f<T, 3>(i + x) << std::endl;
  std::cout << f<T, 4>(i + x) << std::endl;
  std::cout << f<T, 5>(i + x) << std::endl;
  std::cout << f<T, 6>(i + x) << std::endl;
  std::cout << f<T, 7>(i + x) << std::endl;
  std::cout << f<T, 8>(i + x) << std::endl;
  std::cout << f<T, 9>(i + x) << std::endl;
}

template <typename T, int i>
void gg(T x) {
  g<T, i + 0>(i + x);
  g<T, i + 1>(i + x);
  g<T, i + 2>(i + x);
  g<T, i + 3>(i + x);
  g<T, i + 4>(i + x);
  g<T, i + 5>(i + x);
  g<T, i + 6>(i + x);
  g<T, i + 7>(i + x);
  g<T, i + 8>(i + x);
  g<T, i + 9>(i + x);
}

template <typename T, int i>
void ggg(T x) {
  gg<T, i + 0>(i + x);
  gg<T, i + 1>(i + x);
  gg<T, i + 2>(i + x);
  gg<T, i + 3>(i + x);
  gg<T, i + 4>(i + x);
  gg<T, i + 5>(i + x);
  gg<T, i + 6>(i + x);
  gg<T, i + 7>(i + x);
  gg<T, i + 8>(i + x);
  gg<T, i + 9>(i + x);
}

template<typename T>
void gggg(T x) {
  ggg<T, 0>(x);
  ggg<T, 1>(x);
  ggg<T, 2>(x);
  ggg<T, 3>(x);
  ggg<T, 4>(x);
  ggg<T, 5>(x);
  ggg<T, 6>(x);
  ggg<T, 7>(x);
  ggg<T, 8>(x);
  ggg<T, 9>(x);
}
