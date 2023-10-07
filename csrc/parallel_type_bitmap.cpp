// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <parallel_type_bitmap.h>

namespace nvfuser {

constexpr std::bitset<ParallelTypeBitmap::kNumParallelTypes>
    ParallelTypeBitmap::kTIDBits;
constexpr std::bitset<ParallelTypeBitmap::kNumParallelTypes>
    ParallelTypeBitmap::kBIDBits;
constexpr std::bitset<ParallelTypeBitmap::kNumParallelTypes>
    ParallelTypeBitmap::kCIDBits;

std::string ParallelTypeBitmap::toString() const {
  std::stringstream ss;
  ss << "(";
  bool is_first = true;
  for (ParallelType pt : *this) {
    if (!is_first) {
      ss << " ";
    }
    ss << pt;
    is_first = false;
  }
  ss << ")";
  return ss.str();
}

} // namespace nvfuser
