// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <swizzle.h>

#include <ir/builder.h>
#include <ops/arith.h>

namespace nvfuser {
namespace swizzles {

// ------------------------------------------------------------
// Swizzle Definitions
//   for each swizzle name:
// un(Swizzle Name) e.g. unZShape is the inverse of ZShape,
//  (unswizzle is needed for inlining and is currently not actively used.)
// ------------------------------------------------------------

// Unit Z swizzle:
//  Alternate directions of Y dimension:
//    1 2 3      1 2 3
//    4 5 6  =>  6 5 4
//    7 8 9      7 8 9

// ZShape is inverse of itself

// Block cyclic Xor swizzle: (bank conflict removal)
//  Apply cyclic Xor within blocks:
//   Example: cyclic Xor
//    1   2  3  4       1   2   3  4
//    5   6  7  8       6   5   8  7
//    9  10 11 12  =>   11  12  9 10
//    13 14 15 16       16  15 14 13
std::pair<Val*, Val*> Xor(Val* x, Val* y) {
  // Need to validate in swizzle configuration:
  //  size_x == size_y
  return {x, bitwise_xor(x, y)};
}

// Xor is inverse of itself
std::pair<Val*, Val*> unXor(Val* x, Val* y) {
  return Xor(x, y);
}

// Block cyclic shift swizzle: (bank conflict removal)
//  Apply cyclic shift within blocks:
//   Example: cyclic shift
//    1   2  3  4       1   2   3   4
//    5   6  7  8       8   5   6   7
//    9  10 11 12  =>   11  12  9  10
//    13 14 15 16       14  15  16 13
std::pair<Val*, Val*> CyclicShift(Val* x, Val* y, Val* size_x) {
  return {x, mod(add(x, y), size_x)};
}

std::pair<Val*, Val*> unCyclicShift(Val* x, Val* y, Val* size_x) {
  return {x, mod(sub(add(size_x, y), x), size_x)};
}

} // namespace swizzles

std::pair<Val*, Val*> dispatchSwizzle(
    SwizzleType type,
    Val* x,
    Val* y,
    Val* maybe_size_x,
    Val* maybe_size_y) {
  switch (type) {
    case SwizzleType::XOR:
      return swizzles::Xor(x, y);
    default:
      NVF_THROW("Unsupported swizzle type");
  }
}

std::pair<Val*, Val*> dispatchUnSwizzle(
    SwizzleType type,
    Val* x,
    Val* y,
    Val* maybe_size_x,
    Val* maybe_size_y) {
  switch (type) {
    case SwizzleType::XOR:
      return swizzles::unXor(x, y);
    default:
      NVF_THROW("Unsupported swizzle type");
  }
}

} // namespace nvfuser
