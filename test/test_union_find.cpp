// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <test/utils.h>
#include <union_find.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, FusionUnionFind) {
  UnionFind<uint8_t> uf(5);

  uf.merge(3, 4);

  assert(uf.equiv(3, 4));
  assert(!uf.equiv(2, 3));

  uf.enlarge(8);
  assert(!uf.equiv(3, 7));

  EXPECT_ANY_THROW(uf.enlarge(270); // Try to enlarge past capacity of IndexType
  );

  assert(uf.size() == 8);
  EXPECT_ANY_THROW(uf.find(8); // Try to index past current size
  );
}

// Test that joining two UnionFinds results in a partition containing A and B as
// refinements
TEST_F(NVFuserTest, FusionUnionFindJoin) {
  UnionFind<uint8_t> a(5);
  UnionFind<uint8_t> b(5);

  a.merge(2, 3);
  b.merge(3, 4);

  auto c = a.join(b);

  assert(a <= c);
  assert(b <= c);
}

// Test that meeting two UnionFinds results in a refinement of both A and B
TEST_F(NVFuserTest, FusionUnionFindMeet) {
  using IndexType = uint8_t;
  UnionFind<IndexType> a(5);
  UnionFind<IndexType> b(5);
  UnionFind<IndexType> c(5);

  // a and b overlap in all the singleton sets
  a.merge(2, 3);
  b.merge(3, 4);

  // c is single-set partition holding all items
  c.merge(0, 1);
  c.merge(1, 2);
  c.merge(2, 3);
  c.merge(3, 4);

  auto d = a.meet(b);
  assert(d <= a);
  assert(d <= b);
  assert(d == UnionFind<IndexType>(5));
  assert(d != a);

  auto e = a.meet(c);
  assert(e <= a);
  assert(e <= c);
  // Since a <= c, meet(a, c) == a
  assert(a <= c);
  assert(e == a);
}

} // namespace nvfuser
