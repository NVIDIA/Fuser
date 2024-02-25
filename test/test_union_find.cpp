// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <simplification/union_find.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;

class UnionFindTest : public NVFuserTest {};

TEST_F(UnionFindTest, Basic) {
  UnionFind<uint8_t> uf(5);

  uf.merge(3, 4);

  EXPECT_TRUE(uf.equiv(3, 4));
  EXPECT_FALSE(uf.equiv(2, 3));

  uf.enlarge(8);
  EXPECT_FALSE(uf.equiv(3, 7));

  EXPECT_ANY_THROW(uf.enlarge(270); // Try to enlarge past capacity of IndexType
  );

  EXPECT_EQ(uf.size(), 8);
  EXPECT_ANY_THROW(uf.find(8); // Try to index past current size
  );
}

TEST_F(UnionFindTest, EquivalenceClasses) {
  UnionFind<uint8_t> uf(5);
  //   Class 0:
  //     0  *
  //   Class 1:
  //     1  *
  //   Class 2:
  //     2  *
  //   Class 3:
  //     3  *
  //   Class 3:
  //     4  *
  for (const auto i : c10::irange(uf.size())) {
    EXPECT_EQ(uf.find(i), i);
  }

  uf.merge(3, 4);
  //   Class 0:
  //     0  *
  //   Class 1:
  //     1  *
  //   Class 2:
  //     2  *
  //   Class 3:
  //     3  *
  //     4
  EXPECT_TRUE(uf.equiv(3, 4));
  EXPECT_FALSE(uf.equiv(2, 3));
  for (const auto i : c10::irange(4)) {
    EXPECT_EQ(uf.find(i), i);
  }
  EXPECT_EQ(uf.find(4), 3);

  uf.enlarge(8);
  //   Class 0:
  //     0  *
  //   Class 1:
  //     1  *
  //   Class 2:
  //     2  *
  //   Class 3:
  //     3  *
  //     4
  //   Class 4:
  //     5  *
  //   Class 5:
  //     6  *
  //   Class 6:
  //     7  *
  EXPECT_FALSE(uf.equiv(4, 7));
  EXPECT_EQ(uf.find(0), 0);
  EXPECT_EQ(uf.find(1), 1);
  EXPECT_EQ(uf.find(2), 2);
  EXPECT_EQ(uf.find(3), 3);
  EXPECT_EQ(uf.find(4), 3);
  EXPECT_EQ(uf.find(5), 5);
  EXPECT_EQ(uf.find(6), 6);
  EXPECT_EQ(uf.find(7), 7);

  // Perform a couple more merges to check that ordering is sane
  uf.merge(6, 0); // 0 -> 6
  uf.merge(2, 0); // 0 -> 6 -> 2
  uf.merge(5, 7); // 7 -> 5
  uf.merge(7, 3); // 4 -> 3 -> 7 -> 5
  //   Class 0:
  //     0
  //     2  *
  //     6
  //   Class 1:
  //     1  *
  //   Class 2:
  //     3
  //     4
  //     5  *
  //     7
  EXPECT_TRUE(uf.equiv(4, 7));
  for (auto expected_root : {1, 2, 5}) {
    EXPECT_EQ(uf.find(expected_root), expected_root);
  }
  EXPECT_EQ(uf.find(0), 2);
  EXPECT_EQ(uf.find(2), 2);
  EXPECT_EQ(uf.find(6), 2);

  EXPECT_EQ(uf.find(1), 1);

  EXPECT_EQ(uf.find(3), 5);
  EXPECT_EQ(uf.find(4), 5);
  EXPECT_EQ(uf.find(5), 5);
  EXPECT_EQ(uf.find(7), 5);

  // Test printing to string
  const auto s = uf.toString();
  EXPECT_GT(s.size(), 0);
}

// Test that joining two UnionFinds results in a partition containing A and B as
// refinements
TEST_F(UnionFindTest, Join) {
  UnionFind<uint8_t> a(5);
  UnionFind<uint8_t> b(5);

  a.merge(2, 3);
  b.merge(3, 4);

  auto c = a.join(b);

  EXPECT_LE(a, c);
  EXPECT_LE(b, c);
}

// Test that meeting two UnionFinds results in a refinement of both A and B
TEST_F(UnionFindTest, Meet) {
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
  EXPECT_LE(d, a);
  EXPECT_LE(d, b);
  EXPECT_EQ(d, UnionFind<IndexType>(5));
  EXPECT_NE(d, a);

  auto e = a.meet(c);
  EXPECT_LE(e, a);
  EXPECT_LE(e, c);
  // Since a <= c, meet(a, c) == a
  EXPECT_LE(a, c);
  EXPECT_EQ(e, a);
}

} // namespace nvfuser
