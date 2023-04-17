// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/test_gpu_validator.h>
#include <test/test_utils.h>
#include <union_find.h>
#include <val_equivalence.h>

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

TEST_F(NVFuserTest, FusionValEquivalence) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto ve = ValEquivalence<uint8_t>(fusion);
}

} // namespace nvfuser
