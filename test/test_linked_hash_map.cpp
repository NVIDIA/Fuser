// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <linked_hash_map.h>

namespace nvfuser {

using testing::ElementsAre;
using testing::Pair;

TEST(LinkedHashMapTest, PushBack) {
  LinkedHashMap<std::string, int> map;
  map.pushBack("a", 1);
  map.pushBack("b", 2);
  map.pushBack("c", 3);

  EXPECT_THAT(map, ElementsAre(Pair("a", 1), Pair("b", 2), Pair("c", 3)));
}

TEST(LinkedHashMapTest, Insert) {
  LinkedHashMap<std::string, int> map;
  map.pushBack("a", 1);
  map.insert(map.begin(), "b", 2);
  map.pushBack("c", 3);

  EXPECT_THAT(map, ElementsAre(Pair("b", 2), Pair("a", 1), Pair("c", 3)));
}

TEST(LinkedHashMapTest, Erase) {
  LinkedHashMap<std::string, int> map;
  map.pushBack("a", 1);
  map.pushBack("b", 2);
  map.pushBack("c", 3);

  auto [v, i] = map.erase("b");
  EXPECT_EQ(v, 2);
  EXPECT_EQ(i->first, "c");
  EXPECT_THAT(map, ElementsAre(Pair("a", 1), Pair("c", 3)));

  std::tie(v, i) = map.erase("c");
  EXPECT_EQ(v, 3);
  EXPECT_EQ(i, map.end());
  EXPECT_THAT(map, ElementsAre(Pair("a", 1)));
}

} // namespace nvfuser
