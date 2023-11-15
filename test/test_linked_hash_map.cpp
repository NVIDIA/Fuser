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

namespace {
class CopyableKey {
 public:
  explicit CopyableKey(std::string data) : data_(std::move(data)) {}

  size_t hash() const {
    return std::hash<std::string>()(data_);
  }

  bool operator==(const CopyableKey& other) const {
    return data_ == other.data_;
  }

 private:
  std::string data_;
};

class MovableValue {
 public:
  explicit MovableValue(int data) : data_(data) {}

  MovableValue(const MovableValue&) = delete;
  MovableValue& operator=(const MovableValue&) = delete;

  MovableValue(MovableValue&&) = default;
  MovableValue& operator=(MovableValue&&) = default;

  int data() const {
    return data_;
  }

 private:
  int data_;
};
} // namespace

namespace std {
template <>
struct hash<CopyableKey> {
  size_t operator()(const CopyableKey& key) const {
    return key.hash();
  }
};
} // namespace std

namespace nvfuser {

using testing::ElementsAre;
using testing::Eq;
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

TEST(LinkedHashMapTest, EraseThenPushBack) {
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

  map.pushBack("b", 4);
  EXPECT_THAT(map, ElementsAre(Pair("a", 1), Pair("b", 4)));
}

namespace {
MATCHER_P(DataIs, data, "") {
  return arg.data() == data;
}
} // namespace

TEST(LinkedHashMapTest, MovableValue) {
  LinkedHashMap<CopyableKey, MovableValue> map;
  map.pushBack(CopyableKey("a"), MovableValue(1));
  map.pushBack(CopyableKey("b"), MovableValue(2));
  map.erase(CopyableKey("b"));

  EXPECT_THAT(map, ElementsAre(Pair(CopyableKey("a"), DataIs(1))));
}

} // namespace nvfuser
