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
class Key {
 public:
  explicit Key(std::string data) : data_(std::move(data)) {}

  size_t hash() const {
    return std::hash<std::string>()(data_);
  }

  bool operator==(const Key& other) const {
    return data_ == other.data_;
  }

 private:
  std::string data_;
};

class Value {
 public:
  explicit Value(int data) : data_(data) {}

  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;

  Value(Value&&) = default;
  Value& operator=(Value&&) = default;

  bool operator==(const Value& other) const {
    return data_ == other.data_;
  }

  int data() const {
    return data_;
  }

 private:
  int data_;
};
} // namespace
} // namespace nvfuser

namespace std {
template <>
struct hash<nvfuser::Key> {
  size_t operator()(const nvfuser::Key& key) const {
    return key.hash();
  }
};
} // namespace std

namespace nvfuser {

namespace {
MATCHER_P(DataIs, data, "") {
  return arg.data() == data;
}
} // namespace

TEST(LinkedHashMapTest, MovableValue) {
  LinkedHashMap<Key, Value> map;
  map.pushBack(Key("a"), Value(1));
  map.pushBack(Key("b"), Value(2));
  map.erase(Key("b"));

  EXPECT_THAT(map, ElementsAre(Pair(Key("a"), DataIs(1))));
}

} // namespace nvfuser
