// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include <device_lower/dependencies.h>
#include <device_lower/interval_tree.h>

namespace nvfuser {

//! Naive reference implementation for testing
//!
//! This implementation uses simple linear search for all operations.
//! Complexity analysis (where N = number of intervals, k = number of
//! overlapping results):
//!
//! Method              | NaiveIntervalTree | CenteredIntervalTree
//! --------------------|-------------------|--------------------
//! insert()            | O(1)             | O(log N)
//! remove()            | O(N)             | O(log N)
//! query(point)        | O(N)             | O(log N + k)
//! query(start, end)   | O(N)             | O(log N + k)
//! size()              | O(1)             | O(N) - must traverse tree
//! empty()             | O(1)             | O(1)
//! clear()             | O(1)             | O(1)
//!
//! The naive implementation is simpler but less efficient for large datasets.
//! The tree-based implementation provides logarithmic query time at the cost
//! of logarithmic insertion/removal time and more complex implementation.
template <typename CoordT, typename PayloadT>
class NaiveIntervalTree {
 public:
  struct Interval {
    CoordT start;
    CoordT end;
    PayloadT data;

    Interval(CoordT start, CoordT end, const PayloadT& data)
        : start(start), end(end), data(data) {}

    bool overlaps(const Interval& other) const {
      return start <= other.end && end >= other.start;
    }

    bool contains(CoordT point) const {
      return start <= point && point <= end;
    }
  };

  void insert(CoordT start, CoordT end, const PayloadT& data) {
    intervals_.emplace_back(start, end, data);
  }

  void remove(CoordT start, CoordT end, const PayloadT& data) {
    auto it = std::find_if(
        intervals_.begin(), intervals_.end(), [&](const Interval& interval) {
          return interval.start == start && interval.end == end &&
              interval.data == data;
        });
    if (it != intervals_.end()) {
      intervals_.erase(it);
    }
  }

  std::vector<PayloadT> query(CoordT point) const {
    std::vector<PayloadT> result;
    for (const auto& interval : intervals_) {
      if (interval.contains(point)) {
        result.push_back(interval.data);
      }
    }
    return result;
  }

  std::vector<PayloadT> query(CoordT start, CoordT end) const {
    std::vector<PayloadT> result;
    Interval query_interval(start, end, PayloadT());
    for (const auto& interval : intervals_) {
      if (interval.overlaps(query_interval)) {
        result.push_back(interval.data);
      }
    }
    return result;
  }

  size_t size() const {
    return intervals_.size();
  }

  bool empty() const {
    return intervals_.empty();
  }

  void clear() {
    intervals_.clear();
  }

 private:
  std::vector<Interval> intervals_;
};

class IntervalTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up test data
    test_intervals_ = {
        {1, 5, "A"},
        {3, 7, "B"},
        {2, 4, "C"},
        {6, 10, "D"},
        {8, 12, "E"},
        {0, 2, "F"},
        {11, 15, "G"},
        {4, 6, "H"},
    };
  }

  std::vector<CenteredIntervalTree<int, std::string>::Interval> test_intervals_;
};

//! Test basic insertion and querying
TEST_F(IntervalTreeTest, BasicInsertAndQuery) {
  CenteredIntervalTree<int, std::string> tree;
  NaiveIntervalTree<int, std::string> naive;

  // Insert test intervals
  for (const auto& interval : test_intervals_) {
    tree.insert(interval.start, interval.end, interval.data);
    naive.insert(interval.start, interval.end, interval.data);
  }

  // Test point queries
  std::vector<int> test_points = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15};
  for (int point : test_points) {
    auto tree_result = tree.query(point);
    auto naive_result = naive.query(point);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Point query failed for point " << point;
    EXPECT_EQ(tree_result.size(), naive_result.size())
        << "Result size mismatch for point " << point;
  }

  // Test range queries
  std::vector<std::pair<int, int>> test_ranges = {
      {0, 2}, {1, 5}, {3, 7}, {6, 10}, {8, 12}, {11, 15}, {0, 15}};

  for (const auto& [start, end] : test_ranges) {
    auto tree_result = tree.query(start, end);
    auto naive_result = naive.query(start, end);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Range query failed for range [" << start << ", " << end << "]";
    EXPECT_EQ(tree_result.size(), naive_result.size())
        << "Result size mismatch for range [" << start << ", " << end << "]";
  }
}

//! Test the fromIntervals factory method
TEST_F(IntervalTreeTest, FromIntervals) {
  auto tree =
      CenteredIntervalTree<int, std::string>::fromIntervals(test_intervals_);
  NaiveIntervalTree<int, std::string> naive;

  for (const auto& interval : test_intervals_) {
    naive.insert(interval.start, interval.end, interval.data);
  }

  // Verify the tree was built correctly
  EXPECT_EQ(tree.size(), test_intervals_.size());
  EXPECT_EQ(tree.size(), naive.size());

  // Test a few queries to ensure correctness
  auto tree_result = tree.query(3);
  auto naive_result = naive.query(3);

  std::sort(tree_result.begin(), tree_result.end());
  std::sort(naive_result.begin(), naive_result.end());

  EXPECT_EQ(tree_result, naive_result);
}

//! Test removal functionality
TEST_F(IntervalTreeTest, Remove) {
  CenteredIntervalTree<int, std::string> tree;
  NaiveIntervalTree<int, std::string> naive;

  // Insert test intervals
  for (const auto& interval : test_intervals_) {
    tree.insert(interval.start, interval.end, interval.data);
    naive.insert(interval.start, interval.end, interval.data);
  }

  // Remove some intervals
  std::vector<CenteredIntervalTree<int, std::string>::Interval> to_remove = {
      {1, 5, "A"},
      {6, 10, "D"},
      {0, 2, "F"},
  };

  for (const auto& interval : to_remove) {
    tree.remove(interval.start, interval.end, interval.data);
    naive.remove(interval.start, interval.end, interval.data);
  }

  // Verify sizes match
  EXPECT_EQ(tree.size(), naive.size());

  // Test queries after removal
  std::vector<int> test_points = {3, 4, 5, 6, 7, 8, 9, 10};
  for (int point : test_points) {
    auto tree_result = tree.query(point);
    auto naive_result = naive.query(point);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Query failed after removal for point " << point;
  }
}

//! Test empty tree behavior
TEST_F(IntervalTreeTest, EmptyTree) {
  CenteredIntervalTree<int, std::string> tree;

  EXPECT_TRUE(tree.empty());
  EXPECT_EQ(tree.size(), 0);

  // Queries on empty tree should return empty results
  EXPECT_TRUE(tree.query(5).empty());
  EXPECT_TRUE(tree.query(1, 10).empty());
}

//! Test clear functionality
TEST_F(IntervalTreeTest, Clear) {
  CenteredIntervalTree<int, std::string> tree;

  // Insert some intervals
  for (const auto& interval : test_intervals_) {
    tree.insert(interval.start, interval.end, interval.data);
  }

  EXPECT_FALSE(tree.empty());
  EXPECT_EQ(tree.size(), test_intervals_.size());

  // Clear the tree
  tree.clear();

  EXPECT_TRUE(tree.empty());
  EXPECT_EQ(tree.size(), 0);
  EXPECT_TRUE(tree.query(5).empty());
  EXPECT_TRUE(tree.query(1, 10).empty());
}

//! Test with different coordinate types
TEST_F(IntervalTreeTest, DifferentCoordinateTypes) {
  // Test with double coordinates
  CenteredIntervalTree<double, int> double_tree;
  NaiveIntervalTree<double, int> double_naive;

  std::vector<std::pair<double, double>> double_intervals = {
      {1.5, 3.5}, {2.0, 4.0}, {3.0, 5.0}, {1.0, 2.0}};

  for (size_t i = 0; i < double_intervals.size(); ++i) {
    const auto& [start, end] = double_intervals[i];
    double_tree.insert(start, end, static_cast<int>(i));
    double_naive.insert(start, end, static_cast<int>(i));
  }

  // Test queries
  std::vector<double> test_points = {1.0, 2.0, 3.0, 4.0, 5.0};
  for (double point : test_points) {
    auto tree_result = double_tree.query(point);
    auto naive_result = double_naive.query(point);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Double query failed for point " << point;
  }
}

//! Test with complex payload types
TEST_F(IntervalTreeTest, ComplexPayload) {
  struct ComplexData {
    int id;
    std::string name;
    double value;

    bool operator==(const ComplexData& other) const {
      return id == other.id && name == other.name && value == other.value;
    }
  };

  CenteredIntervalTree<int, ComplexData> tree;
  NaiveIntervalTree<int, ComplexData> naive;

  std::vector<ComplexData> complex_data = {
      {1, "Alice", 10.5},
      {2, "Bob", 20.3},
      {3, "Charlie", 15.7},
  };

  for (size_t i = 0; i < complex_data.size(); ++i) {
    int start = static_cast<int>(i * 2);
    int end = start + 3;
    tree.insert(start, end, complex_data[i]);
    naive.insert(start, end, complex_data[i]);
  }

  // Test queries
  for (int point = 0; point < 8; ++point) {
    auto tree_result = tree.query(point);
    auto naive_result = naive.query(point);

    EXPECT_EQ(tree_result.size(), naive_result.size())
        << "Size mismatch for point " << point;

    // Sort by id for comparison
    std::sort(
        tree_result.begin(),
        tree_result.end(),
        [](const ComplexData& a, const ComplexData& b) { return a.id < b.id; });
    std::sort(
        naive_result.begin(),
        naive_result.end(),
        [](const ComplexData& a, const ComplexData& b) { return a.id < b.id; });

    EXPECT_EQ(tree_result, naive_result)
        << "Complex query failed for point " << point;
  }
}

//! Test edge cases and boundary conditions
TEST_F(IntervalTreeTest, EdgeCases) {
  CenteredIntervalTree<int, std::string> tree;
  NaiveIntervalTree<int, std::string> naive;

  // Test single-point intervals
  tree.insert(5, 5, "single");
  naive.insert(5, 5, "single");

  EXPECT_EQ(tree.query(5), std::vector<std::string>{"single"});
  EXPECT_EQ(naive.query(5), std::vector<std::string>{"single"});

  // Test overlapping intervals
  tree.insert(1, 10, "wide");
  tree.insert(5, 5, "point");
  naive.insert(1, 10, "wide");
  naive.insert(5, 5, "point");

  auto result = tree.query(5);
  std::sort(result.begin(), result.end());
  std::vector<std::string> expected = {"point", "single", "wide"};
  EXPECT_EQ(result, expected);

  // Test non-overlapping intervals
  tree.insert(20, 30, "far");
  naive.insert(20, 30, "far");

  EXPECT_TRUE(tree.query(15).empty());
  EXPECT_TRUE(naive.query(15).empty());
}

//! Test random stress testing
TEST_F(IntervalTreeTest, StressTest) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> coord_dist(0, 1000);
  std::uniform_int_distribution<> id_dist(1, 1000);

  CenteredIntervalTree<int, int> tree;
  NaiveIntervalTree<int, int> naive;

  // Insert random intervals
  const int num_intervals = 100;
  for (int i = 0; i < num_intervals; ++i) {
    int start = coord_dist(gen);
    int end = start + coord_dist(gen) % 100 + 1; // Ensure end > start
    int id = id_dist(gen);

    tree.insert(start, end, id);
    naive.insert(start, end, id);
  }

  // Verify sizes match
  EXPECT_EQ(tree.size(), naive.size());
  EXPECT_EQ(tree.size(), num_intervals);

  // Test random point queries
  const int num_point_queries = 50;
  for (int i = 0; i < num_point_queries; ++i) {
    int point = coord_dist(gen);
    auto tree_result = tree.query(point);
    auto naive_result = naive.query(point);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Stress test failed for point " << point;
    EXPECT_EQ(tree_result.size(), naive_result.size())
        << "Result size mismatch for point " << point;
  }

  // Test random range queries
  const int num_range_queries = 50;
  for (int i = 0; i < num_range_queries; ++i) {
    int start = coord_dist(gen);
    int end = start + coord_dist(gen) % 200;

    auto tree_result = tree.query(start, end);
    auto naive_result = naive.query(start, end);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Stress test failed for range [" << start << ", " << end << "]";
    EXPECT_EQ(tree_result.size(), naive_result.size())
        << "Result size mismatch for range [" << start << ", " << end << "]";
  }
}

//! Test with DependencyMapper::Coords coordinate type
TEST_F(IntervalTreeTest, DependencyMapperCoords) {
  using Coords = DependencyMapper::Coords;
  using CoordsTree = CenteredIntervalTree<Coords, std::string>;
  using CoordsNaive = NaiveIntervalTree<Coords, std::string>;
  using CoordsInterval = CoordsTree::Interval;

  // Create test intervals with DependencyMapper::Coords
  std::vector<CoordsInterval> coords_intervals = {
      {{0, 1, 2}, {0, 1, 5}, "A"}, // [0,1,2] to [0,1,5]
      {{0, 2, 1}, {0, 2, 4}, "B"}, // [0,2,1] to [0,2,4]
      {{1, 0, 0}, {1, 0, 3}, "C"}, // [1,0,0] to [1,0,3]
      {{1, 1, 1}, {1, 1, 6}, "D"}, // [1,1,1] to [1,1,6]
      {{2, 0, 0}, {2, 0, 2}, "E"}, // [2,0,0] to [2,0,2]
  };

  CoordsTree tree;
  CoordsNaive naive;

  // Insert test intervals
  for (const auto& interval : coords_intervals) {
    tree.insert(interval.start, interval.end, interval.data);
    naive.insert(interval.start, interval.end, interval.data);
  }

  // Test point queries
  std::vector<Coords> test_points = {
      {0, 1, 3}, // Should match A
      {0, 2, 2}, // Should match B
      {1, 0, 1}, // Should match C
      {1, 1, 4}, // Should match D
      {2, 0, 1}, // Should match E
      {0, 0, 0}, // Should match none
      {3, 0, 0}, // Should match none
  };

  for (const auto& point : test_points) {
    auto tree_result = tree.query(point);
    auto naive_result = naive.query(point);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Point query failed for point [" << point[0] << ", " << point[1]
        << ", " << point[2] << "]";
    EXPECT_EQ(tree_result.size(), naive_result.size())
        << "Result size mismatch for point [" << point[0] << ", " << point[1]
        << ", " << point[2] << "]";
  }

  // Test range queries
  using CoordsRange = std::pair<Coords, Coords>;
  std::vector<CoordsRange> test_ranges = {
      {{0, 1, 1}, {0, 1, 4}}, // Should match A
      {{0, 0, 0}, {0, 3, 6}}, // Should match A, B
      {{1, 0, 0}, {1, 2, 7}}, // Should match C, D
      {{0, 0, 0}, {3, 3, 10}}, // Should match all
      {{3, 0, 0}, {3, 0, 5}}, // Should match none
  };

  for (const auto& [start, end] : test_ranges) {
    auto tree_result = tree.query(start, end);
    auto naive_result = naive.query(start, end);

    std::sort(tree_result.begin(), tree_result.end());
    std::sort(naive_result.begin(), naive_result.end());

    EXPECT_EQ(tree_result, naive_result)
        << "Range query failed for range [[" << start[0] << "," << start[1]
        << "," << start[2] << "], [" << end[0] << "," << end[1] << "," << end[2]
        << "]]";
    EXPECT_EQ(tree_result.size(), naive_result.size())
        << "Result size mismatch for range [[" << start[0] << "," << start[1]
        << "," << start[2] << "], [" << end[0] << "," << end[1] << "," << end[2]
        << "]]";
  }

  // Test the fromIntervals factory method
  auto built_tree = CoordsTree::fromIntervals(coords_intervals);
  EXPECT_EQ(built_tree.size(), coords_intervals.size());

  // Test a query on the built tree
  auto result = built_tree.query({0, 1, 3});
  std::sort(result.begin(), result.end());
  EXPECT_EQ(result, std::vector<std::string>{"A"});
}

//! Test edge cases with DependencyMapper::Coords
TEST_F(IntervalTreeTest, DependencyMapperCoordsEdgeCases) {
  using Coords = DependencyMapper::Coords;
  using CoordsIntTree = CenteredIntervalTree<Coords, int>;
  using CoordsIntNaive = NaiveIntervalTree<Coords, int>;

  CoordsIntTree tree;
  CoordsIntNaive naive;

  // Test single-point intervals
  Coords point = {5, 5, 5};
  tree.insert(point, point, 42);
  naive.insert(point, point, 42);

  EXPECT_EQ(tree.query(point), std::vector<int>{42});
  EXPECT_EQ(naive.query(point), std::vector<int>{42});

  // Test overlapping intervals
  tree.insert({1, 1, 1}, {10, 10, 10}, 100);
  tree.insert({5, 5, 5}, {5, 5, 5}, 200);
  naive.insert({1, 1, 1}, {10, 10, 10}, 100);
  naive.insert({5, 5, 5}, {5, 5, 5}, 200);

  auto result = tree.query({5, 5, 5});
  std::sort(result.begin(), result.end());
  std::vector<int> expected = {42, 100, 200};
  EXPECT_EQ(result, expected);

  // Test non-overlapping intervals
  tree.insert({20, 20, 20}, {30, 30, 30}, 300);
  naive.insert({20, 20, 20}, {30, 30, 30}, 300);

  EXPECT_TRUE(tree.query({15, 15, 15}).empty());
  EXPECT_TRUE(naive.query({15, 15, 15}).empty());
}

} // namespace nvfuser
