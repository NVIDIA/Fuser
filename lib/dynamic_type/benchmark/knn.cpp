// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <benchmark/benchmark.h>

#include <dynamic_type/dynamic_type.h>

#include <cmath>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>

using namespace dynamic_type;

// Given a vector of points with their (x, y, z) coordinates and values, find
// the k nearest neighbors of a given point and compute their average value.
// This benchmark compares the performance of the native struct implementation
// with the dynamic struct implementation.

struct Point {
  double x, y, z;
};

struct PointAndValue {
  Point point;
  double value;
};

static std::vector<PointAndValue> getRandomPoints(int64_t size) {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);

  std::vector<PointAndValue> result;
  result.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    result.push_back(
        {{distribution(generator),
          distribution(generator),
          distribution(generator)},
         distribution(generator)});
  }
  return result;
}

const std::vector<PointAndValue> random_data = getRandomPoints(10000);

static double kNN_Native(
    const std::vector<PointAndValue>& data,
    const Point& query_point,
    int64_t k) {
  using DistanceAndValue = std::pair<double, double>;
  auto compare = [](const DistanceAndValue& a, const DistanceAndValue& b) {
    return a.first < b.first;
  };
  std::priority_queue<
      DistanceAndValue,
      std::vector<DistanceAndValue>,
      decltype(compare)>
      distances_and_values(compare);
  for (const auto& point_and_value : data) {
    const auto& point = point_and_value.point;
    const auto& value = point_and_value.value;
    double dx = point.x - query_point.x;
    double dy = point.y - query_point.y;
    double dz = point.z - query_point.z;
    double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    if ((int64_t)distances_and_values.size() < k) {
      distances_and_values.push({distance, value});
    } else if (distance < distances_and_values.top().first) {
      distances_and_values.pop();
      distances_and_values.push({distance, value});
    }
  }
  double sum = 0;
  while (!distances_and_values.empty()) {
    sum += distances_and_values.top().second;
    distances_and_values.pop();
  }
  return sum / k;
}

static void kNN_Native(benchmark::State& state) {
  for (auto _ : state) {
    kNN_Native(random_data, {0, 0, 0}, 10);
  }
}

BENCHMARK(kNN_Native)->Unit(benchmark::kMillisecond);

template <typename T>
struct DynamicStruct {
  std::unordered_map<std::string, std::shared_ptr<T>> fields;
  const T& operator[](const std::string& key) const {
    return *fields.at(key);
  }
  T& operator[](const std::string& key) {
    if (fields.count(key) == 0) {
      fields[key] = std::make_unique<T>();
    }
    return *fields.at(key);
  }
};

using StructVecDouble =
    DynamicType<Containers<DynamicStruct, std::vector>, double>;

static StructVecDouble kNN_Dictionary(
    const StructVecDouble& data,
    const StructVecDouble& query_point,
    int64_t k) {
  using DistanceAndValue = std::pair<StructVecDouble, StructVecDouble>;
  auto compare = [](const DistanceAndValue& a, const DistanceAndValue& b) {
    return a.first < b.first;
  };
  std::priority_queue<
      DistanceAndValue,
      std::vector<DistanceAndValue>,
      decltype(compare)>
      distances_and_values(compare);
  for (const auto& point_and_value : data.as<std::vector>()) {
    const auto& point = point_and_value["point"];
    const auto& value = point_and_value["value"];
    StructVecDouble dx = point["x"] - query_point["x"];
    StructVecDouble dy = point["y"] - query_point["y"];
    StructVecDouble dz = point["z"] - query_point["z"];
    StructVecDouble distance =
        std::sqrt((dx * dx + dy * dy + dz * dz).as<double>());
    if ((int64_t)distances_and_values.size() < k) {
      distances_and_values.push({distance, value});
    } else if (distance < distances_and_values.top().first) {
      distances_and_values.pop();
      distances_and_values.push({distance, value});
    }
  }
  StructVecDouble sum = 0.0;
  while (!distances_and_values.empty()) {
    sum += distances_and_values.top().second;
    distances_and_values.pop();
  }
  return sum / k;
}

static void kNN_Dictionary(benchmark::State& state) {
  StructVecDouble data = std::vector<StructVecDouble>{};
  auto& data_vector = data.as<std::vector>();
  for (const auto& point_and_value : random_data) {
    DynamicStruct<StructVecDouble> point;
    point["x"] = point_and_value.point.x;
    point["y"] = point_and_value.point.y;
    point["z"] = point_and_value.point.z;
    DynamicStruct<StructVecDouble> point_and_value_struct;
    point_and_value_struct["point"] = point;
    point_and_value_struct["value"] = point_and_value.value;
    data_vector.push_back(point_and_value_struct);
  }
  DynamicStruct<StructVecDouble> query_point;
  query_point["x"] = 0.0;
  query_point["y"] = 0.0;
  query_point["z"] = 0.0;
  for (auto _ : state) {
    kNN_Dictionary(data, query_point, 10);
  }
}

BENCHMARK(kNN_Dictionary)->Unit(benchmark::kMillisecond);
