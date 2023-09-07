// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <benchmark/benchmark.h>

#include <dynamic_type/dynamic_type.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <vector>

using namespace dynamic_type;

std::vector<int64_t> getRandomInt(int64_t size) {
  std::vector<int64_t> result;
  result.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    result.push_back(rand());
  }
  return result;
}

static const std::vector<int64_t> data_raw = getRandomInt(1000000);

static void Sort_Vector(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<int64_t> data_copy(data_raw);
    std::sort(data_copy.begin(), data_copy.end());
    state.counters["size"] = sizeof(int64_t);
  }
}

template <typename... Ts>
void BenchmarkDynamicType(benchmark::State& state) {
  using IntAndMore = DynamicType<NoContainers, Ts...>;
  std::vector<IntAndMore> data(data_raw.begin(), data_raw.end());
  for (auto _ : state) {
    std::vector<IntAndMore> data_copy(data);
    std::sort(data_copy.begin(), data_copy.end());
    state.counters["size"] = sizeof(IntAndMore);
  }
}

static void Sort_DynamicType1(benchmark::State& state) {
  BenchmarkDynamicType<int64_t>(state);
}

static void Sort_DynamicType2(benchmark::State& state) {
  BenchmarkDynamicType<int64_t, double>(state);
}

static void Sort_DynamicType3(benchmark::State& state) {
  BenchmarkDynamicType<int64_t, double, bool>(state);
}

static void Sort_DynamicType4(benchmark::State& state) {
  BenchmarkDynamicType<int64_t, double, bool, std::complex<double>>(state);
}

static void Sort_DynamicType5(benchmark::State& state) {
  BenchmarkDynamicType<int64_t, double, bool, std::complex<double>, float*>(
      state);
}

static void Sort_DynamicType6(benchmark::State& state) {
  BenchmarkDynamicType<
      int64_t,
      double,
      bool,
      std::complex<double>,
      float*,
      std::string>(state);
}

static void Sort_DynamicType7(benchmark::State& state) {
  BenchmarkDynamicType<
      int64_t,
      double,
      bool,
      std::complex<double>,
      float*,
      std::string,
      double*>(state);
}

static void Sort_DynamicType8(benchmark::State& state) {
  BenchmarkDynamicType<
      int64_t,
      double,
      bool,
      std::complex<double>,
      float*,
      std::string,
      double*,
      std::vector<int64_t>>(state);
}

static void Sort_DynamicType9(benchmark::State& state) {
  BenchmarkDynamicType<
      int64_t,
      double,
      bool,
      std::complex<double>,
      float*,
      std::string,
      double*,
      std::vector<int64_t>,
      int64_t*>(state);
}

static void Sort_DynamicType10(benchmark::State& state) {
  struct SomeType {};
  BenchmarkDynamicType<
      int64_t,
      double,
      bool,
      std::complex<double>,
      float*,
      std::string,
      double*,
      std::vector<int64_t>,
      int64_t*,
      SomeType>(state);
}

BENCHMARK(Sort_Vector)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType1)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType2)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType3)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType4)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType5)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType6)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType7)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType8)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType9)->Unit(benchmark::kMillisecond);
BENCHMARK(Sort_DynamicType10)->Unit(benchmark::kMillisecond);
