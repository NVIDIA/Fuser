// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "utils.h"

#include <options.h>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

std::string getHostName() {
#if defined(__linux__)
  constexpr int len = 128;
  char name[len];
  gethostname(name, len);
  return std::string(name);
#else
  return "UNKNOWN";
#endif
}

std::string getDeviceName() {
  int dev_idx;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&dev_idx));

  cudaDeviceProp prop;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, dev_idx));

  return std::string(prop.name);
}

} // namespace

// Copied from BENCHMARK_MAIN with extra custom settings
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  ::benchmark::AddCustomContext("Host", getHostName());
  ::benchmark::AddCustomContext("GPU", getDeviceName());

  // Disable kernel reuse during all benchmarks.
  // This is important since some benchmarks use FusionExecutorCache in order to
  // benchmark scheduling and segmentation. However, when benchmarking multiple
  // input shapes, this can lead to re-using suboptimal FusionKernelRuntimes, in
  // some extreme cases resulting in the use of a cached segmented runtime when
  // it would be possible to schedule the Fusion without segmentation.
  // See https://github.com/NVIDIA/Fuser/pull/563 for more info
  DisableOptionsGuard og;
  DisableOptionsGuard::getCurOptions().set(DisableOption::KernelReuse);

  ::benchmark::RunSpecifiedBenchmarks();

  ::benchmark::Shutdown();
  return 0;
}
