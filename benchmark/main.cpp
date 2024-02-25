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

void addGPUBenchmarkContext() {
  int dev_idx = 0;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&dev_idx));

  cudaDeviceProp prop;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, dev_idx));

  ::benchmark::AddCustomContext("gpu_name", prop.name);
  ::benchmark::AddCustomContext(
      "gpu_gmem_bytes", std::to_string(prop.totalGlobalMem));
  ::benchmark::AddCustomContext(
      "gpu_smem_bytes_per_block", std::to_string(prop.sharedMemPerBlock));
  ::benchmark::AddCustomContext(
      "gpu_regs_per_block", std::to_string(prop.regsPerBlock));
  ::benchmark::AddCustomContext(
      "gpu_clock_khz", std::to_string(prop.clockRate));
  ::benchmark::AddCustomContext(
      "gpu_mem_clock_khz", std::to_string(prop.memoryClockRate));
  ::benchmark::AddCustomContext(
      "gpu_mem_bus_width_bits", std::to_string(prop.memoryBusWidth));
  ::benchmark::AddCustomContext(
      "gpu_compute_capability_major", std::to_string(prop.major));
  ::benchmark::AddCustomContext(
      "gpu_compute_capability_minor", std::to_string(prop.minor));
  ::benchmark::AddCustomContext(
      "gpu_sm_count", std::to_string(prop.multiProcessorCount));
  ::benchmark::AddCustomContext(
      "gpu_l2_bytes", std::to_string(prop.l2CacheSize));
  ::benchmark::AddCustomContext(
      "gpu_max_threads_per_sm",
      std::to_string(prop.maxThreadsPerMultiProcessor));
  ::benchmark::AddCustomContext(
      "gpu_max_threads_per_block", std::to_string(prop.maxThreadsPerBlock));
}

} // namespace

// Copied from BENCHMARK_MAIN with extra custom settings
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  ::benchmark::AddCustomContext("host", getHostName());

  addGPUBenchmarkContext();

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
