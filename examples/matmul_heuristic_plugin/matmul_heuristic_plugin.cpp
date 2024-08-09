// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <matmul_heuristic_plugin_api.h>

#include <cstdint>
#include <iostream>
#include <memory>

using namespace nvfuser::matmul_heuristic_plugin;

// This example heuristic simply prints the problem description then sets a
// fixed kernel configuration.
struct MyKernelConfig : KernelConfig {
  void configure() final {
    std::cout << "Using example heuristic for problem: ";
    std::cout << "m=" << problem.m << " ";
    std::cout << "n=" << problem.n << " ";
    std::cout << "k=" << problem.k << " ";
    std::cout << "batch_size=" << problem.batch_size << " ";
    std::cout << "layout=";
    switch (problem.layout) {
      case KernelConfig::ProblemDescription::Layout::NN:
        std::cout << "NN ";
        break;
      case KernelConfig::ProblemDescription::Layout::NT:
        std::cout << "NT ";
        break;
      case KernelConfig::ProblemDescription::Layout::TN:
        std::cout << "TN ";
        break;
      case KernelConfig::ProblemDescription::Layout::TT:
        std::cout << "TT ";
        break;
    }
    std::cout << "precision=" << problem.precision << std::endl;

    cta_tile = {128, 128, 32};
    warp_tile = {64, 64, 32};
    instruction_tile = {16, 8, 16};
    splitk_factor = 2;
    load_stages = 3;
    async_gmem_load_operands = true;
    grid_swizzle_factor = 1;
    cta_order = 0;
  };

  ~MyKernelConfig() {
    std::cout << "~MyKernelConfig" << std::endl;
  }
};

extern "C" std::unique_ptr<KernelConfig> makeConfig() {
  return std::unique_ptr<KernelConfig>(new MyKernelConfig);
}
