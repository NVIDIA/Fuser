// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <fstream>
#include <iostream>
#include <string>

#include <nvrtc.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/jit_utils.h>
#include <torch/csrc/jit/resource_guard.h>

#include <gtest/gtest.h>

#include <cuda_utils.h>
#include <executor_utils.h>

namespace nvfuser {

TEST(NvrtcTest, Compile) {
  std::ifstream fin("__tmp_kernel_reduction_f0_c1_r0_g0.cu");
  std::string full_src_code(
      (std::istreambuf_iterator<char>(fin)),
      (std::istreambuf_iterator<char>()));

  std::string id = "reduction_f0_c1_r0_g0";
  std::string func_name = "nvfuser_reduction_f0_c1_r0_g0";

  nvrtcProgram program;
  torch::jit::ResourceGuard holdProgram(
      [&] { NVFUSER_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program)); });
  executor_utils::createNvrtcProgram(program, id, full_src_code);
  NVFUSER_NVRTC_SAFE_CALL(nvrtcAddNameExpression(program, func_name.c_str()));

  executor_utils::NvrtcCompileDriver nvrtc_compile_driver;
  nvrtc_compile_driver.setOption("--std=c++17");
  nvrtc_compile_driver.setOption("--diag-suppress=177");
  nvrtc_compile_driver.setOption("--gpu-architecture=sm_70");
  nvrtc_compile_driver.setOption("-default-device");
  nvrtc_compile_driver.setOption("--fmad=true");

  nvrtc_compile_driver.invoke(program, full_src_code);
}

} // namespace nvfuser
