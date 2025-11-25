// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <fstream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>
#include <nvrtc.h>

#include "cute.cuh"
#include "utils.h"

// GCC/Clang/CUDA (Linux)
#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>

template <typename T>
std::string type_to_string() {
  int status = 0;
  // abi::__cxa_demangle converts the internal compiler ID to human C++ code
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(typeid(T).name(), NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : typeid(T).name();
}

// MSVC (Windows)
#elif defined(_MSC_VER)
template <typename T>
std::string type_to_string() {
  // MSVC name() is already human readable usually, but struct/class might
  // differ
  return typeid(T).name();
}
#endif

std::string read_file(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    exit(1);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

std::vector<std::byte> convertTensorToBytes(at::Tensor& tensor) {
  std::vector<std::byte> buffer;
  void* data = tensor.data_ptr();
  buffer.reserve(sizeof(void*));
  buffer.insert(
      buffer.end(), (std::byte*)&data, (std::byte*)&data + sizeof(void*));
  return buffer;
}

template <class T>
std::vector<std::byte> convertToBytes(T data) {
  std::vector<std::byte> buffer;
  buffer.reserve(sizeof(data));
  buffer.insert(
      buffer.end(), (std::byte*)&data, (std::byte*)&data + sizeof(data));
  return buffer;
}

std::string getGpuArch() {
  constexpr int device_index = 0;
  auto properties = at::cuda::getDeviceProperties(device_index);
  return std::to_string(properties->major) + std::to_string(properties->minor);
}

int main() {
  // 1. Read the CUDA source file
  std::string filename = "bulk_load.cu";
  std::string cuda_source = read_file(filename);

  // 2. Create the NVRTC Program
  nvrtcProgram prog;
  // Parameters: program ptr, source string, name for errors, num headers,
  // headers, header names
  nvrtcCreateProgram(
      &prog, cuda_source.c_str(), filename.c_str(), 0, NULL, NULL);

  auto smem_layout = cute::make_layout(
      cute::Shape<cute::_32, cute::_32>{}, cute::GenRowMajor{});
  auto gmem_layout = smem_layout;
  int32_t smem_size =
      static_cast<int32_t>(sizeof(SharedStorage<float, decltype(smem_layout)>));

  std::string name_expression = "cute_bulk_copy<float," +
      type_to_string<decltype(gmem_layout)>() + "," +
      type_to_string<decltype(smem_layout)>() + ">";
  nvrtcAddNameExpression(prog, name_expression.c_str());

  std::string gpu_arch = "--gpu-architecture=compute_" + getGpuArch() + "a";
  std::string cutlass_arch = "-DCUTLASS_NVCC_ARCHS=" + getGpuArch() + "a\"";

  // 3. Compile the Program
  std::vector<const char*> opts;
  opts.push_back("-std=c++20");
  opts.push_back("-default-device");
  opts.push_back(gpu_arch.c_str());
  opts.push_back(cutlass_arch.c_str());
  opts.push_back("-I/usr/local/cuda/include/");
  opts.push_back("-I/usr/local/cuda/include/cccl/");
  opts.push_back("-I../../../third_party/cutlass/include");
  opts.push_back("-I../../../third_party/cutlass/include/tools/util/include");
  NVRTC_CHECK(nvrtcCompileProgram(prog, opts.size(), opts.data()));

  const char* mangled_name;
  nvrtcGetLoweredName(prog, name_expression.c_str(), &mangled_name);

  // 5. Get the PTX (The compiled binary)
  size_t ptx_size;
  nvrtcGetPTXSize(prog, &ptx_size);
  assert(ptx_size > 0);

  std::vector<char> ptx(ptx_size);
  nvrtcGetPTX(prog, ptx.data());

  // **********************************************************

  // Create ATen arguments before getting CUfunction to create cuContext because
  // manually creating cuContext fails
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto d_in = at::randn({32, 32}, options);
  auto d_out = at::empty({32, 32}, options);

  std::vector<std::vector<std::byte>> data;
  std::vector<void*> pointers;

  data.push_back(convertTensorToBytes(d_in));
  pointers.emplace_back(data.back().data());

  data.push_back(convertTensorToBytes(d_out));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(gmem_layout));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(smem_layout));
  pointers.emplace_back(data.back().data());

  // Create module from binary file (FATBIN)
  CUmodule cu_module;
  CUDA_CHECK(cuModuleLoadData(&cu_module, (void*)ptx.data()));

  // Get function handle from module
  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, cu_module, mangled_name));

  // Cleanup NVRTC after cuModuleGetFunction to avoid destroying mangled_name
  nvrtcDestroyProgram(&prog);

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

  CUlaunchConfig config = {};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 128;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = smem_size;
  config.attrs = nullptr;
  config.numAttrs = 0;

  CUDA_CHECK(cuLaunchKernelEx(&config, kernel, pointers.data(), nullptr));

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

  // Flatten the tensors for easier access
  auto d_in_flat = d_in.flatten();
  auto d_out_flat = d_out.flatten();

  // Validate the results
  for (int i = 0; i < cute::size(gmem_layout); ++i) {
    int k = gmem_layout(i);
    // TODO: Replace with ASSERT_EQ if using gtest framework
    float in_val = d_in_flat[k].item<float>();
    float out_val = d_out_flat[k].item<float>();
    if (in_val != out_val) {
      printf("d_in[%d] = %f, d_out[%d] = %f\n", k, in_val, k, out_val);
    }
  }

  return 0;
}
