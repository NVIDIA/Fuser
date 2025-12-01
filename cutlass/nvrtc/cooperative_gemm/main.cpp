// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <chrono>
#include <fstream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/swizzle.hpp> // cute::Swizzle
#include <cute/swizzle_layout.hpp> // cute::compose(cute::Swizzle)
#include <cute/tensor.hpp>
#include <nvrtc.h>

#include "cute.cuh"
#include "utils.h"

using namespace cute;

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

int roundUp(int a, int b) {
  return ((a + b - 1) / b) * b;
}

std::string getGpuArch() {
  constexpr int device_index = 0;
  auto properties = at::cuda::getDeviceProperties(device_index);
  return std::to_string(properties->major) + std::to_string(properties->minor);
}

int main() {
  constexpr uint32_t thread_block_size = 128;
  constexpr int max_vec_bits = 16;
  constexpr uint32_t copy_max_vec_bytes = max_vec_bits / 8;
  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using ALoadTransform = cute::identity;
  using BLoadTransform = cute::identity;
  using CLoadTransform = cute::identity;
  using CStoreTransform = cute::identity;

  const auto alpha = static_cast<TC>(1.0);
  const auto beta = static_cast<TC>(0.0);

  auto tiled_mma = TiledMMA<
      MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
      Layout<Shape<_2, _2, _1>, Stride<_1, _2, _0>>,
      Tile<_32, _32, _16>>{};

  auto a_layout = Layout<Shape<_64, _64>, Stride<_64, _1>>{};
  auto b_layout = Layout<Shape<_64, _64>, Stride<_64, _1>>{};
  auto c_layout = Layout<Shape<_64, _64>, Stride<_64, _1>>{};

  ALoadTransform a_load_transform;
  BLoadTransform b_load_transform;
  CLoadTransform c_load_transform;
  CStoreTransform c_store_transform;
  auto a_smem_copy_op = SM75_U32x4_LDSM_N{};
  auto b_smem_copy_op = SM75_U32x4_LDSM_N{};
  auto c_smem_copy_ld_op = SM75_U32x4_LDSM_N{};
  auto c_smem_copy_st_op = SM90_U32x4_STSM_N{};

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto d_a = at::randn({64, 64}, options);
  auto d_b = at::randn({64, 64}, options);
  auto d_c = at::zeros({64, 64}, options);
  auto d_c_out = at::empty({64, 64}, options);
  const size_t smem_size =
      roundUp(sizeof(TA) * d_a.numel(), copy_max_vec_bytes) +
      roundUp(sizeof(TB) * d_b.numel(), copy_max_vec_bytes) +
      sizeof(TC) * d_c.numel();

  // 1. Read the CUDA source file
  std::string filename = "cooperative_gemm.cu";
  std::string cuda_source = read_file(filename);

  // 2. Create the NVRTC Program
  nvrtcProgram prog;
  // Parameters: program ptr, source string, name for errors, num headers,
  // headers, header names
  nvrtcCreateProgram(
      &prog, cuda_source.c_str(), filename.c_str(), 0, NULL, NULL);

  std::string name_expression = "cooperative_gemm_kernel<" +
      std::to_string(thread_block_size) + "," + std::to_string(max_vec_bits) +
      "," + type_to_string<decltype(a_layout)>() + "," +
      type_to_string<decltype(b_layout)>() + "," +
      type_to_string<decltype(c_layout)>() + "," +
      type_to_string<decltype(a_layout)>() + "," +
      type_to_string<decltype(b_layout)>() + "," +
      type_to_string<decltype(c_layout)>() + "," + type_to_string<TA>() + "," +
      type_to_string<TB>() + "," + type_to_string<TC>() + "," +
      type_to_string<decltype(alpha)>() + "," +
      type_to_string<decltype(beta)>() + "," +
      type_to_string<decltype(tiled_mma)>() + "," +
      type_to_string<ALoadTransform>() + "," +
      type_to_string<BLoadTransform>() + "," +
      type_to_string<CLoadTransform>() + "," +
      type_to_string<CStoreTransform>() + "," +
      type_to_string<decltype(a_smem_copy_op)>() + "," +
      type_to_string<decltype(b_smem_copy_op)>() + "," +
      type_to_string<decltype(c_smem_copy_ld_op)>() + "," +
      type_to_string<decltype(c_smem_copy_st_op)>() + ">";
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

  auto start = std::chrono::high_resolution_clock::now();
  NVRTC_CHECK(nvrtcCompileProgram(prog, opts.size(), opts.data()));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::cout << "NVRTC compilation time: " << duration.count() << " seconds"
            << std::endl;

  const char* mangled_name;
  nvrtcGetLoweredName(prog, name_expression.c_str(), &mangled_name);

  // 5. Get the PTX (The compiled binary)
  size_t ptx_size;
  nvrtcGetPTXSize(prog, &ptx_size);
  assert(ptx_size > 0);

  std::vector<char> ptx(ptx_size);
  nvrtcGetPTX(prog, ptx.data());

  // **********************************************************

  // Run Kernel
  std::vector<std::vector<std::byte>> data;
  std::vector<void*> pointers;

  data.push_back(convertToBytes(a_layout));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(b_layout));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(c_layout));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(a_layout));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(b_layout));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(c_layout));
  pointers.emplace_back(data.back().data());

  data.push_back(convertTensorToBytes(d_a));
  pointers.emplace_back(data.back().data());

  data.push_back(convertTensorToBytes(d_b));
  pointers.emplace_back(data.back().data());

  data.push_back(convertTensorToBytes(d_c));
  pointers.emplace_back(data.back().data());

  data.push_back(convertTensorToBytes(d_c_out));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(alpha));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(beta));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(tiled_mma));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(a_load_transform));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(b_load_transform));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(c_load_transform));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(c_store_transform));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(a_smem_copy_op));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(b_smem_copy_op));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(c_smem_copy_ld_op));
  pointers.emplace_back(data.back().data());

  data.push_back(convertToBytes(c_smem_copy_st_op));
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
  config.blockDimX = thread_block_size;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = smem_size;
  config.attrs = nullptr;
  config.numAttrs = 0;

  CUDA_CHECK(cuLaunchKernelEx(&config, kernel, pointers.data(), nullptr));

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

  // TODO Validate the results
  constexpr float atol = 1e-2;
  constexpr float rtol = 1e-2;
  auto ref = at::linear(d_a, d_b);
  bool result = d_c_out.allclose(ref, atol, rtol);
  std::cout << "result allclose\t" << result << std::endl;

  return 0;
}
