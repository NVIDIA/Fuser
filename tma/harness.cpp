#include <cstdio>
#include <iostream>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

// Only if nvfuser_index_t is int64_t
std::vector<std::byte> convertTensorToBytes(at::Tensor& tensor) {
  std::vector<std::byte> buffer;
  void* data = tensor.data_ptr();
  c10::IntArrayRef logical_size = tensor.sizes();
  c10::IntArrayRef alloc_stride = tensor.strides();
  buffer.reserve(
      sizeof(void*) + sizeof(int64_t) * logical_size.size() +
      sizeof(int64_t) * alloc_stride.size());
  buffer.insert(
      buffer.end(), (std::byte*)&data, (std::byte*)&data + sizeof(void*));
  buffer.insert(
      buffer.end(),
      (std::byte*)logical_size.data(),
      (std::byte*)logical_size.data() +
      sizeof(int64_t) * logical_size.size());
  buffer.insert(
      buffer.end(),
      (std::byte*)alloc_stride.data(),
      (std::byte*)alloc_stride.data() +
      sizeof(int64_t) * alloc_stride.size());
  return buffer;
}

int main(int argc, char *argv[]) {
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  constexpr int M = 64, N = 128;
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::ones({M, N}, options);
  at::Tensor at_tv1 = at::ones({M, N}, options);
  at::Tensor at_tv2 = at::empty({M, N}, options);

  std::vector<std::vector<std::byte>> data;
  std::vector<void*> pointers;

  data.push_back(convertTensorToBytes(at_tv0));
  pointers.emplace_back(data.back().data());

  data.push_back(convertTensorToBytes(at_tv1));
  pointers.emplace_back(data.back().data());

  data.push_back(convertTensorToBytes(at_tv2));
  pointers.emplace_back(data.back().data());

  // Initialize
  CUDA_CHECK(cuInit(0));

  CUdevice cuDevice;
  int devID = 0;
  CUDA_CHECK(cuDeviceGet(&cuDevice, devID));

  // Create context
  CUcontext cuContext;
  CUDA_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

  // Create module from binary file (FATBIN)
  CUmodule cuModule;
  std::string filepath = "pointwise_smem.cubin";
  CUDA_CHECK(cuModuleLoad(&cuModule, filepath.c_str()));

  // Get function handle from module
  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, cuModule, "pointwise_smem"));

  constexpr int64_t dynamic_smem_size = 65536;

  // Increase smem limit if it exceeds current limit
  CUDA_CHECK(cuFuncSetAttribute(
        kernel,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        dynamic_smem_size));

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cuLaunchKernel(
      kernel,
      /*gdimx=*/256,
      /*gdimy=*/1,
      /*gdimz=*/1,
      /*bdimx=*/32,
      /*bdimy=*/1,
      /*bdimz=*/1,
      /*smem=*/dynamic_smem_size,
      /*stream=*/0,
      pointers.data(),
      nullptr));

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

  auto at_output = at_tv0 + at_tv1;
  bool result = at_output.allclose(at_tv2);
  std::cout << "result allclose\t" << result << std::endl;

  return 0;
}
