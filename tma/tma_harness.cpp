#include <cstdio>
#include <iostream>
#include <torch/torch.h>
#include <any>

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

template <typename T>
std::vector<std::byte> convertAnyToBytes(const std::any& a) {
  return std::vector<std::byte>(
      (const std::byte*)&a.as<T>(), (const std::byte*)(&a.as<T>() + 1));
}

std::vector<std::byte> convertIntToBytes(int a) {
  return std::vector<std::byte>((std::byte*)&v32, (std::byte*)&v32 + 4);
}

int main(int argc, char *argv[]) {
  constexpr size_t W_global = 1024; // Width of tensor (in # elements)
  constexpr size_t H_global = 1024; // Height of tensor (in # elements)

  constexpr int SMEM_W = 32;     // Width of shared memory buffer (in # elements)
  constexpr int SMEM_H = 8;      // Height of shared memory buffer (in # elements)
                                 
  CUtensorMap tma_desc{};
  CUtensorMapDataType dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32;
  auto rank = 2;
  uint64_t size[rank] = {W_global, H_global};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {W_global * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_W, SMEM_H};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};
  // Interleave patterns are sometimes used to accelerate loading of values that
  // are less than 4 bytes long.
  CUtensorMapInterleave interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
  // Swizzling can be used to avoid shared memory bank conflicts.
  CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMapL2promotion l2_promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
  // Any element that is outside of bounds will be set to zero by the TMA transfer.
  CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  // Create the tensor descriptor.
  CUDA_CHECK(cuTensorMapEncodeTiled(
      &tma_desc,    // CUtensorMap *tensorMap,
      dtype,        // CUtensorMapDataType tensorDataType,
      rank,         // cuuint32_t tensorRank,
      tensor,       // void *globalAddress,
      size,         // const cuuint64_t *globalDim,
      stride,       // const cuuint64_t *globalStrides,
      box_size,     // const cuuint32_t *boxDim,
      elem_stride,  // const cuuint32_t *elementStrides,
      interleave,   // CUtensorMapInterleave interleave,
      swizzle,      // CUtensorMapSwizzle swizzle,
      l2_promotion, // CUtensorMapL2promotion l2Promotion,
      oob_fill      // CUtensorMapFloatOOBfill oobFill);
    ));

  // Convert CuTensorMap to Opaque PolymorphicValue
  // alignas(64) uint64_t opaque[16]
  // map CuTensorMap to std::vector<std::byte>
  
  std::vector<std::vector<std::byte>> data;
  std::vector<void*> pointers;

  data.push_back(convertAnyToBytes(tma_desc));
  pointers.emplace_back(data.back().data());

  data.push_back(convertIntToBytes(/*a=*/0));
  pointers.emplace_back(data.back().data());

  data.push_back(convertIntToBytes(/*a=*/1));
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
  CUDA_CHECK(cuModuleGetFunction(&kernel, cuModule, "tma_kernel"));

  /*
  constexpr int64_t dynamic_smem_size = 65536;
  // Increase smem limit if it exceeds current limit
  CUDA_CHECK(cuFuncSetAttribute(
        kernel,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        dynamic_smem_size));
  */

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

  // cuda-runtime-api reference
  // dim3 grid(1);
  // dim3 block(128);
  // kernel<<<grid, block>>>(tma_desc, 0, 0);
  CUDA_CHECK(cuLaunchKernel(
      kernel,
      /*gdimx=*/1,
      /*gdimy=*/1,
      /*gdimz=*/1,
      /*bdimx=*/128,
      /*bdimy=*/1,
      /*bdimz=*/1,
      /*smem=*/0,
      /*stream=*/0,
      pointers.data(),
      nullptr));

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

  return 0;
}
