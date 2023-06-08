#include <cuda.h>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// struct TensorMap {
//   alignas(64) uint64_t opaque[16];
// };
using TensorMap = CUtensorMap;

__device__ inline unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));
  return smem_ptr_uint;
}

__device__ inline void cpAsyncBulkCommit() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("cp.async.bulk.commit_group;");
#else
  assert(false);
#endif
}

template <int keep_stages>
__device__ inline void cpAsyncBulkPartialReadBarrier() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("cp.async.bulk.wait_group.read %0;"
               :
               : "n"(keep_stages)
               : "memory");
#else
  assert(false);
#endif
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const TensorMap& dest,
    int32_t crd0,
    uint32_t smem_addr) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // TODO: remove this cast?
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&dest);
  asm volatile(
      "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_addr), "r"(crd0)
      : "memory");
  // TODO: this is not correct, and is only a temporary solution for the
  // build-out stage
  cpAsyncBulkCommit();
  cpAsyncBulkPartialReadBarrier<0>();
#else
  assert(false);
#endif
}

__global__ void kernel(float* output, const __grid_constant__ TensorMap tensormap) {
  __shared__ float numbers[32];
  for (int i = 0; i < 32; ++i) {
    numbers[i] = blockIdx.x * 32 + i;
  }

  cpAsyncBulkTensorTileS2G(tensormap, blockIdx.x, toSmem(numbers));
}

std::string getErrorMsg(CUresult error) {
  const char *error_name, *error_string;
  cuGetErrorName(error, &error_name);
  cuGetErrorString(error, &error_string);
  return std::string(error_name) + ": " + error_string;
}

CUtensorMap getTensorMap(float* ptr, size_t N) {
  CUtensorMap tmap;
  unsigned int dtype_size = sizeof(float);

  std::vector<uint64_t> evaluated_gmem_shape{N * 32};
  std::vector<uint64_t> evaluated_gmem_strides{dtype_size};
  std::vector<uint32_t> evaluated_box_shape{32};
  std::vector<uint32_t> evaluated_box_strides{1};

  CUresult result = cuTensorMapEncodeTiled(
      &tmap,
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      1,
      ptr,
      evaluated_gmem_shape.data(),
      evaluated_gmem_strides.data() + 1, // gmem_strides[0] implicitly 1
      evaluated_box_shape.data(),
      evaluated_box_strides.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (result != CUDA_SUCCESS) {
    std::cout << getErrorMsg(result) << std::endl;
  }
  return tmap;
}

int main() {
  float* output;
  size_t N = 32;
  cudaMalloc(&output, N * 32 * sizeof(float));

  kernel<<<N, 1>>>(output, getTensorMap(output, N));
  cudaDeviceSynchronize();

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
  }

  float result[N * 32];
  cudaMemcpy(result, output, N * 32 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N * 32; ++i) {
    std::cout << result[i] << " ";
  }

  return 0;
}