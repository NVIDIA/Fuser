// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda_utils.h>
#include <tests/cpp/cluster_runtime_test/cluster_test_helper.h>

namespace nvfuser {
// copied from runtime/memory.cu to avoid too many includes
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__device__ inline unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }\n"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));

  return smem_ptr_uint;
}
namespace Hopper {
__device__ inline bool electSync(const uint32_t& membermask) {
  uint32_t is_elected;
  asm volatile(
      "{\n\t .reg .pred P_OUT; \n\t"
      "elect.sync _|P_OUT, %1;\n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(is_elected)
      : "r"(membermask)
      :);
  return static_cast<bool>(is_elected);
}
} // namespace Hopper
#endif
// must include mbarrier.cu before cluster.cu
// since mbarrier is used in cluster.cu
// clang-format off
#include <runtime/mbarrier.cu>
#include <runtime/cluster.cu>
// clang-format on
template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE>
__global__ void storeSharedRemoteTestKernel(T* input, T* output) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // load from input to register
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  T value = input[global_tid];

  // Shared memory for data exchange between CTAs in cluster
  __shared__ T shared_data[BLOCK_SIZE];
  __shared__ uint64_t mbarrier;
  uint32_t mbarrier_addr = toSmem(&mbarrier);

  // Initialize barrier
  if (threadIdx.x == 0) {
    mbarrier::init(mbarrier_addr, 1);
  }
  nvf::cluster::clusterSync();

  // Each thread writes to its peer CTA's shared memory
  auto cluster_id = nvf::cluster::blockIdInCluster().x;
  if (threadIdx.x == 0) {
    uint32_t expected_bytes = BLOCK_SIZE * sizeof(T);
    mbarrier::arriveExpectTX(mbarrier_addr, expected_bytes);
  }
  uint32_t peer_cta_rank_in_cluster = (cluster_id + 1) % CLUSTER_SIZE;
  uint32_t buffer_addr = toSmem(&shared_data[threadIdx.x]);
  nvf::cluster::storeSharedRemote<T>(
      value, buffer_addr, mbarrier_addr, peer_cta_rank_in_cluster);

  // wait for all writings are done, then write from shared memory to output
  mbarrier::waitParity(mbarrier_addr, 0);
  output[global_tid] = shared_data[threadIdx.x];
#endif
}

template <typename T>
struct AddOp {
  __device__ __forceinline__ void operator()(T& a, const T& b) const {
    a += b;
  }
};

// Reduce BLOCK_SIZE x CLUSTER_SIZE values across a cluster of CLUSTER_SIZE CTAs
template <
    typename T,
    int BLOCK_SIZE,
    int CLUSTER_SIZE,
    int WARPS_PER_BLOCK,
    bool is_all_reduce>
__global__ void clusterReduceTestKernel(T* input, T* output) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  T value = input[global_tid];

  __shared__ uint64_t mbarrier;
  __shared__ T reduction_buffer[CLUSTER_SIZE * WARPS_PER_BLOCK];
  uint32_t mbarrier_addr = toSmem(&mbarrier);

  if (threadIdx.x == 0) {
    mbarrier::init(mbarrier_addr, 1);
  }
  nvf::cluster::clusterSync();

  T result;
  nvf::cluster::
      clusterReduce<CLUSTER_SIZE, WARPS_PER_BLOCK, is_all_reduce, T, AddOp<T>>(
          result,
          value,
          static_cast<T>(0),
          mbarrier_addr,
          reduction_buffer,
          AddOp<T>());

  if constexpr (is_all_reduce) {
    // All-reduce: each thread writes the result to its corresponding output
    // element
    output[global_tid] = result;
  } else {
    // Reduce: only the first thread of the last block writes the scalar result
    constexpr uint32_t last_block_rank = CLUSTER_SIZE - 1;
    uint32_t my_block_rank = nvf::cluster::blockIdInCluster().x;
    if (my_block_rank == last_block_rank && threadIdx.x == 0) {
      output[0] = result;
    }
  }
#endif
}

//============================================================================
// Launch function implementations
//============================================================================
template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE>
void launchStoreSharedRemoteTestKernel(T* input, T* output) {
  // Configure cluster launch
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(CLUSTER_SIZE, 1, 1);
  config.blockDim = dim3(BLOCK_SIZE, 1, 1);

  // Set cluster dimensions
  cudaLaunchAttribute cluster_attr = {};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = CLUSTER_SIZE;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;

  // Launch kernel with cluster configuration
  NVFUSER_CUDA_RT_SAFE_CALL(cudaLaunchKernelEx(
      &config,
      storeSharedRemoteTestKernel<T, BLOCK_SIZE, CLUSTER_SIZE>,
      input,
      output));
}

template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE, bool is_all_reduce>
void launchClusterReduceTestKernel(T* input, T* output) {
  constexpr int WARPS_PER_BLOCK = (BLOCK_SIZE + 31) / 32;

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(CLUSTER_SIZE, 1, 1);
  config.blockDim = dim3(BLOCK_SIZE, 1, 1);

  cudaLaunchAttribute cluster_attr = {};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = CLUSTER_SIZE;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;

  // Use the unified kernel for both all-reduce and reduce cases
  NVFUSER_CUDA_RT_SAFE_CALL(cudaLaunchKernelEx(
      &config,
      clusterReduceTestKernel<
          T,
          BLOCK_SIZE,
          CLUSTER_SIZE,
          WARPS_PER_BLOCK,
          is_all_reduce>,
      input,
      output));
}

// explicit template instantiations
template void launchStoreSharedRemoteTestKernel<float, 32, 2>(
    float* input,
    float* output);

template void launchStoreSharedRemoteTestKernel<double, 32, 2>(
    double* input,
    double* output);

template void launchClusterReduceTestKernel<float, 128, 2, true>(
    float* input,
    float* output);

template void launchClusterReduceTestKernel<double, 128, 2, true>(
    double* input,
    double* output);

template void launchClusterReduceTestKernel<float, 128, 2, false>(
    float* input,
    float* output);

template void launchClusterReduceTestKernel<double, 128, 2, false>(
    double* input,
    double* output);
} // namespace nvfuser
