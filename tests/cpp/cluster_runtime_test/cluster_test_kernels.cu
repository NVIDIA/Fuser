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

namespace nvf {
__device__ inline unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }\n"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));

  return smem_ptr_uint;
}
#include <runtime/mbarrier.cu>
} // namespace nvf
#include <runtime/cluster.cu>

template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE>
__global__ void storeSharedRemoteTestKernel(T* input, T* output) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // load from input to register
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  T value = input[global_tid];

  // Shared memory for data exchange between CTAs in cluster
  __shared__ T shared_data[BLOCK_SIZE];
  __shared__ uint64_t mbarrier;
  uint32_t mbarrier_addr = nvf::toSmem(&mbarrier);

  // Initialize barrier
  if (threadIdx.x == 0) {
    nvf::mbarrier::init(mbarrier_addr, 1);
  }
  nvf::cluster::clusterSync();

  // Each thread writes to its peer CTA's shared memory
  auto cluster_id = nvf::cluster::blockIdInCluster().x;
  if (threadIdx.x == 0) {
    uint32_t expected_bytes = BLOCK_SIZE * sizeof(T);
    nvf::mbarrier::arriveExpectTX(mbarrier_addr, expected_bytes);
  }
  uint32_t peer_cta_rank_in_cluster = (cluster_id + 1) % CLUSTER_SIZE;
  uint32_t buffer_addr = nvf::toSmem(&shared_data[threadIdx.x]);
  nvf::cluster::storeSharedRemote<T>(
      value, buffer_addr, mbarrier_addr, peer_cta_rank_in_cluster);

  // wait for all writings are done, then write from shared memory to output
  nvf::mbarrier::waitParity(mbarrier_addr, 0);
  output[global_tid] = shared_data[threadIdx.x];
#endif
}

//============================================================================
// Launch function implementations
//============================================================================
template <typename T, int BLOCK_SIZE, int CLUSTER_SIZE>
void launchStoreSharedRemoteTestKernel(
    cudaStream_t stream,
    T* input,
    T* output,
    int cluster_x,
    int cluster_y,
    int cluster_z) {
  // Configure cluster launch
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(CLUSTER_SIZE, 1, 1);
  config.blockDim = dim3(BLOCK_SIZE, 1, 1);
  config.stream = stream;

  // Set cluster dimensions
  cudaLaunchAttribute cluster_attr = {};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = cluster_x;
  cluster_attr.val.clusterDim.y = cluster_y;
  cluster_attr.val.clusterDim.z = cluster_z;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;

  // Launch kernel with cluster configuration
  NVFUSER_CUDA_RT_SAFE_CALL(cudaLaunchKernelEx(
      &config,
      storeSharedRemoteTestKernel<T, BLOCK_SIZE, CLUSTER_SIZE>,
      input,
      output));
}

// explicit template instantiations
template void launchStoreSharedRemoteTestKernel<float, 32, 2>(
    cudaStream_t stream,
    float* input,
    float* output,
    int cluster_x,
    int cluster_y,
    int cluster_z);

template void launchStoreSharedRemoteTestKernel<double, 32, 2>(
    cudaStream_t stream,
    double* input,
    double* output,
    int cluster_x,
    int cluster_y,
    int cluster_z);
} // namespace nvfuser
