// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace nvf {
namespace cluster {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

// The optional .relaxed qualifier on barrier.cluster.arrive specifies that
// there are no memory ordering and visibility guarantees provided for the
// memory accesses performed prior to barrier.cluster.arrive.
__device__ void clusterArriveRelaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" : :);
}

// A thread arrives at barrier but it does not have to wait for threads in other
// participating warps.
__device__ void clusterArrive() {
  asm volatile("barrier.cluster.arrive.aligned;" : :);
}

// A thread waits for all non-exited threads of the cluster to perform
// cluster_arrive.
__device__ void clusterWait() {
  asm volatile("barrier.cluster.wait.aligned;" : :);
}

// Synchronize threads in cluster
__device__ void clusterSync() {
  clusterArrive();
  clusterWait();
}

// Returns the dim3 grid size in terms of number of clusters.
__device__ dim3 clusterGridDims() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%nclusterid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%nclusterid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster rank in the grid.
__device__ dim3 clusterIdInGrid() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Returns the relative dim3 block rank local to the cluster.
__device__ dim3 blockIdInCluster() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster shape.
__device__ dim3 clusterShape() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Get 1D ctaid in a cluster.
__device__ uint32_t blockRankInCluster() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank) :);
  return rank;
}

// Set the destination block-ID in cluster for a given SMEM Address
__device__ uint32_t mapSharedRank(uint32_t smemAddr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;"
               : "=r"(result)
               : "r"(smemAddr), "r"(rank));
  return result;
}

// Async store operations - only supports float and double types
template <typename T>
__device__ __forceinline__ void storeSharedRemote(
    T value,
    uint32_t smem_addr,
    uint32_t mbarrier_addr,
    uint32_t dst_cta_rank) {
  static_assert(
      sizeof(T) == 0, "storeSharedRemote only supports float and double types");
}

// Specialization for float (32-bit)
template <>
__device__ __forceinline__ void storeSharedRemote<float>(
    float value,
    uint32_t smem_addr,
    uint32_t mbarrier_addr,
    uint32_t dst_cta_rank) {
  uint32_t dsmem_addr = mapSharedRank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = mapSharedRank(mbarrier_addr, dst_cta_rank);
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, "
      "[%2];"
      :
      : "r"(dsmem_addr), "f"(value), "r"(remote_barrier_addr));
}

// Specialization for double (64-bit)
template <>
__device__ __forceinline__ void storeSharedRemote<double>(
    double value,
    uint32_t smem_addr,
    uint32_t mbarrier_addr,
    uint32_t dst_cta_rank) {
  uint32_t dsmem_addr = mapSharedRank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = mapSharedRank(mbarrier_addr, dst_cta_rank);
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.f64 [%0], %1, "
      "[%2];"
      :
      : "r"(dsmem_addr), "d"(value), "r"(remote_barrier_addr));
}

template <typename T, typename Func>
__device__ __forceinline__ T warpReduce(T val, Func reduction_op) {
  T reduce_val = val;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    reduction_op(reduce_val, __shfl_xor_sync(0xffffffff, reduce_val, offset));
  }
  return reduce_val;
}

// Cluster reduction in x direction
// Algorithm:
// 1. Each warp does a warp reduction
// 2. All warps async store reduction results to its and clustered CTA's shared
// memories
// 3. All warps read from its CTA's shared memory and do a warp reduction
// TODO: we can represent this cluster reduction in fusion IR after we have new
// parallel types to represent warp reduction.
template <int CLUSTER_SIZE, int WARPS_PER_BLOCK, typename T, typename Func>
__device__ __forceinline__ void clusterReduce(
    T& res,
    T inp,
    T init,
    uint32_t barrier_smem_addr,
    T* reduction_buffer,
    Func reduction_op) {
  uint32_t my_block_rank = blockIdInCluster().x;
  uint32_t lane_idx = threadIdx.x % 32;
  uint32_t warp_idx = threadIdx.x / 32;

  T thread_val = inp;

  // 1. Perform warp reduction
  T warp_sum = warpReduce(thread_val, reduction_op);

  // 2. All warps store their results to distributed shared memory
  // Each warp uses N threads to write to N CTAs, e.g. thread-i write to CTA-i
  // Buffer layout: reduction_buffer[CLUSTER_SIZE][WARPS_PER_BLOCK]
  if (warp_idx == 0 && Hopper::electSync(4294967295U)) {
    uint32_t expected_bytes = WARPS_PER_BLOCK * CLUSTER_SIZE * sizeof(T);
    mbarrier::arriveExpectTX(barrier_smem_addr, expected_bytes);
  }
  if (lane_idx < CLUSTER_SIZE) {
    uint32_t peer_cta_rank_in_cluster = lane_idx;
    uint32_t buffer_offset = my_block_rank * WARPS_PER_BLOCK + warp_idx;
    uint32_t buffer_addr = toSmem(&reduction_buffer[buffer_offset]);
    storeSharedRemote<T>(
        warp_sum, buffer_addr, barrier_smem_addr, peer_cta_rank_in_cluster);
  }

  // mbarrier is not repeatedly used, parity phase is set to 0. Otherwise,
  // should flip parity phase, e.g. when used in persistent CTA kernels.
  mbarrier::waitParity(barrier_smem_addr, 0);

  // 3. Each CTA has a copy of the warp reduction results from all warps in the
  // cluster
  // Finish reduction with a warp reduction
  T block_reduce_val = init;
  constexpr int num_iter = (WARPS_PER_BLOCK * CLUSTER_SIZE + 31) / 32;
#pragma unroll
  for (int i = 0; i < num_iter; i++) {
    int idx = lane_idx + i * 32;
    if (idx < CLUSTER_SIZE * WARPS_PER_BLOCK) {
      reduction_op(block_reduce_val, reduction_buffer[idx]);
    }
  }
  // 4. Each CTA performs a warp reduction on its shared memory
  // Get final result using warp reduction
  res = warpReduce(block_reduce_val, reduction_op);
}
#endif // Arch 90
} // namespace cluster
} // namespace nvf
