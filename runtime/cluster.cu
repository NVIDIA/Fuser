// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

// The optional .relaxed qualifier on barrier.cluster.arrive specifies that
// there are no memory ordering and visibility guarantees provided for the
// memory accesses performed prior to barrier.cluster.arrive.
void clusterArriveRelaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" : :);
}

// A thread arrives at barrier but it does not have to wait for threads in other
// participating warps.
void clusterArrive() {
  asm volatile("barrier.cluster.arrive.aligned;" : :);
}

// A thread waits for all non-exited threads of the cluster to perform
// cluster_arrive.
void clusterWait() {
  asm volatile("barrier.cluster.wait.aligned;" : :);
}

// Synchronize threads in cluster
void clusterSync() {
  clusterArrive();
  clusterWait();
}

// Returns the dim3 grid size in terms of number of clusters.
dim3 clusterGridDims() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%nclusterid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%nclusterid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster rank in the grid.
dim3 clusterIdInGrid() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Returns the relative dim3 block rank local to the cluster.
dim3 blockIdInCluster() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster shape.
dim3 clusterShape() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(z) :);
  return {x, y, z};
}

// Get 1D ctaid in a cluster.
uint32_t blockRankInCluster() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank) :);
  return rank;
}

// Set the destination block-ID in cluster for a given SMEM Address
uint32_t mapSharedRank(uint32_t smemAddr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;"
               : "=r"(result)
               : "r"(smemAddr), "r"(rank));
  return result;
}

__device__ __forceinline__ uint32_t blockIdInClusterDimx() {
  uint32_t my_block_rank;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(my_block_rank));
  return my_block_rank;
}

__device__ __forceinline__ uint32_t mapSharedRank(uint32_t smem_addr, uint32_t dst_cta_rank) {
  uint32_t dsmem_addr;
  asm volatile(
      "mapa.shared::cluster.u32 %0, %1, %2;\n"
      : "=r"(dsmem_addr)
      : "r"(smem_addr), "r"(dst_cta_rank));
  return dsmem_addr;
}
// Async store operations - template specializations for different data types
template<typename T>
__device__ __forceinline__ void store_shared_remote(T value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank);

// Specialization for int (32-bit)
template<>
__device__ __forceinline__ void store_shared_remote<int>(int value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank) {
uint32_t dsmem_addr = mapSharedRank(smem_addr, dst_cta_rank);
uint32_t remote_barrier_addr = mapSharedRank(mbarrier_addr, dst_cta_rank);
asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];"
             : : "r"(dsmem_addr), "r"(value), "r"(remote_barrier_addr));
}

// Specialization for float (32-bit)
template<>
__device__ __forceinline__ void store_shared_remote<float>(float value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank) {
uint32_t dsmem_addr = mapSharedRank(smem_addr, dst_cta_rank);
uint32_t remote_barrier_addr = mapSharedRank(mbarrier_addr, dst_cta_rank);
asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
             : : "r"(dsmem_addr), "f"(value), "r"(remote_barrier_addr));
}

// ========== Kernel ==========

// Warp reduction using inline PTX
template<typename T, typename Func>
__device__ __forceinline__ T warp_reduce(T val, Func reduction_op) {
  T reduce_val = val;
for (int offset = 16; offset > 0; offset /= 2) {
  reduction_op(reduce_val, __shfl_xor_sync(0xffffffff, val, offset));
}
return reduce_val;
}

// Cluster reduction in x direction
// blockIdx --> clusterd reduction dimension
// blockIdx.y --> iteration dimension
// blockIdx.z --> Not Used
// CLUSTER_SIZE equals gridDim.x, known at compile time
// Algorithm:
// 1. Each warp does a warp reduction
// 2. All warps async store reduction results to its and clustered CTA's shared memories
// 3. All warps read from its CTA's shared memory and do a warp reduction
template<int CLUSTER_SIZE, typename T, typename Func>
__device__ __forceinline__ void clusterReduce(T& res, T inp, Func reduction_op, T* reduction_buffer)
{
// assume only cluster in x direction
uint32_t my_block_rank = blockIdx.x;
const int warps_per_block = blockDim.x / 32;
uint32_t lane_idx = threadIdx.x % 32;
uint32_t warp_idx = threadIdx.x / 32;

// Initialize barrier and buffers
// barrier for writing to distributed shared memory using st.async
__shared__ uint64_t barrier_storage;
uint32_t barrier_smem_addr = toSmem(&barrier_storage);
if (threadIdx.x == 0) {
  mbarrier::init(barrier_smem_addr, 1);
}
__syncthreads();
clusterSync();

T thread_val = inp;

// 1. Perform warp reduction
T warp_sum = warpReduce(thread_val, reduction_op);

// 2. All warps store their results to distributed shared memory
// Each warp uses N threads to write to N CTAs, e.g. thread-i write to CTA-i
// Buffer layout: reduction_buffer[CLUSTER_SIZE][warps_per_block]
uint64_t arrival_token;
if (threadIdx.x == 0) {
  uint32_t expected_bytes = warps_per_block * CLUSTER_SIZE * sizeof(T);
  arrival_token = mbarrier::arriveExpectTX(barrier_smem_addr, expected_bytes);
}
if (lane_idx < CLUSTER_SIZE) {
  uint32_t peer_cta_rank_in_cluster = lane_idx;
  uint32_t buffer_offset = my_block_rank * warps_per_block + warp_idx;
  uint32_t buffer_addr = toSmem(&reduction_buffer[buffer_offset]);
  store_shared_remote<T>(
    warp_sum,
    buffer_addr,
    barrier_smem_addr, 
    peer_cta_rank_in_cluster
  );
}
mbarrier::wait(barrier_smem_addr, arrival_token);

// 3. Each CTA has a copy of the warp reduction results from all warps in the cluster
//    Finish reduction with a warp reduction
T block_reduce_val = reduction_buffer[lane_idx];
int num_iter = (warps_per_block * CLUSTER_SIZE + 31) / 32;
for(int i = 0; i < num_iter; i++){
  int idx = lane_idx + i * 32;
  if(idx < CLUSTER_SIZE * warps_per_block){
    block_reduce_val += reduction_buffer[idx];
  }
}
// 4. Each CTA performs a warp reduction on its shared memory
// Get final result using warp reduction
res = warp_reduce_sum(block_reduce_val);
}
#endif // Arch 90
