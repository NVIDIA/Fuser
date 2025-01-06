// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

namespace Hopper {

// The optional .relaxed qualifier on barrier.cluster.arrive specifies that
// there are no memory ordering and visibility guarantees provided for the
// memory accesses performed prior to barrier.cluster.arrive.
void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
}

// A thread arrives at barrier but it does not have to wait for threads in other
// participating warps.
void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;\n" : :);
}

// A thread waits for all non-exited threads of the cluster to perform
// cluster_arrive.
void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

// Synchronize threads in cluster
void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

// Returns the dim3 grid size in terms of number of clusters.
dim3 cluster_grid_dims() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%nclusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%nclusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster rank in the grid.
dim3 cluster_id_in_grid() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the relative dim3 block rank local to the cluster.
dim3 block_id_in_cluster() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Returns the dim3 cluster shape.
dim3 cluster_shape() {
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
}

// Get 1D ctaid in a cluster.
uint32_t block_rank_in_cluster() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
}

// Set the destination block-ID in cluster for a given SMEM Address
uint32_t map_shared_rank(uint32_t smemAddr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
               : "=r"(result)
               : "r"(smemAddr), "r"(rank));
  return result;
}

} // namespace Hopper

#endif // Arch 90
