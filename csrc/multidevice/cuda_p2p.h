// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/Tensor.h>
#include <cuda.h>
#include <cstdint>

#include "multidevice/ipc_handle.h"

namespace nvfuser {

enum class P2pProtocol : std::uint8_t { Get, Put };

// Prescribed P2P protocol from NVFUSER_ENABLE (p2p_protocol put|get).
P2pProtocol getP2pProtocol();

std::ostream& operator<<(std::ostream& os, P2pProtocol protocol);

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);

void recvWait(const P2pIpcHandle& ipc_handles, CUstream stream);

void sendPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);

void sendWait(const P2pIpcHandle& ipc_handles, CUstream stream);

void postWithCudaBackend(
    Communication* communication,
    at::Tensor input,
    at::Tensor output,
    SymmetricMemoryHandle* symmetric_memory_handle,
    CUstream stream,
    int64_t root);

void waitWithCudaBackend(
    Communication* communication,
    SymmetricMemoryHandle* multicast_handle,
    CUstream stream,
    int64_t root);

struct AlltoallvMetadata {
  at::Tensor send_counts; // CUDA [R]
  at::Tensor recv_counts; // CUDA [R]
  at::Tensor send_offsets; // CUDA [R]
  at::Tensor recv_offsets; // CUDA [R]

  // CPU scalars — upper bounds from the caller, NOT read from GPU.
  // Using upper bounds (instead of exact GPU values) avoids CPU-GPU
  // sync and keeps the data path CUDA-graph-capturable.
  int64_t total_recv = 0; // upper bound on sum(recv_counts)
  int64_t max_recv = 0; // recv buffer first dim
  int64_t max_send_total = 0; // send buffer first dim = sum(send_counts)
  int64_t max_send_bytes = 0; // max per-peer send count (kernel grid X)
  int64_t world_size = 0;
};

AlltoallvMetadata prepareAlltoallvMetadata(
    const at::Tensor& send_counts,
    const std::string& tag);

void alltoallvWithCudaBackend(
    const at::Tensor& send,
    const at::Tensor& recv,
    const AlltoallvMetadata& metadata,
    const at::Tensor& recv_ptrs_gpu, // CUDA [R] int64, from
                                     // SymmetricTensor::remotePointersTensor
    CUstream stream);

void alltoallvBarrier(const std::string& tag);

// Launch ld_reduce kernel (multimem) for NVLS reduce. Used by multicast reduce
// path and by tests. mc_src and dst must be 16-byte aligned; size multiple
// of 16.
void launchMulticastReduceKernel(
    const void* mc_src,
    void* dst,
    size_t size,
    CUstream stream);

} // namespace nvfuser
