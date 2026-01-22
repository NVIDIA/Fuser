// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda.h>

#include <ATen/core/Tensor.h>
#include <string>
#include <vector>

#include "multidevice/ipc_handle.h"

namespace nvfuser {

enum class P2pProtocol { Get, Put };

P2pProtocol getP2pProtocol();

std::ostream& operator<<(std::ostream& os, P2pProtocol protocol);

// Returns the prescribed P2P protocol based on NVFUSER_ENABLE option
P2pProtocol getP2pProtocol();

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);

void recvWait(const P2pIpcHandle& ipc_handles, CUstream stream);

void sendPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);

void sendWait(const P2pIpcHandle& ipc_handles, CUstream stream);

void postWithCudaBackend(
    Communication* communication,
    at::Tensor input,
    SymmetricMemoryHandle* multicast_handle,
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
  int64_t total_recv = 0;
  int64_t max_recv = 0;
  int64_t max_send_total = 0;
  int64_t max_send_bytes = 0;
  int64_t world_size = 0;
};

AlltoallvMetadata prepareAlltoallvMetadata(
    const at::Tensor& send_counts,
    const std::string& tag);

void alltoallvWithCudaBackend(
    const at::Tensor& send,
    const at::Tensor& recv,
    const AlltoallvMetadata& metadata,
    const std::vector<void*>& recv_ptrs,
    CUstream stream);

void alltoallvBarrier(const std::string& tag);

} // namespace nvfuser
