// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda.h>

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
    CUstream stream);

void waitWithCudaBackend(
    Communication* communication,
    SymmetricMemoryHandle* multicast_handle,
    CUstream stream);

} // namespace nvfuser
