// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <cuda.h>
#include <multidevice/ipc_handle.h>

namespace nvfuser {

enum class P2pProtocol { Get, Put };

inline std::ostream& operator<<(std::ostream& os, P2pProtocol protocol) {
  switch (protocol) {
    case P2pProtocol::Get:
      os << "Get";
      break;
    case P2pProtocol::Put:
      os << "Put";
      break;
    default:
      os << "Unknown";
      break;
  }
  return os;
}

// Returns the prescribed P2P protocol based on NVFUSER_ENABLE option
P2pProtocol getPrescribedP2pProtocol();

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);

void recvWait(const P2pIpcHandle& ipc_handles, CUstream stream);

void sendPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);

void sendWait(const P2pIpcHandle& ipc_handles, CUstream stream);

} // namespace nvfuser
