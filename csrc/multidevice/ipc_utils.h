// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nvfuser {

// Helper functions for serializing data to bytes for TCP store
template <typename T>
std::vector<uint8_t> toBytes(const T& data) {
  return std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(&data),
      reinterpret_cast<const uint8_t*>(&data) + sizeof(T));
}

template <typename T>
const T& fromBytes(const std::vector<uint8_t>& bytes) {
  return *reinterpret_cast<const T*>(bytes.data());
}

// IPC Utils for sharing file descriptors

enum class MulticastProtocol { Memcpy, Multimem, BatchMemcpy };

MulticastProtocol getMulticastProtocol();

// Creates a listening Unix domain socket bound to path.
// If path starts with '@', it uses the abstract namespace (replaced with \0).
// Returns the socket file descriptor.
int createIpcSocket(const std::string& path);

// Connects to the Unix domain socket at path and sends the file descriptor fd.
// Optionally sends header_data of size header_len along with the FD.
void sendFd(
    const std::string& path,
    int fd,
    const void* header_data = nullptr,
    size_t header_len = 0);

// Accepts a connection on the listening socket_fd and receives a file
// descriptor. Optionally receives header_data of size header_len. Returns the
// received file descriptor.
int recvFd(int socket_fd, void* header_data = nullptr, size_t header_len = 0);

} // namespace nvfuser
