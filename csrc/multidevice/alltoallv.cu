// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

extern "C" __global__ void alltoallv_kernel(
    const unsigned char* send,
    const unsigned long long* recv_ptrs,
    const long long* send_offsets,
    const long long* send_sizes,
    const long long* recv_offsets,
    long long world_size,
    long long elem_size,
    long long max_send_bytes) {
  const long long peer = static_cast<long long>(blockIdx.y);
  if (peer >= world_size) {
    return;
  }
  const long long bytes = send_sizes[peer] * elem_size;
  if (bytes == 0) {
    return;
  }
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= bytes) {
    return;
  }
  const long long send_byte_offset = send_offsets[peer] * elem_size + idx;
  const long long recv_byte_offset = recv_offsets[peer] * elem_size + idx;
  auto* dst = reinterpret_cast<unsigned char*>(
      static_cast<unsigned long long>(recv_ptrs[peer]));
  dst[recv_byte_offset] = send[send_byte_offset];
}

