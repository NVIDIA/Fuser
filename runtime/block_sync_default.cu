// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Default block synchronization. Just use __barrier_sync
namespace block_sync {

__forceinline__ __device__ void init() {}

// Thread-block synchronization
template <bool aligned>
__forceinline__ __device__ void sync(dim3 block_dim={}) {
  if constexpr (aligned) {
    __syncthreads();
  } else {
    uint32_t num_threads = block_dim.x * block_dim.y * block_dim.z;
    asm volatile("bar.sync 0, %0;" : : "r"(num_threads) : "memory");
  }
}

} // namespace block_sync
