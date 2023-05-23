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
__forceinline__ __device__ void sync() {
  if constexpr (aligned) {
    __syncthreads();
  } else {
    __barrier_sync(0);
  }
}

} // namespace block_sync
