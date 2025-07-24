// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Basically just blockDim, but wrapped as a struct so that we have a mechanism
// to know at compile time that whether we are just using blockDim or some
// custom value. For a kernel without warp specialization, we just use blockDim,
// but for a kernel with warp specialization, we use a custom block_dim whose
// dimension are the dimensions of the compute warps.
struct DefaultBlockDim {
  const uint32_t x, y, z;
  __device__ DefaultBlockDim() : x(blockDim.x), y(blockDim.y), z(blockDim.z) {}
  __device__ operator dim3() const {
    return blockDim;
  }
};

// Default block synchronization. Just use __barrier_sync
namespace block_sync {

__forceinline__ __device__ void init() {}

// Thread-block synchronization
// barrier 0 is reserved for the default block synchronization
template <bool aligned, typename BlockDimT>
__forceinline__ __device__ void sync(
    BlockDimT block_dim,
    uint32_t barrier_id = 1) {
  if constexpr (aligned) {
    __syncthreads();
  } else if constexpr (std::is_same_v<BlockDimT, DefaultBlockDim>) {
    __barrier_sync(0);
  } else {
    uint32_t num_threads = block_dim.x * block_dim.y * block_dim.z;
    asm volatile("bar.sync %0, %1;"
                 :
                 : "r"(barrier_id), "r"(num_threads)
                 : "memory");
  }
}

} // namespace block_sync
