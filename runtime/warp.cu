// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace warp {

template <typename T>
__device__ __forceinline__ T shfl_xor(T var, int laneMask, int width = 32) {
  return __shfl_xor_sync(0xffffffff, var, laneMask, width);
}
template <typename T>
__device__ __forceinline__ std::complex<T> shfl_xor(
    std::complex<T> var,
    int laneMask,
    int width = 32) {
  T real = __shfl_xor_sync(0xffffffff, var.real(), laneMask, width);
  T imag = __shfl_xor_sync(0xffffffff, var.imag(), laneMask, width);
  return std::complex<T>(real, imag);
}

template <typename T>
__device__ __forceinline__ T
warp_broadcast(T var, int srcLane, int width = 32) {
  return __shfl_sync(0xffffffff, var, srcLane, width);
}

template <typename T>
__device__ __forceinline__ std::complex<T> warp_broadcast(
    std::complex<T> var,
    int width = 32) {
  T real = __shfl_xor_sync(0xffffffff, var.real(), srcLane, width);
  T imag = __shfl_xor_sync(0xffffffff, var.imag(), srcLane, width);
  return std::complex<T>(real, imag);
}

template <
    bool SINGLE_WARP,
    bool Aligned,
    bool Padded,
    typename T,
    typename Func>
__device__ void warpReduceTIDX(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  constexpr int WARP_SIZE = 32;

  T reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  // Register usage is reduced when Padded is true due to the elimination of the
  // if-statement.
  unsigned int warp_idx = threadIdx.x / WARP_SIZE;
  unsigned int lane_idx = threadIdx.x % WARP_SIZE;
  unsigned int reduction_size = blockDim.x;
  unsigned int num_of_warps = (reduction_size + WARP_SIZE - 1) / WARP_SIZE;
  int hoist_idx = reduction_size - lane_idx - warp_idx * WARP_SIZE;
  for (int i = 16; i >= 1; i /= 2) {
    T shuffled_value = shfl_xor(reduce_val, i, WARP_SIZE);
    if (Padded || i < hoist_idx) {
      reduction_op(reduce_val, shuffled_value);
    }
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (!SINGLE_WARP) {
    unsigned int reduce_group_id = threadIdx.z * blockDim.y + threadIdx.y;
    bool is_warp_head = lane_idx == 0;

    unsigned int smem_offset = reduce_group_id * num_of_warps;

    block_sync::sync<Aligned>();

    if (is_warp_head) {
      shared_mem[smem_offset + warp_idx] = reduce_val;
    }

    block_sync::sync<Aligned>();

    if (warp_idx == 0) {
      // This assumes num_of_warps will be < 32, meaning < 1024 threads.
      //  Should be true for long enough.
      assert(num_of_warps <= 32);

      reduce_val = lane_idx < num_of_warps ? shared_mem[smem_offset + lane_idx]
                                           : init_val;

      // Reduce within warp 0
      for (int i = 16; i >= 1; i /= 2) {
        reduction_op(reduce_val, shfl_xor(reduce_val, i, 32));
      }
    }

    if (is_warp_head) {
      reduction_op(out, reduce_val);
    }
    // needs sync, otherwise other warps may access shared memory before this
    // reduction is done.
    block_sync::sync<Aligned>();
  } else {
    if (!Padded) {
      reduce_val = warp_broadcast(reduce_val, 0);
    }
    reduction_op(out, reduce_val);
  }
}

} // namespace warp
