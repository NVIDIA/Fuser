// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace warp {

template <typename T>
__device__ __forceinline__ T
shfl_xor(unsigned int participating_mask, T var, int laneMask, int width = 32) {
  return __shfl_xor_sync(participating_mask, var, laneMask, width);
}
template <typename T>
__device__ __forceinline__ std::complex<T> shfl_xor(
    unsigned int participating_mask,
    std::complex<T> var,
    int laneMask,
    int width = 32) {
  T real = __shfl_xor_sync(participating_mask, var.real(), laneMask, width);
  T imag = __shfl_xor_sync(participating_mask, var.imag(), laneMask, width);
  return std::complex<T>(real, imag);
}

template <typename T>
__device__ __forceinline__ T warp_broadcast(
    unsigned int participating_mask,
    T var,
    int srcLane,
    int width = 32) {
  return __shfl_sync(participating_mask, var, srcLane, width);
}

template <typename T>
__device__ __forceinline__ std::complex<T> warp_broadcast(
    unsigned int participating_mask,
    std::complex<T> var,
    int srcLane,
    int width = 32) {
  T real = __shfl_sync(participating_mask, var.real(), srcLane, width);
  T imag = __shfl_sync(participating_mask, var.imag(), srcLane, width);
  return std::complex<T>(real, imag);
}

// This function is used when Padded = false, a spceial overload for Padded =
// true is also implemented.
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
  // unpadded version only support 1D thread block.
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);

  T reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  unsigned int warp_idx = threadIdx.x / WARP_SIZE;
  unsigned int lane_idx = threadIdx.x % WARP_SIZE;
  unsigned int reduction_size = blockDim.x;
  unsigned int num_of_warps = (reduction_size + WARP_SIZE - 1) / WARP_SIZE;
  bool is_last_warp = warp_idx == num_of_warps - 1;
  // The mask sets the participating threads, e.g. for a block with 35 threads,
  // the active threads of the last warp is [32, 33, 34], the mask is
  // [00,...,00111].
  const unsigned int mask =
      is_last_warp ? __ballot_sync(0xffffffff, true) : 0xffffffff;

  // The shfl offset begins at np2, which represents the greatest power of 2
  // less than valid_lanes. For instance, consider a block of 35 threads: warp_0
  // has 32 active threads, making np2 equal to 16; meanwhile, warp_1 only has 3
  // active threads, setting np2 to 2. In the case of warp_0, 32 threads engage
  // in the shfl, with the offset cycling through 16, 8, 4, 2, and 1. As for
  // warp_1, only 3 threads engage in the shfl, using offsets of 2 and 1. At an
  // offset of 2, threads 0-2 and 1-3 swap data. Since thread-3 is not active,
  // its data isn't valid, thread-1 disregards it. Threads 0 and 2 then
  // aggregate their data. With an offset of 1, threads 0-1 and 2-3 exchange
  // data, leading to threads 0 and 1 obtaining the final result.
  const unsigned int valid_lanes =
      is_last_warp ? reduction_size - warp_idx * WARP_SIZE : WARP_SIZE;
  int np2 = valid_lanes > 1 ? 1 << (31 - __clz(valid_lanes - 1)) : 0;
  for (int offset = np2; offset >= 1; offset /= 2) {
    T other_val = shfl_xor(mask, reduce_val, offset);
    bool other_valid = !is_last_warp || (offset ^ lane_idx) < valid_lanes;
    if (other_valid) {
      reduction_op(reduce_val, other_val);
    }
  }
  // Reduce across warp if needed, sometimes codegen can't detect it is a single
  // warp. Directly check blockDim.x to make sure it is really multiple warps.
  if (!SINGLE_WARP && blockDim.x > 32) {
    block_sync::sync<Aligned>();

    if (lane_idx == 0) {
      shared_mem[warp_idx] = reduce_val;
    }
    block_sync::sync<Aligned>();

    if (warp_idx == 0) {
      assert(num_of_warps <= 32);
      const bool launch_condition = lane_idx < num_of_warps;
      const unsigned int mask = __ballot_sync(0xffffffff, launch_condition);
      reduce_val = launch_condition ? shared_mem[lane_idx] : init_val;
      if (launch_condition) {
        int np2 = num_of_warps > 1 ? 1 << (31 - __clz(num_of_warps - 1)) : 0;
        for (int offset = np2; offset >= 1; offset /= 2) {
          T other_val = shfl_xor(mask, reduce_val, offset);
          bool other_valid = (offset ^ lane_idx) < num_of_warps;
          if (other_valid) {
            reduction_op(reduce_val, other_val);
          }
        }
      }
    }
    if (lane_idx == 0) {
      reduction_op(out, reduce_val);
    }
    // needs sync, otherwise other warps may access shared memory before this
    // reduction is done.
    block_sync::sync<Aligned>();
  } else {
    reduce_val = warp_broadcast(mask, reduce_val, 0);
    reduction_op(out, reduce_val);
  }
}

// Overload specifically for Padded == true:
template <bool SINGLE_WARP, bool Aligned, typename T, typename Func>
__device__ void warpReduceTIDX<SINGLE_WARP, Aligned, true>(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  constexpr int WARP_SIZE = 32;

  // Assume input padded to multiples of a warp
  T reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  for (int i = 16; i >= 1; i /= 2) {
    reduction_op(reduce_val, shfl_xor(0xffffffff, reduce_val, i, WARP_SIZE));
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (!SINGLE_WARP) {
    unsigned int warp_idx = threadIdx.x / WARP_SIZE;
    unsigned int lane_idx = threadIdx.x % WARP_SIZE;
    unsigned int reduce_group_id = threadIdx.z * blockDim.y + threadIdx.y;
    bool is_warp_head = lane_idx == 0;
    unsigned int reduction_size = blockDim.x;
    unsigned int num_of_warps = reduction_size / WARP_SIZE;
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
        reduction_op(reduce_val, shfl_xor(0xffffffff, reduce_val, i, 32));
      }
    }

    if (is_warp_head) {
      reduction_op(out, reduce_val);
    }
    // needs sync, otherwise other warps may access shared memory before this
    // reduction is done.
    block_sync::sync<Aligned>();
  } else {
    reduction_op(out, reduce_val);
  }
}

} // namespace warp
