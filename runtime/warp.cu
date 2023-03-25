// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace warp {

template <
    bool SINGLE_WARP,
    typename T,
    typename Func,
    typename _dim3ti,
    typename _dim3bd>
__device__ void warpReduceTIDX(
    T& out,
    const T& inp_val,
    Func reduction_op,
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim,
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
    reduction_op(
        reduce_val, __shfl_xor_sync(0xffffffff, reduce_val, i, WARP_SIZE));
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (!SINGLE_WARP) {
    unsigned int warp_idx = thread_idx.x / WARP_SIZE;
    unsigned int lane_idx = thread_idx.x % WARP_SIZE;
    unsigned int reduce_group_id = thread_idx.z * block_dim.y + thread_idx.y;
    bool is_warp_head = lane_idx == 0;
    unsigned int reduction_size = block_dim.x;
    unsigned int num_of_warps = reduction_size / WARP_SIZE;
    unsigned int smem_offset = reduce_group_id * num_of_warps;

    block_sync::sync();

    if (is_warp_head) {
      shared_mem[smem_offset + warp_idx] = reduce_val;
    }

    block_sync::sync();

    if (warp_idx == 0) {
      // This assumes num_of_warps will be < 32, meaning < 1024 threads.
      //  Should be true for long enough.
      assert(num_of_warps <= 32);

      reduce_val = lane_idx < num_of_warps ? shared_mem[smem_offset + lane_idx]
                                           : init_val;

      // Reduce within warp 0
      for (int i = 16; i >= 1; i /= 2) {
        reduction_op(
            reduce_val, __shfl_xor_sync(0xffffffff, reduce_val, i, 32));
      }
    }

    if (is_warp_head) {
      reduction_op(out, reduce_val);
    }
  } else {
    reduction_op(out, reduce_val);
  }
}

// specialized version with T = std::complex<float>
template <
    bool SINGLE_WARP,
    typename Func,
    typename _dim3ti,
    typename _dim3bd>
__device__ void warpReduceTIDX(
    std::complex<float>& out,
    const std::complex<float>& inp_val,
    Func reduction_op,
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim,
    std::complex<float>* shared_mem,
    bool read_write_pred,
    std::complex<float> init_val) {
  constexpr int WARP_SIZE = 32;

  // Assume input padded to multiples of a warp
  std::complex<float> reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  for (int i = 16; i >= 1; i /= 2) {
    float real = __shfl_xor_sync(0xffffffff, reduce_val.real(), i, WARP_SIZE);
    float imag = __shfl_xor_sync(0xffffffff, reduce_val.imag(), i, WARP_SIZE);
    std::complex<float> other_val(real, imag);
    reduction_op(
        reduce_val, other_val);
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (!SINGLE_WARP) {
    unsigned int warp_idx = thread_idx.x / WARP_SIZE;
    unsigned int lane_idx = thread_idx.x % WARP_SIZE;
    unsigned int reduce_group_id = thread_idx.z * block_dim.y + thread_idx.y;
    bool is_warp_head = lane_idx == 0;
    unsigned int reduction_size = block_dim.x;
    unsigned int num_of_warps = reduction_size / WARP_SIZE;
    unsigned int smem_offset = reduce_group_id * num_of_warps;

    block_sync::sync();

    if (is_warp_head) {
      shared_mem[smem_offset + warp_idx] = reduce_val;
    }

    block_sync::sync();

    if (warp_idx == 0) {
      // This assumes num_of_warps will be < 32, meaning < 1024 threads.
      //  Should be true for long enough.
      assert(num_of_warps <= 32);

      reduce_val =
          lane_idx < num_of_warps ? shared_mem[smem_offset + lane_idx]
                                  : init_val;

      // Reduce within warp 0
      for (int i = 16; i >= 1; i /= 2) {
        float real = __shfl_xor_sync(0xffffffff, reduce_val.real(), i, WARP_SIZE);
        float imag = __shfl_xor_sync(0xffffffff, reduce_val.imag(), i, WARP_SIZE);
        std::complex<float> other_val(real, imag);
        reduction_op(
            reduce_val, other_val);
      }
    }

    if (is_warp_head) {
      reduction_op(out, reduce_val);
    }
  } else {
    reduction_op(out, reduce_val);
  }
}

// specialized versionwith T = std::complex<double>
template <
    bool SINGLE_WARP,
    typename Func,
    typename _dim3ti,
    typename _dim3bd>
__device__ void warpReduceTIDX(
    std::complex<double>& out,
    const std::complex<double>& inp_val,
    Func reduction_op,
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim,
    std::complex<double>* shared_mem,
    bool read_write_pred,
    std::complex<double> init_val) {
  constexpr int WARP_SIZE = 32;

  // Assume input padded to multiples of a warp
  std::complex<double> reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  for (int i = 16; i >= 1; i /= 2) {
    double real = __shfl_xor_sync(0xffffffff, reduce_val.real(), i, WARP_SIZE);
    double imag = __shfl_xor_sync(0xffffffff, reduce_val.imag(), i, WARP_SIZE);
    std::complex<double> other_val(real, imag);
    reduction_op(
        reduce_val, other_val);
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (!SINGLE_WARP) {
    unsigned int warp_idx = thread_idx.x / WARP_SIZE;
    unsigned int lane_idx = thread_idx.x % WARP_SIZE;
    unsigned int reduce_group_id = thread_idx.z * block_dim.y + thread_idx.y;
    bool is_warp_head = lane_idx == 0;
    unsigned int reduction_size = block_dim.x;
    unsigned int num_of_warps = reduction_size / WARP_SIZE;
    unsigned int smem_offset = reduce_group_id * num_of_warps;

    block_sync::sync();

    if (is_warp_head) {
      shared_mem[smem_offset + warp_idx] = reduce_val;
    }

    block_sync::sync();

    if (warp_idx == 0) {
      // This assumes num_of_warps will be < 32, meaning < 1024 threads.
      //  Should be true for long enough.
      assert(num_of_warps <= 32);

      reduce_val =
          lane_idx < num_of_warps ? shared_mem[smem_offset + lane_idx]
                                  : init_val;

      // Reduce within warp 0
      for (int i = 16; i >= 1; i /= 2) {
        double real = __shfl_xor_sync(0xffffffff, reduce_val.real(), i, WARP_SIZE);
        double imag = __shfl_xor_sync(0xffffffff, reduce_val.imag(), i, WARP_SIZE);
        std::complex<double> other_val(real, imag);
        reduction_op(
            reduce_val, other_val);
      }
    }

    if (is_warp_head) {
      reduction_op(out, reduce_val);
    }
  } else {
    reduction_op(out, reduce_val);
  }
}

} // namespace warp
