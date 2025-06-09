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

template <
    bool SINGLE_WARP,
    bool Aligned,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void warpReduceTIDX(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim) {
  constexpr int WARP_SIZE = 32;

  // Assume input padded to multiples of a warp
  T reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  for (int i = 16; i >= 1; i /= 2) {
    reduction_op(reduce_val, shfl_xor(reduce_val, i, WARP_SIZE));
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (!SINGLE_WARP) {
    unsigned int warp_idx = threadIdx.x / WARP_SIZE;
    unsigned int lane_idx = threadIdx.x % WARP_SIZE;
    unsigned int reduce_group_id = threadIdx.z * block_dim.y + threadIdx.y;
    bool is_warp_head = lane_idx == 0;
    unsigned int reduction_size = block_dim.x;
    unsigned int num_of_warps = reduction_size / WARP_SIZE;
    unsigned int smem_offset = reduce_group_id * num_of_warps;

    block_sync::sync<Aligned>(block_dim);

    if (is_warp_head) {
      shared_mem[smem_offset + warp_idx] = reduce_val;
    }

    block_sync::sync<Aligned>(block_dim);

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
    block_sync::sync<Aligned>(block_dim);
  } else {
    reduction_op(out, reduce_val);
  }
}

template <
    int BDIMX,
    int BDIMY,
    bool Aligned,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void warpReduceTIDXY(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim) {
  constexpr int WARP_SIZE = 32;
  constexpr int num_of_warps = BDIMX * BDIMY / WARP_SIZE;

  // Assume input padded to multiples of a warp
  T reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  for (int i = 16; i >= 1; i /= 2) {
    reduction_op(reduce_val, shfl_xor(reduce_val, i, WARP_SIZE));
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (num_of_warps > 1) {
    unsigned int idx = threadIdx.x + threadIdx.y * BDIMX;
    unsigned int warp_idx = idx / WARP_SIZE;
    unsigned int lane_idx = idx % WARP_SIZE;
    block_sync::sync<Aligned>(block_dim);
    if (lane_idx == 0) {
      shared_mem[warp_idx] = reduce_val;
    }
    block_sync::sync<Aligned>(block_dim);

    if (warp_idx == 0) {
      reduce_val = lane_idx < num_of_warps ? shared_mem[lane_idx] : init_val;
      // Reduce within warp 0
      for (int i = 16; i >= 1; i /= 2) {
        reduction_op(reduce_val, shfl_xor(reduce_val, i, 32));
      }
    }

    if (lane_idx == 0) {
      reduction_op(out, reduce_val);
    }
    // needs sync, otherwise other warps may access shared memory before this
    // reduction is done.
    block_sync::sync<Aligned>(block_dim);
  } else {
    reduction_op(out, reduce_val);
  }
}

// sizeof(T) * K = sizeof(uint64_t)
// Array structure ensures data is aligned for safe casting to uint64_t
template <int K, typename T, typename Func>
__device__ __forceinline__ void packedWarpReduce(
    Array<T, K, K>& val,
    Func reduction_op) {
  constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    Array<T, K, K> remote;
    *reinterpret_cast<uint64_t*>(remote.array) = __shfl_xor_sync(
        FINAL_MASK, *reinterpret_cast<uint64_t*>(val.array), mask, 32);
#pragma unroll
    for (int i = 0; i < K; i++) {
      reduction_op(val[i], remote[i]);
    }
  }
}

template <
    bool SINGLE_WARP,
    bool Aligned,
    int N, // Number of elements per input array
    int n_threads,
    typename T,
    typename Func>
__device__ void iterGroupedStaticWarpAllReduce(
    T out[N],
    const T inp_val[N],
    Func reduction_op,
    T* shared_mem,
    uint32_t barrier_id = 1) {
  // pack T into uint64_t to reduce number of shuffles
  // sizeof(T) * K = sizeof(uint64_t), e.g. T = float, K = 2.
  constexpr int K = sizeof(uint64_t) / sizeof(T);
  constexpr dim3 block_dim = dim3(n_threads, 1, 1);
  Array<T, K, K> packed_reduce_val;
  // original N reductions are reduced to N / K reductions
  static_assert(N % K == 0, "N must be a multiple of K");
#pragma unroll
  for (int i = 0; i < N; i += K) {
#pragma unroll
    for (int j = 0; j < K; j++) {
      packed_reduce_val[j] = inp_val[i + j];
    }
    packedWarpReduce(packed_reduce_val, reduction_op);
#pragma unroll
    for (int j = 0; j < K; j++) {
      out[i + j] = packed_reduce_val[j];
    }
  }

  // short circuit if we only have one warp
  if constexpr (SINGLE_WARP) {
    return;
  }

  // cross warp reduction using shared memory
  constexpr int WARP_SIZE = 32;
  constexpr int num_of_warps = n_threads / WARP_SIZE;
  unsigned int warp_idx = threadIdx.x / WARP_SIZE;
  unsigned int lane_idx = threadIdx.x % WARP_SIZE;
  constexpr unsigned int align_size = sizeof(T) * N;
  static_assert(align_size <= 16, "max allowed vect r/w is 16 bytes");
  // [warp_idx, N]
  // [w0r0, w0r1, w0r2, w0r3, w1r0, w1r1, w1r2, w1r3]
  if (lane_idx == 0) {
    loadGeneric<T, N>(shared_mem + N * warp_idx, out);
  }
  block_sync::sync<Aligned>(block_dim, barrier_id);

  // All reduce
  loadGeneric<T, N>(out, shared_mem);
  __align__(align_size) T other[N];
#pragma unroll
  for (int i = 1; i < num_of_warps; i++) {
    loadGeneric<T, N>(other, shared_mem + i * N);
#pragma unroll
    for (int j = 0; j < N; j++) {
      out[j] += other[j];
    }
  }
  // needs sync, otherwise other warps may access shared memory before this
  // reduction is done.
  block_sync::sync<Aligned>(block_dim, barrier_id);
}

template <
    bool SINGLE_WARP,
    bool Aligned,
    int n_threads,
    typename T,
    typename Func>
__device__ void staticWarpAllReduceTIDX(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    uint32_t barrier_id = 1) {
  constexpr int WARP_SIZE = 32;
  constexpr dim3 block_dim = dim3(n_threads, 1, 1);
  constexpr unsigned int num_of_warps = n_threads / WARP_SIZE;
  T reduce_val = inp_val;
  // Reduce within each warp
  for (int i = 16; i >= 1; i /= 2) {
    reduction_op(reduce_val, shfl_xor(reduce_val, i, WARP_SIZE));
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if constexpr (!SINGLE_WARP) {
    unsigned int warp_idx = threadIdx.x / WARP_SIZE;
    unsigned int lane_idx = threadIdx.x % WARP_SIZE;

    if (lane_idx == 0) {
      shared_mem[warp_idx] = reduce_val;
    }
    block_sync::sync<Aligned>(block_dim, barrier_id);

    // All reduce
    out = shared_mem[0];
#pragma unroll
    for (int i = 1; i < num_of_warps; i++) {
      out += shared_mem[i];
    }
    // needs sync, otherwise other warps may access shared memory before this
    // reduction is done.
    block_sync::sync<Aligned>(block_dim, barrier_id);
  } else {
    reduction_op(out, reduce_val);
  }
}
} // namespace warp
