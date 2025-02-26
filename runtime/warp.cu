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

// sizeof(T) * K = sizeof(uint64_t)
// require alginment of sizeof(T) * K to safely cast between T and uint64_t
// shfl uses uint64_t to reduce number of shuffles
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
#pragma unroll K
    for (int i = 0; i < K; i++) {
      reduction_op(val[i], remote[i]);
    }
  }
}

template <
    bool SINGLE_WARP,
    bool Aligned,
    int N, // Number of elements per input array
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void iterGroupedWarpReduce(
    T out[N],
    const T inp_val[N],
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim,
    uint32_t barrier_id = 1) {
  // pack T into uint64_t to reduce number of shuffles
  // sizeof(T) * K = sizeof(uint64_t), e.g. T = float, K = 2.
  constexpr int K = sizeof(uint64_t) / sizeof(T);
  Array<T, K, K> packed_reduce_val;
  T result[N];
  // original N reductions are reduced to N / K reductions
  static_assert(N % K == 0, "N must be a multiple of K");
#pragma unroll N / K
  for (int i = 0; i < N; i += K) {
    packed_reduce_val.set(init_val);
    if (read_write_pred) {
#pragma unroll K
      for (int j = 0; j < K; j++) {
        packed_reduce_val[j] = inp_val[i + j];
      }
    }
    // reduce within each warp
    packedWarpReduce(packed_reduce_val, reduction_op);

// unpack
#pragma unroll K
    for (int j = 0; j < K; j++) {
      result[i + j] = packed_reduce_val[j];
    }
  }

  // short circuit if we only have one warp
  if (SINGLE_WARP) {
    return;
  }

  // cross warp reduction using shared memory
  constexpr int WARP_SIZE = 32;
  unsigned int warp_idx = threadIdx.x / WARP_SIZE;
  unsigned int lane_idx = threadIdx.x % WARP_SIZE;
  unsigned int num_of_warps = block_dim.x / WARP_SIZE;

  // [warp_idx, N]
  if (lane_idx == 0) {
#pragma unroll N
    for (int i = 0; i < N; i++) {
      shared_mem[N * warp_idx + i] = result[i];
    }
  }
  block_sync::sync<Aligned>(block_dim);

  if (warp_idx == 0) {
    // assuming N = 4, num_of_warps = 4, the warp reduction results are written
    // to smem: [a0 b0 c0 d0, a1 b1 c1 d1, a2 b2 c2 d2, a3 b3 c3 d3] we need
    // to further compute [a,b,c,d], where a = sum(a0,a1,a2,a3).
    int np2 = 1U << (32 - __clz(num_of_warps - 1));
    if (np2 * N <= 32) {
      // collect results to threads [0, N-1]
      T myVal = lane_idx < num_of_warps * N ? shared_mem[lane_idx] : init_val;
      for (int i = np2 / 2; i >= 1; i /= 2) {
        T peer = __shfl_down_sync(0xffffffff, myVal, i * N);
        reduction_op(myVal, peer);
      }
      // thread 0 collect the final results from threads 1 to N-1
      out[0] = myVal;
      for (int i = 1; i < N; i++) {
        T peer = __shfl_sync(0xffffffff, myVal, i);
        if (lane_idx == 0) {
          out[i] = peer;
        }
      }
    } else {
// serial reduction
#pragma unroll N
      for (int j = 0; j < N; j++) {
        out[j] = shared_mem[j];
      }
      for (int i = 1; i < num_of_warps; i++) {
#pragma unroll N
        for (int j = 0; j < N; j++) {
          out[j] += shared_mem[i * N + j];
        }
      }
    }
  }
  // needs sync, otherwise other warps may access shared memory before this
  // reduction is done.
  block_sync::sync<Aligned>(block_dim);
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
    BlockDimT block_dim,
    uint32_t barrier_id = 0) {
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

    block_sync::sync<Aligned>(block_dim, barrier_id);

    if (is_warp_head) {
      shared_mem[smem_offset + warp_idx] = reduce_val;
    }

    block_sync::sync<Aligned>(block_dim, barrier_id);

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
    block_sync::sync<Aligned>(block_dim, barrier_id);
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
} // namespace warp
