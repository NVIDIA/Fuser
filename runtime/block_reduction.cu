// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to false the dimension doesn't
// participate in the reduction. We could start with warp reductions, then
// reduce the warps, this could save some shared memory, but could be slower in
// some instances.
//
//  EXAMPLE USAGE:
//  blockReduceSum<X_THREADS, Y_THREADS, Z_THREADS>
//    (output[output_index], inputs[input_index],
//      [] __device__ (T& a, const T b) { a += b; });
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename Func>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // If this thread will output a final result
  bool should_write =
      index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(threadIdx);

  // Size of the reduction segments
  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(blockDim);

  // Index into the reduction segment
  unsigned int reduction_tid =
      index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
          threadIdx, blockDim);

  // Index of the reduction segment
  unsigned int reduction_idx =
      index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
          threadIdx, blockDim);

  // Offset into smem for the current thread
  unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

  // Initialize shared memory
  if (read_pred) {
    shared_mem[smem_offset] = inp_val;
  } else {
    shared_mem[smem_offset] = init_val;
  }

  block_sync::sync<Aligned>();
  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + np2]);
  }
  block_sync::sync<Aligned>();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + factor]);
    }
    block_sync::sync<Aligned>();
  }

  if (should_write && write_pred) {
    T result = out;
    reduction_op(result, shared_mem[smem_offset]);
    if (reduction_size > 1) {
      reduction_op(result, shared_mem[smem_offset + 1]);
    }
    out = result;
  }
  block_sync::sync<Aligned>();
}

// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,

    typename Func>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  blockReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, Aligned, T, Func>(
      out,
      inp_val,
      reduction_op,
      shared_mem,
      read_write_pred,
      read_write_pred,
      init_val);
}

// Each thread in the iteration dimension processes N elements
// Typical usage is in outer reduction where the iteration dimension
// is parallelized by vectorized loads, bidmx. The reduction dimension
// is parallelized by bdimy. This function works as follows:
// (1) Each thread vectorized loads N elements from input register array to
// smem. (2) do N * bdimx parallel reductions in smem.

// TODO: merge `blockIterGroupedReduce` with `blockReduce`
// (1) for-loops are fully unrolled should not cause overhead for `blockReduce`
// (2) used in gridReduce, needs to change correspodning gridReduce function
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    int N, // Number of elements per input array
    typename T,
    typename Func>
__device__ void blockIterGroupedReduce(
    T out[N],
    const T inp_val[N],
    Func reduction_op,
    T* shared_mem,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // N should be a valid vectorization factor
  static_assert(
      N == 2 || N == 4 || N == 8 || N == 16,
      "N should be a valid vectorization factor, one of (2, 4, 8, 16)!");

  bool should_write =
      index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(threadIdx);

  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(blockDim);

  unsigned int reduction_tid =
      index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
          threadIdx, blockDim);

  unsigned int reduction_idx =
      index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
          threadIdx, blockDim);

  // Adjust shared memory offset for array processing
  unsigned int smem_offset =
      (reduction_idx * reduction_size + reduction_tid) * N;
  if (read_pred) {
    // This section calculates the number of vectorized load operations required
    // to fetch all elements of an array into shared memory, assuming each load
    // can transfer up to 16 bytes. For example, with fusion input vectorized by
    // 8 (N = 8) and computations in fp32 (sizeof(T) = 4 bytes), the total data
    // size is 4 * 8 = 32 bytes, necessitating 32 / 16 = 2 load transactions.
    // Each transaction loads 16 / 4 (bytes per element) = 4 elements.
    if constexpr (sizeof(T) * N <= 16) {
      loadGeneric<T, N>(shared_mem + smem_offset, const_cast<T*>(inp_val));
    } else {
      constexpr unsigned int total_loads = sizeof(T) * N / 16;
      constexpr unsigned int elements_per_load = 16 / sizeof(T);
      static_assert(
          sizeof(T) * N == 16 * total_loads,
          "This combination of vectorization factor and data type is not supported!");
      static_assert(
          sizeof(T) * elements_per_load == 16,
          "This data type is not supported!");
#pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        loadGeneric<T, elements_per_load>(
            shared_mem + smem_offset + i * elements_per_load,
            const_cast<T*>(inp_val) + i * elements_per_load);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      shared_mem[smem_offset + i] = init_val;
    }
  }

  block_sync::sync<Aligned>();

  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  // Perform parallel reduction for each element in the array
  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      reduction_op(
          shared_mem[smem_offset + i], shared_mem[smem_offset + np2 * N + i]);
    }
  }

  block_sync::sync<Aligned>();

  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        reduction_op(
            shared_mem[smem_offset + i],
            shared_mem[smem_offset + factor * N + i]);
      }
    }
    block_sync::sync<Aligned>();
  }

  if (should_write && write_pred) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      T result = out[i];
      reduction_op(result, shared_mem[smem_offset + i]);
      if (reduction_size > 1) {
        reduction_op(
            result,
            shared_mem[smem_offset + N + i]); // Handle the last element if
                                              // reduction size is odd
      }
      out[i] = result;
    }
  }
  block_sync::sync<Aligned>();
}

// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    int N, // Number of elements per input array
    typename T,
    typename Func>
__device__ void blockIterGroupedReduce(
    T out[N],
    const T inp_val[N],
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  blockIterGroupedReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, Aligned, N, T, Func>(
      out,
      inp_val,
      reduction_op,
      shared_mem,
      read_write_pred,
      read_write_pred,
      init_val);
}
