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

  // number of reductions per block
  unsigned int reduction_num =
      index_utils::maskedSize<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(blockDim);

  // smem_offset is the offset into shared memory for the current thread.
  // To ensure coalesced access to shared memory, we need to ensure
  // each transaction is accessing a contiguous block of 128 bytes.
  // For outer reduction where TIDy is in the reduction dimension and TIDx
  // is in the iteration dimension and TIDz is not used. We have
  // reduction_tid = TIDy and reduction_idx = TIDx. If we directly use the
  // offset based on reduction_tid and reduction_idx, we will have stride
  // access to shared memory. For example:
  // offset = reduction_idx * reduction_size + reduction_tid
  //        = TIDx * blockDim.y + TIDy
  // To avoid this, we should always use the offset based on the indexing of
  // threads within a block.
  // Offset into smem for the current thread
  unsigned int smem_offset = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;

  // The peer stride represents the distance between the current element and its
  // nearest reduction peer. It depends on the reduction dimension. A reduction
  // peer refers to elements that belong to the same reduction segment. For
  // example, if the reduction is across TIDy, all the elements in the same
  // column (with the same TIDx) are considered peers of each other. The
  // distance between an element and its nearest peer is blockDim.x.
  constexpr int num_redu_dims = (int)X_REDUCE + (int)Y_REDUCE + (int)Z_REDUCE;
  constexpr bool xz_reduce = (num_redu_dims == 2 && !Y_REDUCE);
  // reduction in 3 dimensions, XYZ, stride is 1
  unsigned int peer_stride = 1;
  if (num_redu_dims == 1) {
    // Reduction only in 1 dimension, X or Y or Z
    // e.g. inner or outer reduction
    // If X_REDUCE, reducing in neighbor cols in smem, peer_stride is 1
    // If Y_REDUCE, reducing in neighbor rows in smem, peer_stride is blockDim.x
    // If Z_REDUCE, reducing in neighbor planes in smem, peer_stride is
    // blockDim.x * blockDim.y
    peer_stride = X_REDUCE ? 1
        : Y_REDUCE         ? blockDim.x
                           : blockDim.x * blockDim.y;
  } else if (num_redu_dims == 2) {
    // Reduction in 2 dimensions, only one dimension is not reduced, !X, !Y, !Z
    // If !Z_REDUCE, merge XY, reducing neighbor cols, peer_stride is 1
    // If !X_REDUCE, merge ZY, reducing neighbor rows, peer_stride is blockDim.x
    // If !Y_REDUCE, if blockDim.y == 1, merge XZ, peer_stride is 1.
    // otherwise, needs carefully calculate offset to the reduction peer:
    // (1) redu_offset = reduction_tid + tree_fold_factor
    // (2) idz = redu_offset / blockDim.x
    // (3) idx = redu_offset % blockDim.x
    // (4) smem_offset = idx + threadIdx.y * blockDim.x + idz * blockDim.x *
    // blockDim.y
    if (!Y_REDUCE) {
      peer_stride = 1;
    } else {
      peer_stride = !Z_REDUCE ? 1 : blockDim.x;
    }
  }

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
    int peer_offset = smem_offset + np2 * peer_stride;
    if constexpr (xz_reduce) {
      if (blockDim.y > 1) {
        int redu_offset = reduction_tid + np2;
        int idz = redu_offset / blockDim.x;
        int idx = redu_offset % blockDim.x;
        peer_offset =
            idx + threadIdx.y * blockDim.x + idz * blockDim.x * blockDim.y;
      }
    }
    reduction_op(shared_mem[smem_offset], shared_mem[peer_offset]);
  }
  block_sync::sync<Aligned>();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      int peer_offset = smem_offset + factor * peer_stride;
      if constexpr (xz_reduce) {
        if (blockDim.y > 1) {
          int redu_offset = reduction_tid + factor;
          int idz = redu_offset / blockDim.x;
          int idx = redu_offset % blockDim.x;
          peer_offset =
              idx + threadIdx.y * blockDim.x + idz * blockDim.x * blockDim.y;
        }
      }
      reduction_op(shared_mem[smem_offset], shared_mem[peer_offset]);
    }
    block_sync::sync<Aligned>();
  }

  if (should_write && write_pred) {
    T result = out;
    reduction_op(result, shared_mem[smem_offset]);
    if (reduction_size > 1) {
      reduction_op(result, shared_mem[smem_offset + peer_stride]);
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
  static_assert(
      !X_REDUCE && Y_REDUCE && !Z_REDUCE, "Only support reduction in Y dim!");

  bool should_write = threadIdx.y == 0;
  unsigned int reduction_size = blockDim.y;
  unsigned int reduction_tid = threadIdx.y;

  // In shared memory, each row has 128 bytes, if sizeof(T) * N = 32 bytes, each
  // row has 128 / 32 = 4 threads. Each transaction can only load data from one
  // row, with a max of 16 bytes per thread. So the total bytes per transaction
  // is 4 x 16 = 64 bytes which is only half of the maximum 128 bytes per
  // transaction. we should change the layout from [TIDy, TIDx, N] to [N/4,
  // TIDy, TIDx, 4]
  constexpr unsigned int total_loads =
      sizeof(T) * N / 16 > 1 ? sizeof(T) * N / 16 : 1;
  constexpr unsigned int elements_per_load =
      16 / sizeof(T) > N ? N : 16 / sizeof(T);

  // assume TIDy is the reduction dimension, TIDx is the iteration dimension
  // TIDz is not used
  unsigned int peer_stride = elements_per_load * blockDim.x;

  unsigned int smem_offset_inter = blockDim.x * blockDim.y * elements_per_load;
  unsigned int smem_offset_intra =
      (threadIdx.y * blockDim.x + threadIdx.x) * elements_per_load;

// load to [total_loads] sections of shared memory
#pragma unroll
  for (unsigned int i = 0; i < total_loads; ++i) {
    loadGeneric<T, elements_per_load>(
        shared_mem + smem_offset_inter * i + smem_offset_intra,
        const_cast<T*>(inp_val) + i * elements_per_load);
  }
  block_sync::sync<Aligned>();

  // Reduce down to nearest power of 2 for the tree reduction:
  // Perform parallel reduction for each element in the array
  int np2 = 1 << (31 - __clz(reduction_size));
  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    // vectorized load from smem to regs
    T self[N];
    T peer[N];
#pragma unroll
    for (unsigned int i = 0; i < total_loads; ++i) {
      int self_offset = smem_offset_inter * i + smem_offset_intra;
      int peer_offset = self_offset + np2 * peer_stride;
      loadGeneric<T, elements_per_load>(
          self + i * elements_per_load, shared_mem + self_offset);
      loadGeneric<T, elements_per_load>(
          peer + i * elements_per_load, shared_mem + peer_offset);
    }
// reduction
#pragma unroll
    for (int i = 0; i < N; ++i) {
      reduction_op(self[i], peer[i]);
    }
// write self back to smem
#pragma unroll
    for (unsigned int i = 0; i < total_loads; ++i) {
      int self_offset = smem_offset_inter * i + smem_offset_intra;
      loadGeneric<T, elements_per_load>(
          shared_mem + self_offset, self + i * elements_per_load);
    }
  }
  block_sync::sync<Aligned>();

  // Tree reduction
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      // vectorized load from smem to regs
      T self[N];
      T peer[N];
#pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        int self_offset = smem_offset_inter * i + smem_offset_intra;
        int peer_offset = self_offset + factor * peer_stride;
        loadGeneric<T, elements_per_load>(
            self + i * elements_per_load, shared_mem + self_offset);
        loadGeneric<T, elements_per_load>(
            peer + i * elements_per_load, shared_mem + peer_offset);
      }
// reduction
#pragma unroll
      for (int i = 0; i < N; ++i) {
        reduction_op(self[i], peer[i]);
      }
// write self back to smem
#pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        int self_offset = smem_offset_inter * i + smem_offset_intra;
        loadGeneric<T, elements_per_load>(
            shared_mem + self_offset, self + i * elements_per_load);
      }
    }
    block_sync::sync<Aligned>();
  }

  // last reduction
  if (should_write && write_pred) {
    // init result
    T result[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = out[i];
    }

    // copy first element to result
    T self[N];
#pragma unroll
    for (unsigned int i = 0; i < total_loads; ++i) {
      int self_offset = smem_offset_inter * i + smem_offset_intra;
      loadGeneric<T, elements_per_load>(
          self + i * elements_per_load, shared_mem + self_offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      reduction_op(result[i], self[i]);
    }

    // reduction of the 2nd last element
    if (reduction_size > 1) {
      T peer[N];
#pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        int peer_offset =
            smem_offset_inter * i + smem_offset_intra + peer_stride;
        loadGeneric<T, elements_per_load>(
            peer + i * elements_per_load, shared_mem + peer_offset);
      }
#pragma unroll
      for (int i = 0; i < N; ++i) {
        reduction_op(result[i], peer[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out[i] = result[i];
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
