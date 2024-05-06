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

  // The peer stride is the distance between the current element and its
  // reduction peer, it depends on the reduction dimension.
  constexpr int num_redu_dims = (int)X_REDUCE + (int)Y_REDUCE + (int)Z_REDUCE;
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
    // If !Y_REDUCE, if blockDim.y == 1, merge XZ, peer_stride is 1, otherwise
    //               different strides for different reduction stages, change
    //               smem layout based on reduction_idx and reduction_tid, may
    //               have stride access and bank conflict, rare case, may happen
    //               for batch norm doing multiple reductions per block.
    if (!Y_REDUCE) {
      smem_offset = blockDim.y > 1
          ? reduction_idx * reduction_size + reduction_tid
          : smem_offset;
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
    reduction_op(
        shared_mem[smem_offset], shared_mem[smem_offset + np2 * peer_stride]);
  }
  block_sync::sync<Aligned>();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(
          shared_mem[smem_offset],
          shared_mem[smem_offset + factor * peer_stride]);
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



// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
  
  // Grouped block welford optimized for outer reductions with
  // TIDx and TIDy mapped to non-reduction and reduction domains,
  // respectively with unused TIDz.
  //
  // The main motivation of this optimized version is the same as the
  // grouped grid reduction, i.e, by doing multiple reductions together,
  // it is possible to reduce the number of synchronizations. However,
  // unlike the grouped grid reduction, the cost of grouping can be
  // prohitively high, i.e., the size of the work buffer must be
  // expanded by a factor of grouping. In the case of grid
  // reductions, the buffer is on global memory, so the space requirement
  // is not a concern, but that isn't the case with block reductions,
  // since the buffer is on shared memory, which has a limited
  // capacity.
  //
  // This implementation tries to benefit from aggregated block
  // synchronizations while minimizing the cost of the expanded buffer
  // size by first partially reducing the input within each warp. It
  // would save the required buffer size by a factor of WARP_SIZE /
  // blockDim.x as the reduction is done along threadIdx.y. So to be
  // effective, blockDim.x needs to be smaller than WARP_SIZE, and in the
  // case of grouped grid welford, it should be typically 8 or 16.
  //
  // The algorithm is an adaptation of scattered butterfly reduction,
  // aka recursive halving, commonly used for implementing
  // MPI_Reduce_scatter. For a visual illustration of the data
  // organization, see, for example, page 22 of Solomonik,
  // Design of Parallel and High-Performance Computing:
  // Distributed-Memory Models and Algorithms, 2015
  // (https://solomonik.cs.illinois.edu/talks/dphpc-dec-2015.pdf)
  //
  // Assumptions:
  // - blockDim.x and blockDim.y are statically known values so that all
  // loops can be completely unrolled
  // - blockDim.x is smaller than WARP_SIZE
  // - blockDim.x evenly divides WARP_SIZE
  // - There are multiple warps per block
  // - The gouping factor, N, is at least as large as the warp
  // dimY and is divisible by the warp dimY.
  //
  // This is meant to be used as part of the grouped grid welford
  // reduction but should be usable as a standalone block welford routine as
  // long as the above assumptions hold.
  //
  // Note: Having an output reference parameter resulted in using more
  // registers than just returing the output. Results would vary
  // depending on compiler versions, but it seems safer to return outputs
  // as a new value.
  template <bool Aligned, int N, typename T, int BDIMX, int BDIMY>
  __inline__ __device__ void blockIterGroupedWarpReduce(
    T out[N],
    T inp_val[N],
    T* smem) {

    constexpr int num_warps = BDIMX * BDIMY / 32;
    static_assert(num_warps >= 1, "There must be at least a single warp");
    static_assert(32 % BDIMX == 0, "blockDimx.x must be able to divide 32");
  
    const int tid = threadIdx.x + threadIdx.y * BDIMX;
    const int wid = tid / 32;
  
    // Dimension of the Y axis within each warp
    constexpr int wdimy = 32 / BDIMX;
    static_assert(N >= wdimy, "N must be >= 32 / blockDim.x");
    static_assert(
        N % wdimy == 0, "N must be divisible by 32 / blockDim.x");
    // There must be at least a single warp
  
    // Y index within each warp
    const int warp_tidy = threadIdx.y % wdimy;
  
    // Thread index in each warp
    const int lane_id = threadIdx.x + warp_tidy * BDIMX;
  
    // We define a chunk as a value in a group and a chunk size as the
    // number of group values per thread. Initially, the chunk size is
    // N. After the initial warp reduction, the chunk size is
    // reduced to N/wdimy. For example, suppose N=8,
    // blockDim.x=8, blockDim.y=32, then wdimy=4, so after the initial
    // warp reduction, the chunk size is 2. This is the number of
    // elements each thread stores to shared memory.
  
    int chunk_size = N;
  
    // Butterfly reduction, a.k.a. recursive halving as each iteration
    // halves the number of values
  #pragma unroll
    for (int lane_mask = 16; lane_mask >= BDIMX; lane_mask /= 2) {
      chunk_size /= 2;
    
  #pragma unroll
      for (int index_in_chunk = 0; index_in_chunk < chunk_size;
           ++index_in_chunk) {
        T pushed_avg = 0;
        T self_avg = 0;
        // Divergent branch. Not a big deal with independent scheduling?
        if (lane_id & lane_mask) {
          // Push first half
          auto push_offset = index_in_chunk;
          auto self_offset = index_in_chunk + chunk_size;
          pushed_avg = inp_val[push_offset];
          self_avg = inp_val[self_offset];
          // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && wid == 0){
          //   printf("tidxy=%d lane_id: %d, lane_mask: %d, xor=%d index_in_chunk: %d, self_offset: %d, push_offset: %d\n", threadIdx.y, lane_id, lane_mask, lane_id ^ lane_mask, index_in_chunk, self_offset, push_offset);
          // }         
        } else {
          // Push second half
          auto push_offset = index_in_chunk + chunk_size;
          auto self_offset = index_in_chunk;
          pushed_avg = inp_val[push_offset];
          self_avg = inp_val[self_offset];
          // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && wid == 0){
          //   printf("tidxy=%d lane_id: %d, lane_mask: %d, xor=%d index_in_chunk: %d, self_offset: %d, push_offset: %d\n", threadIdx.y, lane_id, lane_mask, lane_id ^ lane_mask, index_in_chunk, self_offset, push_offset);
          // }          
        }

        auto peer_avg = __shfl_xor_sync(0xffffffff, pushed_avg, lane_mask);
        self_avg += peer_avg;
        inp_val[index_in_chunk] = self_avg;
      }
    }

    // At this point, chunk_size is reduced to N/wdimy as
    // mentioned above. Each thread has warp-reduced chunk_size values
    // in array inp. This chunk_size_post_reduction should be equal to
    // chunk_size at this point.
    constexpr int chunk_size_post_reduction = N / wdimy;
  
    // More specifically, the warp_tidy of each thread defines
    // the chunk IDs held by the thread as follows:
    //
    // [warp_tidy * chunk_size_post_reduction, warp_tidy *
    // chunk_size_post_reduction + chunk_size_post_reduction]
    //
    // Each thread uploads the chunk_size_post_reduction values one by
    // one. Each chunk is spread by BDIMX * BDIMY values. The data
    // layout of the shared memory is:
    //
    // [chunk_size, wid, warp_tidy, TIDx]
    // [2, 32, 4, 8], bdimx = 8, bdimy = 128, N = 8, wdimy = 4
    // The remaining reduction is done on the WID
    // dimension. More specifically, we assign one warp per chunk (or
    // a value of the group). The wdimy threads of the same threadId.x
    // collectively reduce num_warps partial results, each of which is
    // stored with stride 32. This means that there will be wdimy-way
    // bank conflicts, so to avoid that, swizzling is also employed.
  #pragma unroll
    for (int i = 0; i < chunk_size; ++i) {
      // Accumulating smem offset from the innermost dimension
      int smem_offset = 0;
      // TIDx
      smem_offset += threadIdx.x;
      // Warp_TIDy with swizzle
      smem_offset += ((warp_tidy + wid) % wdimy) * BDIMX;
      // WID
      smem_offset += wid * 32;
      // chunk_size
      smem_offset += i * BDIMX * BDIMY;
      smem[smem_offset] = inp_val[i];

      // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && wid >= 2 && wid <= 3){
      //   printf("tidx=%d warp_tidy: %d, wid: %d, ichunk=%d smem_offset: %d\n", threadIdx.x, warp_tidy, wid, i, smem_offset);
      // }         
    }
  
    block_sync::sync<Aligned>();
  
    // The next step is to let each thread of a warp independently
    // accumulate the partial results on the shared memory
    // reduction. A single warp is used to accumulate of the partial
    // results for a single chunk, so warp wid takes care of the wid-th
    // chunk.
    //
    // The starting offset of partial results of a chunk is:
    //
    // (wid % chunk_size_post_reduction) * BDIMX * BDIMY + (wid /
    // chunk_size_post_reduction) * BDIMX
    //
    // Note that each thread had chunk_size_post_reduction contiguous
    // chunks, so when uploaded to shmem, they are strided by
    // BDIMX*BDIMY, hence (wid % chunk_size_post_reduction) * BDIMX *
    // BDIMY.
  
    // The vector width is likely at least 4, so at least 4 warps should
    // be used, which is
    // enough to occupy an SM. When N=8, it might be more
    // efficient to use just 4 warps with each warp taking care of two
    // groups, but the difference would be pretty small.
  
    // Also, the number of warps should be at least 8 and can be 16
    // too. N should be 8 at largest, so it's always num_warps >=
    // N.
  
    T avg = 0;
    static_assert(
        num_warps >= N,
        "Number of warps must be at least as large as N");
    if (wid < N) {
      // num_warps = 32, warp_tidy = [0,1,2,3], loop 8 times
  #pragma unroll
      for (int i = warp_tidy; i < num_warps; i += wdimy) {
        int offset = 0;
        offset += threadIdx.x;
        // Offset to the partial results of the i-th warp
        offset += i * 32;
        // Offset to the chunk for this warp. Swizzled to avoid bank
        // conflicts.
        offset += ((wid / chunk_size + i) % wdimy) * BDIMX;
        offset += (wid % chunk_size) * BDIMX * BDIMY;
        avg += smem[offset];
        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && wid == 0 && warp_tidy == 0){
        //   printf("tidx=%d warp_tidy: %d, wid: %d, irow=%d smem_offset: %d\n", threadIdx.x, warp_tidy, wid, i, offset);
        // }         
      }
    }
    block_sync::sync<Aligned>();
  
    // Nothing to do for warps whose wid is larger than NunVals
    if (wid >= N) {
      return ;
    }

    // Standard binary-exchange reduction within wdimy intra-warp
    // threads.
  #pragma unroll
    for (int lane_mask = 16; lane_mask >= BDIMX; lane_mask /= 2) {
      auto avg_peer = __shfl_xor_sync(0xffffffff, avg, lane_mask);
      avg += avg_peer;
    }
     
    // 1st warp has the final result of the first chunk
    // 2nd warp has the final result of the second chunk
    // ...
    // N-th warp has the final result of the N-th chunk
    // We need threads with TIDy == 0 to have all the final results
    if(warp_tidy==0){
      const int iter_idx = threadIdx.x;
      const int valid_group_idx = wid;
      int offset = valid_group_idx * BDIMX + iter_idx;
      smem[offset] = avg;
    }
    block_sync::sync<Aligned>();

    if(threadIdx.y == 0){
      for(int i = 0; i<N; i++){
        int offset = i * BDIMX + threadIdx.x;    
        out[i] = smem[offset];
      }
    }
    block_sync::sync<Aligned>();


    // // vectorized version
    // // [BDIMX, N]
    // if(warp_tidy==0){
    //   const int iter_idx = threadIdx.x;
    //   const int valid_group_idx = wid;
    //   int offset = valid_group_idx + iter_idx * N;
    //   smem[offset] = avg;
    //   if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && wid <=1 ){
    //     printf("tidx=%d, wid: %d, smem_offset: %d\n", threadIdx.x, wid,offset);
    //   }      
    // }
    // block_sync::sync<Aligned>();

    // if(threadIdx.y == 0){
    //   constexpr unsigned int total_loads = sizeof(T) * N / 16;
    //   constexpr unsigned int elements_per_load = 16 / sizeof(T);
    //   #pragma unroll
    //   for (unsigned int i = 0; i < total_loads; ++i) {
    //     loadGeneric<T, elements_per_load>(
    //       out + i * elements_per_load,
    //       smem + N * threadIdx.x + i * elements_per_load);
    //   }      
    // }
    block_sync::sync<Aligned>();
  }