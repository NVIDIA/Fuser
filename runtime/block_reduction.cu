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
  unsigned int smem_offset = (threadIdx.y * blockDim.x + threadIdx.x) * N;

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


  constexpr unsigned int total_loads = sizeof(T) * N / 16 > 1 ? sizeof(T) * N / 16 : 1;
  constexpr unsigned int elements_per_load = 16 / sizeof(T);
  // Perform parallel reduction for each element in the array
  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    T self[N];
    T peer[N];
    #pragma unroll
    for (unsigned int i = 0; i < total_loads; ++i) {
      loadGeneric<T, elements_per_load>(
        self + i * elements_per_load,
          shared_mem + smem_offset + i * elements_per_load);
      loadGeneric<T, elements_per_load>(
        peer + i * elements_per_load,
          shared_mem + smem_offset + np2 * N * blockDim.x + i * elements_per_load);          
    }
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      reduction_op(self[i], peer[i]);
    }

    // write self back to smem
    #pragma unroll
    for (unsigned int i = 0; i < total_loads; ++i) {
      loadGeneric<T, elements_per_load>(shared_mem + smem_offset + i * elements_per_load, self + i * elements_per_load);
    }
  }
  block_sync::sync<Aligned>();

  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      // vectorized load from smem to regs
      T self[N];
      T peer[N];
      #pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        loadGeneric<T, elements_per_load>(
          self + i * elements_per_load,
            shared_mem + smem_offset + i * elements_per_load);
        loadGeneric<T, elements_per_load>(
          peer + i * elements_per_load,
            shared_mem + smem_offset + factor * N * blockDim.x + i * elements_per_load);          
      }

      #pragma unroll
      for (int i = 0; i < N; ++i) {
        reduction_op(self[i], peer[i]);
      }

      // write self back to smem
      #pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        loadGeneric<T, elements_per_load>(shared_mem + smem_offset + i * elements_per_load, self + i * elements_per_load);
      }      
    }
    block_sync::sync<Aligned>();
  }

  if (should_write && write_pred) {
    T result[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = out[i];
    }

    T self[N];
    #pragma unroll
    for (unsigned int i = 0; i < total_loads; ++i) {
      loadGeneric<T, elements_per_load>( self + i * elements_per_load, shared_mem + smem_offset + i * elements_per_load);
    }
    #pragma unroll
    for (int i = 0; i < N; ++i) {
      reduction_op(result[i], self[i]);
    }

    if(reduction_size > 1){
      T peer[N];
      #pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        loadGeneric<T, elements_per_load>( peer + i * elements_per_load, 
          shared_mem + smem_offset + N * blockDim.x + i * elements_per_load);
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
        } else {
          // Push second half
          auto push_offset = index_in_chunk + chunk_size;
          auto self_offset = index_in_chunk;
          pushed_avg = inp_val[push_offset];
          self_avg = inp_val[self_offset];
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
    // if(warp_tidy==0){
    //   const int iter_idx = threadIdx.x;
    //   const int valid_group_idx = wid;
    //   int offset = valid_group_idx * BDIMX + iter_idx;
    //   smem[offset] = avg;
    // }
    // block_sync::sync<Aligned>();

    // if(threadIdx.y == 0){
    //   for(int i = 0; i<N; i++){
    //     int offset = i * BDIMX + threadIdx.x;    
    //     out[i] = smem[offset];
    //   }
    // }
    // block_sync::sync<Aligned>();


    // vectorized version
    // [BDIMX, N]
    if(warp_tidy==0){
      const int iter_idx = threadIdx.x;
      const int valid_group_idx = wid;
      int offset = valid_group_idx + iter_idx * N;
      smem[offset] = avg;
    }
    block_sync::sync<Aligned>();

    if(threadIdx.y == 0){
      constexpr unsigned int total_loads = sizeof(T) * N / 16;
      constexpr unsigned int elements_per_load = 16 / sizeof(T);
      #pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        loadGeneric<T, elements_per_load>(
          out + i * elements_per_load,
          smem + N * threadIdx.x + i * elements_per_load);
      }      
    }
    block_sync::sync<Aligned>();
  }