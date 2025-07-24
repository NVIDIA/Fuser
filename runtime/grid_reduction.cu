// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// Inter-block reduction.
//
// The gridReduce function performs point-wise reductions of scalars across
// thread blocks. Thread blocks are disjointly partitioned into groups,
// "reduction segments", that are collectively defined by boolean template
// parameters, X_BLOCK, Y_BLOCK and Z_BLOCK. Each of X/Y/Z_BLOCK determines
// whether thread blocks along the dimension should be grouped into the same
// reduction segment. Cross-block reducitons are independently done within each
// segment and generates distinctive results per segment. For instance, if all
// of X/Y/Z_BLOCK are true, reductions will be done across all thread blocks
// since there will be just a single segment consisting of all thread blocks. If
// none of them are true, each thread block will become a segment by itself, so
// no reduction will be performed.
//
// The input scalars to reduce within each segment are a certain subset of
// thread-private scalars provided as part of the gridReduce function
// parameters. Boolean template parameters, X_THREAD, Y_THREAD and Z_THREAD,
// determine which subset of the scalars should be used for inter-block
// reductions. Specifically, all the input scalars of threads along each
// dimension will be used when X/Y/Z_THREAD are true. Otherwise, only the value
// held at offset 0 of each dimension will be used. Thus, for example, if all of
// X/Y/Z_THREAD are true, the scalars of all threads in each block will
// participate in inter-block reductions. If all of them are false, only one
// scalar of the thread at threadIdx.x == threadIdx.y == threadIdx.z == 0 will
// be used. In the code below, we call the subset of threads a "reduction
// block". "Participating" thread dimensions here are similar to the
// "non-participating" block dimensions. They come from a block dimension that
// has not been reduced before hitting this grid reduction.
//
// Inter-block reductions perform point-wise reductions of scalars of reduction
// blocks within each reduction segment. More specifically, let rb be a
// reduction block and rs be a reduction segment. Let IN(thread_idx, block_idx)
// denote the input scalar of thread at thread_idx and block_idx. The result of
// each reduction segment, OUT(thread_idx, block_idx_out), is defined only for
// each thread_idx in thread block block_idx_out in the segment as follows:
//
//   OUT(thread_idx, block_idx_out) =
//     Reduction of IN(thread_idx, block_idx) for
//       all block_idx in a reduction segment
//
// OUT is not given for all threads that are not in block_idx_out and the
// reduction block.
//
// See also the function comment of gridReduce.

namespace reduction {

// Reduces all the reduction blocks in each reduction segment. This is the
// "cleanup" stage of a grid reduction.
//
// This is only called by one thread block per reduction segment. The input
// reduction blocks of the segment are stored in an intermediate buffer pointed
// by parameter in. Template parameters X/Y/Z_THREAD denote how the reduction
// block is formed.
//
// The size of a reduction block is by definition smaller or equal to the size
// of a thread block. We use the remaining threads to parallelize reductions
// across reduction blocks. For example, when X/Y/Z_THREAD = {true, false,
// false}, we use blockDim.y*blockDim.z threads for each output value. This is
// done first by loading the input values in parallel and then by reducing
// across threads of dimensions whose XYZ_THREAD are false.
//
// Note that what is done here after the loading from global memory is similar
// to what the existing blockReduce function does.
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void gridReduceLastBlock(
    T& out,
    const volatile T* in,
    const nvfuser_index_t
        grid_reduction_segment_size, // Number of reductions across
                                     // grid reduce dimensions
    const nvfuser_index_t
        block_reduction_segment_size, // Number of reductions across the block
    Func reduction_op,
    T* shared_buf,
    bool write_pred,
    T init_val,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim) {
  // We have to do num_reductions across reduction_size. The reductions are
  // contiguous, but offset by reduction_size. There is an entry in "in" for
  // every block, and every thread marked as true. Threads in dimensions marked
  // as false can be used to parallelize the reduction.

  // Find the reduction id of the participating threads
  const auto block_reduction_segment_idx =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, block_dim);

  // Find an id associated within a reduction segment for all
  // "non-participating" threads, which will parallelize the reductions for the
  // "participating" threads
  const auto id_in_block_segment =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, block_dim);

  // Stride by the "non-participating" threads
  const auto input_stride_for_thread_in_segment =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(block_dim);

  T inp = init_val;

  // Block stride across the reduction until we only have one value per thread
  for (nvfuser_index_t reduction_i = id_in_block_segment;
       reduction_i < grid_reduction_segment_size;
       reduction_i += input_stride_for_thread_in_segment) {
    auto work_buf_offset = reduction_i * block_reduction_segment_size +
        block_reduction_segment_idx;
    reduction_op(inp, in[work_buf_offset]);
  }

  // Block reduce the per thread values into per "participating" thread values
  T inp_tmp = init_val;
  blockReduce<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
      inp_tmp, inp, reduction_op, shared_buf, true, init_val, block_dim);
  const bool should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);
  if (should_write && write_pred) {
    reduction_op(out, inp_tmp);
  }
}

// Reduces per-thread values across threads and thread blocks.
//
// Function parameters:
// - out: Per-thread output location
// - inp_val: Per-thread input value
// - reduction_op: Scalar reduction function
// - work_buf: Temporary buffer for cross-block reductions
// - sync_flags: A vector of integers for synchronizations
// - shared_buf: Shared memory buffer for intra-block reduction
//
// Thread has valid results based on if it's the last block in the grid
// reduction dimension
//
// Template parameters:
// - X/Y/Z_BLOCK/THREAD: When true, reduces across thread blocks along the X/Y/Z
//   dimensions
// - PERSISTENT_REDUCTION: Indicates grid reduction will be called in a loop, or
//   the result of the grid reduction will be broadcasted and used across the
//   grid. These requires cross grid communication and the grid synchronizations
//   here to actually synchronize across the entire grid. When false the grid is
//   not synchronized, the last block just waits for everyone else to finish and
//   the other blocks can exit early.
// - T: Scalar data type of input/output data
// - Func: Type of scalara reduction function
//
// Template parameters X/Y/Z_BLOCK define a group of thread blocks that are
// reduced together. We call it a reduction segment. Some examples are:
//
// Case 1: X/Y/Z_BLOCK == true/true/true -> There is only one segment, which
// includes all thread blocks. It is effecively the same as the grid.
//
// Case 2: X/Y/Z_BLOCK == false/false/false -> Each thread block comprises an
// individual segment by itself.
//
// Case 3: X/Y/Z_BLOCK == true/false/false -> Each segment contains thread
// blocks that have the same blockDim.x. There will be blockDim.y*blockDim.z
// such segments.
//
// X/Y/Z_THREAD also works similarly as X/Y/Z_BLOCK and defines a
// group of threads that are reduced togather.
//
// After the function completes, only one thread block per reduction segment
// gets valid reduction results. There is no guarantee which particular block
// gets the final results.
//
// entrance_ind and n_entrances are allowed when PERSISTENT_REDUCTION = false.
// If a grid reduction call is only called once per thread, entrance_ind == 0
// and n_entrances == 1. However, grid reduction can be called in a loop in a
// thread, in that case entrance_ind is the count of times the function has been
// called, and n_entrances is the total number of times it will be called.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void gridReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim) {
  T block_reduction_val = init_val;

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
    blockReduce<X_THREAD, Y_THREAD, Z_THREAD, Aligned>(
        block_reduction_val,
        inp_val,
        reduction_op,
        shared_buf,
        read_pred,
        true,
        init_val,
        block_dim);
  } else if (read_pred) {
    block_reduction_val = inp_val;
  }

  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(block_dim);

  // Number of reductions in the grid
  const nvfuser_index_t grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, block_dim);
    auto work_buf_offset =
        block_offset * block_reduction_segment_size + thread_offset;
    work_buf[work_buf_offset] = block_reduction_val;
  }
  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);

  } else {
    // Use a different sync flag for each call
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[entrance_ind * grid_segment_size + idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
        out,
        (T*)work_buf,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op,
        shared_buf,
        write_pred,
        init_val,
        block_dim);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);
  }
}

// This is just a wrapper of the above grid reduction routine to
// measure the elapsed cycles. The measurement must be done just by
// one thread, and in this case it should be done by one of the
// threads in the last thread block.
#ifdef NVFUSER_PROFILE_KERNEL
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void gridReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim,
    int64_t& cycles,
    int64_t& count) {
  int64_t start_counter = 0;

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  gridReduce<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      PERSISTENT_REDUCTION,
      Aligned,
      T,
      Func>(
      out,
      inp_val,
      reduction_op,
      work_buf,
      sync_flags,
      shared_buf,
      read_pred,
      write_pred,
      init_val,
      entrance_ind,
      n_entrances,
      block_dim);

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}
#endif // NVFUSER_PROFILE_KERNEL

template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void gridReduce2PartialReduction(
    const T& inp_val,
    T init_val,
    Func reduction_op,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim,
    volatile T* work_buf,
    T* shared_buf,
    bool read_pred,
    nvfuser_index_t grid_reduction_segment_size,
    nvfuser_index_t idx_in_grid_segment,
    nvfuser_index_t block_reduction_segment_size) {
  T block_reduction_val = init_val;

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
    blockReduce<X_THREAD, Y_THREAD, Z_THREAD, Aligned>(
        block_reduction_val,
        inp_val,
        reduction_op,
        shared_buf,
        read_pred,
        true,
        init_val,
        block_dim);
  } else if (read_pred) {
    block_reduction_val = inp_val;
  }

  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, block_dim);
    auto work_buf_offset =
        block_offset * block_reduction_segment_size + thread_offset;
    work_buf[work_buf_offset] = block_reduction_val;
  }
}

// 2-way horizontally fused grid reduction
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T1,
    typename Func1,
    typename T2,
    typename Func2,
    typename BlockDimT>
__device__ void gridReduceGroup(
    T1& out1,
    const T1& inp_val1,
    T1 init_val1,
    Func1 reduction_op1,
    volatile T1* work_buf1,
    T2& out2,
    const T2& inp_val2,
    T2 init_val2,
    Func2 reduction_op2,
    volatile T2* work_buf2,
    int64_t* sync_flags,
    void* shared_buf,
    bool read_pred,
    bool write_pred,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim) {
  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(block_dim);

  // Number of reductions in the grid
  const nvfuser_index_t grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf1 += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  work_buf2 += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  gridReduce2PartialReduction<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      Aligned>(
      inp_val1,
      init_val1,
      reduction_op1,
      block_dim,
      work_buf1,
      (T1*)shared_buf,
      read_pred,
      grid_reduction_segment_size,
      idx_in_grid_segment,
      block_reduction_segment_size);

  gridReduce2PartialReduction<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      Aligned>(
      inp_val2,
      init_val2,
      reduction_op2,
      block_dim,
      work_buf2,
      (T2*)shared_buf,
      read_pred,
      grid_reduction_segment_size,
      idx_in_grid_segment,
      block_reduction_segment_size);

  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);
  } else {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[entrance_ind * grid_segment_size + idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
        out1,
        work_buf1,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op1,
        (T1*)shared_buf,
        write_pred,
        init_val1,
        block_dim);
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
        out2,
        work_buf2,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op2,
        (T2*)shared_buf,
        write_pred,
        init_val2,
        block_dim);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);
  }
}

#ifdef NVFUSER_PROFILE_KERNEL
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T1,
    typename Func1,
    typename T2,
    typename Func2,
    typename BlockDimT>
__device__ void gridReduceGroup(
    T1& out1,
    const T1& inp_val1,
    T1 init_val1,
    Func1 reduction_op1,
    volatile T1* work_buf1,
    T2& out2,
    const T2& inp_val2,
    T2 init_val2,
    Func2 reduction_op2,
    volatile T2* work_buf2,
    int64_t* sync_flags,
    void* shared_buf,
    bool read_pred,
    bool write_pred,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim,
    int64_t& cycles,
    int64_t& count) {
  int64_t start_counter = 0;

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  gridReduceGroup<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      PERSISTENT_REDUCTION,
      Aligned,
      T1,
      Func1,
      T2,
      Func2>(
      out1,
      inp_val1,
      init_val1,
      reduction_op1,
      work_buf1,
      out2,
      inp_val2,
      init_val2,
      reduction_op2,
      work_buf2,
      sync_flags,
      shared_buf,
      read_pred,
      write_pred,
      entrance_ind,
      n_entrances,
      block_dim);

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}
#endif // NVFUSER_PROFILE_KERNEL

// This performs a single reduction step, combining a single element "in" with
// a previous value "work". For a serial grid reduction, "work" resides in
// global memory, while "in" and "out" are in registers.
//
// If the write predicate is false, this function returns early (noop). If the
// read predicate is false, "init" is used in place of "in".
//
// If first_step is false, "work" will be read and reduction_op will be called.
// The result will be written back to "work" unless last_step is true.
template <int64_t vec_size, typename T, typename Func>
__device__ void serialReductionStep(
    T* out,
    T* in,
    T init,
    volatile T* work,
    Func reduction_op,
    bool first_step,
    bool last_step,
    bool read_pred,
    bool write_pred) {
  if (!write_pred) {
    return;
  }
  if (read_pred) {
    loadGeneric<T, vec_size>(out, in);
  } else {
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out[i] = init;
    }
  }
  if (!first_step) {
    T work_reg[vec_size];
    loadGlobalToLocal<T, vec_size, true, CacheOp::Global>(work_reg, work);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      reduction_op(out[i], work_reg[i]);
    }
  }
  if (!last_step) {
    loadLocalToGlobal<T, vec_size, true>(work, out);
  }
}

// check required transactions based on data type and vectorization factor
// ensure each thread in each transaction has no more than 16 bytes which
// is the maximum allowed vectorization width.
template <typename T, int vec_size>
constexpr __device__ int getTransactions() {
  constexpr int total_bytes = vec_size * sizeof(T);
  return total_bytes <= 16 ? 1 : total_bytes / 16;
}

template <typename T, int vec_size>
constexpr __device__ int getElementsPerTransaction() {
  return vec_size * sizeof(T) <= 16 ? vec_size : 16 / sizeof(T);
}

// calculate elements per section
__inline__ __device__ nvfuser_index_t getElementsPerSection(
    nvfuser_index_t row_len,
    nvfuser_index_t col_len,
    nvfuser_index_t elements_per_thread) {
  return row_len * col_len * elements_per_thread;
}

// calculate offset within a section
__inline__ __device__ nvfuser_index_t getOffsetWithinSection(
    nvfuser_index_t row_len,
    nvfuser_index_t row_id,
    nvfuser_index_t col_id,
    nvfuser_index_t elements_per_thread) {
  return (row_id * row_len + col_id) * elements_per_thread;
}
// vectorized reduction
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    int vec_size,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void iterGroupedGridReduceLastBlock(
    T* out,
    const volatile T* in,
    const nvfuser_index_t
        grid_reduction_segment_size, // Number of reductions across
                                     // grid reduce dimensions
    const nvfuser_index_t
        block_segment_size, // Number of reductions across the block
    Func reduction_op,
    T* shared_buf,
    bool write_pred,
    T init_val,
    const nvfuser_index_t grid_segment_size,
    const nvfuser_index_t idx_in_grid_segment,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim) {
  // We have to do num_reductions across reduction_size. The reductions are
  // contiguous, but offset by reduction_size. There is an entry in "in" for
  // every block, and every thread marked as true. Threads in dimensions marked
  // as false can be used to parallelize the reduction.

  // Find the reduction id of the participating threads
  const auto block_reduction_segment_idx =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, block_dim);

  // Find an id associated within a reduction segment for all
  // "non-participating" threads, which will parallelize the reductions for the
  // "participating" threads
  const auto id_in_block_segment =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, block_dim);

  // index into iteration dim.
  // Its calculation is same to that in [iterGroupedGridReduce]. Becuase when
  // [iterGroupedGridReduceLastBlock] is called from [iterGroupedGridReduce],
  // X_THREAD, Y_THREAD, Z_THREAD are flipped.
  const auto thread_offset =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, block_dim);

  // Stride by the "non-participating" threads
  const auto input_stride_for_thread_in_segment =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(block_dim);

  constexpr unsigned int max_align_bytes = 16;
  constexpr unsigned int vec_bytes = sizeof(T) * vec_size;
  constexpr unsigned int align_bytes =
      vec_bytes > max_align_bytes ? max_align_bytes : vec_bytes;
  // Ensure alignment for vectorized load/store to smem in grouped block
  // reduction
  __align__(align_bytes) T inp[vec_size];
#pragma unroll
  for (int i = 0; i < vec_size; i++) {
    inp[i] = init_val;
  }

  // Max vectorized load/store size is 16 bytes, if each thread has more than
  // 16 bytes, split into multiple sections to ensure each thread occupies only
  // 16 bytes at most. For example, if each thread has 8 fp32 which occupies 32
  // bytes, split into 2 sections, in each secdtion each thread holds 4 fp32 or
  // 16 bytes. Thread-0 processes elements [0,7], the first 4 elements [0,3] are
  // stored in the first section and the last 4 elements [4,7] are stored in the
  // 2nd section. The data layout in gmem is:
  //         |-----------section 1-----------|-----------section 2-----------|
  // TIDx:   |000|001|002|003|004|005|006|007|000|001|002|003|004|005|006|007|
  // GMEM:   |000|016|032|048|064|080|096|112|128|144|160|176|192|208|224|240|
  // Element:|000|008|016|024|032|040|048|056|004|012|020|028|036|044|052|060|
  // This layout ensures coalesced access to gmem and each transaction loads 128
  // bytes.
  constexpr auto n_transactions = getTransactions<T, vec_size>();
  constexpr auto n_elements_per_transaction =
      getElementsPerTransaction<T, vec_size>();
  const auto elements_per_section = getElementsPerSection(
      block_segment_size * grid_segment_size, // row len
      grid_reduction_segment_size, // col len
      n_elements_per_transaction);
  // Block stride across the reduction until we only have one value per thread
  for (nvfuser_index_t reduction_i = id_in_block_segment;
       reduction_i < grid_reduction_segment_size;
       reduction_i += input_stride_for_thread_in_segment) {
    auto offset_in_section = getOffsetWithinSection(
        block_segment_size * grid_segment_size, // row len
        reduction_i, // row id
        block_segment_size * idx_in_grid_segment + thread_offset, // col id
        n_elements_per_transaction);

#pragma unroll
    for (auto i = 0; i < n_transactions; i++) {
      auto i_offset = i * n_elements_per_transaction;
      T in_reg[n_elements_per_transaction];
      loadGlobalToLocal<T, n_elements_per_transaction, true, CacheOp::Global>(
          &in_reg[0],
          const_cast<T*>(in + elements_per_section * i + offset_in_section));
#pragma unroll
      for (auto j = 0; j < n_elements_per_transaction; j++) {
        reduction_op(inp[i_offset + j], in_reg[j]);
      }
    }
  }

  // Block reduce the per thread values into per "participating" thread values.
  // inp_tmp stores output results, not being vectorized loaded to smem, no need
  // to enforce alignment.
  T inp_tmp[vec_size];
#pragma unroll
  for (int i = 0; i < vec_size; i++) {
    inp_tmp[i] = init_val;
  }
  blockIterGroupedYdimReduce<Aligned, vec_size>(
      inp_tmp, inp, reduction_op, shared_buf, true, init_val, block_dim);
  const bool should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);
  if (should_write && write_pred) {
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      reduction_op(out[i], inp_tmp[i]);
    }
  }
}

// Main algorithm is same to gridReduce: start with block reduce then write
// results to gmem, the last block load from gmem and finalize with a block
// reduction. Main differences:
// (1) each thread in the iter dim does [vec_size] reductions instead of 1.
// (2) using [blockIterGroupedYdimReduce] instead of [blockReduce].
// (3) ensures vectorized load/store to gmem.
// Specifically, the new para [vec_size] is the vecotrization factor in the
// iteration dimension. It is used in outer reduction to reduce calling this
// grid reduction from [vec_size] times to only 1 time. Its value is limited
// to 1, 2, 4, 8, 16 based on the hardware support and input data type.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    int vec_size,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void iterGroupedGridReduce(
    T* out,
    const T* inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val,
    // block_dim is basically just blockDim (wrapped as DefaultBlockDim) if
    // there is no warp specialization in the kernel. If there is warp
    // specialization, block_dim is the the dimension of the compute warps.
    BlockDimT block_dim) {
  // inp or block reduction results
  T block_reduction_val[vec_size];

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      block_reduction_val[i] = init_val;
    }
    blockIterGroupedYdimReduce<Aligned, vec_size>(
        block_reduction_val,
        inp_val,
        reduction_op,
        shared_buf,
        read_pred,
        true,
        init_val,
        block_dim);
  } else if (read_pred) {
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      block_reduction_val[i] = inp_val[i];
    }
  }

  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of reductions in each block
  const auto block_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(block_dim);

  // Number of reductions in the grid
  const nvfuser_index_t grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, block_dim);

    // Max vectorized load/store size is 16 bytes, if each thread has more than
    // 16 bytes, split into multiple sections to ensure each thread occupies
    // only 16 bytes at most. For example, if each thread has 8 fp32 which
    // occupies 32 bytes, split into 2 sections, in each secdtion each thread
    // holds 4 fp32 or 16 bytes. Thread-0 processes elements [0,7], the first 4
    // elements [0,3] are stored in the first section and the last 4 elements
    // [4,7] are stored in the 2nd section. The data layout in gmem is:
    //         |-----------section 1-----------|-----------section 2-----------|
    // TIDx:   |000|001|002|003|004|005|006|007|000|001|002|003|004|005|006|007|
    // GMEM:   |000|016|032|048|064|080|096|112|128|144|160|176|192|208|224|240|
    // Element:|000|008|016|024|032|040|048|056|004|012|020|028|036|044|052|060|
    // This layout ensures coalesced access to gmem and each transaction loads
    // 128 bytes.
    constexpr auto n_transactions = getTransactions<T, vec_size>();
    constexpr auto n_elements_per_transaction =
        getElementsPerTransaction<T, vec_size>();

    // get elements per section, used to offset between different sections
    // number of elements in each thread: [n_elements_per_transaction]
    // number of threads in each row: [block_segment_size] * [grid_segment_size]
    // number of rows in each section: [grid_reduction_segment_size]
    auto elements_per_section = getElementsPerSection(
        block_segment_size * grid_segment_size, // row len
        grid_reduction_segment_size, // col len
        n_elements_per_transaction);

    // index to the right position in [work_buf] to store block reduction
    // results. Consider a typical outer reduction case where iteration dim is
    // TIDx and BIDx and reduction dim is TIDy and BIDy. block_offset = BIDy
    // block_segment_size = blockDim.x
    // grid_segment_size = gridDim.x
    // idx_in_grid_segment = BIDx
    // thread_offset = TIDx
    auto offset_in_section = getOffsetWithinSection(
        block_segment_size * grid_segment_size, // row len
        block_offset, // row id
        block_segment_size * idx_in_grid_segment + thread_offset, // col id
        n_elements_per_transaction);

#pragma unroll
    for (int i = 0; i < n_transactions; i++) {
      loadLocalToGlobal<T, n_elements_per_transaction, true>(
          &work_buf[elements_per_section * i + offset_in_section],
          &block_reduction_val[i * n_elements_per_transaction]);
    }
  }

  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);

  } else {
    // there is only one vectorized call
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    iterGroupedGridReduceLastBlock<
        !X_THREAD,
        !Y_THREAD,
        !Z_THREAD,
        Aligned,
        vec_size>(
        out,
        (T*)work_buf,
        grid_reduction_segment_size,
        block_segment_size,
        reduction_op,
        shared_buf,
        write_pred,
        init_val,
        grid_segment_size,
        idx_in_grid_segment,
        block_dim);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment],
        grid_reduction_segment_size,
        block_dim);
  }
}
} // namespace reduction
