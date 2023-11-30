// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace grid_sync {

// Get the first bit in a 64 bit integer
#define FIRST_UINT64_BIT ((uint64_t)1 << (sizeof(uint64_t) * 8 - 1))

template <typename T>
__device__ T globalAsVolatile(volatile T& global_val) {
  return global_val;
}

// A grid synchronization that can be called multiple times in a kernel assuming
// all the blocks fit on device at once. The semaphore is an integer semaphore
// assumed to be initialized to 0 before launching the kernel. The persistent
// option should be envoked if this sync will be called multiple times in one
// kernel (i.e. having a grid reduce within a loop). Having multiple grid syncs
// called once in the same kernel does not require persistent mode. Segment size
// is the number of blocks participating in the sync in the dimensions marked by
// [X,Y,Z]_BLOCK. The granularity of this sync are those dimensions. I.E.
// Marking X and Y but not Z means there should be Z semaphores of size X*Y.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool PERSISTENT,
    bool Aligned>
__device__ void sync(
    int64_t& semaphore,
    const uint64_t& segment_size,
    const bool last_block) {
  // Finish all global memory transactions before synchronizing
  __threadfence();

  // Synchronize all threads in a block before synchronizing blocks
  block_sync::sync<Aligned>();

  // Only allow linear_tid == 0 to participate in the synchronization
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Get increment value, only want a single block to have the large
    // increment, doesn't really matter which one, the goal is to flip/flop the
    // first bit of a uint64_t value, since our semaphores are actualy int64_t
    // we will just reinterpret_cast it to act as a uint64_t
    uint64_t semaphore_increment = 1;

    // Makes the assumption that blocks are in increasing order, this is not
    // guaranteed by CUDA but this is the current behavior, and unlikely to
    // change.
    if (last_block) {
      semaphore_increment = FIRST_UINT64_BIT - (segment_size - 1);
    }

    uint64_t oldArrive =
        atomicAdd(reinterpret_cast<uint64_t*>(&semaphore), semaphore_increment);

    // If for persistent kernels, lock all blocks until the semaphore has been
    // reached. Make sure we access semaphore as a volatile address so we get
    // the global memory updates.
    unsigned int ns = 8;
    while ((PERSISTENT || last_block) &&
           ((oldArrive ^ globalAsVolatile(semaphore)) & FIRST_UINT64_BIT) ==
               0) {
      // Put a sleep here so we have some breaks in probing the global
      // semaphore, giving a better chance for other warps/blocks to catch up.
#if __CUDA_ARCH__ >= 700
      // __nanosleep only available on compute capability 7.0 or higher
      __nanosleep(ns); // avoids busy waiting
      if (ns < 256) {
        ns *= 2;
      }
#endif
    }
  }

  // Sync block to make sure all other threads are waiting on the sync
  block_sync::sync<Aligned>();
}

template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool PERSISTENT,
    bool Aligned>
__device__ void sync(int64_t& semaphore, const uint64_t& segment_size) {
  sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT, Aligned>(
      semaphore,
      segment_size,
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim));
}

// Grid sync that can be called multiple times in the same kernel without all
// blocks being resident on device. This allows grid sync to be called multiple
// times as long as it's not broadcasted on the parallel axis it was reduced on.
//
// n_entrances is how many times every block is expected to enter into this
// function. All blocks must enter n_entrances times. The last block is only
// allowed to proceed once all other blocks have entered n_entrance
// times.
//
// Note that this is not currently used by grid and welford reduction
// as they use a separate sync flag for each each grid sync call.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, bool Aligned>
__device__ void sync(
    int64_t& semaphore,
    const uint64_t& segment_size,
    const nvfuser_index_t n_entrances) {
  // Finish all global memory transactions before synchronizing
  __threadfence();

  // Synchronize all threads in a block before synchronizing blocks
  block_sync::sync<Aligned>();

  // Only allow linear_tid == 0 to participate in the synchronization
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Makes the assumption that blocks are in increasing order, this is not
    // guaranteed by CUDA but this is the current behavior, and unlikely to
    // change.
    bool last_block =
        index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    if (last_block) {
      int64_t finished_val =
          ((int64_t)(index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(
                         gridDim) -
                     1)) *
          ((int64_t)n_entrances);

      unsigned int ns = 8;
      // Last block needs to wait for all other blocks to finish
      while (globalAsVolatile(semaphore) < finished_val) {
#if __CUDA_ARCH__ >= 700
        // __nanosleep only available on compute capability 7.0 or higher
        __nanosleep(ns); // avoids busy waiting
        if (ns < 256) {
          ns *= 2;
        }
#endif
      }
    } else {
      auto old = atomicAdd(reinterpret_cast<uint64_t*>(&semaphore), 1);
    }
  }

  // Sync block to make sure all other threads are waiting on the sync
  block_sync::sync<Aligned>();
}

// Non-blocking function to acquire the semaphore value in each calling thread
__device__ int64_t semaphoreFetch(int64_t* semaphore) {
  int64_t state;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.gpu.b64 %0, [%1];\n"
               : "=l"(state)
               : "l"(semaphore));
#else
  asm volatile("ld.global.cg.b64 %0, [%1];\n" : "=l"(state) : "l"(semaphore));
#endif
  return state;
}

// Sync block then et semaphore to new_value
__device__ void semaphoreRelease(int64_t* semaphore, int64_t new_value) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.gpu.b64 [%0], %1;\n"
                 :
                 : "l"(semaphore), "l"(new_value));
#else
    asm volatile("st.global.cg.b64 [%0], %1;\n"
                 :
                 : "l"(semaphore), "l"(new_value));
#endif
  }
}

// Block waits until fetched semaphore value matches trigger
__device__ void semaphoreWait(int64_t* semaphore, int64_t trigger_value) {
  int64_t status = -1;
  // Cutlass uses a loop like this, and has a facility where any thread can
  // fetch the semaphore value ahead of waiting. This could reduce the wait
  // time potentially but requires placement of the early fetch.
  // https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/semaphore.h
  // while (__syncthreads_and(status != trigger_value)) {
  // As soon as any thread in the block observes the trigger then it is
  // safe to proceed
  while (status != trigger_value) {
    status = semaphoreFetch(semaphore);
  }
  __syncthreads();
}

// Serialize blocks in segments indicated by the [XYZ]_BLOCK template arguments.
// This should be called at the beginning of the section to be serialized.
// Persistent parameter indicates whether first block needs to wait
// (PERSISTENT==true) or if it can proceed assuming the semaphore is
// initialized to zero.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, bool PERSISTENT>
__device__ void blockSerializeWait(int64_t* semaphore) {
  int segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);
  int block_idx_in_segment =
      index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (PERSISTENT || block_idx_in_segment > 0) {
    semaphoreWait(semaphore, block_idx_in_segment);
  }
}

// Serialize blocks in segments indicated by the [XYZ]_BLOCK template arguments.
// This should be called at the end of the section to be serialized.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, bool PERSISTENT>
__device__ void blockSerializeRelease(int64_t* semaphore) {
  int segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);
  int block_idx_in_segment =
      index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
  bool last_block = block_idx_in_segment == segment_size - 1;

  if (last_block) {
    if (PERSISTENT) {
      semaphoreRelease(semaphore, 0);
    }
  } else {
    semaphoreRelease(semaphore, block_idx_in_segment + 1);
  }
}

} // namespace grid_sync
