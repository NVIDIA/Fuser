#include <cstdio>               // printf
#include <cuda.h>               // CUtensormap

#include <cuda_awbarrier_primitives.h> // __mbarrier_*

#include "utils.h"                      // CUDA_CHECK macro

/*
 * Constants.
 */
constexpr int SMEM_W = 32;     // Width of shared memory buffer (in # elements)
constexpr int SMEM_H = 8;      // Height of shared memory buffer (in # elements)


/*
 * PTX wrappers
 */

inline __device__ __mbarrier_token_t barrier_arrive1_tx(
  __mbarrier_t *barrier, uint32_t expected_tx_count
)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
  __mbarrier_token_t token;

  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
               : "=l"(token)
               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count)
               : "memory");
  return token;
}

inline __device__ bool barrier_try_wait_token(__mbarrier_t *barrier, __mbarrier_token_t token)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
  //
  // This function returns a bool, so that software can retry.
  //
  //  The HW only provides best-effort waiting support. The wait time is limited
  //  by the HW capability, after which a fail occurs, in which case the SW is
  //  responsible for retrying.
  int __ready;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(__ready)
               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))),
                 "l"(token)
               : "memory");
  return __ready;
}

inline __device__ void cp_async_bulk_tensor_2d(
  __mbarrier_t *barrier, void *dst, int access_coord_x, int access_coord_y, const CUtensorMap *tensor_desc)
{
  unsigned smem_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
  unsigned smem_barrier_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  uint64_t tensor_desc_ptr = reinterpret_cast<uint64_t>(tensor_desc);

  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2, %3}], [%4];\n"
    :
    : "r"(smem_int_ptr),
      "l"(tensor_desc_ptr),
      "r"(access_coord_x),
      "r"(access_coord_y),
      "r"(smem_barrier_int_ptr)
    : "memory");
}

// Layout of shared memory. It contains:
//
// - a buffer to hold a subset of a tensor,
// - a shared memory barrier.
template <int H, int W>
struct smem_t {

  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  struct alignas(128) tensor_buffer {
    int data[H][W];

    __device__ constexpr int width() {return W;}
    __device__ constexpr int height() {return H;}
  };

  tensor_buffer buffer;

  // Put the barrier behind the tensor buffer to prevent 100+ bytes of padding.
  __mbarrier_t bar;

  __device__ constexpr int buffer_size_in_bytes() {
    return sizeof(tensor_buffer::data);
  }
};


/*
 * Main kernel: takes a TMA descriptor and two coordinates.
 *
 * Loads a tile into shared memory using TMA and prints the tile.
 *
 */
extern "C" __global__ void tma_kernel(const __grid_constant__ CUtensorMap tma_desc, int x_0, int y_0) {
  /*
   * ***NOTE***:
     A CUtensorMap can only be passed as a `const __grid_constant__`
     parameter. Passing a CUtensorMap in any other way from the host to
     device can result in difficult if not impossible to debug failures.
  */

  // Declare shared memory to hold tensor buffer and shared memory barrier.
  __shared__ smem_t<SMEM_H, SMEM_W> smem;

  // Utility variable to elect a leader thread.
  bool leader = (threadIdx.x == 0);


  if (leader) {
    // Initialize barrier. We will participate in the barrier with `blockDim.x`
    // threads.
    __mbarrier_init(&smem.bar, blockDim.x);
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();


  // This token is created when arriving on the shared memory barrier. It is
  // used again when waiting on the barrier.
  __mbarrier_token_t token;

  // Load first  batch
  if (leader) {
    // Initiate bulk tensor copy.
    cp_async_bulk_tensor_2d(&smem.bar, &smem.buffer.data, x_0, y_0, &tma_desc);
    // Arrive with arrival count of 1 and expected transaction count equal to
    // the number of bytes that are copied by cp_async_bulk_tensor_2d.
    token = barrier_arrive1_tx(&smem.bar, smem.buffer_size_in_bytes());
  } else {
    // Other threads arrive with arrival count of 1 and expected tx count of 0.
    token = barrier_arrive1_tx(&smem.bar, 0);
  }

  // The barrier will flip when the following two conditions have been met:
  //
  // - Its arrival count reaches blockDim.x (see __mbarrier_init above).
  //   Typically, each thread will arrive with an arrival count of one so this
  //   indicates that all threads have arrived.
  //
  // - Its expected transaction count reaches smem.buffer_size_in_bytes(). The
  //   bulk tensor operation will increment the transaction count as it copies
  //   bytes.

  // Wait for barrier to flip. Try_wait puts the thread to sleep while waiting.
  // It is woken up when the barrier flips or when a hardware-defined number of
  // clock cycles have passed. In the second case, we retry waiting.
  while(! barrier_try_wait_token(&smem.bar, token)) { };

  // From this point onwards, the data in smem.buffer is readable by all threads
  // participating the in the barrier.

  // Print the data:
  if (leader) {
    printf("\n\nPrinting tile at coordinates x0 = %d, y0 = %d\n", x_0, y_0);

    // Print global x coordinates
    printf("global->\t");
    for (int x = 0; x < smem.buffer.width(); ++x) {
      printf("[%4d] ", x_0 + x);
    }
    printf("\n");

    // Print local x coordinates
    printf("local ->\t");
    for (int x = 0; x < smem.buffer.width(); ++x) {
      printf("[%4d] ", x);
    }
    printf("\n");

    for (int y = 0; y < smem.buffer.height(); ++y) {
      // Print global and local y coordinates
      printf("[%4d] [%2d]\t", y_0 + y, y);
      for (int x = 0; x < smem.buffer.width(); ++x) {
        printf(" %4d  ", smem.buffer.data[y][x]);
      }
      printf("\n");
    }

    // Invalidate barrier. If further computations were to take place in the
    // kernel, this allows the memory location of the shared memory barrier to
    // be repurposed.
    __mbarrier_inval(&smem.bar);
  }
}
