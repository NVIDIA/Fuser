// Simplified CLC Demo Kernel
// This demonstrates Cluster Launch Control without the complexity of TMA
// operations Based on the reference implementation pattern from
// add_clusterlaunchcontrol.cu

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

// ============================================================================
// Utility Functions (simplified from tma1d_clc.cu)
// ============================================================================

__device__ inline unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));
  return smem_ptr_uint;
}

namespace mbarrier {

__device__ inline void init(
    uint32_t smem_barrier_ptr,
    uint32_t thread_count = 1) {
  asm volatile(
      "mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_barrier_ptr),
      "r"(thread_count));
}

__device__ inline uint64_t arriveExpectTX(
    uint32_t smem_barrier_ptr,
    uint32_t tx_count) {
  volatile uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;\n"
               : "=l"(state)
               : "r"(smem_barrier_ptr), "r"(tx_count));
  return state;
}

__device__ inline void waitParity(uint32_t smem_barrier_ptr, uint32_t parity) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile(
      "{\n"
      ".reg .pred                complete;\n"
      "waitLoop:\n"
      "mbarrier.try_wait.parity.shared.b64 complete, [%0], %1;\n"
      "@!complete bra waitLoop;\n"
      "}\n" ::"r"(smem_barrier_ptr),
      "r"(parity));
#else
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.test_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "nanosleep.u32 20;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_barrier_ptr),
      "r"(parity));
#endif
}
} // namespace mbarrier

namespace clc {

// Query next work ID asynchronously (pass pointers directly like reference)
__device__ inline void try_cancel(uint4* nextworkid, uint64_t* bar) {
  asm volatile(
      "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx:"
      ":bytes.b128 [%0], [%1];"
      :
      : "l"(nextworkid), "l"(bar)
      : "memory");
}

// Check if more work is available (pass uint4 directly)
__device__ inline bool query_cancel_is_canceled(uint4 response) {
  int is_canceled;
  asm volatile(
      "{\n"
      " .reg .b128 B128_try_cancel_response;\n"
      " mov.b128 B128_try_cancel_response, {%1, %2};\n"
      " {\n"
      "  .reg .pred P_OUT;\n"
      "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P_OUT, "
      "B128_try_cancel_response;\n"
      "  selp.b32 %0, 1, 0, P_OUT;\n"
      " }\n"
      "}"
      : "=r"(is_canceled)
      : "l"(((uint64_t*)&response)[0]), "l"(((uint64_t*)&response)[1])
      : "memory");
  return is_canceled != 0;
}

// Extract the new block ID from CLC response (pass uint4 directly)
__device__ inline int query_cancel_get_first_ctaid_x(uint4 response) {
  int new_cta_id;
  asm volatile(
      "{\n"
      " .reg .b128 B128_try_cancel_response;\n"
      " mov.b128 B128_try_cancel_response, {%1, %2};\n"
      " clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, "
      "B128_try_cancel_response;\n"
      "}"
      : "=r"(new_cta_id)
      : "l"(((uint64_t*)&response)[0]), "l"(((uint64_t*)&response)[1])
      : "memory");
  return new_cta_id;
}

} // namespace clc

// ============================================================================
// Simple CLC Kernel: Element-wise Multiplication with Dynamic Work Assignment
// ============================================================================

__global__ void simple_clc_multiply(
    float* out,
    const float* a,
    const float* b,
    int total_elements) {
  const int elements_per_block = blockDim.x;

  // ============ CLC SETUP ============
  __shared__ uint64_t bar;
  __shared__ uint4 nextworkid[2];

  // Get shared memory addresses
  uint32_t bar_addr = toSmem(&bar);

  // Initialize barrier (thread 0 only)
  if (threadIdx.x == 0) {
    mbarrier::init(bar_addr, blockDim.x);
  }

  __syncthreads();

  // CLC state variables
  int parity = 0;
  int bx = blockIdx.x;
  bool valid = true;
  uint32_t arvtx =
      threadIdx.x == 0 ? sizeof(uint4) : 0; // Only thread 0 has non-zero
  int iteration = 0;

  // ============ CLC WORK LOOP ============
  do {
    // Store current bx for CTA-0
    if (threadIdx.x == 0) {
      printf("CTA-%d processing bx-%d\n", blockIdx.x, bx); // Print as we go
    }

    // Query next work ID (thread 0 only)
    if (threadIdx.x == 0) {
      clc::try_cancel(
          nextworkid + parity, &bar); // Pointer arithmetic like reference
    }

    // All threads arrive at barrier, thread 0 has non-zero arvtx
    mbarrier::arriveExpectTX(bar_addr, arvtx);

    // ============ DO WORK ============
    int global_idx = bx * elements_per_block + threadIdx.x;

    if (global_idx < total_elements) {
      out[global_idx] = a[global_idx] * b[global_idx];
    }

    // ============ CLC COMPLETION (matching reference pattern) ============
    // Wait for CLC response (barrier ensures nextworkid[parity^1] consumed)
    mbarrier::waitParity(bar_addr, parity);

    // Decode response (pass uint4 directly like reference)
    valid = clc::query_cancel_is_canceled(nextworkid[parity]);
    bx = clc::query_cancel_get_first_ctaid_x(nextworkid[parity]);

    // Toggle parity
    parity ^= 1;
    iteration++;

  } while (valid);

  if (threadIdx.x == 0) {
    printf(
        "\n[CLC Demo] CTA-%d completed %d different tiles\n",
        blockIdx.x,
        iteration);
  }
}

// ============================================================================
// Host Code
// ============================================================================

int main() {
  printf("=== CLC Demo: Simple Element-wise Multiplication ===\n\n");

  // Problem size
  const int block_size = 1024;
  const int grid_size = 1024;
  const int N = block_size * grid_size;

  printf("Problem size: %d elements\n", N);
  printf("Block size: %d threads\n", block_size);
  printf("Grid size: %d blocks\n\n", grid_size);

  // Allocate host memory
  float* h_a = new float[N];
  float* h_b = new float[N];
  float* h_out = new float[N];

  // Initialize input data
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f + i * 0.001f;
    h_b[i] = 2.0f - i * 0.0005f;
  }

  // Allocate device memory
  float *d_a, *d_b, *d_out;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));

  // Copy to device
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

  printf("Test 1: Standard launch (no cluster)\n");
  printf("--------------------------------------\n");
  simple_clc_multiply<<<grid_size, block_size>>>(d_out, d_a, d_b, N);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("ERROR: %s\n", cudaGetErrorString(err));
  } else {
    printf("Success!\n");
  }

  // Copy results back
  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify results
  printf("\nVerifying results...\n");
  bool correct = true;
  for (int i = 0; i < N; i++) {
    float expected = h_a[i] * h_b[i];
    if (fabsf(h_out[i] - expected) > 1e-5) {
      printf(
          "Mismatch at index %d: got %f, expected %f\n", i, h_out[i], expected);
      correct = false;
      break;
    }
  }

  if (correct) {
    printf("✓ Results verified correctly!\n");
  } else {
    printf("✗ Results incorrect!\n");
  }

  // Cleanup
  delete[] h_a;
  delete[] h_b;
  delete[] h_out;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  return 0;
}
