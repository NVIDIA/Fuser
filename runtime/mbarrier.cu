// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Reference:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-barrier
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

namespace mbarrier {

__device__ inline void init(uint64_t& smem_barrier, int thread_count = 1) {
  uint32_t smem_int_ptr = toSmem(&smem_barrier);
  asm volatile(
      "mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
      "r"(thread_count));
}

__device__ inline void inval(uint64_t& smem_barrier, int thread_count = 1) {
  uint32_t smem_int_ptr = toSmem(&smem_barrier);
  asm volatile(
      "mbarrier.inval.shared.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
      "r"(thread_count));
}

__device__ inline void arrive(uint64_t& smem_barrier) {
  uint32_t smem_int_ptr = toSmem(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .b64 state; \n"
      "mbarrier.arrive.shared.b64   state, [%0];\n"
      "}\n" ::"r"(smem_int_ptr));
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

__device__ inline void arriveExpectTx(uint64_t& smem_barrier, uint32_t bytes) {
  uint32_t smem_int_ptr = toSmem(&smem_barrier);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem_int_ptr),
      "r"(bytes));
}

#endif // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

__device__ inline void wait(uint64_t& smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = toSmem(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
}

} // namespace mbarrier

#endif // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
