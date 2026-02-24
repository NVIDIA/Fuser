// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on

#include "tests/cpp/tma_test_kernels.h"

#include <cassert>
#include <cstdint>

namespace nvfuser {

// TMA 1D bulk copy kernel: GMEM(src) -> SMEM -> GMEM(dst).
// Inspired by DeepEP's tma_load_1d / tma_store_1d pattern.
// A single elected thread issues all TMA operations while the rest of the warp
// idles. mbarrier synchronization ensures the async TMA load completes before
// the TMA store reads from shared memory.
//
// Dynamic shared memory layout (128-byte aligned):
//   [0, num_bytes)           : data staging buffer
//   [num_bytes, num_bytes+8) : mbarrier (uint64_t, 16-byte aligned since
//                              num_bytes is a multiple of 16)
__global__ void __launch_bounds__(32, 1)
    tma_copy_1d_kernel(
        void* __restrict__ dst,
        const void* __restrict__ src,
        int num_bytes) {
  extern __shared__ __align__(128) uint8_t smem[];

  auto* mbar = reinterpret_cast<uint64_t*>(smem + num_bytes);
  auto smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  auto mbar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(mbar));

  if (threadIdx.x == 0) {
    // Initialize mbarrier with arrival count = 1
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        ::"r"(mbar_addr), "r"(1));
    // Ensure init is visible cluster-wide before any use
    asm volatile(
        "fence.mbarrier_init.release.cluster;" :::);
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    // Announce expected number of transaction bytes on the mbarrier
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        ::"r"(mbar_addr), "r"(num_bytes));

    // TMA Load: GMEM -> SMEM (async, completed via mbarrier)
    asm volatile(
        "cp.async.bulk.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];\n"
        ::"r"(smem_addr),
          "l"(src),
          "r"(num_bytes),
          "r"(mbar_addr)
        : "memory");

    // Block until the mbarrier phase flips (TMA load completed).
    // Phase 0 is the initial phase after mbarrier.init.
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "TMA_COPY_WAIT_LOAD:\n"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1;\n"
        "@P1 bra TMA_COPY_LOAD_DONE;\n"
        "bra TMA_COPY_WAIT_LOAD;\n"
        "TMA_COPY_LOAD_DONE:\n"
        "}"
        ::"r"(mbar_addr), "r"(0));

    // TMA Store: SMEM -> GMEM
    // No fence.proxy.async needed here because both the load and store
    // operate through the async proxy; the mbarrier completion already
    // establishes the necessary ordering (cf. DeepEP intranode.cu).
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group"
        " [%0], [%1], %2;\n"
        ::"l"(dst),
          "r"(smem_addr),
          "r"(num_bytes)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile(
        "cp.async.bulk.wait_group.read 0;" ::: "memory");

    // Invalidate mbarrier before kernel exit
    asm volatile(
        "mbarrier.inval.shared::cta.b64 [%0];"
        ::"r"(mbar_addr));
  }
}

void launchTmaCopy1D(
    void* dst,
    const void* src,
    int num_bytes,
    cudaStream_t stream) {
  assert(num_bytes > 0 && "num_bytes must be positive");
  assert(
      num_bytes % 16 == 0 &&
      "cp.async.bulk requires size to be a multiple of 16 bytes");

  // data buffer + mbarrier (8 bytes)
  int smem_size = num_bytes + static_cast<int>(sizeof(uint64_t));
  tma_copy_1d_kernel<<<1, 32, smem_size, stream>>>(dst, src, num_bytes);
}

} // namespace nvfuser
