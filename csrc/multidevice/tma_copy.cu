// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
//
// TMA 1D bulk copy kernel (SM90+ / Hopper).
//
// This file implements a TMA-based data copy kernel. The build system
// stringifies it into nvfuser_resources/tma_copy.h (a const char*),
// which is compiled at runtime via NVRTC. The file is never compiled
// statically by nvcc.
//
// Currently used by tests (test_multidevice_tma.cpp). In a future PR
// this kernel will be integrated as a P2P and multicast transport
// alongside the existing SM-based and copy-engine transports in
// csrc/multidevice/cuda_p2p.cpp.
//
// TMA (cp.async.bulk) is a GMEM<->SMEM transfer engine — there is no
// GMEM-to-GMEM variant. Shared memory staging is inherent to the
// hardware, so the kernel performs a two-phase copy:
//
//   GMEM(src) --[TMA load]--> SMEM --[TMA store]--> GMEM(dst)
//
// A single elected thread (thread 0) drives both phases:
//   1. mbarrier.init (arrival count = 1)
//   2. mbarrier.arrive.expect_tx (announce expected bytes)
//   3. cp.async.bulk.shared::cluster.global  (TMA load, async)
//   4. mbarrier.try_wait.parity (block until load completes)
//   5. cp.async.bulk.global.shared::cta      (TMA store)
//   6. cp.async.bulk.commit_group + wait_group.read 0
//
// Dynamic shared memory layout (128-byte aligned):
//   [0, num_bytes)           : staging buffer
//   [num_bytes, num_bytes+8) : mbarrier (uint64_t)

extern "C" __global__ void __launch_bounds__(32, 1) tma_copy_1d(
    void* __restrict__ dst,
    const void* __restrict__ src,
    int num_bytes) {
  extern __shared__ __align__(128) unsigned char smem[];

  unsigned long long* mbar =
      reinterpret_cast<unsigned long long*>(smem + num_bytes);
  unsigned int smem_addr =
      static_cast<unsigned int>(__cvta_generic_to_shared(smem));
  unsigned int mbar_addr =
      static_cast<unsigned int>(__cvta_generic_to_shared(mbar));

  if (threadIdx.x == 0) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(1));
    asm volatile("fence.mbarrier_init.release.cluster;" :::);
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    // Announce expected transaction bytes on the mbarrier
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
            mbar_addr),
        "r"(num_bytes));

    // TMA Load: GMEM -> SMEM (async, completed via mbarrier)
    asm volatile(
        "cp.async.bulk.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];\n" ::"r"(smem_addr),
        "l"(src),
        "r"(num_bytes),
        "r"(mbar_addr)
        : "memory");

    // Block until the mbarrier phase flips (TMA load completed)
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "TMA_COPY_WAIT_LOAD:\n"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1;\n"
        "@P1 bra TMA_COPY_LOAD_DONE;\n"
        "bra TMA_COPY_WAIT_LOAD;\n"
        "TMA_COPY_LOAD_DONE:\n"
        "}" ::"r"(mbar_addr),
        "r"(0));

    // TMA Store: SMEM -> GMEM
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group"
        " [%0], [%1], %2;\n" ::"l"(dst),
        "r"(smem_addr),
        "r"(num_bytes)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");

    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mbar_addr));
  }
}
