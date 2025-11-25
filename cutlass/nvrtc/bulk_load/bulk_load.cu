// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cute/tensor.hpp>
#include "cute.cuh"

using namespace cute;

template <class T, class GmemLayout, class SmemLayout>
__global__ void cute_bulk_copy(
    T const* g_in,
    T* g_out,
    GmemLayout gmem_layout,
    SmemLayout smem_layout) {
  // Use Shared Storage structure to allocate and distribute aligned SMEM
  // addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sA =
      make_tensor(make_smem_ptr(shared_storage.smem.data()), smem_layout);
  // Construct the GMEM tensor
  Tensor gA = make_tensor(make_gmem_ptr(g_in), gmem_layout);

  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* bulk_copy_mbar = shared_storage.bulk_copy_mbar;

  //
  // Perform the BULK_COPY load
  //

  auto blkcp = Copy_Traits<SM90_BULK_COPY_AUTO>{};

  // Set the bytes transferred in this transaction (may involve multiple issues)
  constexpr int transaction_bytes = size(sA) * sizeof(float);

  if (threadIdx.x == 0) {
    /// Initialize shared memory barrier
    bulk_copy_mbar[0] = 0;
    initialize_barrier(bulk_copy_mbar[0], 1 /*numThreads*/);
    set_barrier_transaction_bytes(bulk_copy_mbar[0], transaction_bytes);

    copy(blkcp.with(bulk_copy_mbar[0]), gA, sA);
  }
  __syncthreads();

  /// Wait on the shared memory barrier until the phase bit flips from kPhaseBit
  /// value
  constexpr int kPhaseBit = 0;
  wait_barrier(bulk_copy_mbar[0], kPhaseBit);

  //
  // Write out trivially
  //

  Tensor gA_out = make_tensor(make_gmem_ptr(g_out), gmem_layout);

  // Output smem -> gmem
  for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
    gA_out(i) = sA(i);
  }
}
