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

template <
    class T,
    class TiledCopy,
    class CTA_Tiler,
    class GmemLayout,
    class SmemLayout>
__global__ void cute_tma_copy(
    T const* g_in,
    T* g_out,
    CUTE_GRID_CONSTANT TiledCopy const tma,
    CTA_Tiler cta_tiler,
    GmemLayout gmem_layout,
    SmemLayout smem_layout) {
  using namespace cute;
  CUTE_STATIC_ASSERT_V(
      product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM
  // addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  // (CTA_TILE_M,CTA_TILE_N,...)
  Tensor sA =
      make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);
  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

  // TMA requires special handling of strides to deal with coord codomain
  // mapping Represent the full tensors -- get these from TMA
  Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
  Tensor mB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

  constexpr int R = rank_v<CTA_Tiler>;
  // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gA = flat_divide(mA, cta_tiler);
  // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gB = flat_divide(mB, cta_tiler);

  //
  // Prepare the TMA_LOAD
  //

  // CTA slice
  auto cta_tma = tma.get_slice(Int<0>{});
  // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAgA_x = cta_tma.partition_S(gA);
  // (TMA,TMA_M,TMA_N)
  Tensor tAsA_x = cta_tma.partition_D(sA);

#if 0
  if (thread0()) {
    print(tma);
    print("TILE  :  "); print(cta_tiler); print("\n");
    print("  mA  :  "); print(  mA);   print("\n");
    print("  mB  :  "); print(  mB);   print("\n");
    print("  gA  :  "); print(  gA);   print("\n");
    print("  gB  :  "); print(  gB);   print("\n");
    print("  sA  :  "); print(  sA);   print("\n");
    print("tAgA_x:  "); print(tAgA_x); print("\n");
    print("tAsA_x:  "); print(tAsA_x); print("\n");
  }
#endif

  //
  // Perform the TMA_LOAD
  //

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through
  // the tiles (TMA,REST)
  Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x);
  // (TMA,REST)
  Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);
  static_assert(size<1>(tAsA) == 1);

  // OUTPUT: Group the CTA_TILE_X modes and REST_X modes for output
  // (CTA_TILE, REST)
  Tensor tBgB = group_modes<0, R>(group_modes<R, rank(gB)>(gB));

#if 0
  if (thread0()) {
    print("tAgA  :  "); print(tAgA); print("\n");
    print("tAsA  :  "); print(tAsA); print("\n");
    print("tBgB  :  "); print(tBgB); print("\n");
  }
#endif

  // Test L2 prefetch
  if (threadIdx.x == 0) {
    prefetch(tma, tAgA);
  }

  // Loop over the TMA stages, using smem as our buffer
  for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    // Set the bytes transferred in this TMA transaction (may involve multiple
    // issues)
    constexpr int kTmaTransactionBytes =
        sizeof(make_tensor_like(tensor<0>(tAsA)));

    if (threadIdx.x == 0) {
      /// Initialize shared memory barrier
      tma_load_mbar[0] = 0;
      cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
      cute::set_barrier_transaction_bytes(
          tma_load_mbar[0], kTmaTransactionBytes);

      copy(tma.with(tma_load_mbar[0]), tAgA(_, stage), tAsA(_, 0));
    }
    __syncthreads();

    /// Wait on the shared memory barrier until the phase bit flips from
    /// kPhaseBit value
    constexpr int kPhaseBit = 0;
    cute::wait_barrier(tma_load_mbar[0], kPhaseBit);

    //
    // Write out trivially smem -> gmem
    //

    // Subbyte elements could cause race conditions, so be even more
    // conservative
    if (thread0()) {
      copy(sA, tBgB(_, stage));
    }

    __syncthreads();
  }
}
