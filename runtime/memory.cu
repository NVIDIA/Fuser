// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Utility for converting generic pointer to SMEM pointer in PTX.
//  We should review vectorized load/stores with shared memory.
//  SMEM memory movement PTX is only Global -> SMEM, SMEM -> Local, Local ->
//  SMEM, and this is needed for these PTX instructions to provide the SMEM
//  pointer.
__device__ inline unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));

  return smem_ptr_uint;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))

namespace Turing {

// LdMatrix has .x1, .x2 and .x4 options, currently we actively use .x2 and
//  .x4. In .x2 option. the the address register of upper half warp (lane 16-31)
//  are un-used but on Turing [sm75,sm80) architecture these un-used addresses
//  need to be valid, in the sense that:
//     1. The data it points to has to be within allocated shared mem buffer.
//     2. The address needs to be aligned to 16 byte.
//  See also:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
//  This function addresses 2. above by masking out the sub-16B component
//    of the address in upper warp and 1. is guaranteed by ldmatrix swizzle
//    util.
//  This will **not** affect any functionality. This is just modification
//    of unused pointers to satisfy the alignment requirement on Turing
//    hardware.
//  The alignment requirement is lifted on sm80+,
//    so this function is a no-op on Ampere or above.
template <unsigned num_valid_addresses>
__device__ inline unsigned adjustPartialLdMatrixAddrInTuring(
    unsigned addr_in_byte) {
  const unsigned lane = threadIdx.x % 32;
  if (lane >= num_valid_addresses) {
    return 0;
  }
  return addr_in_byte;
}

} // namespace Turing

#endif // Arch 75

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

namespace Hopper {

// Description: Elect a leader thread from a set of threads in a warp
//
// The common pattern is to select any thread from the first warp without
// creating a serialized, peeling loop.
//
// Code example: threadIdx.x / 32 == 0 && ptx::elect_sync(~0)
//
// Compile Explorer Reference: https://ce.nvidia.com/z/d9x4q8
//
// Document Reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync
__device__ inline bool electSync(const uint32_t& membermask) {
  uint32_t is_elected;
  asm volatile(
      "{\n\t .reg .pred P_OUT; \n\t"
      "elect.sync _|P_OUT, %1;\n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(is_elected)
      : "r"(membermask)
      :);
  return static_cast<bool>(is_elected);
}

// References:
//
// TMA:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp
//
// Tensor map:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

// 1D TMA load:
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1400

// 1D TMA Load:
struct CpAsyncBulkG2SIndex {
  const void* raw_gmem_addr;
  uint32_t bytes;
  uint32_t mbarrier;
};
__device__ inline void cpAsyncBulkG2S(
    const CpAsyncBulkG2SIndex& src,
    uint32_t smem_addr) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      :
      : "r"(smem_addr),
        "l"(src.raw_gmem_addr),
        "r"(src.bytes),
        "r"(src.mbarrier)
      : "memory");
}

// 1D TMA Store:
struct CpAsyncBulkS2GIndex {
  const void* raw_gmem_addr;
  uint32_t bytes;
};
__device__ inline void cpAsyncBulkS2G(
    const CpAsyncBulkS2GIndex& dst,
    uint32_t smem_addr) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
               :
               : "l"(dst.raw_gmem_addr), "r"(smem_addr), "r"(dst.bytes)
               : "memory");
}

// TMA Loads:

template <int dim>
struct CpAsyncBulkTensorTileG2SIndex {
  const TensorMap* descriptor;
  Array<int32_t, dim> crds;
  uint32_t mbarrier;
};

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<1>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3}], [%2];"
      :
      : "r"(smem_addr), "l"(gmem_int_desc), "r"(src.mbarrier), "r"(src.crds[0])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2SMulticast(
    const CpAsyncBulkTensorTileG2SIndex<1>& src,
    uint32_t smem_addr,
    uint16_t cta_mask) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "h"(cta_mask)
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<2>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2SMulticast(
    const CpAsyncBulkTensorTileG2SIndex<2>& src,
    uint32_t smem_addr,
    uint16_t cta_mask) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "h"(cta_mask)
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<3>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2SMulticast(
    const CpAsyncBulkTensorTileG2SIndex<3>& src,
    uint32_t smem_addr,
    uint16_t cta_mask) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast_cluster"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2]),
        "h"(cta_mask)
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<4>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2]),
        "r"(src.crds[3])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2SMulticast(
    const CpAsyncBulkTensorTileG2SIndex<4>& src,
    uint32_t smem_addr,
    uint16_t cta_mask) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast_cluster"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2]),
        "r"(src.crds[3]),
        "h"(cta_mask)
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<5>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2]),
        "r"(src.crds[3]),
        "r"(src.crds[4])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2SMulticast(
    const CpAsyncBulkTensorTileG2SIndex<5>& src,
    uint32_t smem_addr,
    uint16_t cta_mask) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast_cluster"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2]),
        "r"(src.crds[3]),
        "r"(src.crds[4]),
        "h"(cta_mask)
      : "memory");
}

// TMA Stores:

template <int dim>
struct CpAsyncBulkTensorTileS2GIndex {
  const TensorMap* descriptor;
  Array<int32_t, dim> crds;
};

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<1>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_addr), "r"(dest.crds[0])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<2>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_addr), "r"(dest.crds[0]), "r"(dest.crds[1])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<3>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(gmem_int_desc),
        "r"(smem_addr),
        "r"(dest.crds[0]),
        "r"(dest.crds[1]),
        "r"(dest.crds[2])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<4>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(gmem_int_desc),
        "r"(smem_addr),
        "r"(dest.crds[0]),
        "r"(dest.crds[1]),
        "r"(dest.crds[2]),
        "r"(dest.crds[3])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<5>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc),
        "r"(smem_addr),
        "r"(dest.crds[0]),
        "r"(dest.crds[1]),
        "r"(dest.crds[2]),
        "r"(dest.crds[3]),
        "r"(dest.crds[4])
      : "memory");
}

} // namespace Hopper

#endif // Arch 90
