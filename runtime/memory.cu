// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// Utility macro for this file
#define DEVICE_INLINE __device__ inline

// Utility for converting generic pointer to SMEM pointer in PTX.
//  We should review vectorized load/stores with shared memory.
//  SMEM memory movement PTX is only Global -> SMEM, SMEM -> Local, Local ->
//  SMEM, and this is needed for these PTX instructions to provide the SMEM
//  pointer.
DEVICE_INLINE unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));

  return smem_ptr_uint;
}

DEVICE_INLINE unsigned toSmem(unsigned addr) {
  // already converted
  return addr;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))

namespace Turing {

namespace util {

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
DEVICE_INLINE void adjustPartialLdMatrixAddrInTuring(unsigned& addr_in_byte) {
#if (__CUDA_ARCH__ < 800)
  const unsigned thread_id = threadIdx.x;
  // Upper half warp has 8 bytes offset from aligned in .x2 option
  //  of ldmatrix. Currently no support for .x1 so assume always
  //  adjust by half warp.
  constexpr unsigned half_warp = 16;
  // Need to adjust to 16 byte alignment, mask out un-aligned component.
  constexpr unsigned mask_out = 16 - 1;
  // Adjust only in upper half warp.
  // use bit math to reduce strength
  if (thread_id & half_warp) {
    // mask out the bits where adjust_mask has 1.
    addr_in_byte &= (~mask_out);
  }
#endif //(__CUDA_ARCH__ < 800)
}

} // namespace util

// Load Matrix (per warp instruction) is to take data from SMEM to Local Memory.
//   Automatically handles vectorized loads/stores in the MMA operation.
//   Loads 8x8 matrix into a warp. Thread 0-7 provide the ptr that is the start
//   of each row. All other threads can simply point to something valid
//   (including 0).
// The x2 modifier on the instruction will actually load 2x8 rows to make a
// 16x8,
//   then thread 0-15 will specify the start of each row.
// Finally is an x4 modifier producing a 32x8 using addrs from 0-31 in each
// warp.
DEVICE_INLINE void ldMatrix(Array<__half, 4, 4>& out, unsigned addr) {
  uint2& val = reinterpret_cast<uint2&>(out);
  util::adjustPartialLdMatrixAddrInTuring(addr);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
               : "=r"(val.x), "=r"(val.y)
               : "r"(addr));
}

// Same as previous, 8x8 matrix is vectorized loaded, then scattered (to perform
// transpose) so threads will hold 2 values down a column (instead of the
// previous instruction that's across a row).
DEVICE_INLINE void ldMatrixT(Array<__half, 4, 4>& out, unsigned addr) {
  uint2& val = reinterpret_cast<uint2&>(out);
  util::adjustPartialLdMatrixAddrInTuring(addr);
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];"
               : "=r"(val.x), "=r"(val.y)
               : "r"(addr));
}

DEVICE_INLINE void ldMatrix(Array<__half, 8, 8>& out, unsigned addr) {
  uint4& val = reinterpret_cast<uint4&>(out);
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "r"(addr));
}

DEVICE_INLINE void ldMatrixT(Array<__half, 8, 8>& out, unsigned addr) {
  uint4& val = reinterpret_cast<uint4&>(out);
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
      : "r"(addr));
}

} // namespace Turing

#endif // Arch 75

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

namespace Ampere {

// MMA instruction wrappers (sm_80+):
template<int dst_size, typename word_type>
DEVICE_INLINE void cpGlobalToShared(
        unsigned smem_addr,
        const void *gmem_ptr,
        int src_size) {
    constexpr int word_size = sizeof(word_type);
    const auto *__restrict src = reinterpret_cast<const word_type *>(__builtin_assume_aligned(gmem_ptr, word_size));
    auto *__restrict dst = reinterpret_cast<word_type *>(__builtin_assume_aligned(__cvta_shared_to_generic(smem_addr), word_size));
    // these allow to generate ld.global.u8 and st.shared.u8 properly.
    bool s = __isShared(dst);
    __builtin_assume(s);
    bool g = __isGlobal(src);
    __builtin_assume(g);

    bool b = src_size >= 0 && src_size < dst_size;
    __builtin_assume(b);

#pragma unroll
    for (int i = 0; i < dst_size; ++i) {
        dst[i] = i < src_size ? src[i] : word_type{0};
    }
}

/**
 * Copies src_size bytes from gmem_ptr to smem_addr. 
 * Zero fills upper bytes if src_size < dst_size.
 * @tparam dst_size 
 * @param smem_addr 
 * @param gmem_ptr 
 * @param src_size_or_predicate 
 */
template<int dst_size, typename Predicate_T>
DEVICE_INLINE void cpAsyncCaAligned(
        unsigned smem_addr,
        const void *gmem_ptr,
        Predicate_T src_size_or_predicate) {
    static_assert(dst_size == 4 || dst_size == 8 || dst_size == 16, "cp_async : unsupported byte size");

    static_assert(is_same_v < Predicate_T, bool > || is_same_v < Predicate_T, int > );

    if constexpr(is_same_v < Predicate_T, bool > )
    {
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.eq.b32 p, %3, 0;\n"
                "cp.async.ca.shared.global [%0], [%1], %2, p;\n"
                "}\n"::"r"(smem_addr),
        "l"(__builtin_assume_aligned(gmem_ptr, dst_size)),
        "n"(dst_size),
        "r"((int) src_size_or_predicate) : "memory");
    } else if constexpr(is_same_v < Predicate_T, int > )
    {
        asm volatile(
                "{\n"
                "cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                "}\n"::"r"(smem_addr),
        "l"(__builtin_assume_aligned(gmem_ptr, dst_size)),
        "n"(dst_size),
        "r"(src_size_or_predicate): "memory");
    }
}


/**
 * Async load from gmem to smem. 
 * @tparam T elt type
 * @tparam count elt count
 * @tparam assume_aligned_data Assumes that gmem is aligned on sizeof(T) * count
 * @param smem_addr 
 * @param gmem_ptr must be at least aligned on sizeof(T)
 * @param src_size_or_predicate if predicate is false, zeroes are written to smem and gmem is not accessed
 */
template<typename Predicate_T, typename T, int count, bool assume_aligned_data = true>
DEVICE_INLINE void cpAsyncCa(
        unsigned smem_addr,
        const T *gmem_ptr,
        Predicate_T src_size_or_predicate) {

    static_assert(is_same_v < Predicate_T, bool > || is_same_v < Predicate_T, int > );

    constexpr unsigned byte_size = sizeof(T) * count;
    static_assert(byte_size == 4 || byte_size == 8 || byte_size == 16, "cp_async : unsupported byte size");

    if constexpr(assume_aligned_data || sizeof(T) == byte_size)
    {
        const char *__restrict src = reinterpret_cast< const char *>(__builtin_assume_aligned(gmem_ptr, byte_size));
        cpAsyncCaAligned<byte_size, Predicate_T>(smem_addr, src, src_size_or_predicate);
    } else {
        const auto ptr_low_bytes = static_cast<unsigned> (reinterpret_cast<size_t>(gmem_ptr) & 0xFFFFFFFF); //Should be std::intptr_t
        if (ptr_low_bytes % byte_size == 0) {
            // ptr is aligned on cp size, 
            const char *__restrict src = reinterpret_cast< const char *>(__builtin_assume_aligned(gmem_ptr, byte_size));
            cpAsyncCaAligned<byte_size, Predicate_T>(smem_addr, src, src_size_or_predicate);
        } else { // Unaligned case, fallback on min_alignment
            constexpr unsigned min_alignment = sizeof(T); // min alignment guaranteed for the type
            static_assert(byte_size % min_alignment == 0);
            if constexpr(min_alignment == 1)
            { // fallback to copy through registers using bytes
                const char *__restrict src = reinterpret_cast< const char *>(__builtin_assume_aligned(gmem_ptr, 1));
                cpGlobalToShared<byte_size, unsigned char>(smem_addr, src, src_size_or_predicate);
            } else if constexpr(min_alignment == 2)
            { // fallback to copy through registers using shorts
                const char *__restrict src = reinterpret_cast< const char *>(__builtin_assume_aligned(gmem_ptr, 2));
                cpGlobalToShared<byte_size / 2, unsigned short>(smem_addr, src, src_size_or_predicate);
            } else {
                static_assert(min_alignment == 4 || min_alignment == 8 || min_alignment == 16, "cp_async : unsupported elt size");
                const char *__restrict src = reinterpret_cast< const char *>(__builtin_assume_aligned(gmem_ptr, min_alignment));
#pragma unroll
                for (unsigned i = 0; i < byte_size; i += min_alignment) {
                    cpAsyncCaAligned<min_alignment>(smem_addr + i, src + i, src_size_or_predicate);
                }
            }
        }
    }
}


template<typename Predicate_T>
DEVICE_INLINE void cpAsyncCgAligned(
        unsigned smem_addr,
        void const *__restrict gmem_ptr,
        Predicate_T src_size_or_predicate) {
    static_assert(is_same_v < Predicate_T, bool > || is_same_v < Predicate_T, int > );
    if constexpr(is_same_v < Predicate_T, bool > )
    {
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.eq.b32 p, %2, 0;\n"
                "cp.async.cg.shared.global [%0], [%1], 16, p;\n"
                "}\n"::"r"(smem_addr),
        "l"(__builtin_assume_aligned(gmem_ptr, 16)),
        "r"((int) src_size_or_predicate) : "memory");
    } else if constexpr(is_same_v < Predicate_T, int > )
    {
        asm volatile(
                "{\n"
                "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                "}\n"::"r"(smem_addr),
        "l"(__builtin_assume_aligned(gmem_ptr, 16)),
        "r"(src_size_or_predicate) : "memory");
    }
}


// Global to SMEM load that is asynchronous,
// not guaranteed to be completed until cpAsyncBarrier() is called.
template<typename Predicate_T, typename T, int len, bool assume_aligned_data = true>
DEVICE_INLINE void cpAsyncCg(
        unsigned smem_addr,
        const void *gmem_ptr,
        Predicate_T src_size_or_predicate) {
    constexpr int byte_size = sizeof(T) * len;
    static_assert(byte_size == 16, "cp_async : unsupported byte size");

    if constexpr(assume_aligned_data || sizeof(T) == 16)
    {
        cpAsyncCgAligned(smem_addr, gmem_ptr, src_size_or_predicate);
    } else {
        const auto ptr_low = static_cast<unsigned> (reinterpret_cast<size_t>(gmem_ptr) & 0xFFFFFFFF); //Should be std::intptr_t
        unsigned misalignment = ptr_low % byte_size;
        __builtin_assume(misalignment < byte_size);
        if (misalignment == 0) {
            cpAsyncCgAligned(smem_addr, gmem_ptr, src_size_or_predicate);
        } else {
            cpAsyncCa<Predicate_T, T, len, assume_aligned_data>(smem_addr, gmem_ptr, src_size_or_predicate);
        }
    }
}

// TODO: Might have a different category of sync if we want to build out this:
DEVICE_INLINE void cpAsyncBarrier() {
  asm volatile("cp.async.wait_all;");
}

DEVICE_INLINE void cpAsyncCommit() {
  asm volatile("cp.async.commit_group;");
}

template <int keep_stages>
DEVICE_INLINE void cpAsyncPartialBarrier() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(keep_stages));
}

} // namespace Ampere

#endif // Arch 80

// Double buffer calculation utilities:

// In place update of double buffer index that has been accumulated to the data
// buffer.
template <int number_of_stage, int loop_offset>
DEVICE_INLINE void doubleBufferUpdate(
    DataPointer& data_buffer,
    const nvfuser_index_t& loop_index,
    nvfuser_index_t buffer_size) {
  // static_assert(
  //     loop_offset < number_of_stage && loop_offset > -number_of_stage);

  // convert offset to [0, number_of_stage)
  constexpr nvfuser_index_t offset =
      loop_offset < 0 ? (loop_offset + number_of_stage) : loop_offset;

  // Rewind back at number_of_stage-1, otherwise increment by 1.
  nvfuser_index_t increment =
      (loop_index % number_of_stage) == (number_of_stage - 1 - offset)
      ? buffer_size * (-number_of_stage + 1)
      : buffer_size;
  data_buffer += increment;
}

template <int number_of_stage, int loop_offset>
DEVICE_INLINE void doubleBufferUpdate(
    unsigned& data_buffer,
    const nvfuser_index_t& loop_index,
    nvfuser_index_t buffer_size) {
  // static_assert(
  //     loop_offset < number_of_stage && loop_offset > -number_of_stage);

  // convert offset to [0, number_of_stage)
  constexpr nvfuser_index_t offset =
      loop_offset < 0 ? (loop_offset + number_of_stage) : loop_offset;

  // Rewind back at number_of_stage-1, otherwise increment by 1.
  nvfuser_index_t increment =
      (loop_index % number_of_stage) == (number_of_stage - 1 - offset)
      ? buffer_size * (-number_of_stage + 1)
      : buffer_size;
  data_buffer += (unsigned)increment;
}

// Update double buffer offset value for smem double buffered tensors.
// See [Uniform Double Buffer Offset]
template <int number_of_stage, int loop_offset>
DEVICE_INLINE void doubleBufferSwitch(
    int& buffer_offset,
    const nvfuser_index_t& loop_index,
    nvfuser_index_t buffer_size) {
  constexpr nvfuser_index_t offset =
      loop_offset < 0 ? (loop_offset + number_of_stage) : loop_offset;

  // Rewind back at number_of_stage-1, otherwise increment by 1.
  nvfuser_index_t increment =
      (loop_index % number_of_stage) == (number_of_stage - 1 - offset)
      ? buffer_size * (-number_of_stage + 1)
      : buffer_size;
  buffer_offset += (int)increment;
}

// Reset smem space to zero
// TODO: try cp.async.ignore-source ?
template <typename dtype, int len>
DEVICE_INLINE void smemReset(SmemAddress smem_addr) {
  constexpr int byte_size = sizeof(dtype) * len;

  static_assert(
      byte_size == 4 || byte_size == 8 || byte_size == 16,
      "cp_async : unsupported byte size");

  switch (byte_size) {
    case 4:
      asm volatile(
          "{\n"
          "st.shared.u32 [%0], {%1};\n"
          "}\n"
          :
          : "r"(smem_addr), "r"(0));
      break;
    case 8:
      asm volatile(
          "{\n"
          "st.shared.v2.u32 [%0], {%1, %2};\n"
          "}\n"
          :
          : "r"(smem_addr), "r"(0), "r"(0));
      break;
    case 16:
      asm volatile(
          "{\n"
          "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
          "}\n"
          :
          : "r"(smem_addr), "r"(0), "r"(0), "r"(0), "r"(0));
      break;
  }
}

#undef DEVICE_INLINE
