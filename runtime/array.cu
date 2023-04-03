// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size>
struct alignas(sizeof(scalar_t) * align_size) Array {
  scalar_t array[size];

  __device__ void set(scalar_t v) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = v;
    }
  }

  __device__ scalar_t& operator[](const unsigned int i) {
    return array[i];
  }
};

// Used for vectorized allocations that are not in registers
template <typename scalar_t, int vec_size>
__device__ void arraySet(scalar_t* buff, scalar_t val) {
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    buff[i] = val;
  }
}

template <typename scalar_t, int vec_size>
__device__ void loadGeneric(scalar_t* to, scalar_t* from) {
  // It would be really nice to use memcpy here, but one example was failing
  // with:
  //
  //  memcpy(to, from, vec_size * sizeof(scalar_t));
  //
  // Yet passing with:
  //
  // for(int i = 0; i < vec_size; i++){
  //   to[i] = from[i];
  // }

  switch (sizeof(scalar_t) * vec_size) {
    case 1:
      *reinterpret_cast<uchar1*>(to) = *reinterpret_cast<uchar1*>(from);
      break;
    case 2:
      *reinterpret_cast<uchar2*>(to) = *reinterpret_cast<uchar2*>(from);
      break;
    case 4:
      *reinterpret_cast<uint1*>(to) = *reinterpret_cast<uint1*>(from);
      break;
    case 8:
      *reinterpret_cast<uint2*>(to) = *reinterpret_cast<uint2*>(from);
      break;
    case 12:
      *reinterpret_cast<uint3*>(to) = *reinterpret_cast<uint3*>(from);
      break;
    case 16:
      *reinterpret_cast<uint4*>(to) = *reinterpret_cast<uint4*>(from);
      break;
  }
}

// Volatile version only works with c++ fundamnetal types
template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGenericVolatile(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned char, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned char, is_volatile_from>::type*>(
              from);
      break;
    case 2:
      *reinterpret_cast<typename MaybeVolatile<short, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<short, is_volatile_from>::type*>(from);
      break;
    case 4:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned int, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned int, is_volatile_from>::type*>(
              from);
      break;
    case 8:
      *reinterpret_cast<typename MaybeVolatile<double, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<double, is_volatile_from>::type*>(from);
      break;
  }
}

template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadLocalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile>::type* to,
    scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, is_volatile, false>(to, from);
      break;
    case 8: {
      uint2 const& data = *reinterpret_cast<uint2*>(from);
      if (is_volatile) {
        asm volatile(
            "st.volatile.global.v2.s32 [%0], {%1,%2};" ::"l"(
                (typename MaybeVolatile<uint2, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y));
      } else {
        asm volatile(
            "st.global.cs.v2.s32 [%0], {%1,%2};" ::"l"(
                (typename MaybeVolatile<uint2, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y));
      }
      break;
    }
    case 16: {
      uint4 const& data = *reinterpret_cast<uint4*>(from);
      if (is_volatile) {
        asm volatile(
            "st.volatile.global.v4.s32 [%0], {%1,%2,%3,%4};" ::"l"(
                (typename MaybeVolatile<uint4, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y),
            "r"(data.z),
            "r"(data.w));
      } else {
        asm volatile(
            "st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" ::"l"(
                (typename MaybeVolatile<uint4, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y),
            "r"(data.z),
            "r"(data.w));
      }
      break;
    }
  }
}

template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadGlobalToLocal(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, is_volatile>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, false, is_volatile>(to, from);
      break;
    case 8: {
      if (is_volatile) {
        uint2& data = *reinterpret_cast<uint2*>(to);
        asm volatile("ld.volatile.global.v2.s32 {%0,%1}, [%2];"
                     : "=r"(data.x), "=r"(data.y)
                     : "l"((uint2*)from));
        break;
      } else {
        uint2& data = *reinterpret_cast<uint2*>(to);
        asm volatile("ld.global.cs.v2.s32 {%0,%1}, [%2];"
                     : "=r"(data.x), "=r"(data.y)
                     : "l"((uint2*)from));
      }
      break;
    }
    case 16: {
      if (is_volatile) {
        uint4& data = *reinterpret_cast<uint4*>(to);
        asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                     : "l"((uint4*)from));
      } else {
        uint4& data = *reinterpret_cast<uint4*>(to);
        asm volatile("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                     : "l"((uint4*)from));
      }
      break;
    }
  }
}

template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGlobalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
    case 2:
    case 4:
    case 8:
      loadGenericVolatile<scalar_t, vec_size, is_volatile_to, is_volatile_from>(
          to, from);
      break;
    case 12: {
      uint3 local_intermediate;
      loadGlobalToLocal<scalar_t, vec_size, is_volatile_from>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
    case 16: {
      uint4 local_intermediate;
      loadGlobalToLocal<scalar_t, vec_size, is_volatile_from>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
  }
}

template <int size>
__device__ inline void loadLocalToGlobalAlignedImpl(
    char* __restrict to,
    const char* __restrict from);

template <>
__device__ inline void loadLocalToGlobalAlignedImpl<1>(
    char* __restrict to,
    const char* __restrict from) {
  asm volatile(
      "st.global.cs.u8 [%0], {%1};" ::"l"(__builtin_assume_aligned(to, 1)),
      "h"((unsigned short)*from)
      : "memory");
}

template <>
__device__ inline void loadLocalToGlobalAlignedImpl<2>(
    char* __restrict to,
    const char* __restrict from) {
  unsigned short const& data = *reinterpret_cast<const unsigned short*>(from);
  asm volatile(
      "st.global.cs.u16 [%0], {%1};" ::"l"(__builtin_assume_aligned(to, 1)),
      "h"((unsigned short)data)
      : "memory");
}

template <>
__device__ inline void loadLocalToGlobalAlignedImpl<4>(
    char* __restrict to,
    const char* __restrict from) {
  unsigned const& data = *reinterpret_cast<const unsigned*>(from);
  asm volatile("st.global.cs.u32 [%0], {%1};" ::"l"(reinterpret_cast<unsigned*>(
                   __builtin_assume_aligned(to, 4))),
               "r"(data)
               : "memory");
}

template <>
__device__ inline void loadLocalToGlobalAlignedImpl<8>(
    char* __restrict to,
    const char* __restrict from) {
  uint2 const& data = *reinterpret_cast<const uint2*>(from);
  asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};" ::"l"(
                   reinterpret_cast<uint2*>(__builtin_assume_aligned(to, 8))),
               "r"(data.x),
               "r"(data.y)
               : "memory");
}

template <>
__device__ inline void loadLocalToGlobalAlignedImpl<16>(
    char* __restrict to,
    const char* __restrict from) {
  uint4 const& data = *reinterpret_cast<const uint4*>(from);
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(
                   reinterpret_cast<uint4*>(__builtin_assume_aligned(to, 16))),
               "r"(data.x),
               "r"(data.y),
               "r"(data.z),
               "r"(data.w)
               : "memory");
}

constexpr static inline __device__ bool is_power_two(
    unsigned v,
    int result = 1) {
  if (v == 0)
    return false;
  if (v == result)
    return true;
  else if (result < v)
    return is_power_two(v, result * 2);
  else
    return false;
}

constexpr static inline __device__ unsigned prev_or_equal_power_two(
    unsigned v,
    unsigned result = 1) {
  if (v == 0)
    return 0;
  if (result <= v && 2 * result > v)
    return result;
  else
    return prev_or_equal_power_two(v, result * 2);
}

/**
 *
 * @tparam T
 * @tparam count
 * @param to
 * @param from
 */
template <typename T, int count>
__device__ inline void loadLocalToGlobalUnaligned(
    T* __restrict to,
    const T* __restrict from) {
  if constexpr (count == 0)
    return;
  static_assert(is_power_two(sizeof(T)));
  constexpr unsigned byte_size = sizeof(T) * count;
  constexpr unsigned min_alignment = sizeof(T) < 16 ? sizeof(T) : 16;
  constexpr unsigned upper_alignment = prev_or_equal_power_two(byte_size) < 16
      ? prev_or_equal_power_two(byte_size)
      : 16;
  static_assert(byte_size % min_alignment == 0);

  const auto ptr_low = static_cast<unsigned>(
      reinterpret_cast<size_t>(to) & 0xFFFFFFFF); // Should be std::intptr_t

  if (ptr_low % upper_alignment == 0) {
    char* __restrict dst =
        reinterpret_cast<char*>(__builtin_assume_aligned(to, upper_alignment));
    const char* __restrict src = reinterpret_cast<const char*>(from);
    unsigned i = 0;
#pragma unroll
    for (; i + upper_alignment <= byte_size; i += upper_alignment) {
      loadLocalToGlobalAlignedImpl<upper_alignment>(dst + i, src + i);
    }

#pragma unroll
    for (; i < byte_size; i += min_alignment) {
      loadLocalToGlobalAlignedImpl<min_alignment>(dst + i, src + i);
    }
  } else {
    char* __restrict dst =
        reinterpret_cast<char*>(__builtin_assume_aligned(to, min_alignment));
    const char* __restrict src = reinterpret_cast<const char*>(from);
#pragma unroll
    for (unsigned i = 0; i < byte_size; i += min_alignment) {
      loadLocalToGlobalAlignedImpl<min_alignment>(dst + i, src + i);
    }
  }
}