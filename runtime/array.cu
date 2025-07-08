// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size = 1>
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

  __device__ const scalar_t& operator[](const unsigned int i) const {
    return array[i];
  }

  Array& operator=(const Array& a) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = a[i];
    }
    return *this;
  }
};

template <int size, int align_size>
struct alignas(align_size / 2) Array<__e2m1, size, align_size> {
  static_assert(size % 2 == 0, "There must be an even number of fp4 elements");
  __e2m1 array[size / 2];

  __device__ __e2m1& operator[](const unsigned int i) {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(i % 2 == 0);
    return array[i / 2];
  }

  __device__ const __e2m1& operator[](const unsigned int i) const {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(i % 2 == 0);
    return array[i / 2];
  }

  Array& operator=(const Array& a) {
#pragma unroll
    for (int i = 0; i < size / 2; ++i) {
      array[i] = a.array[i];
    }
    return *this;
  }
};

static_assert(
    sizeof(Array<__e2m1, 2, 2>) == 1,
    "sizeof(Array<__e2m1, 2, 2>) must be 1");
static_assert(
    sizeof(Array<__e2m1, 4, 2>) == 2,
    "sizeof(Array<__e2m1, 4, 2>) must be 2");
static_assert(
    sizeof(Array<__e2m1, 4, 4>) == 2,
    "sizeof(Array<__e2m1, 4, 4>) must be 2");
static_assert(
    sizeof(Array<__e2m1, 8, 2>) == 4,
    "sizeof(Array<__e2m1, 8, 4>) must be 4");
static_assert(
    sizeof(Array<__e2m1, 8, 4>) == 4,
    "sizeof(Array<__e2m1, 8, 4>) must be 4");
static_assert(
    sizeof(Array<__e2m1, 8, 8>) == 4,
    "sizeof(Array<__e2m1, 8, 8>) must be 4");
static_assert(
    sizeof(Array<__e2m1, 16, 2>) == 8,
    "sizeof(Array<__e2m1, 16, 2>) must be 8");
static_assert(
    sizeof(Array<__e2m1, 16, 4>) == 8,
    "sizeof(Array<__e2m1, 16, 4>) must be 8");
static_assert(
    sizeof(Array<__e2m1, 16, 8>) == 8,
    "sizeof(Array<__e2m1, 16, 8>) must be 8");
static_assert(
    sizeof(Array<__e2m1, 16, 16>) == 8,
    "sizeof(Array<__e2m1, 16, 16>) must be 8");

// Used for vectorized allocations that are not in registers
template <typename scalar_t, int vec_size>
__device__ void arraySet(scalar_t* buff, scalar_t val) {
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    buff[i] = val;
  }
}

template <typename scalar_t>
constexpr int64_t vecSizeBit(int64_t vec_size) {
  return vec_size * sizeof(scalar_t) * 8;
}

template <>
constexpr int64_t vecSizeBit<__e2m1>(int64_t vec_size) {
  return vec_size * 4;
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

  constexpr int64_t vec_size_bit = vecSizeBit<scalar_t>(vec_size);
  static_assert(vec_size_bit % 8 == 0, "vec_size_bit must be a multiple of 8");
  switch (vec_size_bit) {
    case 8:
      *reinterpret_cast<uchar1*>(to) = *reinterpret_cast<uchar1*>(from);
      break;
    case 16:
      *reinterpret_cast<uchar2*>(to) = *reinterpret_cast<uchar2*>(from);
      break;
    case 32:
      *reinterpret_cast<uint1*>(to) = *reinterpret_cast<uint1*>(from);
      break;
    case 64:
      *reinterpret_cast<uint2*>(to) = *reinterpret_cast<uint2*>(from);
      break;
    case 96:
      *reinterpret_cast<uint3*>(to) = *reinterpret_cast<uint3*>(from);
      break;
    case 128:
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
  constexpr int64_t vec_size_bit = vecSizeBit<scalar_t>(vec_size);
  static_assert(vec_size_bit % 8 == 0, "vec_size_bit must be a multiple of 8");
  switch (vec_size_bit) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 8:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned char, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned char, is_volatile_from>::type*>(
              from);
      break;
    case 16:
      *reinterpret_cast<typename MaybeVolatile<short, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<short, is_volatile_from>::type*>(from);
      break;
    case 32:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned int, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned int, is_volatile_from>::type*>(
              from);
      break;
    case 64:
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
  constexpr int64_t vec_size_bit = vecSizeBit<scalar_t>(vec_size);
  static_assert(vec_size_bit % 8 == 0, "vec_size_bit must be a multiple of 8");
  switch (vec_size_bit) {
    case 8:
    case 16:
    case 32:
      loadGenericVolatile<scalar_t, vec_size, is_volatile, false>(to, from);
      break;
    case 64: {
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
    case 128: {
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

// This is copied from csrc/type.h and should be kept consistent.
enum class CacheOp {
  AllLevels,
  Streaming,
  Global,
};

template <typename T, CacheOp cache_op>
__device__ void loadGlobalToLocalCached(void* to, void* from) {
  T* typed_to = reinterpret_cast<T*>(to);
  T* typed_from = reinterpret_cast<T*>(from);
  switch (cache_op) {
    case CacheOp::AllLevels:
      *typed_to = __ldca(typed_from);
      break;
    case CacheOp::Streaming:
      *typed_to = __ldcs(typed_from);
      break;
    case CacheOp::Global:
      *typed_to = __ldcg(typed_from);
      break;
  }
}

// For simplicity, cache_op is only used for non-volatile loads written in
// inline assembly. Other loads are done with the default cache operator --
// cache all levels. ld.volatile doesn't accept cache operator anyway.
template <typename scalar_t, int vec_size, bool is_volatile, CacheOp cache_op>
__device__ void loadGlobalToLocal(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, is_volatile>::type* from) {
  constexpr int64_t vec_size_bit = vecSizeBit<scalar_t>(vec_size);
  static_assert(vec_size_bit % 8 == 0, "vec_size_bit must be a multiple of 8");
  switch (vec_size_bit) {
    case 8:
    case 16:
    case 32:
      loadGenericVolatile<scalar_t, vec_size, false, is_volatile>(to, from);
      break;
    case 64: {
      if (is_volatile) {
        uint2& data = *reinterpret_cast<uint2*>(to);
        asm volatile("ld.volatile.global.v2.s32 {%0,%1}, [%2];"
                     : "=r"(data.x), "=r"(data.y)
                     : "l"((uint2*)from));
      } else {
        loadGlobalToLocalCached<uint2, cache_op>(
            to, const_cast<scalar_t*>(from));
      }
      break;
    }
    case 128: {
      if (is_volatile) {
        uint4& data = *reinterpret_cast<uint4*>(to);
        asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                     : "l"((uint4*)from));
      } else {
        loadGlobalToLocalCached<uint4, cache_op>(
            to, const_cast<scalar_t*>(from));
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
  constexpr int64_t vec_size_bit = vecSizeBit<scalar_t>(vec_size);
  static_assert(vec_size_bit % 8 == 0, "vec_size_bit must be a multiple of 8");
  switch (vec_size_bit) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 8:
    case 16:
    case 32:
    case 64:
      loadGenericVolatile<scalar_t, vec_size, is_volatile_to, is_volatile_from>(
          to, from);
      break;
    case 96: {
      uint3 local_intermediate;
      loadGlobalToLocal<
          scalar_t,
          vec_size,
          is_volatile_from,
          CacheOp::Streaming>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
    case 128: {
      uint4 local_intermediate;
      loadGlobalToLocal<
          scalar_t,
          vec_size,
          is_volatile_from,
          CacheOp::Streaming>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
  }
}
