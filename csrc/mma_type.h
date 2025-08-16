// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <macros.h>

#include <exceptions.h>
#include <type.h>
#include <visibility.h>

#include <cstring>
#include <ostream>

#include <cstdint>

namespace nvfuser {

constexpr std::string_view MATMUL_LOG_PREFIX = "[MATMUL DEBUG] ";

//! Named descriptors of domains in matmul
enum class MatmulDimRole { M = 0, N, K, Batch };

std::string toString(MatmulDimRole role);

//! Named descriptors of TensorView roles in fusion
//!  OPERAND_A - an input to the fusion that is a producer of a matmul "A" input
//!  OPERAND_B - an input to the fusion that is a producer of a matmul "B" input
//!  OUTPUT - fusion outputs that have the matmul as a dependency
//!  EPILOGUE_INPUT - an input to the fusion that is a producer of an
//!    OUTPUT, but not of an MMA input
//!
//!  Note: bias vector tensors will be assigned to the EPILOGUE_INPUT role.
enum class MatmulTensorRole {
  OPERAND_A = 0,
  OPERAND_B,
  OUTPUT,
  EPILOGUE_INPUT
};

//! The expected number of occurances of core TensorView roles in fusion
static constexpr size_t MATMUL_CORE_ROLES_EXPECTED_COUNT = 1;

//! Utility data structure for recording gemm tiles
struct GemmTile {
  int64_t m, n, k;
  GemmTile(int64_t m_, int64_t n_, int64_t k_) : m(m_), n(n_), k(k_) {}

  bool operator==(const GemmTile& other) const {
    return m == other.m && n == other.n && k == other.k;
  }

  GemmTile operator/(const GemmTile& other) const {
    return GemmTile(m / other.m, n / other.n, k / other.k);
  }

  std::vector<int64_t> toVector() const {
    return {m, n, k};
  }
};

//! Utility data structure for recording gemm tiles
struct MatMulTileOptions {
  GemmTile cta_tile = GemmTile(128, 128, 32);
  GemmTile warp_tile = GemmTile(64, 64, 32);

  MatMulTileOptions() = default;
  MatMulTileOptions(GemmTile cta_tile_, GemmTile warp_tile_)
      : cta_tile(cta_tile_), warp_tile(warp_tile_) {}

  bool operator==(const MatMulTileOptions& other) const {
    return cta_tile == other.cta_tile && warp_tile == other.warp_tile;
  }
};

enum class MmaMacro : uint64_t;

struct MmaMacroEncode {
  enum class Arch : uint16_t {
    NoMma,
    Volta,
    Turing,
    Ampere,
    Hopper,
    Blackwell1CTA,
    Blackwell2CTA
  } arch;
  uint16_t m;
  uint16_t n;
  uint16_t k;

  constexpr operator uint64_t() {
    return (uint64_t)arch << 48 | (uint64_t)m << 32 | (uint64_t)n << 16 |
        (uint64_t)k;
  }

  constexpr operator MmaMacro() {
    return static_cast<MmaMacro>(static_cast<uint64_t>(*this));
  }

  constexpr MmaMacroEncode(MmaMacro macro)
      : arch(Arch(toUnderlying(macro) >> 48)),
        m((toUnderlying(macro) >> 32) & 0xFFFF),
        n((toUnderlying(macro) >> 16) & 0xFFFF),
        k(toUnderlying(macro) & 0xFFFF) {}

  constexpr MmaMacroEncode(Arch arch_, uint16_t m_, uint16_t n_, uint16_t k_)
      : arch(arch_), m(m_), n(n_), k(k_) {}
};

static_assert(sizeof(MmaMacroEncode) == sizeof(uint64_t));

//! Type of mma instrinsic macro to use
//!  This will translate to which mma intrinsic from runtime string
//!    to be generated to implement the mma op. The current plan
//!    is to have exactly one macro for each
//!  (arch, datatype, operand layout) triple, though there
//!  exists multiple possibilities for some cases, e.g. for Turing and fp16
//!  one can use 16_8_8 or 16_8_16.
//! Will consider adding more choices that the scheduler can pick from
//!  when our perf target becomes more fine grained, which is more likely in
//!  latency bound kernels.

#define MACRO(arch, m, n, k) \
  arch##_##m##_##n##_##k = MmaMacroEncode(MmaMacroEncode::Arch::arch, m, n, k)

enum class MmaMacro : uint64_t {
  NoMMA = 0,

  MACRO(Turing, 16, 8, 8),
  MACRO(Turing, 16, 8, 16),
  MACRO(Turing, 16, 16, 16),

  MACRO(Ampere, 16, 8, 16),
  MACRO(Ampere, 16, 16, 16),

  MACRO(Hopper, 64, 8, 16),
  MACRO(Hopper, 64, 16, 16),
  MACRO(Hopper, 64, 24, 16),
  MACRO(Hopper, 64, 32, 16),
  MACRO(Hopper, 64, 40, 16),
  MACRO(Hopper, 64, 48, 16),
  MACRO(Hopper, 64, 56, 16),
  MACRO(Hopper, 64, 64, 16),
  MACRO(Hopper, 64, 72, 16),
  MACRO(Hopper, 64, 80, 16),
  MACRO(Hopper, 64, 88, 16),
  MACRO(Hopper, 64, 96, 16),
  MACRO(Hopper, 64, 104, 16),
  MACRO(Hopper, 64, 112, 16),
  MACRO(Hopper, 64, 120, 16),
  MACRO(Hopper, 64, 128, 16),
  MACRO(Hopper, 64, 136, 16),
  MACRO(Hopper, 64, 144, 16),
  MACRO(Hopper, 64, 152, 16),
  MACRO(Hopper, 64, 160, 16),
  MACRO(Hopper, 64, 168, 16),
  MACRO(Hopper, 64, 176, 16),
  MACRO(Hopper, 64, 184, 16),
  MACRO(Hopper, 64, 192, 16),
  MACRO(Hopper, 64, 200, 16),
  MACRO(Hopper, 64, 208, 16),
  MACRO(Hopper, 64, 216, 16),
  MACRO(Hopper, 64, 224, 16),
  MACRO(Hopper, 64, 232, 16),
  MACRO(Hopper, 64, 240, 16),
  MACRO(Hopper, 64, 248, 16),
  MACRO(Hopper, 64, 256, 16),

  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-kind-shapes
  // (M=64, N=multiple of 8, K=16)
  MACRO(Blackwell1CTA, 64, 8, 16),
  MACRO(Blackwell1CTA, 64, 16, 16),
  MACRO(Blackwell1CTA, 64, 24, 16),
  MACRO(Blackwell1CTA, 64, 32, 16),
  MACRO(Blackwell1CTA, 64, 40, 16),
  MACRO(Blackwell1CTA, 64, 48, 16),
  MACRO(Blackwell1CTA, 64, 56, 16),
  MACRO(Blackwell1CTA, 64, 64, 16),
  MACRO(Blackwell1CTA, 64, 72, 16),
  MACRO(Blackwell1CTA, 64, 80, 16),
  MACRO(Blackwell1CTA, 64, 88, 16),
  MACRO(Blackwell1CTA, 64, 96, 16),
  MACRO(Blackwell1CTA, 64, 104, 16),
  MACRO(Blackwell1CTA, 64, 112, 16),
  MACRO(Blackwell1CTA, 64, 120, 16),
  MACRO(Blackwell1CTA, 64, 128, 16),
  MACRO(Blackwell1CTA, 64, 136, 16),
  MACRO(Blackwell1CTA, 64, 144, 16),
  MACRO(Blackwell1CTA, 64, 152, 16),
  MACRO(Blackwell1CTA, 64, 160, 16),
  MACRO(Blackwell1CTA, 64, 168, 16),
  MACRO(Blackwell1CTA, 64, 176, 16),
  MACRO(Blackwell1CTA, 64, 184, 16),
  MACRO(Blackwell1CTA, 64, 192, 16),
  MACRO(Blackwell1CTA, 64, 200, 16),
  MACRO(Blackwell1CTA, 64, 208, 16),
  MACRO(Blackwell1CTA, 64, 216, 16),
  MACRO(Blackwell1CTA, 64, 224, 16),
  MACRO(Blackwell1CTA, 64, 232, 16),
  MACRO(Blackwell1CTA, 64, 240, 16),
  MACRO(Blackwell1CTA, 64, 248, 16),
  MACRO(Blackwell1CTA, 64, 256, 16),

  // (M=128, N=multiple of 16, K=16)
  MACRO(Blackwell1CTA, 128, 16, 16),
  MACRO(Blackwell1CTA, 128, 32, 16),
  MACRO(Blackwell1CTA, 128, 48, 16),
  MACRO(Blackwell1CTA, 128, 64, 16),
  MACRO(Blackwell1CTA, 128, 80, 16),
  MACRO(Blackwell1CTA, 128, 96, 16),
  MACRO(Blackwell1CTA, 128, 112, 16),
  MACRO(Blackwell1CTA, 128, 128, 16),
  MACRO(Blackwell1CTA, 128, 144, 16),
  MACRO(Blackwell1CTA, 128, 160, 16),
  MACRO(Blackwell1CTA, 128, 176, 16),
  MACRO(Blackwell1CTA, 128, 192, 16),
  MACRO(Blackwell1CTA, 128, 208, 16),
  MACRO(Blackwell1CTA, 128, 224, 16),
  MACRO(Blackwell1CTA, 128, 240, 16),
  MACRO(Blackwell1CTA, 128, 256, 16),

  // (M=128, N=multiple of 32, K=16)
  MACRO(Blackwell2CTA, 128, 32, 16),
  MACRO(Blackwell2CTA, 128, 64, 16),
  MACRO(Blackwell2CTA, 128, 96, 16),
  MACRO(Blackwell2CTA, 128, 128, 16),
  MACRO(Blackwell2CTA, 128, 160, 16),
  MACRO(Blackwell2CTA, 128, 192, 16),
  MACRO(Blackwell2CTA, 128, 224, 16),
  MACRO(Blackwell2CTA, 128, 256, 16),

  // (M=256, N=multiple of 32, K=16)
  MACRO(Blackwell2CTA, 256, 32, 16),
  MACRO(Blackwell2CTA, 256, 64, 16),
  MACRO(Blackwell2CTA, 256, 96, 16),
  MACRO(Blackwell2CTA, 256, 128, 16),
  MACRO(Blackwell2CTA, 256, 160, 16),
  MACRO(Blackwell2CTA, 256, 192, 16),
  MACRO(Blackwell2CTA, 256, 224, 16),
  MACRO(Blackwell2CTA, 256, 256, 16),
};

#undef MACRO

//! [Operand Layout Convention]
//! Operand layout, T=transposed/row_major, N=normal/col_major
//! Ordered by position of K
//! NT : K,M x K,N -> M,N
//! TT : M,K X K,N -> M,N
//! TN : M,K X N,K -> M,N
//! NN : K,M X N,K -> M,N
enum class MmaLayout { NT = 0, TT, TN, NN };

//! Indicates which dimension is innermost in the allocation domain of an
//! operand
enum class UnitDim { K, M_or_N };

//! Utility to annotate which input of mma this option struct describes
enum class MmaOperand { A, B };

//! GPU arch check for macro type
inline bool isTuring(MmaMacro macro) {
  return MmaMacroEncode(macro).arch == MmaMacroEncode::Arch::Turing;
}

inline bool isAmpere(MmaMacro macro) {
  return MmaMacroEncode(macro).arch == MmaMacroEncode::Arch::Ampere;
}

inline bool isHopper(MmaMacro macro) {
  return MmaMacroEncode(macro).arch == MmaMacroEncode::Arch::Hopper;
}

inline bool isBlackwell(MmaMacro macro) {
  auto arch = MmaMacroEncode(macro).arch;
  return arch == MmaMacroEncode::Arch::Blackwell1CTA ||
      arch == MmaMacroEncode::Arch::Blackwell2CTA;
}

inline bool isBlackwell1CTA(MmaMacro macro) {
  return MmaMacroEncode(macro).arch == MmaMacroEncode::Arch::Blackwell1CTA;
}

inline bool isBlackwell2CTA(MmaMacro macro) {
  return MmaMacroEncode(macro).arch == MmaMacroEncode::Arch::Blackwell2CTA;
}

//! Get the m size from macro type
inline int64_t getM(MmaMacro macro) {
  return MmaMacroEncode(macro).m;
}

//! Get the n size from macro type
inline int64_t getN(MmaMacro macro) {
  return MmaMacroEncode(macro).n;
}

//! Get the k size from macro type
inline int64_t getK(MmaMacro macro) {
  return MmaMacroEncode(macro).k;
}

// Unpacked constants from macro type:
//   exact numbers are defined by each individual instruction.
int getOutputRegisterSize(MmaMacro macro);
int getInputARegisterSize(MmaMacro macro);
int getInputBRegisterSize(MmaMacro macro);

// Unpack MMA op shape
GemmTile getMmaOpShape(MmaMacro macro);

// Warning: The values of the enum class must match the matrix descriptor as
// specified in:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
// Do not edit the values of the enum class unless you know what you are doing.
enum class MmaInputSmemSwizzle {
  None = 0,
  B128 = 1,
  B64 = 2,
  B32 = 3,
};

constexpr int64_t core_matrix_width_bytes = 16;

int64_t getSharedMemoryByteAlignment(MmaInputSmemSwizzle swizzle);
int64_t getBytesFromSwizzle(MmaInputSmemSwizzle swizzle);
MmaInputSmemSwizzle getSwizzleFromBytes(int64_t bytes);

// MMA stringify utils
NVF_API std::string toString(MmaLayout input_layout);
NVF_API std::string toString(const GemmTile& tile);
NVF_API std::string toString(const MatMulTileOptions& opts);
NVF_API std::string toString(MmaMacro macro);
NVF_API std::string toString(MmaInputSmemSwizzle swizzle);
inline std::ostream& operator<<(
    std::ostream& os,
    MmaInputSmemSwizzle input_layout) {
  os << toString(input_layout);
  return os;
}

// MMA hash utils
NVF_API size_t hash(MmaMacro macro);
size_t hash(MmaLayout input_layout);
size_t hash(const GemmTile& tile);
NVF_API size_t hash(const MatMulTileOptions& opts);

} // namespace nvfuser
