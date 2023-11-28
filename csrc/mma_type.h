// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <macros.h>

#include <c10/macros/Export.h>
#include <exceptions.h>
#include <fusion.h>

#include <cstring>

#if IS_CPP20
#include <bit>
#endif
#include <cstdint>

namespace nvfuser {

constexpr std::string_view MATMUL_LOG_PREFIX = "[MATMUL DEBUG] ";

//! Named descriptors of domains in matmul
enum class MatmulDomain { M = 0, N, K };

//! Named descriptors of TensorView roles in fusion
//!  INPUT_A - a producer of MMA input A
//!  INPUT_B - a producer of MMA input B
//!  INPUT_C - a producer of a tensor used in fusion epilogue,
//!            for example tensor used in beta scaling fusion
//!  OUTPUT_D - the main consumer of MMA op results
//!  OUTPUT_AUX - fusion outputs that are consumers of OUTPUT_D
//!
//! Naming convention is based on the following formula:
//!    D = alpha * A x B + beta * C
//!    AUX = relu(D)
//!  Note: bias vector tensors will be assigned to INPUT_C role.
enum class MatmulRole { INPUT_A = 0, INPUT_B, OUTPUT_D, INPUT_C, OUTPUT_AUX };

//! The expected number of occurances of core TensorView roles in fusion
static constexpr size_t MATMUL_CORE_ROLES_EXPECTED_COUNT = 1;

//! Utility data structure for recording gemm tiles
struct GemmTile {
  int m, n, k;
  GemmTile(int m_, int n_, int k_) : m(m_), n(n_), k(k_) {}

  bool operator==(const GemmTile& other) const {
    return m == other.m && n == other.n && k == other.k;
  }

  GemmTile operator/(const GemmTile& other) const {
    return GemmTile(m / other.m, n / other.n, k / other.k);
  }

  std::vector<int> toVector() const {
    return {m, n, k};
  }
};

//! Utility data structure for recording gemm tiles
struct MatMulTileOptions {
  GemmTile cta_tile = GemmTile(128, 128, 32);
  GemmTile warp_tile = GemmTile(64, 64, 32);
  GemmTile instruction_tile = GemmTile(16, 8, 16);

  MatMulTileOptions() = default;
  MatMulTileOptions(
      GemmTile cta_tile_,
      GemmTile warp_tile_,
      GemmTile instruction_tile_)
      : cta_tile(cta_tile_),
        warp_tile(warp_tile_),
        instruction_tile(instruction_tile_) {}

  bool operator==(const MatMulTileOptions& other) {
    return cta_tile == other.cta_tile && warp_tile == other.warp_tile &&
        instruction_tile == other.instruction_tile;
  }
};

enum class MmaMacro : uint64_t;

struct MmaMacroEncode {
  enum class Arch { NoMma, Volta, Turing, Ampere, Hopper } arch : 16;
  unsigned m : 16;
  unsigned n : 16;
  unsigned k : 16;

  constexpr operator uint64_t() {
#if IS_CPP20 && !defined(__clang__)
    // std::bit_cast for bit field is not supported by clang yet
    return std::bit_cast<uint64_t>(*this);
#else
    return (uint64_t)arch << 48 | (uint64_t)m << 32 | (uint64_t)n << 16 |
        (uint64_t)k;
#endif
  }

  constexpr operator MmaMacro();

  constexpr MmaMacroEncode(MmaMacro macro);

  constexpr MmaMacroEncode(Arch arch, unsigned m, unsigned n, unsigned k)
      : arch(arch), m(m), n(n), k(k) {}
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
};

#undef MACRO

constexpr MmaMacroEncode::operator MmaMacro() {
#if IS_CPP20 && !defined(__clang__)
  // std::bit_cast for bit field is not supported by clang yet
  return std::bit_cast<MmaMacro>(*this);
#else
  return static_cast<MmaMacro>(static_cast<uint64_t>(*this));
#endif
}

constexpr MmaMacroEncode::MmaMacroEncode(MmaMacro macro)
#if IS_CPP20 && !defined(__clang__)
{
  // std::bit_cast for bit field is not supported by clang yet
  *this = std::bit_cast<MmaMacroEncode>(macro);
}
#else
    : arch((Arch)(toUnderlying(macro) >> 48)),
      m((toUnderlying(macro) >> 32) & 0xFFFF),
      n((toUnderlying(macro) >> 16) & 0xFFFF),
      k(toUnderlying(macro) & 0xFFFF) {
}
#endif

//! [Operand Layout Convention]
//! Operand layout, T=transposed/row_major, N=normal/col_major
//! Ordered by position of K
//! NT : K,M x K,N -> M,N
//! TT : M,K X K,N -> M,N
//! TN : M,K X N,K -> M,N
//! NN : K,M X N,K -> M,N
enum class MmaLayout { NT = 0, TT, TN, NN };

//! Information for configuring and lowering mma ops
struct MmaOptions {
  using MacroType = MmaMacro;

  //! Utility to annotate which input of mma this option struct describes
  enum class Operand { Accumulator = 0, A, B };

  //! Utility to annotate which mma macro this config uses.
  MacroType macro = MacroType::NoMMA;

  //! Utility to annotate which input of mma this option struct describes
  Operand operand = Operand::A;

  bool operator==(const MmaOptions& other) const {
    return macro == other.macro && operand == other.operand;
  }
};

//! User interface for configuring the mma and mma related
//!  operators by specifying the mma instruction tile type
//!  input data layout, and the operand position of a tensor.
class MmaBuilder {
 public:
  //! Initialized a mma builder, for the given mma instruction type.
  MmaBuilder(MmaOptions::MacroType macro);

  //! User configuration function:
  //!  Specifies which element in the mma op this builder is generating
  //!    parameters for, i.e. A or B. This is useful when generating
  //!    data swizzles for different elements of mma.
  //!  - Operand::Accumulator means the parameters describe accumulator in mma
  //!  op.
  //!  - This option is ignored when configuring the mma operator itself.
  MmaBuilder& operand(MmaOptions::Operand a_or_b);

  //! Export all the parameters with user's configurations applied.
  MmaOptions build() const;

 private:
  MmaOptions option_;
};

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

//! Get the m size from macro type
inline int getM(MmaMacro macro) {
  return MmaMacroEncode(macro).m;
}

//! Get the n size from macro type
inline int getN(MmaMacro macro) {
  return MmaMacroEncode(macro).n;
}

//! Get the k size from macro type
inline int getK(MmaMacro macro) {
  return MmaMacroEncode(macro).k;
}

//! Returns true if the given option describes a transposed operand
bool isOperandTransposed(MmaOptions options);

// Unpacked constants from macro type:
//   exact numbers are defined by each individual instruction.
int getOutputRegisterSize(MmaMacro macro);
int getInputARegisterSize(MmaMacro macro);
int getInputBRegisterSize(MmaMacro macro);

// Unpack MMA op shape
GemmTile getMmaOpShape(MmaMacro macro);

// MMA stringify utils
std::string toString(MmaLayout input_layout);
std::string toString(const GemmTile& tile);
std::string toString(const MatMulTileOptions& opts);
std::string toString(MmaMacro macro);

// MMA hash utils
size_t hash(MmaMacro macro);
size_t hash(MmaLayout input_layout);
size_t hash(const GemmTile& tile);
size_t hash(const MatMulTileOptions& opts);
} // namespace nvfuser
