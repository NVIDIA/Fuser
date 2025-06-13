// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <ir/all_nodes.h>
#include <mma_type.h>
#include <functional>

namespace nvfuser {

GemmTile getMmaOpShape(MmaMacro macro) {
  return {getM(macro), getN(macro), getK(macro)};
}

int64_t getSharedMemoryByteAlignment(MmaInputSmemSwizzle swizzle) {
  // References:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#swizzling-modes
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-multi-dim-tma
  switch (swizzle) {
    case MmaInputSmemSwizzle::None:
      return 128;
    case MmaInputSmemSwizzle::B32:
      return 256;
    case MmaInputSmemSwizzle::B64:
      return 512;
    case MmaInputSmemSwizzle::B128:
      return 1024;
    default:
      NVF_CHECK(false, "Unknown swizzle type!");
      break;
  }
}

int64_t getBytesFromSwizzle(MmaInputSmemSwizzle swizzle) {
  switch (swizzle) {
    case MmaInputSmemSwizzle::None:
      return 16;
    case MmaInputSmemSwizzle::B32:
      return 32;
    case MmaInputSmemSwizzle::B64:
      return 64;
    case MmaInputSmemSwizzle::B128:
      return 128;
    default:
      NVF_CHECK(false, "Unknown swizzle type!");
      break;
  }
}

MmaInputSmemSwizzle getSwizzleFromBytes(int64_t bytes) {
  switch (bytes) {
    case 16:
      return MmaInputSmemSwizzle::None;
    case 32:
      return MmaInputSmemSwizzle::B32;
    case 64:
      return MmaInputSmemSwizzle::B64;
    case 128:
      return MmaInputSmemSwizzle::B128;
    default:
      NVF_CHECK(false, "Unknown swizzle size!");
      break;
  }
}

std::string toString(MmaLayout input_layout) {
  std::stringstream ss;
  switch (input_layout) {
    case MmaLayout::TT:
      ss << "TT";
      break;
    case MmaLayout::TN:
      ss << "TN";
      break;
    case MmaLayout::NT:
      ss << "NT";
      break;
    case MmaLayout::NN:
      ss << "NN";
      break;
    default:
      NVF_THROW("unsupported operand layout");
  }
  return ss.str();
}

std::string toString(const GemmTile& tile) {
  std::stringstream ss;
  ss << "[" << tile.m << ", " << tile.n << ", " << tile.k << "]";
  return ss.str();
}

std::string toString(const MatMulTileOptions& opts) {
  std::stringstream ss;
  ss << "MatMulTileOptions:\n  CTA tile " << toString(opts.cta_tile)
     << "\n  warp tile " << toString(opts.warp_tile) << "\n  epilogue tile "
     << toString(opts.epilogue_tile);
  return ss.str();
}

std::string toString(MmaMacro macro) {
  std::stringstream ss;
  auto underlying = static_cast<MmaMacroEncode>(macro);
  switch (underlying.arch) {
    case MmaMacroEncode::Arch::NoMma:
      return "NoOp";
    case MmaMacroEncode::Arch::Volta:
      ss << "Volta";
      break;
    case MmaMacroEncode::Arch::Turing:
      ss << "Turing";
      break;
    case MmaMacroEncode::Arch::Ampere:
      ss << "Ampere";
      break;
    case MmaMacroEncode::Arch::Hopper:
      ss << "Hopper";
      break;
    case MmaMacroEncode::Arch::Blackwell1CTA:
      ss << "Blackwell1CTA";
      break;
    case MmaMacroEncode::Arch::Blackwell2CTA:
      ss << "Blackwell2CTA";
      break;
  }
  ss << "_" << underlying.m << "_" << underlying.n << "_" << underlying.k;
  return ss.str();
}

std::string toString(MmaInputSmemSwizzle swizzle) {
  switch (swizzle) {
    case MmaInputSmemSwizzle::None:
      return "NoSwizzle";
    case MmaInputSmemSwizzle::B32:
      return "32B";
    case MmaInputSmemSwizzle::B64:
      return "64B";
    case MmaInputSmemSwizzle::B128:
      return "128B";
    default:
      NVF_CHECK(false, "Unknown tensor map swizzle type!");
      break;
  }
}

size_t hash(MmaMacro macro) {
  return std::hash<size_t>{}(static_cast<size_t>(macro));
}

size_t hash(MmaLayout input_layout) {
  return std::hash<size_t>{}(static_cast<size_t>(input_layout));
}

size_t hash(const GemmTile& tile) {
  return std::hash<size_t>{}(
      (static_cast<size_t>(tile.m) << 32) +
      (static_cast<size_t>(tile.n) << 16) + (static_cast<size_t>(tile.k)));
}

size_t hash(const MatMulTileOptions& opts) {
  size_t h = hash(opts.warp_tile);
  hashCombine(h, hash(opts.cta_tile));
  return h;
}

std::string toString(const MatmulDimRole role) {
  switch (role) {
    case MatmulDimRole::Batch:
      return "Batch";
    case MatmulDimRole::M:
      return "M";
    case MatmulDimRole::N:
      return "N";
    case MatmulDimRole::K:
      return "K";
  }
  // Unreachable
  return "Unrecognized role";
}

} // namespace nvfuser
