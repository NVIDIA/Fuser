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

MmaOp* MmaOptions::mmaOp() const {
  NVF_ERROR(
      accumulator_tv != nullptr && accumulator_tv->definition() != nullptr,
      "Invalid accumulator_tv.");
  auto mma_op = dynamic_cast<MmaOp*>(accumulator_tv->definition());
  NVF_ERROR(mma_op != nullptr, "accumulator tv not an output of mma op");
  return mma_op;
}

MmaBuilder::MmaBuilder(
    MmaOptions::MacroType macro,
    MatMulTileOptions gemm_tile) {
  option_.macro = macro;
}

MmaBuilder& MmaBuilder::layout(MmaOptions::MmaLayout layout) {
  option_.layout = layout;
  return *this;
}

MmaBuilder& MmaBuilder::operand(MmaOptions::Operand a_or_b) {
  option_.operand = a_or_b;
  return *this;
}

// TODO: validate op config
MmaOptions MmaBuilder::build() const {
  NVF_CHECK(
      option_.accumulator_tv != nullptr,
      "Please configure accumulator tv before using swizzle options.")
  return option_;
}

void MmaBuilder::configureMma(MmaOp* mma) const {
  NVF_CHECK(mma, "configureMma: invalid op object ", mma);
  mma->configureOptions(option_);
}

void MmaBuilder::accumulatorTv(TensorView* tv) {
  NVF_CHECK(
      tv->getMemoryType() == MemoryType::Local, "Mma only outputs to register");
  NVF_CHECK(tv->definition(), "Input cannot be accumulator tv");
  NVF_CHECK(
      tv->definition()->isA<MmaOp>(),
      "Requires mma op output for reduction tv");
  option_.accumulator_tv = tv;
}

namespace {

// Utility to get ldmatrix direction a mma layout and operand
LoadStoreOpType getLdMatrixType(MmaOptions options) {
  bool transpose = false;
  switch (options.macro) {
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
    case MmaOptions::MacroType::Ampere_16_16_16:
    case MmaOptions::MacroType::Turing_16_16_16:
      // Turing mma assumes TN as default
      transpose = (options.operand == MmaOptions::Operand::A &&
                   !isOperandTransposed(options)) ||
          (options.operand == MmaOptions::Operand::B &&
           isOperandTransposed(options));
      break;
    default:
      NVF_ERROR(false, "unsupported op with ldmatrix");
      break;
  }
  return transpose ? LoadStoreOpType::LdMatrixTranspose
                   : LoadStoreOpType::LdMatrix;
}

} // namespace

LoadStoreOpType MmaBuilder::ldMatrix() const {
  return getLdMatrixType(option_);
}

bool isVolta(MmaOptions::MacroType macro) {
  return macro == MmaOptions::MacroType::Volta_16_16_4;
}

bool isTuring(MmaOptions::MacroType macro) {
  return macro == MmaOptions::MacroType::Turing_16_8_16 ||
      macro == MmaOptions::MacroType::Turing_16_16_16;
}

bool isAmpere(MmaOptions::MacroType macro) {
  return macro == MmaOptions::MacroType::Ampere_16_8_8 ||
      macro == MmaOptions::MacroType::Ampere_16_8_16 ||
      macro == MmaOptions::MacroType::Ampere_16_16_16;
}

bool isOperandTransposed(MmaOptions options) {
  switch (options.operand) {
    case MmaOptions::Operand::A:
      return options.layout == MmaOptions::MmaLayout::TT ||
          options.layout == MmaOptions::MmaLayout::TN;
    case MmaOptions::Operand::B:
      return options.layout == MmaOptions::MmaLayout::TT ||
          options.layout == MmaOptions::MmaLayout::NT;
    default:
      NVF_CHECK(false, "isOperandTransposed: please specify operand");
  }
  return false;
}

GemmTile getMmaOpShape(MmaOptions::MacroType macro) {
  switch (macro) {
    case MmaOptions::MacroType::Volta_16_16_4:
      return {16, 16, 4};
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
      return {16, 8, 16};
    case MmaOptions::MacroType::Turing_16_16_16:
    case MmaOptions::MacroType::Ampere_16_16_16:
      return {16, 16, 16};
    case MmaOptions::MacroType::Ampere_16_8_8:
      return {16, 8, 8};
    case MmaOptions::MacroType::NoMMA:
      return {1, 1, 1};
  }

  NVF_ERROR(false, "unknown MMA macro");
}

std::string toString(MmaOptions::MmaLayout input_layout) {
  std::stringstream ss;
  switch (input_layout) {
    case MmaOptions::MmaLayout::TT:
      ss << "TT";
      break;
    case MmaOptions::MmaLayout::TN:
      ss << "TN";
      break;
    case MmaOptions::MmaLayout::NT:
      ss << "NT";
      break;
    case MmaOptions::MmaLayout::NN:
      ss << "NN";
      break;
    default:
      NVF_ERROR(false, "unsupported operand layout");
  }
  return ss.str();
}

std::string toString(MmaOptions::MacroType mt) {
  std::stringstream ss;
  switch (mt) {
    case MmaOptions::MacroType::NoMMA:
      ss << "NoOp";
      break;
    case MmaOptions::MacroType::Volta_16_16_4:
      ss << "M16N16K4";
      break;
    case MmaOptions::MacroType::Turing_16_8_16:
    case MmaOptions::MacroType::Ampere_16_8_16:
      ss << "M16N8K16";
      break;
    case MmaOptions::MacroType::Turing_16_16_16:
    case MmaOptions::MacroType::Ampere_16_16_16:
      ss << "M16N16K16";
      break;
    default:
      NVF_ERROR(false, "undefined mma type");
      break;
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
  ss << "MatMulTileOptions: "
     << "instruction tile " << toString(opts.instruction_tile) << ", "
     << "warp tile " << toString(opts.warp_tile) << ", "
     << "CTA tile " << toString(opts.cta_tile);
  return ss.str();
}

std::string toString(MmaOptions::MacroType mt, bool) {
  switch (mt) {
    case MmaOptions::MacroType::Ampere_16_8_8:
      return "Ampere_16_8_8";
    case MmaOptions::MacroType::Ampere_16_8_16:
      return "Ampere_16_8_16";
    case MmaOptions::MacroType::Ampere_16_16_16:
      return "Ampere_16_16_16";
    case MmaOptions::MacroType::NoMMA:
      return "NoOp";
    case MmaOptions::MacroType::Turing_16_8_16:
      return "Turing_16_8_16";
    case MmaOptions::MacroType::Turing_16_16_16:
      return "Turing_16_16_16";
    case MmaOptions::MacroType::Volta_16_16_4:
      return "Volta_16_16_4";
  }
  NVF_ERROR(false, "Unsupported mma type");
  return "Unsupported";
}

size_t hash(MmaOptions::MacroType macro) {
  return std::hash<size_t>{}(static_cast<size_t>(macro));
}

size_t hash(MmaOptions::MmaLayout input_layout) {
  return std::hash<size_t>{}(static_cast<size_t>(input_layout));
}

size_t hash(const GemmTile& tile) {
  return std::hash<size_t>{}(
      (static_cast<size_t>(tile.m) << 32) +
      (static_cast<size_t>(tile.n) << 16) + (static_cast<size_t>(tile.k)));
}

size_t hash(const MatMulTileOptions& opts) {
  return (hash(opts.instruction_tile) << 0) ^ (hash(opts.warp_tile) << 1) ^
      (hash(opts.cta_tile) << 2);
}

} // namespace nvfuser
