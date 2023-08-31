// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/hash.h>
#include <mma_type.h>
#include <scheduler/heuristic.h>
#include <utils.h>
#include <functional>

#include <sstream>
#include "type.h"

namespace nvfuser {

// Parameters of the matmul heuristic to describe the optimial schedule.
class MatmulParams : public HeuristicParams {
 public:
  //! A list of possible strategies used to define along which axis
  //!  parallelization will be done.
  enum class TileRasterizationOrder { RowMajor = 0, ColumnMajor = 1 };

  //! A wrapper for double buffering config pieces
  struct DoubleBufferOptions {
    bool double_buffer_smem_write = false;
    bool double_buffer_smem_read = false;
    int smem_double_buffer_stage = 2;

    bool operator==(const DoubleBufferOptions& other) const {
      return other.double_buffer_smem_write == double_buffer_smem_write &&
          other.double_buffer_smem_read == double_buffer_smem_read &&
          other.smem_double_buffer_stage == smem_double_buffer_stage;
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "DoubleBufferOptions:\n"
         << "  double_buffer_smem_write: "
         << (double_buffer_smem_write ? "true" : "false") << "\n"
         << "  double_buffer_smem_read: "
         << (double_buffer_smem_read ? "true" : "false") << "\n"
         << "  smem_double_buffer_stage: " << smem_double_buffer_stage;
      return ss.str();
    }

    size_t hash() const {
      return std::hash<size_t>{}(
                 (static_cast<size_t>(smem_double_buffer_stage) << 2) |
                 (static_cast<size_t>(double_buffer_smem_write)) << 1) |
          (static_cast<size_t>(double_buffer_smem_read));
    }
  };

  //! Whether to rotate the ldmatrix out of the main loop
  bool rotate_ldmatrix_out_of_main_loop = true;

  //! (Ampere+) Use cp.async to load operands.
  bool async_gmem_load_operands = false;

  //! Specifies the tiling hierarchy on block,
  //!  warp, and instruction levels.
  MatMulTileOptions tile_sizes = {};

  //! Specify the type of MMA op to be used in generated kernel.
  MmaOptions::MacroType mma_macro = MmaOptions::MacroType::NoMMA;

  //! Specify CTA rastrization order.
  TileRasterizationOrder cta_order = TileRasterizationOrder::RowMajor;

  //! Specify which tensor we double buffer.
  DoubleBufferOptions double_buffer_options = {};

  //! Swizzle factor is used to increase L2 hit rate.
  //!  It horizontally squeezes the grid so that gridDim.x is larger and
  //!  gridDim.y is smaller.
  //!  We rely on the observation that the CTAs are scheduled by the GPU by
  //!  iterating on gridDim.x first. As a result, as blocks are launched, they
  //!  will more likely be forming sub-tiles of the C matrix. This will increase
  //!  L2 hit rate/data reuse of A and B.
  //!
  //! Eg for grid_swizzle_factor=2:
  //!    A1 A2 B1 B2 -->   A1 A2 A3 A4 B1 B2 B3 B4
  //!    A3 A4 B3 B4       C1 C2 C3 C4 D1 D2 D3 D4
  //!    C1 C2 D1 D2
  //!    C3 C4 D3 D4
  int grid_swizzle_factor = 1;

  //! Unswizzle MMA results in shared memory to get
  //!  coalesced write to global memory
  bool use_smem_epilogue = false;

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Matmul Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << "\n"
       << "MMA macro: " << nvfuser::toString(mma_macro, true) << "\n"
       << double_buffer_options.toString() << "\n"
       << nvfuser::toString(tile_sizes) << "\n"
       << "Rotate ldmatrix out of main loop: "
       << (rotate_ldmatrix_out_of_main_loop ? "true" : "false") << "\n"
       << "Async global mem load: "
       << (async_gmem_load_operands ? "true" : "false") << "\n"
       << "Indexing mode: "
       << (cparams.index_type.has_value()
               ? (cparams.index_type.value() == PrimDataType::Int ? "int64_t"
                                                                  : "int32_t")
               : "unavailable")
       << "\n"
       << "Tile rastrization order: "
       << ((cta_order == TileRasterizationOrder::RowMajor) ? "row-major"
                                                           : "column-major")
       << "\n"
       << "Grid swizzle factor: " << grid_swizzle_factor << "\n"
       << "Use shared memory epilogue: " << use_smem_epilogue << "\n"
       << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    // combine boolean flags for hashing
    size_t attr_hash =
        (static_cast<size_t>(rotate_ldmatrix_out_of_main_loop) << 1) |
        (static_cast<size_t>(async_gmem_load_operands));

    // combined hash
    attr_hash = std::hash<size_t>{}(attr_hash) ^
        (nvfuser::hash(mma_macro) << 1) ^ (double_buffer_options.hash() << 2) ^
        (nvfuser::hash(tile_sizes) << 3) ^
        (std::hash<size_t>{}(static_cast<size_t>(cta_order)) << 4) ^
        (std::hash<size_t>{}(grid_swizzle_factor) << 5);
    return attr_hash;
  }

  bool sameAs(
      const std::shared_ptr<HeuristicParams>& other_base) const override {
    auto other_casted = std::dynamic_pointer_cast<MatmulParams>(other_base);
    if (other_casted == nullptr) {
      return false;
    }

    return other_casted->mma_macro == mma_macro &&
        other_casted->async_gmem_load_operands == async_gmem_load_operands &&
        other_casted->rotate_ldmatrix_out_of_main_loop ==
        rotate_ldmatrix_out_of_main_loop &&
        other_casted->tile_sizes == tile_sizes &&
        other_casted->double_buffer_options == double_buffer_options &&
        other_casted->cta_order == cta_order &&
        other_casted->grid_swizzle_factor == grid_swizzle_factor;
  }

  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<MatmulParams>(*this);
  }
};

} // namespace nvfuser
