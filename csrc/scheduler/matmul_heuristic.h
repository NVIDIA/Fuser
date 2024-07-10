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

  //! A wrapper for circular buffering config pieces
  struct CircularBufferOptions {
    bool circular_buffer_smem_write = false;
    bool circular_buffer_smem_read = false;
    // This parameter controls the number of circular
    // buffering stages to use when loading operands a and b.
    //
    // If this value is greater than two then it indicates circular buffering,
    // in which case async_gmem_load_operands must also be true.
    //
    // Note that whenever circular_buffer_smem_write is true, this value must be
    // greater than one. Otherwise it is ignored.
    int smem_circular_buffer_stage = 2;

    bool operator==(const CircularBufferOptions& other) const {
      return other.circular_buffer_smem_write == circular_buffer_smem_write &&
          other.circular_buffer_smem_read == circular_buffer_smem_read &&
          other.smem_circular_buffer_stage == smem_circular_buffer_stage;
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "CircularBufferOptions:\n"
         << "  circular_buffer_smem_write: "
         << (circular_buffer_smem_write ? "true" : "false") << "\n"
         << "  circular_buffer_smem_read: "
         << (circular_buffer_smem_read ? "true" : "false") << "\n"
         << "  smem_circular_buffer_stage: " << smem_circular_buffer_stage;
      return ss.str();
    }

    size_t hash() const {
      return std::hash<size_t>{}(
                 (static_cast<size_t>(smem_circular_buffer_stage) << 2) |
                 (static_cast<size_t>(circular_buffer_smem_write)) << 1) |
          (static_cast<size_t>(circular_buffer_smem_read));
    }
  };

  //! This is the maximum vectorization supported by the inputs and outputs.
  //! This refers to the number of data elements loaded simultaneously, not the
  //! number of bytes.
  struct SupportedVectorization {
    // Each operand load from global to shared memory is vectorized along its
    // inner-most allocation dimension as long as that is an M, N, or K
    // dimension. For example, if the innermost dimension is a batch dimension
    // then we will not vectorize that operand's loads from global to shared
    // memory. If there are multiple dimensions in a given role, such as
    // multiple K dimensions, then we can only vectorize those inner dimensions
    // that are consistent with the canonical dimension ordering shared by all
    // tensors in the Fusion.
    int64_t a;
    int64_t b;

    // The epilogue is handled in a separate loop from the main loop/operand
    // loads. We inline the epilogue expressions as much as possible, and we
    // vectorize all tensors with the same factor for better memory coalescence;
    // i.e. we parallelize the epilogue like [ ... TIDx V ] so we do not
    // introduce any loops between the TIDx and V dimensions. If we used
    // different vectorization for each output or epilogue input, then we would
    // need an unrolled loop between TIDx and V which would interfere with
    // memory coalescence. We assume the decrease in indexing arithmetic from
    // vectorization is not worth the slowdown from non-coalesced accesses, so
    // we prefer to use a smaller vectorization instead.
    //
    // To determine the epilogue vectorization we do the following steps:
    //  - Look at each output, then each epilogue input and find the first
    //    tensor with a non-batch dimension as its innermost allocation
    //    dimension. We will use that as the innermost loop dimension and will
    //    vectorize that dimension. If there are multiple such innermost
    //    dimensions with the same role and full contiguity then we consider all
    //    those dimensions as the merged vectorized dimension. For example if
    //    we have an output whose allocation domain is [ B1 M1 N1 M2 M3 ] then
    //    (M2*M3) will be the vectorized dimension. On the other hand, we would
    //    skip a tensor that had allocation domain [ M1 M2 M3 N1 B1 ] since the
    //    batch dimension is innermost.
    //  - Then we pass over all epilogue inputs and outputs. For each tensor, we
    //    consider all innermost dimensions in order. For example if we have
    //    determined that we will vectorize along M1*M2*M3 and a tensor has
    //    allocation [ B1 M1 N1 M2 M3 ] then we consider dimension M2*M3 (along
    //    with all other strides) to find supported vectorization. If another
    //    tensor has allocation [ B1 M1 M2 M3 N1 ] then we skip it since its
    //    innermost dimension is not an N role dimension so its access will not
    //    be vectorized.
    //  - We store the minimum of all the maximum supported vectorizations
    //    across all epilogue input and output tensors that were not skipped.
    //    That is the value below. If no vectorization is possible, this will be
    //    set to 1.
    int64_t epilogue;

    bool operator==(const SupportedVectorization& other) const {
      return other.a == a && other.b == b && other.epilogue == epilogue;
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "SupportedVectorization:\n"
         << "  a: " << a << "\n"
         << "  b: " << b << "\n"
         << "  epilogue: " << epilogue;
      return ss.str();
    }

    size_t hash() const {
      return std::hash<size_t>{}(
                 (static_cast<size_t>(a) << 8) |
                 (static_cast<size_t>(b)) << 4) |
          (static_cast<size_t>(epilogue));
    }
  } supported_vec_size;

  //! Whether to rotate the ldmatrix out of the main loop
  bool rotate_ldmatrix_out_of_main_loop = true;

  //! (Ampere+) Use cp.async to load operands.
  bool async_gmem_load_operands = false;

  //! Specifies the tiling hierarchy on block,
  //!  warp, and instruction levels.
  MatMulTileOptions tile_sizes = {};

  //! Specify the type of MMA op to be used in generated kernel.
  MmaMacro mma_macro = MmaMacro::NoMMA;

  //! Specify CTA rastrization order.
  TileRasterizationOrder cta_order = TileRasterizationOrder::RowMajor;

  //! Specify which tensor we circular buffer.
  CircularBufferOptions circular_buffer_options = {};

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

  //! Promote reuse of prologue shared memory
  bool promote_prologue_smem_reuse = false;

  //! Whether to do single-kernel split-K. If this is >1, we will rfactor the K
  //! axis and perform a grid reduction before the epilogue.
  int splitk_factor = 1;

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Matmul Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << "\n"
       << "MMA macro: " << nvfuser::toString(mma_macro) << "\n"
       << circular_buffer_options.toString() << "\n"
       << supported_vec_size.toString() << "\n"
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
       << "Promote re-use of prologue shared memory: "
       << promote_prologue_smem_reuse << "\n"
       << "Split-K factor: " << splitk_factor << "\n"
       << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    // combine boolean flags for hashing
    size_t attr_hash = (static_cast<size_t>(promote_prologue_smem_reuse) << 3) |
        (static_cast<size_t>(use_smem_epilogue) << 2) |
        (static_cast<size_t>(rotate_ldmatrix_out_of_main_loop) << 1) |
        (static_cast<size_t>(async_gmem_load_operands));

    // combined hash
    attr_hash = std::hash<size_t>{}(attr_hash) ^
        (nvfuser::hash(mma_macro) << 1) ^
        (circular_buffer_options.hash() << 2) ^
        (nvfuser::hash(tile_sizes) << 3) ^
        (std::hash<size_t>{}(static_cast<size_t>(cta_order)) << 4) ^
        (std::hash<size_t>{}(grid_swizzle_factor) << 5) ^
        (std::hash<size_t>{}(splitk_factor) << 6);
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
        other_casted->circular_buffer_options == circular_buffer_options &&
        other_casted->supported_vec_size == supported_vec_size &&
        other_casted->cta_order == cta_order &&
        other_casted->grid_swizzle_factor == grid_swizzle_factor &&
        other_casted->use_smem_epilogue == use_smem_epilogue &&
        other_casted->promote_prologue_smem_reuse ==
        promote_prologue_smem_reuse &&
        other_casted->splitk_factor == splitk_factor;
  }

  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<MatmulParams>(*this);
  }
};

} // namespace nvfuser
