// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>

#include <fusion.h>
#include <mma_type.h>

namespace nvfuser {

//! Starting point for a matmul scheduler parameters:
class MatmulParam {
 public:
  MatmulParam(MmaBuilder builder) : mma_builder(builder) {}

  struct DoubleBufferOptions {
    bool double_buffer_smem_write = false;
    bool double_buffer_smem_read = false;
    int smem_double_buffer_stage = 2;
  };

  //! Whether to rotate the ldmatrix out of the main loop
  bool rotate_ldmatrix_out_of_main_loop = true;

  //! (Ampere+) Use cp.async to load operands.
  bool async_gmem_load_operands = false;

  //! Specifies the tiling hierarchy on block,
  //!  warp, and instruction levels.
  MatMulTileOptions tile_sizes;

  //! Parameters for configuring mma ops.
  MmaBuilder mma_builder;

  //! Specify which tensor we double buffer.
  DoubleBufferOptions double_buffer_options;

  //! Tunable spec to enable/disable lifting
  //!  memory indexing math out of the main
  //!  loop on the generated kernel.
  //! (All defaults to on).
  //! Note: eventually this part of logic
  //!  will be merged into automatic
  //!  indexing math allocation/placement pass.
  struct IndexLiftingOptions {
    bool lift_smem_read_address = true;
    bool lift_smem_write_address = true;
    bool lift_gmem_read_address = true;
    // TODO: add gmem_write address for
    //  latency bound kernels.
  } index_lift_options;

  //! Configurable rasterization/parallelization order.
  //! Depending on the problem shape, switching blockIdx.x and blockIdx.y can
  //! help improve L2 hit rate.
  enum class TileRasterizationOrder {
    RowMajor = 0,
    ColumnMajor = 1
  } rasterization_order = TileRasterizationOrder::RowMajor;

  //! Swizzle factor is used to increase L2 hit rate.
  //! It horizontally squeezes the grid so that gridDim.x is larger and
  //! gridDim.y is smaller.
  //! We rely on the observation that the CTAs are scheduled by the GPU by
  //! iterating on gridDim.x first. As a result, as blocks are launched, they
  //! will more likely be forming sub-tiles of the C matrix. This will increase
  //! L2 hit rate/data reuse of A and B.
  //!
  //! Eg for swizzle_factor=2:
  //!    A1 A2 B1 B2 -->   A1 A2 A3 A4 B1 B2 B3 B4
  //!    A3 A4 B3 B4       C1 C2 C3 C4 D1 D2 D3 D4
  //!    C1 C2 D1 D2
  //!    C3 C4 D3 D4
  int swizzle_factor = 1;

  //! Enables predicate peeling mainloop:
  bool peel_main_loop = true;
};

//! Prototype auto scheduling function.
//!  Currently only support a pure matmul with no
//!   fused prolog or epilog.
//!
//! TODO:
//!   - will support a range of fusions in a follow up
//!   - will formalize scheduling decisions into
//! matmul params data structure.
TORCH_CUDA_CU_API void scheduleMatmul(
    TensorView* c_tv,
    TensorView* a_tv,
    TensorView* b_tv,
    MatmulParam& params);

} // namespace nvfuser
