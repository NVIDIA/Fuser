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
