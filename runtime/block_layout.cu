// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf::block_layout {

namespace {

// TODO: simplify this maybe?!
template <int BLOCK_ROW_OUTER, int BLOCK_ROW_INNER, int BLOCK_COL>
__device__ nvfuser_index_t offsetAfterSwizzlePadding(
    const nvfuser_index_t row_idx,
    const nvfuser_index_t col_idx,
    const nvfuser_index_t col_size) {
  constexpr nvfuser_index_t BLOCK_ROW_SIZE = BLOCK_ROW_OUTER * BLOCK_ROW_INNER;

  /* logical dimension of matrix [ row_size, col_size]
   *
   * while layout is decomposed as
   *   [ (row_tile*BLOCK_ROW_INNER*BLOCK_ROW_OUTER), (col_tile*BLOCK_COL) ]
   * where
   *   row_tile = ceilDiv(row_size , BLOCK_ROW_OUTER * BLOCK_ROW_INNER)
   *   col_tile = ceilDiv(col_size , BLOCK_COL)
   */
  nvfuser_index_t row_tile_idx = ceilDiv(row_idx, BLOCK_ROW_SIZE);

  nvfuser_index_t row_block_idx = row_idx % BLOCK_ROW_SIZE;
  nvfuser_index_t row_block_inner_idx = row_block_idx / BLOCK_ROW_OUTER;
  nvfuser_index_t row_block_outer_idx = row_block_idx % BLOCK_ROW_OUTER;
  nvfuser_index_t col_tile_idx = ceilDiv(col_idx, BLOCK_COL);
  nvfuser_index_t col_block_idx = col_idx % BLOCK_COL;

  /* layout for matrix [ row_size, col_size]
   * it is indexed as a contiguous buffer with shape
   *   [row_tile, col_tile, BLOCK_ROW_OUTER, BLOCK_ROW_INNER, BLOCK_COL]
   */
  nvfuser_index_t row_tile_stride = (col_size + BLOCK_COL - 1) / BLOCK_COL;
  constexpr nvfuser_index_t COL_TILE_STRIDE = BLOCK_ROW_SIZE * BLOCK_COL;
  constexpr nvfuser_index_t BLOCK_ROW_OUTER_STRIDE =
      BLOCK_ROW_INNER * BLOCK_COL;

  return row_tile_idx * row_tile_stride + col_tile_idx * COL_TILE_STRIDE +
      row_block_outer_idx * BLOCK_ROW_OUTER_STRIDE +
      row_block_inner_idx * BLOCK_COL + col_block_idx;
}

} // namespace

// TODO: I think we can actually not have this handled as an opaque function.
template <
    typename T,
    typename Index_T,
    int BLOCK_ROW_OUTER,
    int BLOCK_ROW_INNER,
    int BLOCK_COL,
    int UNROLL_FACTOR>
__device__ void groupedBlockLayout(
    T* output,
    const T* input,
    const nvfuser_index_t row_idx,
    const nvfuser_index_t col_idx,
    const Index_T* expert_offsets,
    const Index_T* output_offsets,
    const nvfuser_index_t row_size,
    const nvfuser_index_t col_size,
    const nvfuser_index_t group_size) {
  // Compute flattened thread ID within the block
  const int tid = threadIdx.z * blockDim.x * blockDim.y +
      threadIdx.y * blockDim.x + threadIdx.x;

  // Compute flattened block ID within the grid
  const int bid =
      blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

  // Compute threadblock size (total threads per block)
  const int block_size = blockDim.x * blockDim.y * blockDim.z;

  // Compute grid size (total number of blocks)
  const int grid_size = gridDim.x * gridDim.y * gridDim.z;

  int expert_id = 0;
  // find corresponding expert_id
  for (int i = expert_id; i < group_size; ++i) {
    if (row_idx < expert_offsets[i + 1]) {
      expert_id = i;
      break;
    }
  }

  // row idx for current matmul
  nvfuser_index_t c_row_idx = row_idx - expert_offsets[expert_id];
  T* out_offset = output + output_offsets[expert_id];
  // TODO: vectorized load/store; The logic could be simplified afterwards.
  for (int i = 0; i < UNROLL_FACTOR && col_idx + i < col_size; ++i) {
    nvfuser_index_t index =
        offsetAfterSwizzlePadding<BLOCK_ROW_OUTER, BLOCK_ROW_INNER, BLOCK_COL>(
            c_row_idx, col_idx, col_size);
    out_offset[index] = input[i];
  }
}

} // namespace nvf::block_layout
