// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf::block_layout {

namespace {

// TODO: support vectorized store
template <int BLOCK_ROW_OUTER, int BLOCK_ROW_INNER, int BLOCK_COL>
__device__ nvfuser_index_t offsetAfterSwizzlePadding(
    const nvfuser_index_t row_idx,
    const nvfuser_index_t col_idx,
    const nvfuser_index_t padded_col_size) {
  constexpr nvfuser_index_t BLOCK_ROW_SIZE = BLOCK_ROW_OUTER * BLOCK_ROW_INNER;

  /* logical dimension of matrix [ row_size, col_size]
   *
   * while logical domain after padding can be viewed as
   *   [ (row_tile*BLOCK_ROW_INNER*BLOCK_ROW_OUTER), (col_tile*BLOCK_COL) ]
   * where
   *   row_tile = ceilDiv(row_size / BLOCK_ROW_OUTER * BLOCK_ROW_INNER)
   *   col_tile = ceilDiv(col_size / BLOCK_COL)
   */

  // we first convert `row_idx` and `col_idx` to the logical index on the 5d tensor.
  nvfuser_index_t row_tile_idx = row_idx / BLOCK_ROW_SIZE;
  nvfuser_index_t row_block_idx = row_idx % BLOCK_ROW_SIZE;
  nvfuser_index_t row_block_inner_idx = row_block_idx / BLOCK_ROW_OUTER;
  nvfuser_index_t row_block_outer_idx = row_block_idx % BLOCK_ROW_OUTER;
  nvfuser_index_t col_tile_idx = col_idx / BLOCK_COL;
  nvfuser_index_t col_block_idx = col_idx % BLOCK_COL;

  /* layout for matrix [ row_size, col_size]
   * it is viewed
   *   [row_tile, BLOCK_ROW_INNER, BLOCK_ROW_OUTER, col_tile, BLOCK_COL]
   * then transposed with axis (1, 3)
   *   [row_tile, col_tile, BLOCK_ROW_OUTER, BLOCK_ROW_INNER, BLOCK_COL]
   * and then made contiguous
   * So we can compute the corresponding stride for each dimension
   */
  constexpr nvfuser_index_t COL_TILE_STRIDE = BLOCK_ROW_SIZE * BLOCK_COL;
  constexpr nvfuser_index_t BLOCK_ROW_OUTER_STRIDE =
      BLOCK_ROW_INNER * BLOCK_COL;
  constexpr nvfuser_index_t BLOCK_ROW_INNER_STRIDE = BLOCK_COL;

  return row_tile_idx * padded_col_size * BLOCK_ROW_SIZE +
      col_tile_idx * COL_TILE_STRIDE +
      row_block_outer_idx * BLOCK_ROW_OUTER_STRIDE +
      row_block_inner_idx * BLOCK_ROW_INNER_STRIDE + col_block_idx;
}

} // namespace

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
    const nvfuser_index_t col_size,
    const nvfuser_index_t group_size) {
  // find corresponding expert_id
  int expert_id = 0;
  for (int i = 0; i < group_size; ++i) {
    if (row_idx < expert_offsets[i + 1]) {
      expert_id = i;
      break;
    }
  }

  // row idx for current group
  nvfuser_index_t c_row_idx = row_idx - expert_offsets[expert_id];
  // compute output group offset for current group
  nvfuser_index_t padded_col_size =
      (col_size + BLOCK_COL - 1) / BLOCK_COL * BLOCK_COL;
  T* out_group_offset = output + output_offsets[expert_id] * padded_col_size;

  // TODO: vectorized load/store instead of for loop
  for (int i = 0; i < UNROLL_FACTOR && col_idx + i < col_size; ++i) {
    nvfuser_index_t index =
        offsetAfterSwizzlePadding<BLOCK_ROW_OUTER, BLOCK_ROW_INNER, BLOCK_COL>(
            c_row_idx, col_idx + i, padded_col_size);
    out_group_offset[index] = input[i];
  }
}

} // namespace nvf::block_layout
