// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace scatter {

template <typename DataT, bool Aligned>
__device__ void blockScatter(
    DataT& out,
    DataT* out_base,
    const DataT& in,
    nvfuser_index_t index,
    DataT val) {
  // Assumes out is in shared memory
  out = in;
  block_sync::sync<Aligned>(blockDim);

  // TODO: Predicate
  if (threadIdx.x < 5) {
    out_base[index] = val;
  }
  block_sync::sync<Aligned>(blockDim);
}

} // namespace scatter
