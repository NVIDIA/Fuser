// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace broadcast {
// Broadcasts within partitioned groups of threads.
//
// X_THREAD: Broadcast from threadIdx.x == 0 if true
// Y_THREAD: Broadcast from threadIdx.y == 0 if true
// Z_THREAD: Broadcast from threadIdx.z == 0 if true
// Aligned: Called from aligned threads if true
// inp_val: Per-thread source value. Only valid when the thread is a source.
// out: Per-thread output location
//
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, bool Aligned, typename T>
__device__ void blockBroadcast(
    T& out,
    const T& inp_val,
    T* shared_mem,
    bool read_write_pred) {
  const bool has_valid_data = (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) && (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  if (has_valid_data && read_write_pred) {
    shared_mem[shared_offset] = inp_val;
  }

  block_sync::sync<Aligned>();

  if (read_write_pred) {
    out = shared_mem[shared_offset];
  }

  block_sync::sync<Aligned>();
}

} // namespace broadcast
