// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class Fusion;

struct PaddedParallelDimensions {
  bool is_tidx_padded = false;
  bool is_tidx_single_warp = false;
  bool has_warp_reduction = false;
};

// Goes through the parallelized iterdomains of the used TVs and find
// the parallel dimensions that need to be padded to a multiples of
// warp size.
PaddedParallelDimensions collectPaddedParallelDims(Fusion* fusion);

} // namespace nvfuser
