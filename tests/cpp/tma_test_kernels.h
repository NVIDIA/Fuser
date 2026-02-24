// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#pragma once

#include <cuda_runtime.h>

namespace nvfuser {

//! Copies num_bytes from src (GMEM) to dst (GMEM) via TMA 1D bulk copy:
//!   GMEM(src) -> SMEM -> GMEM(dst)
//! Uses cp.async.bulk with mbarrier synchronization (SM90+ / Hopper).
//! num_bytes must be a multiple of 16 and > 0.
void launchTmaCopy1D(
    void* dst,
    const void* src,
    int num_bytes,
    cudaStream_t stream);

} // namespace nvfuser
