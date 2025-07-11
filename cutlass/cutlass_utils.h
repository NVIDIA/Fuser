// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser::cutlass_kernels {

// Get the SM (Streaming Multiprocessor) version of the current GPU
//
// Returns: The SM version as an integer (e.g., 80 for SM 8.0, 75 for SM 7.5)
int getSMVersion();

// Get the number of multiprocessors on the current GPU
//
// Returns: The number of multiprocessors on the GPU
int getMultiProcessorCount();

} // namespace nvfuser::cutlass_kernels
