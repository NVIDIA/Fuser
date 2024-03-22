// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/ATen.h>

namespace nvfuser {

//! This returns a slice of a thread local at::Tensor that is zeroed. Uses of
//! this memory should always "clean up" by resetting the memory to zero at the
//! end of the kernel.
at::Tensor contigZeroTensor(
    const std::vector<int64_t>& sizes,
    const c10::ScalarType& aten_dtype,
    const c10::Device& device);

//! This should be called after each kernel launch to allow subsequent launches
//! to re-use allocated memory.
void releaseZeroedMemory();

} // namespace nvfuser
