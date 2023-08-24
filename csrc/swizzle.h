// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <ir/interface_nodes.h>
#include <type.h>

#include <utility>

//
// The operations defined in this header is intended as user facing functions.
// The user will provide the necessary input Vals and the function will
// create the correct intermediate nodes and return the output Vals.
//

namespace nvfuser {

TORCH_CUDA_CU_API std::pair<Val*, Val*> dispatchSwizzle(
    Swizzle2DType type,
    Val* x,
    Val* y,
    Val* maybe_size_x,
    Val* maybe_size_y);

TORCH_CUDA_CU_API std::pair<Val*, Val*> dispatchUnSwizzle(
    Swizzle2DType type,
    Val* x,
    Val* y,
    Val* maybe_size_x,
    Val* maybe_size_y);

} // namespace nvfuser
