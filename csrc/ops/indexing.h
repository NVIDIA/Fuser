// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <ir/interface_nodes.h>
#include <type.h>

namespace nvfuser {

TensorView* select(TensorView* tv, int dim, Val* index);

// index_select
TensorView* index_select(TensorView* input, int dim, TensorView* index);

// torch.gather
TensorView* torch_gather(TensorView* input, int dim, TensorView* index);

// torch.scatter
TensorView* scatterOp(
    ScatterOpType type,
    TensorView* self,
    int dim,
    TensorView* index,
    TensorView* src);

TensorView* scatter(
    TensorView* self,
    int dim,
    TensorView* index,
    TensorView* src);

//! numpy.take_along_axis
//! (https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
//! Note the order of the parameters follows the numpy order, which is
//! different from torch_gather.
TensorView* take_along_axis(TensorView* input, TensorView* index, int64_t dim);

} // namespace nvfuser
