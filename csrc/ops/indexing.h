// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <ir/interface_nodes.h>
#include <type.h>

namespace nvfuser {

// When keep_reduction_axis is true, all reduction axis are kept in the
// SelectOp's consumer. This is used in the context of HostIr where SelectOp is
// used to index into Stream-parallelized axes.
NVF_API TensorView* select(
    TensorView* tv,
    int64_t dim,
    Val* index,
    bool keep_reduction_axis = false);

// torch.index_select
NVF_API TensorView* indexSelect(
    TensorView* input,
    int64_t dim,
    TensorView* index);

// torch.gather
NVF_API TensorView* gather(TensorView* input, int64_t dim, TensorView* index);

// torch.scatter
TensorView* scatterOp(
    ScatterOpType type,
    TensorView* self,
    int64_t dim,
    TensorView* index,
    TensorView* src);

NVF_API TensorView* scatter(
    TensorView* self,
    int64_t dim,
    TensorView* index,
    TensorView* src);

//! numpy.take_along_axis
//! (https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
//! Note the order of the parameters follows the numpy order, which is
//! different from torchGather.
NVF_API TensorView* takeAlongAxis(
    TensorView* input,
    TensorView* index,
    int64_t dim);

} // namespace nvfuser
