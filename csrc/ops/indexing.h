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

NVF_API TensorView* select(TensorView* tv, int64_t dim, Val* index);

// torch.index_select
NVF_API TensorView* indexSelect(
    TensorView* input,
    int64_t dim,
    TensorView* index);

// This is a restricted version of torch.index_put(..., accumulate=true)
NVF_API TensorView* indexPutAccumulate(
    TensorView* acc_tv,
    TensorView* index_tv,
    TensorView* value_tv);

// torch.gather
NVF_API TensorView* gather(TensorView* input, int64_t dim, TensorView* index);

// Provides torch.scatter. It is designed to represent the ouf-of-place
// scatter operation, i.e., the returned tensor, out_tv, is defined as
// follows:
//
// out_tv = self.clone();
// for (auto i: enumerate(index.size()) {
//   out_tv[index[i]] = src[i]
// }
//
// Thus, in principle, it should be legal to use the self tensor with a
// different operation, and that should still use the original self
// tensor. However, it is currently only supported when it is the
// only use of the self tensor since this operation is internally
// implemented as an in-place operation.
//
// TODO: Allow the self tensor to be used by other ops. We might want
// to consider adding a new preseg pass to insert a copy before a
// scatter. For example, given a fusion as below:
//
//   t3 = scatter(t0, 0, t1, t2);
//   t4 = add(t0, 1);
//
// In this case, since we would need to make sure t3 could alias t0,
// we would insert a copy as shown below:
//
//   t5 = t0.clone();
//   t3 = scatter(t5, 0, t1, t2);
//   t4 = add(t0, 1);
NVF_API TensorView* scatter(
    TensorView* self,
    int64_t dim,
    TensorView* index,
    Val* src,
    std::optional<BinaryOpType> accumulate_op = std::nullopt);

//! numpy.take_along_axis
//! (https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
//! Note the order of the parameters follows the numpy order, which is
//! different from torchGather.
NVF_API TensorView* takeAlongAxis(
    TensorView* input,
    TensorView* index,
    int64_t dim);

// Utility function that constructs the padded allocation domain for
// PreprocessGroupedMatmulInputSf output. Expected inputs are the logical domain
// of the output tensor. It applies maximum padding on each group and returns
// allocation domain with the required padding based on the layout.
std::vector<IterDomain*> layoutAllocationDomain(
    std::vector<IterDomain*> logical_dom,
    Val* num_groups,
    BlockScalingFactorLayout layout);

//! Changes the layout of input to satisfy the requirement of grouped matmul on
//! block scaling factor. see:
//! https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
NVF_API TensorView* preprocessGroupedMatmulInputSf(
    TensorView* input,
    TensorView* input_offsets,
    TensorView* output_offsets,
    BlockScalingFactorLayout layout);

} // namespace nvfuser
