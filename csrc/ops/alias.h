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

//
// The operations defined in this header is intended as user facing functions.
// The user will provide the necessary input TensorViews and the function will
// create the correct intermediate nodes and return the output TensorViews.
//

namespace nvfuser {

TORCH_CUDA_CU_API Val* set(Val*);
TORCH_CUDA_CU_API TensorView* set(TensorView*);

// segment_set hints segmenter to break kernel
TORCH_CUDA_CU_API Val* segment_set(Val*);
TORCH_CUDA_CU_API TensorView* segment_set(TensorView*);

TORCH_CUDA_CU_API TensorView* view(TensorView* x, DataType dtype);

TORCH_CUDA_CU_API TensorView* reshape(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

//! Dynamic version of reshape. The number of dimensions is statically
//! fixed as the length of the new_sizes vector, but the size Vals can be
//! symbolic, which are then concretized at run time with actual
//! fusion inputs.
TORCH_CUDA_CU_API TensorView* reshape(
    TensorView* x,
    const std::vector<Val*>& new_sizes);

TORCH_CUDA_CU_API TensorView* flatten(
    TensorView* x,
    int64_t start_dim = 0,
    int64_t end_dim = -1);

TORCH_CUDA_CU_API TensorView* squeeze(
    TensorView* x,
    const std::vector<bool>& to_squeeze);

TORCH_CUDA_CU_API TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& sizes);

TORCH_CUDA_CU_API TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& sizes,
    int dim);

TORCH_CUDA_CU_API TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dims);

TORCH_CUDA_CU_API TensorView* unsqueeze(TensorView* x, int dim);

//! Permute a tensor as specified by axis mappings.
//!
//! The transposition mapping is specified with a list of pairs from
//! new to old positions. Positions are relative to the noReduction
//! domain.
//!
//! \param inp Tensor to transpose
//! \param new2old vector mapping from new to old positions.
TORCH_CUDA_CU_API TensorView* permute(
    TensorView* x,
    const std::vector<int64_t>& new2old);

//! Transpose a tensor by swapping the two dimensions.
TORCH_CUDA_CU_API TensorView* transpose(
    TensorView* x,
    int64_t dim0,
    int64_t dim1);

//! Transpose a 2D tensor.
TORCH_CUDA_CU_API TensorView* transpose(TensorView* x);

//! Pad a tensor by given widths by specified value. Similar to torch.pad, the
//! pad_widths vector specifies the padding widths of the innermost N
//! dimensions, where N is half the size of the width vector. If value is
//! omitted, a default value of zero is assumed. The provied value will be cast
//! to the dtype of the argument x.
//! TODO: Support other padding types
TORCH_CUDA_CU_API TensorView* pad(
    TensorView* x,
    const std::vector<Val*>& pad_widths,
    Val* value = nullptr,
    std::optional<IterType> iter_type_opt = std::nullopt);

//! Concatenate tensors in the given dimension
TORCH_CUDA_CU_API TensorView* cat(
    const std::vector<TensorView*>& inputs,
    int64_t dim,
    std::optional<IterType> iter_type_opt = std::nullopt);

//! Return a tensor where each dimension is sliced as specified by the
//! ranges parameter. Stepping must be one at this moment.
TORCH_CUDA_CU_API TensorView* slice(
    TensorView* inp,
    const std::vector<Slice>& ranges);

} // namespace nvfuser
