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

//
// The operations defined in this header is intended as user facing functions.
// The user will provide the necessary input TensorViews and the function will
// create the correct intermediate nodes and return the output TensorViews.
//

namespace nvfuser {

NVF_API Val* set(Val*);
NVF_API TensorView* set(TensorView*);

// segment_set hints segmenter to break kernel
NVF_API Val* segment_set(Val*);
NVF_API TensorView* segment_set(TensorView*);

NVF_API TensorView* view(TensorView* x, DataType dtype);

NVF_API TensorView* reshape(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

//! Dynamic version of reshape. The number of dimensions is statically
//! fixed as the length of the new_sizes vector, but the size Vals can be
//! symbolic, which are then concretized at run time with actual
//! fusion inputs.
NVF_API TensorView* reshape(TensorView* x, const std::vector<Val*>& new_sizes);

NVF_API TensorView* flatten(
    TensorView* x,
    int64_t start_dim = 0,
    int64_t end_dim = -1);

//! Squeeze the selected dimensions.
//!
//! NOTE: This function throws an error when encountering an unsqueezable
//! dimension. This behavior differs from PyTorch.
NVF_API TensorView* squeeze(TensorView* x, const std::vector<int64_t>& dims);

//! Squeeze the dimensions corresponding to "true" in to_squeeze, i.e. remove
//! those broadcasted dimensions.
//!
//! NOTE: This function throws an error when encountering an unsqueezable
//! dimension. This behavior differs from PyTorch.
//!
//! If squeeze_expanded is true, then expanded Broadcasts will be removed just
//! as if they were not expanded. If squeeze_expanded is false, then it is an
//! error for an expanded broadcast to have a corresponding "true" value in
//! to_squeeze.
NVF_API TensorView* squeeze(
    TensorView* x,
    const std::vector<bool>& to_squeeze,
    bool squeeze_expanded = false);

NVF_API TensorView* unsqueeze(TensorView* x, int64_t dim);

//! Permute a tensor as specified by axis mappings.
//!
//! The transposition mapping is specified with a list of pairs from
//! new to old positions. Positions are relative to the noReduction
//! domain.
//!
//! \param x Tensor to transpose
//! \param new2old vector mapping from new to old positions.
NVF_API TensorView* permute(TensorView* x, const std::vector<int64_t>& new2old);
NVF_API TensorView* permute(
    TensorView* x,
    const std::initializer_list<int64_t>& new2old);

//! Same as above, but with the TensorView::reorder-like API.
NVF_API TensorView* permute(
    TensorView* x,
    const std::unordered_map<int64_t, int64_t>& old2new);
NVF_API TensorView* permute(
    TensorView* x,
    const std::initializer_list<std::pair<const int64_t, int64_t>>& old2new);

//! Transpose a tensor by swapping the two dimensions.
NVF_API TensorView* transpose(TensorView* x, int64_t dim0, int64_t dim1);

//! Transpose a 2D tensor.
NVF_API TensorView* transpose(TensorView* x);

//! Pad a tensor by given widths by specified value. Similar to torch.pad, the
//! pad_widths vector specifies the padding widths of the innermost N
//! dimensions, where N is half the size of the width vector. If value is
//! omitted, a default value of zero is assumed. The provied value will be cast
//! to the dtype of the argument x.
//! TODO: Support other padding types
NVF_API TensorView* pad(
    TensorView* x,
    const std::vector<Val*>& pad_widths,
    Val* value = nullptr,
    std::optional<IterType> iter_type_opt = std::nullopt);

//! Concatenate tensors in the given dimension
NVF_API TensorView* cat(
    const std::vector<TensorView*>& inputs,
    int64_t dim,
    std::optional<IterType> iter_type_opt = std::nullopt);

//! Return a tensor where each dimension is sliced as specified by the
//! ranges parameter. Stepping must be one at this moment. The semantics of
//! slicing with negative values and values >= extent follow those of numpy and
//! PyTorch.
NVF_API TensorView* slice(TensorView* inp, const std::vector<Slice>& ranges);

//! A variant of the above `slice` function. This is closer to the Python API.
NVF_API TensorView* slice(
    TensorView* inp,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& stops,
    const std::vector<int64_t>& steps);

//! Same as above except that `steps` are all 1.
NVF_API TensorView* slice(
    TensorView* inp,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& stops);

} // namespace nvfuser
