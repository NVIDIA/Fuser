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
#include <scheduler/tools/abstract_tensor.h>
#include <type.h>

#include <functional>

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

// Reshape by manually specify domain transformation
NVF_API TensorView* reshape(
    TensorView* x,
    std::function<void(AbstractTensor&)> transform);

NVF_API TensorView* flatten(
    TensorView* x,
    int64_t start_dim = 0,
    int64_t end_dim = -1);

//! Squeeze the selected dimensions.
//!
//! NOTE: This function throws an error when encountering an unsqueezable
//! dimension. This behavior differs from PyTorch.
NVF_API TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& dims,
    bool squeeze_expanded = false);

TensorView* squeeze(TensorView* x, std::initializer_list<int64_t> dims);

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
//!
//! * manual_padding is a flag to skip the pad operation in the cat composite
//! operation.
NVF_API TensorView* cat(
    const std::vector<TensorView*>& inputs,
    int64_t dim,
    std::optional<IterType> iter_type_opt = std::nullopt,
    bool manual_padding = false);

//! Return a tensor where each dimension is sliced as specified by the
//! ranges parameter. Stepping must be one at this moment. The semantics of
//! slicing with negative values and values >= extent follow those of numpy and
//! PyTorch.
//!
//!  * manual_normalization is a flag to skip using the normalize_slice_range
//! lambda to normalize the ranges arguments for each tensor dimension.
NVF_API TensorView* slice(
    TensorView* inp,
    const std::vector<Slice>& ranges,
    bool manual_normalization = false);

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

// Splits `in`'s dimension `dim` into `chunks` chunks. All but the last chunk
// will be of size `ceil(dim_size/chunks)`. Unlike `torch.chunk` which returns
// only positive-size chunks and therefore may return fewer than `chunks` of
// them, this function returns exactly `chunks` chunks and a chunk of negative
// size will lead to a concretization error. This difference is because that we
// can't precompute the number of positive-size chunks when the dimension size
// is symbolic.
std::vector<TensorView*> chunk(TensorView* in, int64_t chunks, int64_t dim);

// Broadcasts inp based on bool vector. Size of broadcast bool vector should be
// the number of dims desired in the broadcasted tensor. This vector should be
// true if output dim should be a broadcasted dim, and false if it is not a
// broadcasted dim. Number of false entires must match the number of input dims.
NVF_API TensorView* broadcast(
    TensorView* inp,
    const std::vector<bool>& is_broadcast_dim);

// Expands input based on provided sizes. expand_sizes should be larger than
// the input's root domain (really rfactor) and will broadcast on inner
// dimensions. expand_sizes should be -1 for any dimension that should remain a
// symbolic size. For dimensions that remain broadcast after the expand should
// be set to 1, any dimension being expanded must be marked as a broadcast in
// the input and will be expanded to the provided constant size. Any dimension
// that's symbolic in the input but specified as a non -1 value will be set to
// that constant value.
NVF_API TensorView* expand(
    TensorView* inp,
    const std::vector<Val*>& expanded_sizes);

// Expands input based on other. For dimensions in inp that are broadcast with a
// matching entry in other that's either a broadcast with expanded extent or a
// non broadcasted iter domain, inp will be expanded to other's size.
NVF_API TensorView* expand_as(TensorView* inp, TensorView* other);

// Repeat each dimension for a given time. The repeat_times parameter
// must have the same number of elements as the dimensionality of the
// input tensor (excluding reduction IDs).
NVF_API TensorView* repeat(
    TensorView* inp,
    const std::vector<int64_t>& repeat_times);

} // namespace nvfuser
