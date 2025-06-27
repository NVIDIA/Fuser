// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <runtime/executor_kernel_arg.h>

namespace nvfuser {

// [ Note stride order and contiguity vector ]
//
// for n-d tensor. we should have stride_order and contiguity both be a size n
// vector.
//
// `stride order` vector corresponds to the order for each logical domain in
//     physical memory; For any 0 <= i < n , we know the dimension i has the
//     stride_order[i]-th smallest stride.
//     An exception to this are implicit broadcast dimensions, i.e. dimensions
//     with `stride == 0`, where we would maintain their semantical position
// `contiguity` vector to whether or not indexing could be collaped
//     corresponding to each physical domain;
//
// e.g. Given size and stride as follow:
//   sizes   = [2, 2, 2, 2]
//   strides = [8, 4, 2, 1]
// Obviously the stride order as: [3, 2, 1, 0] for row-major order, i.e. stride
// in descending order and contiguity flag will be [True, True, True, True]
//
// e.g. Given size and stride as follow:
//   sizes   = [2, 1, 3, 1, 4]
//   strides = [24, 4, 8, 4, 2]
// Note that there are a few explicit broadcast dimensions, dimensions with size
// == 1 and stride != 0. The stride for explicit broadcast dimensions
// participates in stride order computation. The reason is that, frameworks
// could assign meaningful stride to an explicit broadcast dimensions to hint
// memory format, which could be used to deduce the desired output memory
// format. We use stable sort to break tie when two dimension has equal stride,
// i.e. try to preserve their semantical order. Hence, we would compute stride
// order as: [4, 2, 3, 1, 0]. In the context of index, collapsing, how we
// resolve that shouldn't matter. With sorted sizes & strides:
//   sorted_size    = [2, 3, 1, 1, 4]
//   sorted_strides = [24, 8, 4, 4, 2]
// Here, we compute contiguity as: [True, True, None, None, False]
//
// e.g. Given size and stride as follow:
//   sizes   = [2, 2, 2, 2]
//   strides = [8, 4, 0, 2]
// The stride of implicit broadcast dimensions, dimensions with stride == 0,
// does not participate in stride order computation and preserves their
// semantical position in stride order. The logic behind this is so that we
// would not unnecessarily introduce permutated alloc_domain for a naive
// unsqueeze/expanded operation when it doesn't improve indexing. For the given
// example, computed stride_order would be: [3, 2, 1, 0] and contiguity would
// be: [True, True, None, False]
//
// This function returns a pair of <contiguity, stride_order>
std::pair<std::vector<std::optional<bool>>, std::vector<int64_t>>
computeTensorDescriptor(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides);

// Verify that the shape is valid for nvFuser.
//
// The shape must be a list of integers, where each integer is either a positive
// integer, -1, or 1.
void verifyShape(const std::vector<int64_t>& shape);

// Normalizes and validates the stride order for a tensor.
//
// This function processes the stride order vector to ensure it's valid and
// consistent:
// 1. Handles negative indices by converting them to positive indices
// 2. Validates that all indices are within the valid range
// 3. Ensures there are no duplicate entries
//
// Parameters
// ----------
// stride_order : vector<int64_t>
//     The stride order vector to normalize. Can be empty or contain indices
//     from -rank to rank-1, where rank is the size of the vector.
//
// Notes
// -----
// - Negative indices are converted to positive by adding the rank
// - All indices must be within the range [-rank, rank-1]
// - No duplicate indices are allowed
// - An empty stride_order vector is valid and will be left unchanged
void normalizeStrideOrder(std::vector<int64_t>& stride_order);

// Determines which dimensions of a tensor are expanded.
//
// A dimension is considered expanded if:
// 1. It is marked as a broadcast dimension (contiguity is None)
// 2. It has a non-broadcast size (shape != 1)
//
// Parameters
// ----------
// shape : vector<int64_t>
//     The shape of the tensor
// contiguity : vector<optional<bool>>
//     The contiguity flags for each dimension. None indicates a broadcast
//     dimension.
// stride_order : vector<int64_t>
//     The stride order of dimensions, mapping logical domain to allocation
//     domain.
//
// Returns
// -------
// vector<bool>
//     A vector of boolean flags indicating which dimensions are expanded.
//     The order of this vector corresponds to the logical domain of TensorView.
//
// Notes
// -----
// The function handles the mapping between logical domain and allocation domain
// using the stride order. For each dimension i:
// - If stride_order is empty, contiguity[i] corresponds to dimension i
// - Otherwise, contiguity[i] corresponds to dimension
// (rank - 1 - stride_order[i])
std::vector<bool> getExpanded(
    const std::vector<int64_t>& shape,
    const std::vector<std::optional<bool>>& contiguity,
    const std::vector<int64_t>& stride_order);

// If shape indicates a broadcast dimension, then the contiguity is optional.
// Otherwise, assign each dimension the given contiguity value.
std::vector<std::optional<bool>> getContiguityVec(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride_order,
    bool contiguity);

// This helper function translates python sizes into to TensorViewBuilder's view
// of the world.
std::vector<int64_t> getTensorViewBuilderSizes(
    const std::vector<int64_t>& sizes,
    bool static_sizes);

// A helper function that maps PrimDataType to std::string
NVF_API const char* dtypeToPyString(PrimDataType t);

} // namespace nvfuser
