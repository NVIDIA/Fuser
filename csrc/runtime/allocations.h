// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>

#include <evaluator_common.h>

#include <c10/core/ScalarType.h>

namespace nvfuser {

struct GlobalBufferInfo {
  TensorView* tv = nullptr;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  at::ScalarType type = at::ScalarType::Undefined;
  bool zero_init = false;
  bool resets_to_zero = false;
  bool is_profile_buffer = false;
};

//! This function is useful for parallel compilation of segmented fusions.
//! It returns non-allocated KernelArgumentHolder, representing the output
//! sizes from kernel execution.
//! Notes: 1. This API should ignore aliased outputs instead of
//! pushing scalar int 0 as a place-holder.
//! 2. This API does not allocate output in memory, but only returns the
//! inferred output sizes. Used in runtime/fusion_executor_cache.cpp.
KernelArgumentHolder inferOutputSizes(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    PrecomputedValues* evaluator_precomputed_values = nullptr);

int64_t computeSharedMemory(
    ExpressionEvaluator& expr_eval,
    const std::vector<const kir::Allocate*>& buffers,
    DataType index_type,
    int64_t smem_offset = 0);

// Infer the shape of an intemediate tensor using kir::Allocate. This
// is not ideal but still necessary when tensors are expanded with halo
std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfIntermediate(
    const TensorView* tv,
    const kir::Allocate* alloc,
    ExpressionEvaluator& expr_eval);

bool shouldFillAllocationWithNan();

NVF_API void setFillAllocationWithNan(bool value);

void fillTensorWithNan(at::Tensor& t);

// Infer the sizes and strides of an output tensor
std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfOutput(
    TensorView* tv,
    ExpressionEvaluator& expr_eval);

// Allocate output tensors for a given fusion. Outputs may alias inputs, in
// that case output tensors are shallow copies of the aliased inputs
std::vector<at::Tensor> allocateOutputs(
    const Fusion* fusion,
    const std::vector<GlobalBufferInfo>& output_info,
    const c10::Device& device,
    ExpressionEvaluator& ee);

//! Return information necessary for allocating output tensors. Input
//! and output tensors are allowed to alias each other, which is
//! specified by the list of int pairs of input and output indices
std::vector<GlobalBufferInfo> getBufferInfos(
    ExpressionEvaluator& expr_eval,
    DataType index_dtype,
    const std::vector<Val*>& fusion_outputs);

// Start from a tensor whose dimensions are consistent with the allocation
// domain of tv, apply a sequence of view/permute to the tensor to transform it
// into a format whose dimensions are consistent with the logical domain of tv.
// For example, if the logical domain is [I1, I2], and the allocation domain is
// [I2*I1], then we will allocate as [I2*I1], then do a tensor.view(I2, I1).t()
// to get a tensor whose semantics is [I1, I2] but its memory is [I2*I1].
// Another example, if the logical domain is [I1*I2] and the allocation domain
// is [I1, I2], then we will allocate as [I1, I2] and do a tensor.view(I1*I2) to
// get a tensor whose semantics is [I1*I2] but memory is [I1,I2]
at::Tensor transformFromAllocationToLogical(
    at::Tensor tensor,
    TensorView* tv,
    const ExpressionEvaluator& ee);

at::Tensor transformFromLogicalToAllocation(at::Tensor tensor, TensorView* tv);

} // namespace nvfuser
