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

struct KernelExecutorEntry;

// If not sharded unsharded_logical_sizes is empty.
// If no allocation domain is found, allocation_sizes and allocation_strides
// are empty.
// For intermediate tensors, logical_sizes and logical_strides are used only,
// the rest are empty.
struct TensorShapeInfo {
  std::vector<int64_t> logical_sizes;
  std::vector<int64_t> logical_strides;
  std::vector<int64_t> unsharded_logical_sizes;
  std::vector<int64_t> allocation_sizes;
  std::vector<int64_t> allocation_strides;
};

struct GlobalBufferInfo {
  TensorView* tv = nullptr;
  TensorShapeInfo shape_info;
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

bool shouldFillAllocationWithNan();

NVF_API void setFillAllocationWithNan(bool value);

void fillTensorWithNan(at::Tensor& t);

// Infer the sizes and strides of an output tensor
std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfOutput(
    TensorView* tv,
    const ExpressionEvaluator& expr_eval);

// Infer the sizes and strides of an output tensor
TensorShapeInfo inferTensorShapes(
    TensorView* tv,
    const ExpressionEvaluator& expr_eval);

// Allocate output tensors for a given fusion. Outputs may alias inputs, in
// that case output tensors are shallow copies of the aliased inputs.
//
// If dynamic_evaluate is true, then any argument with AllocationType::Evaluate
// will not be populated, it will be filled with std::monostate.
KernelArgumentHolder allocateOutputs(
    const Fusion* fusion,
    const std::vector<GlobalBufferInfo>& output_infos,
    const std::vector<int>& output_alias_to_input_map,
    const c10::Device& device,
    const KernelArgumentHolder& args,
    bool dynamic_evaluate = false);

//! Return information necessary for allocating output tensors. Input
//! and output tensors are allowed to alias each other, which is
//! specified by the list of int pairs of input and output indices
std::vector<GlobalBufferInfo> getBufferInfos(
    ExpressionEvaluator& expr_eval,
    DataType index_dtype,
    const std::vector<Val*>& fusion_outputs);

} // namespace nvfuser
