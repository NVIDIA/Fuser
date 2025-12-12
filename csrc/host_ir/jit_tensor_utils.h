// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>
#include <unordered_map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

#include <ir/all_nodes.h>
#include <type.h>

namespace nvfuser {

// Creates an LLVM struct type that matches runtime::Tensor<T, Dims, AllocDims>
// from runtime/tensor.cu
llvm::StructType* createRuntimeTensorType(
    int64_t num_dims,
    PrimDataType index_type,
    llvm::LLVMContext& context);

// Helper function to generate LLVM IR that extracts tensor size for a given
// dimension
llvm::Value* createTensorSize(
    llvm::Value* tensor,
    int64_t dim,
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>&
        builder);

// Infers tensor shapes and strides by propagating allocation domain to logical
// domain
void inferTensorShapesAndStrides(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>&
        builder,
    llvm::SmallVectorImpl<llvm::Value*>& sizes,
    llvm::SmallVectorImpl<llvm::Value*>& strides);

// Packs a tensor argument into the runtime::Tensor format expected by CUDA
// kernels
llvm::Value* packTensorArgument(
    llvm::Value* tensor,
    TensorView* tv,
    PrimDataType index_type,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>&
        builder);

} // namespace nvfuser
