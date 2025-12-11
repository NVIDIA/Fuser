// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cstdint>
#include <memory>
#include <ranges>
#include <unordered_map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "bfs.h"
#include "expr_evaluator.h"
#include "ir/all_nodes.h"
#include "linked_hash_map.h"
#include "ops/all_ops.h"
#include "tensor_metadata.h"

namespace nvfuser {

// Forward declarations
llvm::StructType* createRuntimeTensorType(
    int64_t num_dims,
    PrimDataType index_type,
    llvm::LLVMContext& context);

void inferTensorShapesAndStrides(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder,
    llvm::SmallVectorImpl<llvm::Value*>& sizes,
    llvm::SmallVectorImpl<llvm::Value*>& strides);

llvm::Value* packTensorArgument(
    llvm::Value* tensor,
    TensorView* tv,
    PrimDataType index_type,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);

namespace {

constexpr std::string_view kTensorSizeFuncName = "tensor_size";
constexpr std::string_view kTensorStrideFuncName = "tensor_stride";
constexpr std::string_view kTensorDataPtrFuncName = "tensor_data_ptr";

// Helper function to generate LLVM IR that extracts tensor size for a given
// dimension
llvm::Value* createTensorSize(
    llvm::Value* tensor,
    int64_t dim,
    llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Function* tensor_size_func = module->getFunction(kTensorSizeFuncName);
  llvm::Value* dim_val = builder.getInt64(dim);

  return builder.CreateCall(tensor_size_func, {tensor, dim_val});
}

// Helper function to generate LLVM IR that extracts tensor stride for a given
// dimension
llvm::Value* createTensorStride(
    llvm::Value* tensor,
    int64_t dim,
    llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Function* tensor_stride_func =
      module->getFunction(kTensorStrideFuncName);
  llvm::Value* dim_val = builder.getInt64(dim);

  return builder.CreateCall(tensor_stride_func, {tensor, dim_val});
}

// Helper function to generate LLVM IR that extracts tensor data pointer
llvm::Value* createTensorDataPtr(
    llvm::Value* tensor,
    llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Function* tensor_data_ptr_func =
      module->getFunction(kTensorDataPtrFuncName);

  return builder.CreateCall(tensor_data_ptr_func, {tensor});
}

// Forward declarations for value creation functions
llvm::Value* getOrCreateValueForExtent(
    IterDomain* id,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);
llvm::Value* getOrCreateValue(
    Val* val,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);

llvm::Value* createValueForBinaryOp(
    BinaryOp* binary_op,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder) {
  auto* lhs = binary_op->lhs();
  auto* rhs = binary_op->rhs();
  llvm::Value* lhs_value = getOrCreateValue(lhs, val_to_value, builder);
  llvm::Value* rhs_value = getOrCreateValue(rhs, val_to_value, builder);
  if (binary_op->getBinaryOpType() == BinaryOpType::Add) {
    return builder.CreateAdd(lhs_value, rhs_value);
  }
  if (binary_op->getBinaryOpType() == BinaryOpType::Sub) {
    return builder.CreateSub(lhs_value, rhs_value);
  }
  if (binary_op->getBinaryOpType() == BinaryOpType::Mul) {
    return builder.CreateMul(lhs_value, rhs_value);
  }
  if (binary_op->getBinaryOpType() == BinaryOpType::CeilDiv) {
    // Implement ceilDiv as (a + b - 1) / b
    llvm::Value* numerator = builder.CreateAdd(lhs_value, rhs_value);
    llvm::Value* one = builder.getInt64(1);
    numerator = builder.CreateSub(numerator, one);
    return builder.CreateUDiv(numerator, rhs_value);
  }
  NVF_THROW(
      "LLVM Lowering Error: Unsupported binary operation type in extent "
      "calculation: ",
      binary_op->getBinaryOpType());
}

llvm::Value* createValueForUnaryOp(
    UnaryOp* unary_op,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder) {
  auto* in = unary_op->in();
  llvm::Value* in_value = getOrCreateValue(in, val_to_value, builder);
  if (unary_op->getUnaryOpType() == UnaryOpType::Cast) {
    return in_value;
  }
  if (unary_op->getUnaryOpType() == UnaryOpType::Abs) {
    llvm::Value* is_negative =
        builder.CreateICmpSLT(in_value, builder.getInt64(0));
    llvm::Value* negated = builder.CreateNeg(in_value);
    return builder.CreateSelect(is_negative, negated, in_value);
  }
  if (unary_op->getUnaryOpType() == UnaryOpType::Neg) {
    return builder.CreateNeg(in_value);
  }
  NVF_THROW(
      "LLVM Lowering Error: Unsupported unary operation type in extent "
      "calculation: ",
      unary_op->getUnaryOpType());
}

llvm::Value* createValue(
    Val* val,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder) {
  if (val->isConst()) {
    return builder.getInt64(val->value().as<int64_t>());
  }

  if (Expr* def = val->definition()) {
    if (auto* binary_op = dynamic_cast<BinaryOp*>(def)) {
      return createValueForBinaryOp(binary_op, val_to_value, builder);
    }

    if (auto* unary_op = dynamic_cast<UnaryOp*>(def)) {
      return createValueForUnaryOp(unary_op, val_to_value, builder);
    }

    NVF_THROW(
        "LLVM Lowering Error: createValueForExtent called with unsupported "
        "expression type: ",
        def->getOpString());
  }

  NVF_THROW(
      "LLVM Lowering Error: createValueForExtent called with undefined "
      "val: ",
      val->toString());
}

llvm::Value* getOrCreateValue(
    Val* val,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder) {
  if (auto it = val_to_value.find(val); it != val_to_value.end()) {
    return it->second;
  }
  val_to_value[val] = createValue(val, val_to_value, builder);
  return val_to_value[val];
}

llvm::Value* getOrCreateValueForExtent(
    IterDomain* id,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder) {
  return getOrCreateValue(id->getMaybeExpandedExtent(), val_to_value, builder);
}

} // anonymous namespace

// Creates an LLVM struct type that matches runtime::Tensor<T, Dims, AllocDims>
// from runtime/tensor.cu
//
// Memory layout:
// struct Tensor {
//   Pointer<T> data;                    // 8 bytes (field 0)
//   Array<index_t, Dims> sizes; // Dims * index_size (field 1)
//   Array<index_t, AllocDims> strides; // AllocDims * index_size (field 2)
// };
llvm::StructType* createRuntimeTensorType(
    int64_t num_dims,
    PrimDataType index_type,
    llvm::LLVMContext& context) {
  // Field 0: data pointer (always 8 bytes)
  llvm::Type* ptr_type = llvm::PointerType::getUnqual(context);

  // Field 1 & 2: index arrays (int64_t or int32_t based on index_type)
  llvm::Type* index_elem_type = (index_type == PrimDataType::Int)
      ? llvm::Type::getInt64Ty(context)
      : llvm::Type::getInt32Ty(context);

  llvm::ArrayType* sizes_array =
      llvm::ArrayType::get(index_elem_type, num_dims);
  llvm::ArrayType* strides_array =
      llvm::ArrayType::get(index_elem_type, num_dims);

  // Create the struct: {ptr, [N x iXX], [N x iXX]}
  // isPacked=false uses natural C struct alignment
  return llvm::StructType::create(
      context,
      {ptr_type, sizes_array, strides_array},
      "runtime_Tensor",
      /*isPacked=*/false);
}

// Simple permute transformation example:
// logical domain: [a, b, c, d]
// original logical sizes: [a, b, c, d]
// original logical stride: [b*c*d, c*d, d, 1]
// permute(0,1)
// allocation domain: [b, a, c, d]
// refined logical sizes: [a, b, c, d]
// refined logical stride: [c*d, a*c*d, d, 1]
// we want to propagate the allocation domain to the logical domain to get:
// 1. correct order of sizes and strides
// 2. refined logical sizes (divide out device/expanded broadcast dimensions)

void inferTensorShapesAndStrides(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder,
    llvm::SmallVectorImpl<llvm::Value*>& sizes,
    llvm::SmallVectorImpl<llvm::Value*>& strides) {
  const std::vector<IterDomain*>& logical_domain = tv->getLogicalDomain();
  const std::vector<IterDomain*>& allocation_domain =
      tv->getMaybeAllocationDomain();
  LinkedHashMap<IterDomain*, llvm::Value*> id_to_allocation_size;

  // push all allocation domains extents in regular order
  for (auto [i, id] : enumerate(allocation_domain)) {
    llvm::Value* extent = getOrCreateValueForExtent(id, val_to_value, builder);
    if (id->isDeviceDim() || id->isBroadcast()) {
      extent = builder.getInt64(1);
    }
    id_to_allocation_size.pushBack(id, extent);
  }

  // traverse backward from allocation domains to logical domains
  for (Expr* transform :
       DependencyCheck::getAllExprsBetween(
           {logical_domain.begin(), logical_domain.end()},
           {allocation_domain.begin(), allocation_domain.end()}) |
           std::views::reverse) {
    if (auto* split = dynamic_cast<Split*>(transform)) {
      auto [outer_extent, outer_i] =
          id_to_allocation_size.erase(split->outer());
      NVF_ERROR(
          outer_i != id_to_allocation_size.end() &&
              outer_i->first == split->inner(),
          split->toString(),
          " invalid split: inner is expected to appear immediately after "
          "outer");
      auto [inner_extent, inner_i] =
          id_to_allocation_size.erase(split->inner());

      if (split->inner()->isBroadcast()) {
        inner_extent = builder.getInt64(1);
      }

      if (split->outer()->isBroadcast()) {
        outer_extent = builder.getInt64(1);
      }

      llvm::Value* in_extent = builder.CreateMul(outer_extent, inner_extent);
      id_to_allocation_size.insert(inner_i, split->in(), in_extent);
      // NOTE: we probably need to throw error for merge as it's not handle yet
    } else if (auto* merge = dynamic_cast<Merge*>(transform)) {
      const auto [out_extent, out_i] =
          id_to_allocation_size.erase(merge->out());

      // NOTE: we don't have a protocol to decide which iter domain to pad,
      // currently we just pad inner value, so dividend is outer value
      // so inner_extent = (out_extent + outer_extent - 1) / outer_extent, which
      // is a ceilDiv
      llvm::Value* outer_extent =
          getOrCreateValueForExtent(merge->outer(), val_to_value, builder);
      llvm::Value* minus_one =
          builder.CreateSub(outer_extent, builder.getInt64(1));
      llvm::Value* plus_value = builder.CreateAdd(out_extent, minus_one);
      llvm::Value* inner_extent = builder.CreateUDiv(plus_value, outer_extent);

      id_to_allocation_size.insert(out_i, merge->outer(), outer_extent);
      id_to_allocation_size.insert(out_i, merge->inner(), inner_extent);
    }
  }

  auto ids = std::views::keys(id_to_allocation_size);
  std::vector<IterDomain*> logical_domain_reordered(ids.begin(), ids.end());

  auto allocation_order =
      ir_utils::computePermutation(logical_domain, logical_domain_reordered);
  NVF_ERROR(
      allocation_order.has_value(),
      "LLVM Lowering Error: Failed to compute allocation order");

  // Map last level propagated allocation domains to logical domain
  // we should be able to get the permutation between them
  llvm::Value* allocation_order_stride = builder.getInt64(1);
  strides.resize(logical_domain.size());
  sizes.reserve(logical_domain.size());

  for (int64_t logical_idx : allocation_order.value() | std::views::reverse) {
    IterDomain* id = logical_domain[logical_idx];
    if (id->isReduction()) {
      continue;
    }
    if (id->isBroadcast()) {
      strides[logical_idx] = builder.getInt64(0);
      continue;
    }
    auto [extent, out_i] = id_to_allocation_size.erase(id);
    strides[logical_idx] = allocation_order_stride;
    allocation_order_stride =
        builder.CreateMul(allocation_order_stride, extent);
  }

  strides.erase(
      std::remove_if(
          strides.begin(),
          strides.end(),
          [](llvm::Value* stride) { return stride == nullptr; }),
      strides.end());

  for (IterDomain* id : logical_domain) {
    if (id->isReduction()) {
      continue;
    }
    sizes.push_back(getOrCreateValueForExtent(id, val_to_value, builder));
  }

  // Check if sizes and strides are the same size as logical domain
  const auto logical_ndims =
      std::ranges::distance(logical_domain | TensorDomain::kNoReductions);
  NVF_ERROR_EQ(std::ssize(sizes), logical_ndims);
  NVF_ERROR_EQ(std::ssize(strides), logical_ndims);
}

// Packs a tensor argument into the runtime::Tensor format expected by CUDA
// kernels. Returns a pointer to the packed buffer (stack-allocated)
//
// Memory layout matches runtime/tensor.cu
// struct Tensor {
//   Pointer<T> data;                    // 8 bytes (field 0)
//   Array<index_t, Dims> sizes; // Dims * index_size (field 1)
//   Array<index_t, AllocDims> strides; // AllocDims * index_size (field 2)
// };
llvm::Value* packTensorArgument(
    llvm::Value* tensor, // at::Tensor*
    TensorView* tv,
    PrimDataType index_type,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder) {
  // Get allocation sizes/strides as LLVM IR values
  llvm::SmallVector<llvm::Value*, 8> alloc_sizes;
  llvm::SmallVector<llvm::Value*, 8> alloc_strides;
  inferTensorShapesAndStrides(
      tv, val_to_value, builder, alloc_sizes, alloc_strides);

  int64_t num_dims = alloc_sizes.size();

  // Special case: zero-dim tensors only have a data pointer
  if (num_dims == 0) {
    llvm::Value* data_ptr = createTensorDataPtr(tensor, builder);
    llvm::Value* ptr_alloca = builder.CreateAlloca(
        llvm::PointerType::getUnqual(builder.getContext()),
        nullptr,
        "zero_dim_tensor");
    builder.CreateStore(data_ptr, ptr_alloca);
    return builder.CreateBitCast(
        ptr_alloca, llvm::PointerType::getUnqual(builder.getInt8Ty()));
  }

  // Create the runtime tensor struct type
  llvm::StructType* tensor_struct_type =
      createRuntimeTensorType(num_dims, index_type, builder.getContext());

  // Allocate struct on stack
  llvm::Value* runtime_tensor =
      builder.CreateAlloca(tensor_struct_type, nullptr, "runtime_tensor");

  // 1. Store data pointer (field 0)
  llvm::Value* data_ptr = createTensorDataPtr(tensor, builder);
  llvm::Value* data_field_ptr = builder.CreateStructGEP(
      tensor_struct_type, runtime_tensor, 0, "data_field");
  builder.CreateStore(data_ptr, data_field_ptr);

  // Apply AdjustLastDim to sizes
  AdjustLastDim adjust = getLastDimAdjustment(tv->dtype());
  llvm::SmallVector<llvm::Value*, 8> adjusted_sizes = alloc_sizes;

  if (!adjusted_sizes.empty() && !adjust.isTrivial()) {
    llvm::Value* last_size = adjusted_sizes.back();
    llvm::Value* numerator = builder.getInt64(adjust.numerator);
    llvm::Value* denominator = builder.getInt64(adjust.denominator);
    llvm::Value* multiplied = builder.CreateMul(last_size, numerator);
    llvm::Value* adjusted = builder.CreateUDiv(multiplied, denominator);
    adjusted_sizes.back() = adjusted;
  }

  // 2. Store sizes (field 1 array)
  for (int64_t i = 0; i < num_dims; ++i) {
    llvm::Value* size_val = adjusted_sizes[i];

    // Truncate to Int32 if needed
    if (index_type == PrimDataType::Int32) {
      size_val = builder.CreateTrunc(size_val, builder.getInt32Ty());
    }

    // Use multi-index GEP to access sizes[i]
    // Indices: [0, 1, i] = deref pointer, field 1 (sizes), array index i
    llvm::Value* size_elem_ptr = builder.CreateGEP(
        tensor_struct_type,
        runtime_tensor,
        {builder.getInt32(0), builder.getInt32(1), builder.getInt32(i)},
        "sizes_" + std::to_string(i));
    builder.CreateStore(size_val, size_elem_ptr);
  }

  // 3. Store strides (field 2 array) with AdjustLastDim
  for (int64_t i = 0; i < num_dims; ++i) {
    llvm::Value* stride_val = alloc_strides[i];

    // Apply adjustment to all strides except last
    // See [Adjust all strides but not the last one] in tensorToBytes
    if (i < num_dims - 1 && !adjust.isTrivial()) {
      llvm::Value* numerator = builder.getInt64(adjust.numerator);
      llvm::Value* denominator = builder.getInt64(adjust.denominator);
      llvm::Value* multiplied = builder.CreateMul(stride_val, numerator);
      stride_val = builder.CreateUDiv(multiplied, denominator);
    }

    // Truncate to Int32 if needed
    if (index_type == PrimDataType::Int32) {
      stride_val = builder.CreateTrunc(stride_val, builder.getInt32Ty());
    }

    // Use multi-index GEP to access strides[i]
    // Indices: [0, 2, i] = deref pointer, field 2 (strides), array index i
    llvm::Value* stride_elem_ptr = builder.CreateGEP(
        tensor_struct_type,
        runtime_tensor,
        {builder.getInt32(0), builder.getInt32(2), builder.getInt32(i)},
        "strides_" + std::to_string(i));
    builder.CreateStore(stride_val, stride_elem_ptr);
  }

  // Cast to i8* for uniform handling
  return builder.CreateBitCast(
      runtime_tensor, llvm::PointerType::getUnqual(builder.getInt8Ty()));
}

} // namespace nvfuser
