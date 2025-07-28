// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <functional>
#include <memory>
#include <unordered_map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <ATen/ATen.h>
#include <c10/core/MemoryFormat.h>

#include <bfs.h>
#include <host_ir/executor.h>
#include <host_ir/host_ir.h>
#include <host_ir/jit.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <linked_hash_map.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>
#include <val_graph_visitor.h>

namespace nvfuser {

using main_func_t = void (*)(const void**, void**);
constexpr std::string_view kMainFuncName = "main";
constexpr std::string_view kTensorSizeFuncName = "tensor_size";
constexpr std::string_view kTensorStrideFuncName = "tensor_stride";
constexpr std::string_view kNewTensorFuncName = "new_tensor";
constexpr std::string_view kDeleteTensorFuncName = "delete_tensor";
constexpr std::string_view kSetTensorFuncName = "set_tensor";
constexpr std::string_view kAtEmptyStridedCudaWrapper = "at_empty_strided_cuda";
constexpr std::string_view kAtTensorType = "at.Tensor";
constexpr std::string_view kNvtxRangePushFuncName = "nvtx_range_push";
constexpr std::string_view kNvtxRangePopFuncName = "nvtx_range_pop";
constexpr std::string_view kPermuteTensorFuncName = "permute_tensor";
constexpr size_t kMaxTensorDim = 8;

llvm::Value* getOrCreateValueForTensor(
    TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);

llvm::Value* getOrCreateValueForExtent(
    IterDomain* id,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);
llvm::Value* getOrCreateValue(
    Val* val,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);

// Pimpl for HostIrJit
struct HostIrJitImpl {
 public:
  HostIrJitImpl(
      std::unique_ptr<hir::HostIrContainer> container,
      int num_threads);
  ~HostIrJitImpl() = default;

  // Main interface methods, these are the only methods that should be called by
  // HostIrJit wrapper
  KernelArgumentHolder runWithInputs(const KernelArgumentHolder& args);
  const std::vector<Val*>& inputs() const {
    return container_->inputs();
  }
  const std::vector<Val*>& outputs() const {
    return container_->outputs();
  }
  const hir::HostIrContainer& container() const {
    return *container_;
  }

 private:
  void compile();
  void registerExternalFunctions();

  std::unique_ptr<llvm::orc::LLJIT> jit_;
  std::unique_ptr<hir::HostIrContainer> container_;
  main_func_t main_func_;
};

// Helper function to check for and throw errors from LLVM
void throwIfError(llvm::Error err) {
  if (err) {
    NVF_THROW(llvm::toString(std::move(err)));
  }
}

template <typename T>
T throwIfError(llvm::Expected<T>&& E) {
  if (!E) {
    throwIfError(E.takeError());
  }
  return std::move(*E);
}

// Helper functions to get LLVM type for given types
llvm::Type* getInt8PtrType(llvm::LLVMContext& context) {
  return llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
}

llvm::Type* getInt8PtrStaticArrayType(llvm::LLVMContext& context, size_t size) {
  return llvm::ArrayType::get(getInt8PtrType(context), size);
}

llvm::Type* getInt8PtrDynamicArrayType(llvm::LLVMContext& context) {
  return llvm::PointerType::getUnqual(getInt8PtrType(context));
}

// Helper function to get opaque at::Tensor type for better type safety
llvm::Type* getTensorPtrType(llvm::LLVMContext& context) {
  // Create an opaque struct type for at::Tensor
  // This provides better type safety than using void* for tensor pointers
  // while still being compatible with LLVM's type system
  return llvm::StructType::create(context, kAtTensorType)->getPointerTo();
}

llvm::ArrayType* getInt64StaticArrayType(
    llvm::LLVMContext& context,
    size_t size) {
  return llvm::ArrayType::get(llvm::Type::getInt64Ty(context), size);
}

llvm::Type* getInt64PtrType(llvm::LLVMContext& context) {
  return llvm::Type::getInt64Ty(context)->getPointerTo();
}

// Helper function to insert nvtxRangePush call
void insertNvtxRangePush(const char* op_name, llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Function* nvtx_range_push_func =
      module->getFunction(kNvtxRangePushFuncName);

  llvm::Value* op_name_ptr = builder.CreateGlobalString(op_name);
  builder.CreateCall(nvtx_range_push_func, {op_name_ptr});
}

void insertNvtxRangePop(llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Function* nvtx_range_pop_func =
      module->getFunction(kNvtxRangePopFuncName);

  // Call nvtxRangePop function
  builder.CreateCall(nvtx_range_pop_func, {});
}

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

// Helper function to register external functions in JIT
void registerExternalFunction(
    void* func_ptr,
    llvm::orc::SymbolMap& symbolMap,
    llvm::orc::MangleAndInterner& mangler,
    std::string_view func_name) {
  auto addr = llvm::orc::ExecutorAddr::fromPtr(func_ptr);
  symbolMap[mangler(func_name)] =
      llvm::orc::ExecutorSymbolDef(addr, llvm::JITSymbolFlags::Exported);
}

// Helper function to print generated LLVM IR after each node is processed
void printLlvmIr(llvm::Function* func, std::string_view msg) {
  llvm::outs() << "=== LLVM IR After Generating " << msg << " ===\n";
  func->print(llvm::outs(), nullptr);
  llvm::outs() << "\n\n";
}

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
    if (auto* binary_op = def->as<BinaryOp>()) {
      return createValueForBinaryOp(binary_op, val_to_value, builder);
    }

    if (auto* unary_op = def->as<UnaryOp>()) {
      return createValueForUnaryOp(unary_op, val_to_value, builder);
    }

    NVF_THROW(
        "LLVM Lowering Error: createValueForExtent called with unsupported "
        "expression type: ",
        def->getOpString());
  }

  NVF_THROW(
      "LLVM Lowering Error: createValueForExtent called with unfounded "
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


llvm::Value* getOrCreateValueForTensor(
    TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder) {
  return getOrCreateValue(tv, val_to_value, builder);
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
  NVF_ERROR_EQ(sizes.size(), TensorDomain::noReductions(logical_domain).size());
  NVF_ERROR_EQ(
      strides.size(), TensorDomain::noReductions(logical_domain).size());
}

void unpackInputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();

  insertNvtxRangePush("unpackInputs", builder);

  // Get the current function (main) and its first argument
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* aten_tensor_array = func->getArg(0);

  llvm::Type* aten_tensor_array_type = getInt8PtrDynamicArrayType(context);
  llvm::Type* tensor_type = getTensorPtrType(context);

  // bind input aten tensor sizes to val_to_value
  for (const auto [i, input] : enumerate(container->inputs())) {
    auto* tv = dynamic_cast<TensorView*>(input);
    NVF_ERROR(tv != nullptr, "Unsupported expression type: ", input);
    llvm::Value* tensor_addr = builder.CreateGEP(
        aten_tensor_array_type, aten_tensor_array, {builder.getInt64(i)});
    tensor_addr->setName("input_aten_tensor_addr");
    // Load the actual tensor pointer from the array
    llvm::Value* tensor = builder.CreateLoad(tensor_type, tensor_addr);
    tensor->setName("input_aten_tensor");
    // bind input aten tensor sizes to val_to_value
    // TODO: We should validate const size and strides here, ie. dim check
    for (const auto [dim_idx, id] :
         enumerate(TensorDomain::noReductions(tv->getLogicalDomain()))) {
      if (id->isBroadcast()) {
        val_to_value[id->extent()] = builder.getInt64(1);
        if (id->hasExpandedExtent()) {
          val_to_value[id->expandedExtent()] =
              createTensorSize(tensor, dim_idx, builder);
        }
      } else {
        val_to_value[id->extent()] = createTensorSize(tensor, dim_idx, builder);
      }
    }
    // bind input aten tensor to val_to_value
    val_to_value[tv] = tensor;
  }
  insertNvtxRangePop(builder);
  if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
    printLlvmIr(func, "Main Function Inputs");
  }
}

void packOutputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();
  insertNvtxRangePush("packOutputs", builder);
  // Get the current function (main) and its second argument
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* aten_tensor_array = func->getArg(1);

  llvm::Type* aten_tensor_array_type = getInt8PtrDynamicArrayType(context);
  // Store output tensor pointers from val_to_value into the output array
  for (const auto [i, output] : enumerate(container->outputs())) {
    auto* tv = dynamic_cast<TensorView*>(output);
    NVF_ERROR(tv != nullptr, "Unsupported expression type: ", output);
    llvm::Value* tensor_addr = builder.CreateGEP(
        aten_tensor_array_type, aten_tensor_array, {builder.getInt64(i)});
    tensor_addr->setName("output_aten_tensor_addr");

    // Get the tensor pointer from val_to_value and store it in the output
    // array
    llvm::Value* tensor_from_val_to_value = val_to_value[tv];
    builder.CreateStore(tensor_from_val_to_value, tensor_addr);
  }
  insertNvtxRangePop(builder);
  builder.CreateRetVoid();
  if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
    printLlvmIr(func, "Main Function Outputs");
  }
}

void compileFunctionDeclarations(
    llvm::Module* module,
    llvm::LLVMContext& context) {
  // Get the types
  auto* void_type = llvm::Type::getVoidTy(context);
  auto* void_ptr_type = getInt8PtrType(context);
  auto* void_array_ptr_type = getInt8PtrDynamicArrayType(context);
  auto* int64_type = llvm::Type::getInt64Ty(context);
  auto* int64_ptr_type = getInt64PtrType(context);
  auto* int32_type = llvm::Type::getInt32Ty(context);
  auto* tensor_type = getTensorPtrType(context);

  // tensor_size function: int64_t tensor_size(at::Tensor* tensor, int64_t dim)
  auto* tensor_size_type =
      llvm::FunctionType::get(int64_type, {tensor_type, int64_type}, false);
  llvm::Function::Create(
      tensor_size_type,
      llvm::Function::ExternalLinkage,
      kTensorSizeFuncName,
      module);

  // new_tensor function: at::Tensor* new_tensor()
  auto* new_tensor_type = llvm::FunctionType::get(tensor_type, {}, false);
  llvm::Function::Create(
      new_tensor_type,
      llvm::Function::ExternalLinkage,
      kNewTensorFuncName,
      module);

  // set_tensor function: void set_tensor(at::Tensor* tensor, at::Tensor*
  // other_tensor)
  auto* set_tensor_type =
      llvm::FunctionType::get(void_type, {tensor_type, tensor_type}, false);
  llvm::Function::Create(
      set_tensor_type,
      llvm::Function::ExternalLinkage,
      kSetTensorFuncName,
      module);

  // at::native::empty_strided_cuda function: void at_empty_strided_cuda(const
  // int64_t* sizes, int64_t ndim, const int64_t* strides, int64_t strides_ndim,
  // int32_t dtype, int64_t device_index, at::Tensor* out_tensor)
  auto* empty_strided_cuda_type = llvm::FunctionType::get(
      void_type,
      {int64_ptr_type,
       int64_type,
       int64_ptr_type,
       int64_type,
       int32_type,
       int64_type,
       tensor_type},
      false);
  llvm::Function::Create(
      empty_strided_cuda_type,
      llvm::Function::ExternalLinkage,
      kAtEmptyStridedCudaWrapper,
      module);

  // delete_tensor function: void delete_tensor(at::Tensor* tensor)
  auto* delete_tensor_type =
      llvm::FunctionType::get(void_type, {tensor_type}, false);
  llvm::Function::Create(
      delete_tensor_type,
      llvm::Function::ExternalLinkage,
      kDeleteTensorFuncName,
      module);

  // nvtx_range_push function: void nvtx_range_push(const char* name)
  auto* nvtx_range_push_type =
      llvm::FunctionType::get(void_type, {void_ptr_type}, false);
  llvm::Function::Create(
      nvtx_range_push_type,
      llvm::Function::ExternalLinkage,
      kNvtxRangePushFuncName,
      module);

  // nvtx_range_pop function: void nvtx_range_pop()
  auto* nvtx_range_pop_type = llvm::FunctionType::get(void_type, {}, false);
  llvm::Function::Create(
      nvtx_range_pop_type,
      llvm::Function::ExternalLinkage,
      kNvtxRangePopFuncName,
      module);

  // permute_tensor function: void permute_tensor(at::Tensor* out, at::Tensor* in, const int64_t* permutation, int64_t perm_size)
  auto* permute_tensor_type = llvm::FunctionType::get(void_type, {tensor_type, tensor_type, int64_ptr_type, int64_type}, false);
  llvm::Function::Create(
      permute_tensor_type,
      llvm::Function::ExternalLinkage,
      kPermuteTensorFuncName,
      module);

  // main function: void main(void** input_tensors, void** output_tensors)
  auto* main_type = llvm::FunctionType::get(
      void_type, {void_array_ptr_type, void_array_ptr_type}, false);
  llvm::Function::Create(
      main_type, llvm::Function::ExternalLinkage, kMainFuncName, module);
}

// Not handled instructions automatically trigger an error.
class HostIrCompileDispatcher : public OptInDispatch {
 public:
  HostIrCompileDispatcher(
      llvm::IRBuilder<>& builder,
      std::unordered_map<Val*, llvm::Value*>& val_to_value)
      : builder_(builder), val_to_value_(val_to_value) {}
  using OptInDispatch::handle;

  void handle(hir::NewTensor* new_tensor) final {
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::Function* new_tensor_func = module->getFunction(kNewTensorFuncName);
    llvm::Value* tensor =
        builder_.CreateCall(new_tensor_func, {}, "new_tensor");
    val_to_value_[new_tensor->buffer()->as<Val>()] = tensor;
  }

  void handle(LoadStoreOp* load_store_op) final {
    NVF_ERROR(
        load_store_op->opType() == LoadStoreOpType::Set ||
        load_store_op->opType() == LoadStoreOpType::SegmenterSet);
    NVF_ERROR(
    load_store_op->out()->isA<TensorView>(), "out must be a TensorView");
    TensorView* in_tv = load_store_op->in()->as<TensorView>();
    TensorView* out_tv = load_store_op->out()->as<TensorView>();
    llvm::Value* in_tensor = getOrCreateValueForTensor(in_tv, val_to_value_, builder_);
    // we assume all output tensors are already created, either through new or allocated
    llvm::Value* out_tensor = getOrCreateValueForTensor(out_tv, val_to_value_, builder_);

    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::LLVMContext& context = builder_.getContext();
  
    if (out_tv->hasRoot()) {
      std::optional<std::vector<int64_t>> permutation =
          ir_utils::computePermutation(
              out_tv->getRootDomain(), out_tv->getLogicalDomain());
      NVF_ERROR(
          permutation.has_value(),
          "The logical domain of a Set.Permute is supposed to be a permutation"
          " of the root domain: ",
          out_tv);

      llvm::Function* permute_tensor_func =
          module->getFunction(kPermuteTensorFuncName);
      
      // Create array of permutation values
      llvm::ArrayType* perm_array_type = getInt64StaticArrayType(context, permutation.value().size());
      llvm::Value* perm_array = builder_.CreateAlloca(perm_array_type, nullptr, "permutation");

      for (size_t i = 0; i < permutation.value().size(); ++i) {
        llvm::Value* gep = builder_.CreateInBoundsGEP(
            perm_array_type, perm_array, 
            {builder_.getInt32(0), builder_.getInt32(i)});
        builder_.CreateStore(builder_.getInt64(permutation.value()[i]), gep);
      }
      
      llvm::Type* int64_ptr_type = getInt64PtrType(context);
      llvm::Value* perm_ptr = builder_.CreateBitCast(perm_array, int64_ptr_type);
      llvm::Value* perm_size = builder_.getInt64(permutation.value().size());
      builder_.CreateCall(permute_tensor_func, {out_tensor, in_tensor, perm_ptr, perm_size});
    } else {
      llvm::Function* set_tensor_func =
          module->getFunction(kSetTensorFuncName);
      builder_.CreateCall(set_tensor_func, {out_tensor, in_tensor});
    }
  }

  // Create Function LLVM IR Generation
  void handle(kir::Allocate* allocate) final {
    llvm::LLVMContext& context = builder_.getContext();
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();

    llvm::Type* int64_ptr_type = getInt64PtrType(context);

    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_sizes;
    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_strides;
    inferTensorShapesAndStrides(
        allocate->buffer()->as<TensorView>(),
        val_to_value_,
        builder_,
        tensor_sizes,
        tensor_strides);

    // Bounds checking for ndim
    const std::vector<IterDomain*>& logical_domain = TensorDomain::noReductions(
        allocate->buffer()->as<TensorView>()->getLogicalDomain());

    NVF_ERROR_EQ(tensor_sizes.size(), logical_domain.size());

    // Create arrays for sizes and strides
    llvm::ArrayType* sizes_type =
        getInt64StaticArrayType(context, tensor_sizes.size());
    llvm::ArrayType* strides_type =
        getInt64StaticArrayType(context, tensor_strides.size());

    llvm::Value* sizes = builder_.CreateAlloca(sizes_type, nullptr, "sizes");
    llvm::Value* strides =
        builder_.CreateAlloca(strides_type, nullptr, "strides");

    for (const auto [i, size] : enumerate(tensor_sizes)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          sizes_type, sizes, {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(size, gep);
    }

    for (const auto [i, stride] : enumerate(tensor_strides)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          strides_type, strides, {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(stride, gep);
    }

    // Convert arrays to pointers
    llvm::Value* sizes_arg = builder_.CreateBitCast(sizes, int64_ptr_type);
    llvm::Value* strides_arg = builder_.CreateBitCast(strides, int64_ptr_type);

    // Create array size arguments
    llvm::Value* shape_ndim_arg = builder_.getInt64(tensor_sizes.size());
    llvm::Value* strides_ndim_arg = builder_.getInt64(tensor_strides.size());

    // Create output tensor
    llvm::Value* out_tensor = builder_.CreateCall(
        module->getFunction(kNewTensorFuncName), {}, "out_tensor");

    // Create constants for type and device from params
    at::ScalarType data_type = data_type_to_aten(
        allocate->buffer()->dtype() == DataType::Index
            ? PrimDataType::Int
            : allocate->buffer()->dtype());
    llvm::Value* dtype_constant =
        builder_.getInt32(static_cast<int32_t>(data_type));
    llvm::Value* device_index_constant =
        builder_.getInt64(Communicator::getInstance().deviceId());

    // Make allocation to new tensor
    llvm::Function* at_empty_strided_cuda_func =
        module->getFunction(kAtEmptyStridedCudaWrapper);

    // Call at::native::empty_strided_cuda with the computed arguments
    builder_.CreateCall(
        at_empty_strided_cuda_func,
        {sizes_arg,
         shape_ndim_arg,
         strides_arg,
         strides_ndim_arg,
         dtype_constant,
         device_index_constant,
         out_tensor});
    val_to_value_[allocate->buffer()] = out_tensor;
  }

  // Deallocation Function LLVM IR Generation
  void handle(hir::Deallocate* deallocate) final {
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::Function* delete_tensor_func =
        module->getFunction(kDeleteTensorFuncName);
    builder_.CreateCall(
        delete_tensor_func, {val_to_value_.at(deallocate->buffer())});
  }

 private:
  llvm::IRBuilder<>& builder_;
  std::unordered_map<Val*, llvm::Value*>& val_to_value_;
};

void HostIrJitImpl::compile() {
  NVF_ERROR(
      container_ != nullptr,
      "container is nullptr during host ir JIT compilation");
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("host_ir_jit_module", *context);
  llvm::IRBuilder<> builder(*context);
  std::unordered_map<Val*, llvm::Value*> val_to_value;

  // compile external functions and main function declarations
  compileFunctionDeclarations(module.get(), *context);

  // Create entry block and set insertion point
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(
      *context, "entry", module->getFunction(kMainFuncName));
  builder.SetInsertPoint(entry);

  // compile inputs in llvm ir
  unpackInputs(container_.get(), builder, val_to_value);
  HostIrCompileDispatcher dispatcher(builder, val_to_value);
  // compile all top level expressions in host ir container
  for (auto* expr : container_->topLevelExprs()) {
    insertNvtxRangePush(expr->getOpString(), builder);
    dispatcher.dispatch(expr);
    insertNvtxRangePop(builder);
    if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
      printLlvmIr(builder.GetInsertBlock()->getParent(), expr->getOpString());
    }
  }

  // compile outputs in llvm ir
  packOutputs(container_.get(), builder, val_to_value);

  // verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(
      !llvm::verifyModule(*module, &error_stream),
      "LLVM module verification failed: ",
      error);

  // Add the module to the JIT
  throwIfError(jit_->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(module), std::move(context))));

  // Look up the main function
  auto main_func_addr = throwIfError(jit_->lookup(kMainFuncName));
  main_func_ = reinterpret_cast<main_func_t>(main_func_addr.getValue());
}

// Implementation of HostIrJitImpl
HostIrJitImpl::HostIrJitImpl(
    std::unique_ptr<hir::HostIrContainer> container,
    int num_threads)
    : container_(std::move(container)) {
  FUSER_PERF_SCOPE("HostIrJitImpl::HostIrJitImpl");

  // Initialize LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  jit_ = throwIfError(
      llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());

  registerExternalFunctions();
  compile();
}

void HostIrJitImpl::registerExternalFunctions() {
  llvm::orc::JITDylib& dest_dynamic_lib = jit_->getMainJITDylib();
  auto mangler = llvm::orc::MangleAndInterner(
      dest_dynamic_lib.getExecutionSession(), jit_->getDataLayout());
  dest_dynamic_lib.addGenerator(throwIfError(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit_->getDataLayout().getGlobalPrefix())));

  // tensor size extraction function
  void* extract_tensor_size_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* tensor, int64_t dim) -> int64_t {
        NVF_ERROR(tensor != nullptr, kTensorSizeFuncName, " tensor is nullptr");
        NVF_ERROR(dim >= 0 && dim < tensor->dim(), "dim is out of range");
        return tensor->size(dim);
      });

  // tensor stride extraction function
  void* extract_tensor_stride_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* tensor, int64_t dim) -> int64_t {
        NVF_ERROR(
            tensor != nullptr, kTensorStrideFuncName, " tensor is nullptr");
        NVF_ERROR(dim >= 0 && dim < tensor->dim(), "dim is out of range");
        return tensor->stride(dim);
      });

  // new at::Tensor() wrapper instead of real tensor allocation
  void* new_tensor_func_ptr = reinterpret_cast<void*>(
      +[]() -> at::Tensor* { return new at::Tensor(); });

  // in place tensor update
  void* set_tensor_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* out, at::Tensor* in) -> void {
        NVF_ERROR(out != nullptr, kSetTensorFuncName, " out is nullptr");
        NVF_ERROR(in != nullptr, kSetTensorFuncName, " in is nullptr");
        if (!out->defined()) {
          // new tensor
          *out = in->clone(); // Clone the input tensor
        } else {
          // allocated tensor
          out->copy_(*in, /*non_blocking=*/true);
        }
      });

  // at::native::empty_strided_cuda
  void* empty_strided_cuda_func_ptr =
      reinterpret_cast<void*>(+[](const int64_t* sizes,
                                  int64_t ndim,
                                  const int64_t* strides,
                                  int64_t strides_ndim,
                                  int32_t dtype,
                                  int64_t device_index,
                                  at::Tensor* out_tensor) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::IntArrayRef aten_strides(strides, strides_ndim);
        auto scalar_type = static_cast<at::ScalarType>(dtype);
        at::Device device =
            at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index));
        *out_tensor = at::native::empty_strided_cuda(
            aten_sizes,
            aten_strides,
            scalar_type,
            c10::nullopt,
            device,
            c10::nullopt);
      });

  // delete a newed tensor
  void* delete_tensor_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* tensor) -> void { delete tensor; });

  // insert fuser perf scope
  void* nvtx_range_push_func_ptr = reinterpret_cast<void*>(
      +[](const char* name) -> void { nvtxRangePush(name); });

  void* nvtx_range_pop_func_ptr =
      reinterpret_cast<void*>(+[]() -> void { nvtxRangePop(); });

  // permute a tensor
  void* permute_tensor_func_ptr = reinterpret_cast<void*>(
    +[](at::Tensor* out, at::Tensor* in, const int64_t* permutation, int64_t perm_size) -> void {
        // Convert pointer to vector for permute function
        std::vector<int64_t> perm_vec(permutation, permutation + perm_size);
        
        if (!out->defined()) {
          *out = in->permute(perm_vec);
        } else {
          out->copy_(in->permute(perm_vec), /*non_blocking=*/true);
        }
    });

  // Register wrapper functions in JIT
  llvm::orc::SymbolMap name_to_symbol;
  registerExternalFunction(
      extract_tensor_size_func_ptr,
      name_to_symbol,
      mangler,
      kTensorSizeFuncName);
  registerExternalFunction(
      extract_tensor_stride_func_ptr,
      name_to_symbol,
      mangler,
      kTensorStrideFuncName);
  registerExternalFunction(
      new_tensor_func_ptr, name_to_symbol, mangler, kNewTensorFuncName);
  registerExternalFunction(
      delete_tensor_func_ptr, name_to_symbol, mangler, kDeleteTensorFuncName);
  registerExternalFunction(
      set_tensor_func_ptr, name_to_symbol, mangler, kSetTensorFuncName);
  registerExternalFunction(
      empty_strided_cuda_func_ptr,
      name_to_symbol,
      mangler,
      kAtEmptyStridedCudaWrapper);
  registerExternalFunction(
      nvtx_range_push_func_ptr,
      name_to_symbol,
      mangler,
      kNvtxRangePushFuncName);
  registerExternalFunction(
      nvtx_range_pop_func_ptr, name_to_symbol, mangler, kNvtxRangePopFuncName);
  registerExternalFunction(
      permute_tensor_func_ptr, name_to_symbol, mangler, kPermuteTensorFuncName);
  throwIfError(
      dest_dynamic_lib.define(llvm::orc::absoluteSymbols(name_to_symbol)));
}

KernelArgumentHolder HostIrJitImpl::runWithInputs(
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("HostIrJitImpl::runWithInputs");
  // Bind cache id to llvm global variable or align with main function inputs
  NVF_ERROR(args.getCacheId().has_value(), "Cache ID is not set");
  NVF_ERROR_EQ(std::ssize(container_->inputs()), args.size());

  std::vector<const void*> input_aten_tensors;
  // Bind the inputs to the tensor map
  for (auto [in_val, arg] : zip(container_->inputs(), args)) {
    NVF_ERROR(
        arg.is<at::Tensor>(),
        "Unsupported argument type: ",
        arg,
        " for input ",
        in_val);
    input_aten_tensors.push_back(&arg.as<at::Tensor>());
  }

  // Run the main function
  std::vector<void*> output_aten_tensors(container_->outputs().size());
  main_func_(input_aten_tensors.data(), output_aten_tensors.data());

  // Collect the outputs
  KernelArgumentHolder outputs;
  for (const auto [output, tensor] :
       zip(container_->outputs(), output_aten_tensors)) {
    NVF_ERROR(
        output->isA<TensorView>(),
        "Unsupported output type: ",
        output,
        " for output ",
        output);
    // Cast void* to at::Tensor* first, then dereference
    at::Tensor* aten_tensor = static_cast<at::Tensor*>(tensor);
    outputs.push(*aten_tensor);
    // Clean up the individual tensor object (not the array)
    delete aten_tensor;
  }
  // Note: output_aten_tensors points to a global array managed by JIT, don't
  // delete the array itself
  return outputs;
}

// NOTE: We have to keep the destructor here, otherwise the unique_ptr can't
// find complete type of HostIrJitImpl
HostIrJit::~HostIrJit() = default;

// HostIrJit wrapper methods, these are the only methods that should be called
// by the user
HostIrJit::HostIrJit(
    std::unique_ptr<hir::HostIrContainer> container,
    int num_threads)
    : pimpl_(new HostIrJitImpl(std::move(container), num_threads)) {}

KernelArgumentHolder HostIrJit::runWithInputs(
    const KernelArgumentHolder& args) {
  return pimpl_->runWithInputs(args);
}

const std::vector<Val*>& HostIrJit::inputs() const {
  return pimpl_->inputs();
}

const std::vector<Val*>& HostIrJit::outputs() const {
  return pimpl_->outputs();
}

const hir::HostIrContainer& HostIrJit::container() const {
  return pimpl_->container();
}

} // namespace nvfuser
