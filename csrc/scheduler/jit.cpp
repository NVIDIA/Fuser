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
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <linked_hash_map.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>
#include <val_graph_visitor.h>

namespace nvfuser {

// cacheId, inputTensors, outputTensors
using main_func_t = void (*)(int64_t, const void**, void**);
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
constexpr std::string_view kLaunchKernelFuncName = "launch_kernel";
constexpr std::string_view kMatmulOutFuncName = "matmul_out";
constexpr std::string_view kLinearOutFuncName = "linear_out";
constexpr size_t kMaxTensorDim = 8;

llvm::Value* getOrCreateValueForExtent(
    IterDomain* id,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);

llvm::Value* getOrCreateValue(
    Val* val,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder);

// Pimpl for HeuristicJit
struct HeuristicJitImpl {
 public:
  HeuristicJitImpl(
      Fusion* fusion,
      SchedulerType scheduler_type,
      int num_threads);
  ~HeuristicJitImpl() = default;

 private:
  void compile();
  void registerExternalFunctions();

  std::unique_ptr<llvm::orc::LLJIT> jit_;
  Fusion* fusion_;
  SchedulerType scheduler_type_;
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
    const Fusion* fusion,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();

  insertNvtxRangePush("unpackInputs", builder);

  llvm::Function* func = builder.GetInsertBlock()->getParent();


  insertNvtxRangePop(builder);
  if (isDebugDumpEnabled(DebugDumpOption::HeuristicJit)) {
    printLlvmIr(func, "Main Function Inputs");
  }
}

void packOutputs(
    const Fusion* fusion,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();
  insertNvtxRangePush("packOutputs", builder);
  // Get the current function (main) and its output tensor array
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* aten_tensor_array = func->getArg(2);

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
  if (isDebugDumpEnabled(DebugDumpOption::HeuristicJit)) {
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
 
  // main function: void main(void** input_tensors, void** output_tensors)
  auto* main_type = llvm::FunctionType::get(
      void_type, {int64_type, void_array_ptr_type, void_array_ptr_type}, false);
  llvm::Function::Create(
      main_type, llvm::Function::ExternalLinkage, kMainFuncName, module);
}


// Implementation of HeuristicJitImpl
HeuristicJitImpl::HeuristicJitImpl(
    Fusion* fusion,
    SchedulerType scheduler_type,
    int num_threads)
    : fusion_(fusion), scheduler_type_(scheduler_type) {
  FUSER_PERF_SCOPE("HeuristicJitImpl::HeuristicJitImpl");

  // Initialize LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  jit_ = throwIfError(
      llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());

  registerExternalFunctions();
  compile();
}

void HeuristicJitImpl::registerExternalFunctions() {
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

  // insert fuser perf scope
  void* nvtx_range_push_func_ptr = reinterpret_cast<void*>(
      +[](const char* name) -> void { nvtxRangePush(name); });

  void* nvtx_range_pop_func_ptr =
      reinterpret_cast<void*>(+[]() -> void { nvtxRangePop(); });

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
      nvtx_range_push_func_ptr,
      name_to_symbol,
      mangler,
      kNvtxRangePushFuncName);
  registerExternalFunction(
      nvtx_range_pop_func_ptr, name_to_symbol, mangler, kNvtxRangePopFuncName);
  throwIfError(
      dest_dynamic_lib.define(llvm::orc::absoluteSymbols(name_to_symbol)));
}

bool HeuristicJitImpl::canReuse(
    const HeuristicParams* heuristic_params) {
  
}

HeuristicJit::~HeuristicJit() = default;

HeuristicJit::HeuristicJit(
    Fusion* fusion,
    SchedulerType scheduler_type,
    int num_threads)
    : pimpl_(new HeuristicJitImpl(fusion, scheduler_type, num_threads)) {}

} // namespace nvfuser