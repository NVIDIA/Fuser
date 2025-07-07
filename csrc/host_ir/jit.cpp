// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bfs.h>
#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <val_graph_visitor.h>

#include <instrumentation.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <unordered_map>
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
#include <host_ir/jit.h>
#include <functional>
#include <memory>

#include <multidevice/communicator.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

using main_func_fn = std::function<void()>;

// Forward declarations
std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferTensorShapesAndStrides(const TensorView* tv, std::unordered_map<Val*, llvm::Value*>& val2llvmMap, llvm::IRBuilder<>& builder);
std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferTensorShapesAndStridesAndStrides(const TensorView* tv, std::unordered_map<Val*, llvm::Value*>& val2llvmMap, llvm::IRBuilder<>& builder);
std::vector<llvm::Value*> inferTensorStridesReordered(const TensorView* tv, std::unordered_map<Val*, llvm::Value*>& val2llvmMap, llvm::IRBuilder<>& builder);
std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferShapeAndStridesNoReorder(const TensorView* tv, std::unordered_map<Val*, llvm::Value*>& val2llvmMap, llvm::IRBuilder<>& builder);
std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferShapeAndStridesRaw(const TensorView* tv, std::vector<Val*> symbolic_sizes, std::vector<bool> expand_flags, std::unordered_map<Val*, llvm::Value*>& val2llvmMap, llvm::IRBuilder<>& builder);
std::vector<llvm::Value*> getContiguousStrides(const std::vector<llvm::Value*>& sizes, const std::vector<bool>& expand_flags, llvm::IRBuilder<>& builder);
llvm::Value* compileJitImpl(const hir::HostIrContainer* container, LlvmJitImpl* pimpl_, llvm::IRBuilder<>& builder);
llvm::Value* generateTensorSizeExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::IRBuilder<>& builder);
llvm::Value* generateTensorStrideExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::IRBuilder<>& builder);
void livenessAnalysis(const std::vector<Expr*>& top_level_exprs, std::unordered_map<Val*, llvm::Value*>& val2llvmMap);

// PIMPL implementation for HostIrJit - moved to public scope
struct LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  main_func_fn main_func_;
  std::unordered_map<const TensorView*, at::Tensor> tensor_map;
  std::unique_ptr<hir::HostIrContainer> container_;
  LlvmJitImpl() = default;
  ~LlvmJitImpl() = default;
};

// Helper function to check for and throw errors from LLVM
void throwIfError(llvm::Error&& err) {
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

// Generate a function for ForLoop runtime
void compileForLoopFunc(
    const ForLoop* for_loop,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  return;
}

// Generate a function for IfThenElse runtime
void compileIfThenElseFunc(
    const kir::IfThenElse* ifthenelse,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  return;
}

// Generate a function for LinearOp runtime
void compileLinearFunc(
    const LinearOp* linear,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  llvm::LLVMContext& context = builder.getContext();
  auto mod = builder.GetInsertBlock()->getParent()->getParent();

  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  
  // Convert tv_in to void pointer
  uintptr_t tv_in_ptr = reinterpret_cast<uintptr_t>(linear->inA());
  llvm::Value* tv_in_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_in_ptr);
  llvm::Value* tv_in_void_ptr = builder.CreateIntToPtr(tv_in_constant, void_ptr_type);
  
  // Convert tv_weight to void pointer
  uintptr_t tv_weight_ptr = reinterpret_cast<uintptr_t>(linear->inB());
  llvm::Value* tv_weight_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_weight_ptr);
  llvm::Value* tv_weight_void_ptr = builder.CreateIntToPtr(tv_weight_constant, void_ptr_type);
  
  // Convert tv_out to void pointer
  uintptr_t tv_out_ptr = reinterpret_cast<uintptr_t>(linear->out());
  llvm::Value* tv_out_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_out_ptr);
  llvm::Value* tv_out_void_ptr = builder.CreateIntToPtr(tv_out_constant, void_ptr_type);

  // Handle optional bias
  llvm::Value* tv_bias_void_ptr = nullptr;
  
  // Call get_tensor function to get at::Tensor* pointers
  llvm::Value* t_in = builder.CreateCall(mod->getFunction("get_tensor"), {tv_in_void_ptr, pimpl_void_ptr}, "t_in");
  llvm::Value* t_weight = builder.CreateCall(mod->getFunction("get_tensor"), {tv_weight_void_ptr, pimpl_void_ptr}, "t_weight");
  llvm::Value* t_out = builder.CreateCall(mod->getFunction("get_tensor"), {tv_out_void_ptr, pimpl_void_ptr}, "t_out");

  if (linear->hasBias()) {
    llvm::Value* t_bias = builder.CreateCall(mod->getFunction("get_tensor"), {tv_bias_void_ptr, pimpl_void_ptr}, "t_bias");
    builder.CreateCall(mod->getFunction("linear_out_with_bias"), {t_out, t_in, t_weight, t_bias}, "linear_out_with_bias");
  } else {
    builder.CreateCall(mod->getFunction("linear_out_without_bias"), {t_out, t_in, t_weight}, "linear_out_without_bias");
  }

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

// Generate a function for Matmul runtime
void compileMatmulFunc(
    const MatmulOp* matmul,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  llvm::LLVMContext& context = builder.getContext();
  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));

  auto mod = builder.GetInsertBlock()->getParent()->getParent();
  uintptr_t tv_a_ptr = reinterpret_cast<uintptr_t>(matmul->inA());
  llvm::Value* tv_a_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_a_ptr);
  llvm::Value* tv_a_void_ptr = builder.CreateIntToPtr(tv_a_constant, void_ptr_type);
  
  // Convert tv_b to void pointer
  uintptr_t tv_b_ptr = reinterpret_cast<uintptr_t>(matmul->inB());
  llvm::Value* tv_b_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_b_ptr);
  llvm::Value* tv_b_void_ptr = builder.CreateIntToPtr(tv_b_constant, void_ptr_type);
  
  // Convert tv_out to void pointer
  uintptr_t tv_out_ptr = reinterpret_cast<uintptr_t>(matmul->out());
  llvm::Value* tv_out_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_out_ptr);
  llvm::Value* tv_out_void_ptr = builder.CreateIntToPtr(tv_out_constant, void_ptr_type);

  // Call get_tensor function to get at::Tensor* pointers
  llvm::Value* t_a = builder.CreateCall(mod->getFunction("get_tensor"), {tv_a_void_ptr, pimpl_void_ptr}, "t_a");
  llvm::Value* t_b = builder.CreateCall(mod->getFunction("get_tensor"), {tv_b_void_ptr, pimpl_void_ptr}, "t_b");
  llvm::Value* t_out = builder.CreateCall(mod->getFunction("get_tensor"), {tv_out_void_ptr, pimpl_void_ptr}, "t_out");

  // Call matmul_out function
  builder.CreateCall(mod->getFunction("matmul_out"), {t_out, t_a, t_b}, "matmul_out");

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}



// Generate a function for LaunchKernel runtime
void compileLaunchKernelFunc(
    const hir::LaunchKernel* launch_kernel,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  llvm::LLVMContext& context = builder.getContext();
  auto mod = builder.GetInsertBlock()->getParent()->getParent();

  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));

  // Convert input TensorViews to void pointers and get tensor pointers
  llvm::SmallVector<llvm::Value*,16> input_tensors;
  for (const auto& input : launch_kernel->inputs()) {
    // Convert TensorView pointer to void pointer
    uintptr_t tv_ptr = reinterpret_cast<uintptr_t>(input);
    llvm::Value* tv_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_ptr);
    llvm::Value* tv_void_ptr = builder.CreateIntToPtr(tv_constant, void_ptr_type);
    
    // Call get_tensor function to get at::Tensor* pointer
    llvm::Value* tensor_ptr = builder.CreateCall(mod->getFunction("get_tensor"), {tv_void_ptr, pimpl_void_ptr}, "input_tensor");
    input_tensors.push_back(tensor_ptr);
  }

  // Convert output TensorViews to void pointers and get tensor pointers
  llvm::SmallVector<llvm::Value*,16> output_tensors;
  for (const auto& output : launch_kernel->outputs()) {
    // Convert TensorView pointer to void pointer
    uintptr_t tv_ptr = reinterpret_cast<uintptr_t>(output);
    llvm::Value* tv_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_ptr);
    llvm::Value* tv_void_ptr = builder.CreateIntToPtr(tv_constant, void_ptr_type);
    
    // Call get_tensor function to get at::Tensor* pointer
    llvm::Value* tensor_ptr = builder.CreateCall(mod->getFunction("get_tensor"), {tv_void_ptr, pimpl_void_ptr}, "output_tensor");
    output_tensors.push_back(tensor_ptr);
  }
  
  // Get cache ID from val2llvmMap
  llvm::Value* cache_id_arg = val2llvmMap[launch_kernel->cacheId()];
  
  // Create arrays to hold tensor pointers
  llvm::ArrayType* input_array_type = llvm::ArrayType::get(void_ptr_type, input_tensors.size());
  llvm::ArrayType* output_array_type = llvm::ArrayType::get(void_ptr_type, output_tensors.size());
  
  llvm::Value* input_array = builder.CreateAlloca(input_array_type, nullptr, "input_array");
  llvm::Value* output_array = builder.CreateAlloca(output_array_type, nullptr, "output_array");
  
  // Populate input array
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    llvm::Value* gep = builder.CreateInBoundsGEP(input_array_type, input_array, {builder.getInt32(0), builder.getInt32(i)});
    builder.CreateStore(input_tensors[i], gep);
  }
  
  // Populate output array
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    llvm::Value* gep = builder.CreateInBoundsGEP(output_array_type, output_array, {builder.getInt32(0), builder.getInt32(i)});
    builder.CreateStore(output_tensors[i], gep);
  }
  
  // Convert arrays to pointers
  llvm::Value* input_ptr = builder.CreateBitCast(input_array, llvm::PointerType::getUnqual(void_ptr_type));
  llvm::Value* output_ptr = builder.CreateBitCast(output_array, llvm::PointerType::getUnqual(void_ptr_type));
  
  // Create constants for array sizes
  llvm::Value* num_inputs = builder.getInt64(input_tensors.size());
  llvm::Value* num_outputs = builder.getInt64(output_tensors.size());
  
  // Convert LaunchKernel pointer to void pointer
  uintptr_t launch_kernel_ptr = reinterpret_cast<uintptr_t>(launch_kernel);
  llvm::Value* launch_kernel_void_ptr = builder.CreateIntToPtr(
      builder.getInt64(launch_kernel_ptr), void_ptr_type);
  
  // Call launch_kernel function with correct signature
  builder.CreateCall(mod->getFunction("launch_kernel"), 
                     {cache_id_arg, input_ptr, num_inputs, output_ptr, num_outputs, launch_kernel_void_ptr, pimpl_void_ptr}, 
                     "launch_kernel");
  
  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR Generation for LaunchKernel Function ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

void compileDeallocateFunc(
    const hir::Deallocate* deallocate,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  llvm::LLVMContext& context = builder.getContext();
  auto mod = builder.GetInsertBlock()->getParent()->getParent();
  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  
  // Call wrapper function to deallocate tensor from tensor_map
  auto tv = deallocate->buffer()->as<TensorView>();
  auto tv_ptr = reinterpret_cast<uintptr_t>(tv);
  llvm::Value* tv_ptr_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tv_ptr);
  llvm::Value* tv_void_ptr = builder.CreateIntToPtr(tv_ptr_constant, void_ptr_type);
  
  llvm::Function* deallocate_tensor_func = mod->getFunction("deallocate_tensor");
  builder.CreateCall(deallocate_tensor_func, {tv_void_ptr, pimpl_void_ptr}, "deallocate_tensor"); 
  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR Generation for Deallocate Function ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

// Allocate tensor and bind to val2llvmMap
void compileAllocateFunc(
    const kir::Allocate* allocate,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  llvm::LLVMContext& context = builder.getContext();
  auto mod = builder.GetInsertBlock()->getParent()->getParent();

  // Define LLVM types
  llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  llvm::Type* int64_ptr_type = int64_type->getPointerTo();
  llvm::Type* int32_type = llvm::Type::getInt32Ty(context);
  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));

  // Get tensor sizes and strides using the inference function
  auto [tensor_sizes, tensor_strides] = inferTensorShapesAndStrides(allocate->buffer()->as<TensorView>(), val2llvmMap, builder);

  // Bounds checking for ndim
  auto logical_domain = TensorDomain::noReductions(
      allocate->buffer()->as<TensorView>()->getLogicalDomain());

  NVF_ERROR(tensor_sizes.size() == logical_domain.size(), "tensor_sizes.size() != logical_domain.size()");
  NVF_ERROR(tensor_strides.size() == logical_domain.size(), "tensor_strides.size() != logical_domain.size()");

  // Create arrays for sizes and strides
  llvm::ArrayType* sizes_array_type = llvm::ArrayType::get(int64_type, tensor_sizes.size());
  llvm::ArrayType* strides_array_type = llvm::ArrayType::get(int64_type, tensor_strides.size());
  
  llvm::Value* sizes_array = builder.CreateAlloca(sizes_array_type, nullptr, "sizes_array");
  llvm::Value* strides_array = builder.CreateAlloca(strides_array_type, nullptr, "strides_array");
  
  // Populate sizes array
  for (size_t i = 0; i < tensor_sizes.size(); ++i) {
    llvm::Value* gep = builder.CreateInBoundsGEP(sizes_array_type, sizes_array, {builder.getInt32(0), builder.getInt32(i)});
    builder.CreateStore(tensor_sizes[i], gep);
  }
  
  // Populate strides array
  for (size_t i = 0; i < tensor_strides.size(); ++i) {
    llvm::Value* gep = builder.CreateInBoundsGEP(strides_array_type, strides_array, {builder.getInt32(0), builder.getInt32(i)});
    builder.CreateStore(tensor_strides[i], gep);
  }
  
  // Convert arrays to pointers
  llvm::Value* sizes_arg = builder.CreateBitCast(sizes_array, int64_ptr_type);
  llvm::Value* strides_arg = builder.CreateBitCast(strides_array, int64_ptr_type);
  
  // Create array size arguments
  llvm::Value* shape_ndim_arg = builder.getInt64(tensor_sizes.size());
  llvm::Value* strides_ndim_arg = builder.getInt64(tensor_strides.size());

  // Create tensor storage for output
  uintptr_t tv_ptr = reinterpret_cast<uintptr_t>(allocate->buffer()->as<TensorView>());
  llvm::Value* tv_ptr_constant = llvm::ConstantInt::get(int64_type, tv_ptr);
  llvm::Value* tv_void_ptr = builder.CreateIntToPtr(tv_ptr_constant, void_ptr_type);
  llvm::Value* out_tensor_arg = builder.CreateCall(mod->getFunction("allocate_tensor"), {tv_void_ptr, pimpl_void_ptr}, "out_tensor");

  // Create constants for type and device from params
  at::ScalarType data_type = data_type_to_aten(
      allocate->buffer()->dtype() == DataType::Index
          ? PrimDataType::Int
          : allocate->buffer()->dtype());
  llvm::Value* dtype_constant =
      llvm::ConstantInt::get(int32_type, static_cast<int32_t>(data_type));
  llvm::Value* device_index_constant = llvm::ConstantInt::get(
      int64_type, Communicator::getInstance().deviceId());

  // Get the at::native::empty_strided_cuda function pointer (registered in the JIT)
  llvm::Function* at_empty_strided_cuda_func =
      mod->getFunction(kHostIrJitEmptyStridedCudaFuncName);
  if (at_empty_strided_cuda_func == nullptr) {
    llvm::FunctionType* at_empty_strided_cuda_func_type =
        llvm::FunctionType::get(
            builder.getVoidTy(),
            {int64_ptr_type,
             int64_type,
             int64_ptr_type,
             int64_type,
             int32_type,
             int64_type,
             void_ptr_type},
            false);
    at_empty_strided_cuda_func = llvm::Function::Create(
        at_empty_strided_cuda_func_type,
        llvm::Function::ExternalLinkage,
        kHostIrJitEmptyStridedCudaFuncName,
        mod);
  }

  // Call at::native::empty_strided_cuda with the computed arguments
  builder.CreateCall(
      at_empty_strided_cuda_func,
      {sizes_arg,
       shape_ndim_arg,
       strides_arg,
       strides_ndim_arg,
       dtype_constant,
       device_index_constant,
       out_tensor_arg});

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR Generation for Allocate Function ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}
        
void compileMainFuncOutputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
    llvm::Module* mod = builder.GetInsertBlock()->getParent()->getParent();
    builder.CreateRetVoid();
    const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
    if (debug_print) {
      llvm::outs() << "=== LLVM IR Generation for Main Function Outputs ===\n";
      mod->print(llvm::outs(), nullptr);
    }
}



// Mainly bind input tensor sizes to val2llvmMap
void compileMainFuncInputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::Value* pimpl_void_ptr) {
  llvm::LLVMContext& context = builder.getContext();
  llvm::Module* mod = builder.GetInsertBlock()->getParent()->getParent();

  std::vector<llvm::Type*> param_types = {};
  
  llvm::FunctionType* func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(context), param_types, false);
  llvm::Function* func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, "full_graph_induction", mod);
  
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry);
  for(auto* input : container->inputs()) {
    if(auto* tv = dynamic_cast<const TensorView*>(input)) {
      uintptr_t tv_ptr = reinterpret_cast<uintptr_t>(tv);
       // Create LLVM constant from the pointer value
      llvm::Value* tv_ptr_constant = llvm::ConstantInt::get(
          llvm::Type::getInt64Ty(context), 
          tv_ptr
      );
      
      // Cast to void pointer type in LLVM
      llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
      llvm::Value* tv_void_ptr = builder.CreateIntToPtr(tv_ptr_constant, void_ptr_type);
      
      // Call get_tensor function with the void pointer
      llvm::Value* tensor_ptr = builder.CreateCall(
          mod->getFunction("get_tensor"),
          {tv_void_ptr, pimpl_void_ptr}
      );
      // bind input aten tensor sizes to val2llvmMap
      auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());

      for (size_t dim = 0; dim < logical_domain.size(); ++dim) {
        if (logical_domain[dim]->isBroadcast()) {
          val2llvmMap[logical_domain[dim]->extent()] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1);
          if (logical_domain[dim]->hasExpandedExtent()) {
            val2llvmMap[logical_domain[dim]->expandedExtent()] = generateTensorSizeExtraction(tensor_ptr, dim, builder);
          }
        } else {
          val2llvmMap[logical_domain[dim]->extent()] = generateTensorSizeExtraction(tensor_ptr, dim, builder);
        }
      }
    }
  }

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR Generation for Main Function Inputs ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

void compile(
    const hir::HostIrContainer* container,
    llvm::orc::LLJIT* jit,
    main_func_fn& main_func_,
    LlvmJitImpl* pimpl_) {
  FUSER_PERF_SCOPE("HostIrJit::compile");
  NVF_ERROR(
      container != nullptr,
      "container is nullptr during host ir JIT compilation");
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("host_ir_jit_module", *ctx);
  llvm::IRBuilder<> builder(*ctx);
  std::unordered_map<Val*, llvm::Value*> val2llvmMap;
  
  // Generate pimpl_void_ptr once at the top
  llvm::Value* pimpl_void_ptr = compileJitImpl(container, pimpl_, builder);
  
  // bind the constants
  compileMainFuncInputs(container, builder, val2llvmMap, pimpl_void_ptr);
  std::vector<Expr*> top_level_exprs = container->topLevelExprs();
  livenessAnalysis(top_level_exprs, val2llvmMap);
  // Generate the top level functions
  for(auto* input : top_level_exprs) {
    if(auto* allocate = dynamic_cast<const kir::Allocate*>(input)) {
      compileAllocateFunc(allocate, builder, val2llvmMap, pimpl_void_ptr);
    }
    else if(auto* deallocate = dynamic_cast<const hir::Deallocate*>(input)) {
      compileDeallocateFunc(deallocate, builder, val2llvmMap, pimpl_void_ptr);
    }
    else if(auto* launch_kernel = dynamic_cast<const hir::LaunchKernel*>(input)) {
      compileLaunchKernelFunc(launch_kernel, builder, val2llvmMap, pimpl_void_ptr);
    }
    else if(auto* for_loop = dynamic_cast<const ForLoop*>(input)) {
      compileForLoopFunc(for_loop, builder, val2llvmMap, pimpl_void_ptr);
    }
    else if(auto* if_then_else = dynamic_cast<const kir::IfThenElse*>(input)) {
      compileIfThenElseFunc(if_then_else, builder, val2llvmMap, pimpl_void_ptr);
    }
    else if(auto* matmul = dynamic_cast<const MatmulOp*>(input)) {
      compileMatmulFunc(matmul, builder, val2llvmMap, pimpl_void_ptr);
    }
    else if(auto* linear_op = dynamic_cast<const LinearOp*>(input)) {
      compileLinearFunc(linear_op, builder, val2llvmMap, pimpl_void_ptr);
    }
    else{
      NVF_THROW("Unsupported input type: ", input);
    }
  }
  // Collect output tensors and garbage collect intermediate tensors
  compileMainFuncOutputs(container, builder, val2llvmMap, pimpl_void_ptr);
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(
      !llvm::verifyModule(*mod, &error_stream),
      "LLVM module verification failed: " + error);

  // Add the module to the JIT
  throwIfError(jit->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx))));
 
  // Look up the main function
  auto main_func_addr = throwIfError(jit->lookup("full_graph_induction"));
  main_func_ = reinterpret_cast<void(*)()>(main_func_addr.getValue());
}


llvm::Value* compileJitImpl(
    const hir::HostIrContainer* container,
    LlvmJitImpl* pimpl_,
    llvm::IRBuilder<>& builder) {
    llvm::LLVMContext& context = builder.getContext();
    llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  // Convert LlvmJitImpl pointer to void pointer
    uintptr_t pimpl_ptr = reinterpret_cast<uintptr_t>(pimpl_);
    llvm::Value* pimpl_constant = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), pimpl_ptr);
    llvm::Value* pimpl_void_ptr = builder.CreateIntToPtr(pimpl_constant, void_ptr_type);
    return pimpl_void_ptr;
}

void livenessAnalysis(const std::vector<Expr*>& top_level_exprs, std::unordered_map<Val*, llvm::Value*>& val2llvmMap) {
  // TODO: for other types of aten ops, we want to insert proper deallocate calls to intermediate tensors
  return;
}

llvm::Value* traverseExtentDFS(Val* val, std::unordered_map<Val*, llvm::Value*>& val2llvmMap, llvm::IRBuilder<>& builder) {
  if (val2llvmMap.find(val) != val2llvmMap.end()) {
    return val2llvmMap[val];
  }
  if (val->definition() != nullptr) {
    auto* def = val->definition();
    if(auto* binary_op = def->as<BinaryOp>()) {
      auto* left = binary_op->lhs()->as<Val>();
      auto* right = binary_op->rhs()->as<Val>();
      if(left->isConst() && val2llvmMap.find(left) == val2llvmMap.end()) {
        val2llvmMap[left] = builder.getInt64(left->value().as<int64_t>());
      }
      else if(!left->isConst() && val2llvmMap.find(left) == val2llvmMap.end()) {
        traverseExtentDFS(left, val2llvmMap, builder);
      }
      if(right->isConst() && val2llvmMap.find(right) == val2llvmMap.end()) {
        val2llvmMap[right] = builder.getInt64(right->value().as<int64_t>());
      }
      else if(!right->isConst() && val2llvmMap.find(right) == val2llvmMap.end()) {
        traverseExtentDFS(right, val2llvmMap, builder);
      }
      if(binary_op->getBinaryOpType() == BinaryOpType::Add) {
        val2llvmMap[val] = builder.CreateAdd(val2llvmMap[left], val2llvmMap[right]);
      }
      else if(binary_op->getBinaryOpType() == BinaryOpType::Sub) {
        val2llvmMap[val] = builder.CreateSub(val2llvmMap[left], val2llvmMap[right]);
      }
      else if(binary_op->getBinaryOpType() == BinaryOpType::Mul) {
        val2llvmMap[val] = builder.CreateMul(val2llvmMap[left], val2llvmMap[right]);
      }
      else if(binary_op->getBinaryOpType() == BinaryOpType::CeilDiv) {
        // Implement ceilDiv as (a + b - 1) / b
        llvm::Value* numerator = builder.CreateAdd(val2llvmMap[left], val2llvmMap[right]);
        llvm::Value* one = builder.getInt64(1);
        numerator = builder.CreateSub(numerator, one);
        val2llvmMap[val] = builder.CreateUDiv(numerator, val2llvmMap[right]);
      }
    }
  }
  else if(val->isConst()) {
    val2llvmMap[val] = builder.getInt64(val->value().as<int64_t>());
  }
  else{
    NVF_THROW("LLVM Lowering Error: traverseExtentDFS called with non-binary op or constant Val.");
  }
  return val2llvmMap[val];
}

std::vector<llvm::Value*> getContiguousStrides(
    const std::vector<llvm::Value*>& sizes,
    const std::vector<bool>& expand_flags,
    llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& context = builder.getContext();
  NVF_ERROR(sizes.size() == expand_flags.size());

  std::vector<llvm::Value*> strides(sizes.size());
  llvm::Value* cur_stride = builder.getInt64(1);
  for (auto i = sizes.size(); i > 0; --i) {
    llvm::Value* size = sizes.at(i - 1);
    llvm::Value* stride = cur_stride;
    // If expanded, stride is 0
    if (expand_flags.at(i - 1)) {
      stride = builder.getInt64(0);
    } else {
      // Handle null size values by treating them as size 1 (safety check)
      if (size == nullptr) {
        size = builder.getInt64(1);
      }
      // Create comparison: size == 0
      llvm::Value* is_zero = builder.CreateICmpEQ(size, builder.getInt64(0));
      
      // Get current function for creating basic blocks
      llvm::Function* current_function = builder.GetInsertBlock()->getParent();
      
      // Create basic blocks
      llvm::BasicBlock* zero_block = llvm::BasicBlock::Create(context, "size_zero", current_function);
      llvm::BasicBlock* nonzero_block = llvm::BasicBlock::Create(context, "size_nonzero", current_function);
      llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(context, "stride_merge", current_function);
      
      // Conditional branch
      builder.CreateCondBr(is_zero, zero_block, nonzero_block);
      
      // Handle size == 0 case
      builder.SetInsertPoint(zero_block);
      llvm::Value* stride_if_zero = builder.getInt64(1);
      builder.CreateBr(merge_block);
      
      // Handle size != 0 case
      builder.SetInsertPoint(nonzero_block);
      llvm::Value* new_cur_stride = builder.CreateMul(cur_stride, size);
      builder.CreateBr(merge_block);
      
      // Merge the results
      builder.SetInsertPoint(merge_block);
      llvm::PHINode* stride_phi = builder.CreatePHI(llvm::Type::getInt64Ty(context), 2);
      stride_phi->addIncoming(stride_if_zero, zero_block);
      stride_phi->addIncoming(cur_stride, nonzero_block);
      stride = stride_phi;
      
      // Update cur_stride for next iteration
      llvm::PHINode* cur_stride_phi = builder.CreatePHI(llvm::Type::getInt64Ty(context), 2);
      cur_stride_phi->addIncoming(cur_stride, zero_block);  // Don't update if size is 0
      cur_stride_phi->addIncoming(new_cur_stride, nonzero_block);  // Update if size != 0
      cur_stride = cur_stride_phi;
    }

    strides.at(i - 1) = stride;
  }

  return strides;
}

// Infer the size and stride of each dimension
std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferShapeAndStridesRaw(
    const TensorView* tv,
    std::vector<Val*> symbolic_sizes,
    std::vector<bool> expand_flags,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::IRBuilder<>& builder) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferShape");
  std::vector<llvm::Value*> concrete_sizes(symbolic_sizes.size(), nullptr);

  for (const auto i : arange(symbolic_sizes.size())) {
    auto symbolic_size = symbolic_sizes.at(i);
    traverseExtentDFS(symbolic_size, val2llvmMap, builder);
    auto* inferred_val = val2llvmMap[symbolic_size];
    if(inferred_val == nullptr) {
      NVF_THROW("LLVM Lowering Error: inferred_val is nullptr for " + symbolic_size->toString());
    }
    concrete_sizes.at(i) = inferred_val;
  }

  auto strides = getContiguousStrides(concrete_sizes, expand_flags, builder);
  return {concrete_sizes, strides};
}

std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferShapeAndStridesNoReorder(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::IRBuilder<>& builder) {
  std::vector<Val*> symbolic_sizes;
  std::vector<bool> expand_flags;

  // Allocate the allocation domain
  for (const auto id : tv->getMaybeAllocationDomain()) {
    if (id->isReduction() || id->isStride()) {
      continue;
    }

    // Skip DIDx parallel domains to match inferTensorStrides filtering
    if (id->getParallelType() == ParallelType::DIDx || id->getParallelType() == ParallelType::DIDy || id->getParallelType() == ParallelType::DIDz) {
      continue;
    }

    if (id->isDeviceDim()) {
      symbolic_sizes.push_back(id->container()->oneVal());
    } else {
      symbolic_sizes.push_back(id->getMaybeExpandedExtent());
    }
    if (id->hasExpandedExtent()) {
      NVF_ERROR(
          id->isBroadcast(),
          "Non-broadcast domain should not have an expanded extent: ",
          id->toString());
      expand_flags.push_back(true);
    } else {
      expand_flags.push_back(false);
    }
  }
  return inferShapeAndStridesRaw(tv, symbolic_sizes, expand_flags, val2llvmMap, builder);
}

Val* mapToInputDomain(
    Val* currentDomain,
    std::unordered_map<Val*, bool>& boundaryVals
) {
  for(auto it = boundaryVals.begin(); it != boundaryVals.end(); ++it) { 
    auto* domain = it->first->as<IterDomain>();
    if(currentDomain->as<IterDomain>() == domain) {
      return it->first;
    }
  }
  return nullptr;
}

void generate_stride_llvm_ir(
    Val* current_val,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::IRBuilder<>& builder,
    llvm::Value*& running_stride_product,
    std::unordered_map<Val*, bool>& boundary_vals,
    std::vector<llvm::Value*>& strides
) {

    // Check if the current val is nullptr
  if (current_val == nullptr) {
      NVF_ERROR(false, "LLVM Lowering Error: generate_stride_llvm_ir called with nullptr Val.");
      return;
  }
    auto* def_expr = current_val->definition();
    // Check if the current val is missing
    if (def_expr == nullptr) {
      // Check if the current val is a boundary val
      Val* original_val = mapToInputDomain(current_val, boundary_vals);
      if(original_val != nullptr){
        // TODO: If the iter domain is a broadcast domain, then we have multiple inputs values pointing to the same valgroup
        // NVF_ERROR(!original_val->as<IterDomain>()->isBroadcast(), "LLVM Lowering Error: Broadcast domain is not supported in stride inference");
        if(boundary_vals[original_val] == false){
          boundary_vals[original_val] = true;
          strides.push_back(running_stride_product);
          running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[original_val->as<IterDomain>()->extent()], "mapped_stride");
        }
      }
      else if(current_val->as<IterDomain>()->extent()->isConst() && val2llvmMap.find(current_val->as<IterDomain>()->extent()) == val2llvmMap.end()){
        val2llvmMap[current_val->as<IterDomain>()->extent()] = builder.getInt64(current_val->as<IterDomain>()->extent()->value().as<int64_t>());
      }
      return;
    }

    // For each merge op, we need to check if it is valid split, we don't want to merge two values that has gaps in between
    if (def_expr->isA<Merge>()) {
        auto* merge_expr = def_expr->as<Merge>();
        auto* input_inner_val = merge_expr->inner()->as<Val>();
        auto* input_outer_val = merge_expr->outer()->as<Val>();
        auto* inner_mapped_val = mapToInputDomain(input_inner_val, boundary_vals);
        auto* outer_mapped_val = mapToInputDomain(input_outer_val, boundary_vals);
        // Check if the inner val is a boundary val
        if(inner_mapped_val != nullptr){
          if(boundary_vals[inner_mapped_val] == false){
            boundary_vals[inner_mapped_val] = true;
            strides.push_back(running_stride_product);
            running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[inner_mapped_val->as<IterDomain>()->extent()], "mapped_stride");
            return;
          }
        }
        else{
          generate_stride_llvm_ir(input_inner_val, val2llvmMap, builder, running_stride_product, boundary_vals, strides);
        }

        // Check if the outer val is a boundary val
        if(outer_mapped_val != nullptr){
          if(boundary_vals[outer_mapped_val] == false){
            boundary_vals[outer_mapped_val] = true;
            strides.push_back(running_stride_product);
            running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[outer_mapped_val->as<IterDomain>()->extent()], "mapped_stride");
            return;
          }
        }
        else{
          generate_stride_llvm_ir(input_outer_val, val2llvmMap, builder, running_stride_product, boundary_vals, strides);
        }
        
        // Extent of merged domain
        if(val2llvmMap[input_outer_val->as<IterDomain>()->extent()] == nullptr || val2llvmMap[input_inner_val->as<IterDomain>()->extent()] == nullptr || val2llvmMap[current_val->as<IterDomain>()->extent()] != nullptr){
          return;
        }
        else if(val2llvmMap.find(current_val->as<IterDomain>()->extent()) == val2llvmMap.end()){
          val2llvmMap[current_val->as<IterDomain>()->extent()] = builder.CreateMul(
              val2llvmMap[input_outer_val->as<IterDomain>()->extent()],
              val2llvmMap[input_inner_val->as<IterDomain>()->extent()],
              current_val->toString() + "mapped_extent"
          );
        }

    } else if (def_expr->isA<Split>()) {
        auto* split_expr = def_expr->as<Split>();
        auto* input_val = split_expr->in()->as<Val>();
        auto* output_inner_val = split_expr->inner()->as<Val>();
        auto* output_outer_val = split_expr->outer()->as<Val>();
        auto* input_mapped_val = mapToInputDomain(input_val, boundary_vals);

        if(input_mapped_val != nullptr){
          if(boundary_vals[input_mapped_val] == false){
            boundary_vals[input_mapped_val] = true;
            strides.push_back(running_stride_product);
            running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[input_mapped_val->as<IterDomain>()->extent()], "mapped_stride");
            return;
          }
        }
        else{
          generate_stride_llvm_ir(input_val, val2llvmMap, builder, running_stride_product, boundary_vals, strides);
        }

        auto* split_factor = split_expr->factor()->as<Val>();
        if(val2llvmMap.find(split_factor) == val2llvmMap.end()){
          val2llvmMap[split_factor] = traverseExtentDFS(split_factor, val2llvmMap, builder);
        }
        if(split_expr->innerSplit()){
          if(split_factor->isConstInt() && val2llvmMap.find(output_inner_val->as<IterDomain>()->extent()) == val2llvmMap.end()){
            val2llvmMap[output_inner_val->as<IterDomain>()->extent()] = builder.getInt64(split_factor->value().as<int64_t>());
          }
          else{
            if(val2llvmMap.find(split_factor) != val2llvmMap.end()){
              val2llvmMap[output_inner_val->as<IterDomain>()->extent()] = val2llvmMap[split_factor];
            }
            else{
              NVF_ERROR(false, "LLVM Lowering Error: Inner split factor is not a constant and not found in val2stride_map");
              return;
            }
          }
          if(val2llvmMap[input_val->as<IterDomain>()->extent()] == nullptr || val2llvmMap[output_inner_val->as<IterDomain>()->extent()] == nullptr || val2llvmMap[output_outer_val->as<IterDomain>()->extent()] != nullptr){
            return;
          }
          else if(val2llvmMap.find(output_inner_val->as<IterDomain>()->extent()) == val2llvmMap.end()){
          val2llvmMap[output_outer_val->as<IterDomain>()->extent()] = builder.CreateUDiv(
            val2llvmMap[input_val->as<IterDomain>()->extent()],
              val2llvmMap[output_inner_val->as<IterDomain>()->extent()],
              output_outer_val->toString() + "mapped_stride"
            );
          }
        }
        else{
          if(split_expr->factor()->isConstInt() && val2llvmMap.find(output_outer_val->as<IterDomain>()->extent()) == val2llvmMap.end()){
            val2llvmMap[output_outer_val->as<IterDomain>()->extent()] = builder.getInt64(split_factor->value().as<int64_t>());
          }
          else{
            if(val2llvmMap.find(split_factor) != val2llvmMap.end()){
              val2llvmMap[output_outer_val->as<IterDomain>()->extent()] = val2llvmMap[split_factor];
            }
            else{
              NVF_ERROR(false, "LLVM Lowering Error: Outer split factor is not a constant and not found in val2stride_map");
              return;
            }
          }
          if(val2llvmMap[input_val->as<IterDomain>()->extent()] == nullptr || val2llvmMap[output_inner_val->as<IterDomain>()->extent()] == nullptr || val2llvmMap[output_outer_val->as<IterDomain>()->extent()] != nullptr){
            return;
          }
          else if(val2llvmMap.find(output_inner_val->as<IterDomain>()->extent()) == val2llvmMap.end()){
          val2llvmMap[output_inner_val->as<IterDomain>()->extent()] = builder.CreateUDiv(
            val2llvmMap[input_val->as<IterDomain>()->extent()],
            val2llvmMap[output_outer_val->as<IterDomain>()->extent()],
            output_inner_val->toString() + "mapped_stride"
          );
          }
        }

    } else { // Fallback for other ops (e.g., simple unary pass-through)
        NVF_ERROR(false, "LLVM Lowering Error: Unhandled op_type '" + def_expr->toString() + "' for Val " + current_val->toString());
    }
}

// Infer the stride of each dimension
std::vector<llvm::Value*> inferTensorStridesReordered(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::IRBuilder<>& builder) {
  std::vector<llvm::Value*> strides;
  llvm::LLVMContext& context = builder.getContext();
  llvm::Value* running_stride = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1);
   std::unordered_map<Val*, bool> boundaryValVisited;
   auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
   for(auto* val : logical_domain) {
    boundaryValVisited[val] = false;
   }
   for(auto it = tv->getMaybeAllocationDomain().rbegin(); it != tv->getMaybeAllocationDomain().rend(); ++it) {
      auto iter_domain = *it;
      if(iter_domain->getParallelType() == ParallelType::DIDx || iter_domain->getParallelType() == ParallelType::DIDy || iter_domain->getParallelType() == ParallelType::DIDz) {
          continue;
      }
      generate_stride_llvm_ir(iter_domain->as<Val>(), val2llvmMap, builder, running_stride, boundaryValVisited, strides);
    }
  return strides;
}

std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferTensorShapesAndStridesNonAlias(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::IRBuilder<>& builder) {
  // Non-alias handling:
  auto allocation_size_stride = inferShapeAndStridesNoReorder(tv, val2llvmMap, builder);
  if (!tv->hasAllocation()) {
    return {allocation_size_stride.first, allocation_size_stride.second};
  }
  // otherwise we want return the reordered size and stride
  return {allocation_size_stride.first, inferTensorStridesReordered(tv, val2llvmMap,builder)};
}

std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferTensorShapesAndStrides(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::IRBuilder<>& builder) {
  // Alias handling, just return empty vector for now:
  auto alias_info = tv->fusion()->getOutputAlias(tv);
  if (alias_info.type != AllocationType::New) {
    NVF_THROW("Alias handling is not supported yet");
    return {std::vector<llvm::Value*>(), std::vector<llvm::Value*>()};
  }

  return inferTensorShapesAndStridesNonAlias(tv, val2llvmMap, builder);
}


// Helper function to generate LLVM IR that extracts tensor size for a given dimension
llvm::Value* generateTensorSizeExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& context = builder.getContext();
  auto mod = builder.GetInsertBlock()->getParent()->getParent();
  // Look up the tensor_size wrapper function
  llvm::Function* tensor_size_func = mod->getFunction("tensor_size");
  if (!tensor_size_func) {
    // Create function declaration
    llvm::FunctionType* func_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context),
      {llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), llvm::Type::getInt64Ty(context)},
      false
    );
    tensor_size_func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "tensor_size", mod
    );
  }
  
  llvm::Value* dim_val = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), dim);
  return builder.CreateCall(tensor_size_func, {tensor_ptr, dim_val});
}

// Helper function to generate LLVM IR that extracts tensor stride for a given dimension
llvm::Value* generateTensorStrideExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& context = builder.getContext();
  auto mod = builder.GetInsertBlock()->getParent()->getParent();
  // Look up the tensor_stride wrapper function
  llvm::Function* tensor_stride_func = mod->getFunction("tensor_stride");
  if (!tensor_stride_func) {
    // Create function declaration
    llvm::FunctionType* func_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context),
      {llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), llvm::Type::getInt64Ty(context)},
      false
    );
    tensor_stride_func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "tensor_stride", mod
    );
  }
  
  llvm::Value* dim_val = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), dim);
  return builder.CreateCall(tensor_stride_func, {tensor_ptr, dim_val});
}

HostIrJit::HostIrJit(std::unique_ptr<hir::HostIrContainer> container, int num_threads)
    : pimpl_(new LlvmJitImpl) {
  // Initialize params with passed parameters
  pimpl_->container_ = std::move(container);

  // Initialize LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  pimpl_->jit = throwIfError(
      llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());
  llvm::orc::JITDylib& dest_dynamic_lib = pimpl_->jit->getMainJITDylib();
  auto mangler = llvm::orc::MangleAndInterner(
      dest_dynamic_lib.getExecutionSession(), pimpl_->jit->getDataLayout());
  dest_dynamic_lib.addGenerator(throwIfError(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          pimpl_->jit->getDataLayout().getGlobalPrefix())));


  // raw tensor allocation and deallocation
  void* allocate_tensor_func_ptr = reinterpret_cast<void*>(
      +[](TensorView* tv, void* host_ir_jit_impl_ptr) -> at::Tensor* {
        auto* host_ir_jit_impl = static_cast<LlvmJitImpl*>(host_ir_jit_impl_ptr);
        NVF_ERROR(host_ir_jit_impl->tensor_map.find(tv) == host_ir_jit_impl->tensor_map.end(), "tensor_ptr is already allocated");
        host_ir_jit_impl->tensor_map[tv] = at::Tensor();
        return &host_ir_jit_impl->tensor_map[tv];
      });

  void* deallocate_tensor_func_ptr = reinterpret_cast<void*>(
      +[](TensorView* tv, void* host_ir_jit_impl_ptr) -> void {
        auto* host_ir_jit_impl = static_cast<LlvmJitImpl*>(host_ir_jit_impl_ptr);
        NVF_ERROR(host_ir_jit_impl->tensor_map.find(tv) != host_ir_jit_impl->tensor_map.end(), "tensor_ptr is not found");
        host_ir_jit_impl->tensor_map.erase(tv);
      });

  void* get_tensor_func_ptr = reinterpret_cast<void*>(
      +[](TensorView* tv, void* host_ir_jit_impl_ptr) -> at::Tensor* {
        auto* host_ir_jit_impl = static_cast<LlvmJitImpl*>(host_ir_jit_impl_ptr);
        return &host_ir_jit_impl->tensor_map[tv];
      });

  // native at:: functions
  void* empty_strided_cuda_func_ptr =
      reinterpret_cast<void*>(+[](const int64_t* sizes,
                                  int64_t ndim,
                                  const int64_t* strides,
                                  int64_t strides_ndim,
                                  int32_t dtype,
                                  int64_t device_index,
                                  at::Tensor& out_tensor) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::IntArrayRef aten_strides(strides, strides_ndim);
        at::ScalarType scalar_type = static_cast<at::ScalarType>(dtype);
        at::Device device =
            at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index));
        out_tensor = at::native::empty_strided_cuda(
            aten_sizes,
            aten_strides,
            scalar_type,
            c10::nullopt,
            device,
            c10::nullopt);
      });

  // matmul function
  void* matmul_out_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* t_out, at::Tensor* t_a, at::Tensor* t_b) {
        at::matmul_out(*t_out, *t_a, *t_b);
      });

  // linear function
  void* linear_out_func_ptr_with_bias = reinterpret_cast<void*>(
      +[](at::Tensor* t_out, at::Tensor* t_in, at::Tensor* t_weight, at::Tensor* t_bias) {
        at::linear_out(*t_out, *t_in, t_weight->squeeze(), t_bias->squeeze());
      });
  void* linear_out_func_ptr_without_bias = reinterpret_cast<void*>(
      +[](at::Tensor* t_out, at::Tensor* t_in, at::Tensor* t_weight) {
        at::linear_out(*t_out, *t_in, t_weight->squeeze());
      });

  // launch kernel function
  void* launch_kernel_func_ptr = reinterpret_cast<void*>(
      +[](size_t id, 
      at::Tensor** input_tensors, int64_t num_inputs, 
      at::Tensor** output_tensors, int64_t num_outputs,
      void* launch_kernel_ptr, void* host_ir_jit_impl_ptr) {
        auto* host_ir_jit_impl = static_cast<LlvmJitImpl*>(host_ir_jit_impl_ptr);
        auto launch_kernel = reinterpret_cast<hir::LaunchKernel*>(launch_kernel_ptr);
        KernelArgumentHolder input_args, output_args;
        input_args.setCacheId(id);
        for (int64_t i = 0; i < num_inputs; i++) {
          input_args.push(input_tensors[i]);
        }
        for (int64_t i = 0; i < num_outputs; i++) {
          output_args.push(output_tensors[i]);
        }
        input_args.setDeviceIndex();
        host_ir_jit_impl->container_->getKernelExecutor(launch_kernel->groupId())
            ->run(
                input_args,
                output_args,
                launch_kernel->launchParams(),
                launch_kernel->compileParams());
      });

  // tensor size and stride extraction functions
  void* extract_tensor_size_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* tensor_ptr, int64_t dim) -> int64_t {
        if (tensor_ptr == nullptr) {
          return 0;
        }
        return tensor_ptr->size(dim);
      });

  void* extract_tensor_stride_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* tensor_ptr, int64_t dim) -> int64_t {
        if (tensor_ptr == nullptr) {
          return 0;
        }
        return tensor_ptr->stride(dim);
      });

  // Register wrapper functions in JIT
  auto empty_strided_cuda_addr = llvm::orc::ExecutorAddr::fromPtr(empty_strided_cuda_func_ptr);
  auto launch_kernel_addr = llvm::orc::ExecutorAddr::fromPtr(launch_kernel_func_ptr);
  auto tensor_size_addr = llvm::orc::ExecutorAddr::fromPtr(extract_tensor_size_func_ptr);
  auto tensor_stride_addr = llvm::orc::ExecutorAddr::fromPtr(extract_tensor_stride_func_ptr);
  auto allocate_tensor_addr = llvm::orc::ExecutorAddr::fromPtr(allocate_tensor_func_ptr);
  auto deallocate_tensor_addr = llvm::orc::ExecutorAddr::fromPtr(deallocate_tensor_func_ptr);
  auto get_tensor_addr = llvm::orc::ExecutorAddr::fromPtr(get_tensor_func_ptr);
  auto matmul_out_addr = llvm::orc::ExecutorAddr::fromPtr(matmul_out_func_ptr);
  auto linear_out_addr_with_bias = llvm::orc::ExecutorAddr::fromPtr(linear_out_func_ptr_with_bias);
  auto linear_out_addr_without_bias = llvm::orc::ExecutorAddr::fromPtr(linear_out_func_ptr_without_bias);
  // Register wrapper functions in JIT
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler(kHostIrJitEmptyStridedCudaFuncName)] =
      llvm::orc::ExecutorSymbolDef(
          empty_strided_cuda_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("launch_kernel")] = 
      llvm::orc::ExecutorSymbolDef(launch_kernel_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("tensor_size")] = 
      llvm::orc::ExecutorSymbolDef(tensor_size_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("tensor_stride")] = 
      llvm::orc::ExecutorSymbolDef(tensor_stride_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("allocate_tensor")] = 
      llvm::orc::ExecutorSymbolDef(allocate_tensor_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("deallocate_tensor")] = 
      llvm::orc::ExecutorSymbolDef(deallocate_tensor_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("get_tensor")] = 
      llvm::orc::ExecutorSymbolDef(get_tensor_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("matmul_out")] = 
      llvm::orc::ExecutorSymbolDef(matmul_out_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("linear_out_with_bias")] = 
      llvm::orc::ExecutorSymbolDef(linear_out_addr_with_bias, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("linear_out_without_bias")] = 
      llvm::orc::ExecutorSymbolDef(linear_out_addr_without_bias, llvm::JITSymbolFlags::Exported);
  throwIfError(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));

  // Compile the module
  compile(
      pimpl_->container_.get(),
      pimpl_->jit.get(),
      pimpl_->main_func_,
      pimpl_.get());
}

HostIrJit::~HostIrJit() = default;

KernelArgumentHolder HostIrJit::runWithInputs(
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("HostIrJit::runWithInputs");
  // Bind cache id to llvm global variable
  NVF_ERROR(args.getCacheId().has_value(), "Cache ID is not set");
  NVF_ERROR_EQ(std::ssize(pimpl_->container_->inputs()), args.size());
  // Bind the inputs to the tensor map

  for (auto&& [in_val, arg] : zip(pimpl_->container_->inputs(), args)) {
    if (arg.is<at::Tensor>()) {
      pimpl_->tensor_map[in_val->as<TensorView>()] = arg.as<at::Tensor>();
    }
    else{
      // TODO: handle other primitive types so we can just align them to input of main function
    }
  }
  // Run the main function
  pimpl_->main_func_();
  // Collect the outputs
  KernelArgumentHolder outputs;
  for(auto* output : pimpl_->container_->outputs()) {
    if(auto* tv = dynamic_cast<const TensorView*>(output)) {
      outputs.push(pimpl_->tensor_map[tv]);
    }
  }
  
  // Clear the entire tensor map after collecting outputs, garbage collect intermediate tensors
  pimpl_->tensor_map.clear();
  
  return outputs;
}



} // namespace nvfuser
