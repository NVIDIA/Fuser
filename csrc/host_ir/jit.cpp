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
#include <chrono>
#include <queue>
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
#include <c10/core/MemoryFormat.h> // for c10::optional
#include <host_ir/jit.h>
#include <functional>

namespace nvfuser {

using allocate_fn = std::function<void*(int64_t*, int64_t, void*)>;

// PIMPL implementation for HostIrJit
struct HostIrJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unordered_map<const kir::Allocate*, allocate_fn> allocate_funcs_;
};

// Helper function to exit on error on LLVM JIT initialization
template <typename T>
T ExitOnErr(llvm::Expected<T>&& E) {
  if (!E) {
    NVF_ERROR(
        false,
        "LLVM JIT Initialization Error: ",
        llvm::toString(E.takeError()));
    exit(1);
  }
  return std::move(*E);
}

inline void ExitOnErr(llvm::Error&& Err) {
  if (Err) {
    NVF_ERROR(
        false,
        "LLVM JIT Initialization Error: " + llvm::toString(std::move(Err)));
    exit(1);
  }
}

// Generate a function that calls at::empty_strided or at::empty
void generateAllocateFunc(
    const kir::Allocate* allocate,
    llvm::Module* mod) {
  llvm::LLVMContext& context = mod->getContext();
  llvm::IRBuilder<> builder(context);

  std::string func_name = "create_tensor_from_sizes_" +
          std::to_string(reinterpret_cast<uintptr_t>(allocate));
  // Define function signature: at::Tensor func(const std::vector<int64_t>&
  // strides)
  llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  llvm::Type* int64_ptr_type = int64_type->getPointerTo();

  // Create function type: at::Tensor (*)(const std::vector<int64_t>&)
  // For simplicity, we'll use void* for at::Tensor return type
  llvm::PointerType* void_ptr_type =
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  llvm::FunctionType* func_type = llvm::FunctionType::get(
      void_ptr_type, {int64_ptr_type, int64_type, void_ptr_type}, false);

  // Create the function with the custom name
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, func_name, mod);

  // Create basic block
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_block);

  // Get function arguments
  llvm::Value* sizes_arg = func->getArg(0); // int64_t* (buffer)
  llvm::Value* ndim_arg = func->getArg(1); // int64_t   (length)
  llvm::Value* options_arg = func->getArg(2); // void*    (TensorOptions*)

  // Get the at::empty function pointer (registered in the JIT)
  llvm::Function* at_empty_func = mod->getFunction("at::empty");
  if (at_empty_func == nullptr) {
    llvm::FunctionType* at_empty_func_type = llvm::FunctionType::get(
        void_ptr_type, {int64_ptr_type, int64_type, void_ptr_type}, false);
    at_empty_func = llvm::Function::Create(
        at_empty_func_type, llvm::Function::ExternalLinkage, "at::empty", mod);
  }

  // Call at::empty
  llvm::Value* result =
      builder.CreateCall(at_empty_func, {sizes_arg, ndim_arg, options_arg});

  // Return the result
  builder.CreateRet(result);

  // Verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(!llvm::verifyModule(*mod, &error_stream), "LLVM module verification failed: " + error);

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    // Print the LLVM IR module
    llvm::outs() << "=== LLVM IR ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

void compile(const hir::HostIrContainer* container, llvm::orc::LLJIT* jit, std::unordered_map<const kir::Allocate*, allocate_fn>& allocate_funcs_) {
  if (allocate_funcs_.size() > 0) {
    return;
  }
  if (container == nullptr) {
    NVF_ERROR(false, "container is nullptr during host ir JIT compilation");
    return;
  }
  FUSER_PERF_SCOPE("HostIrJit::compile");
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>(
      "host_ir_container_" +
          std::to_string(reinterpret_cast<uintptr_t>(container)),
      *ctx);
  std::unordered_map<const kir::Allocate*, std::string> allocate_func_names;
  for (auto expr : container->topLevelExprs()) {
    if (auto allocate = dynamic_cast<const kir::Allocate*>(expr)) {
      // Generate a unique function name for this allocate
      generateAllocateFunc(allocate, mod.get());
      // Store the mapping from allocate to function name
      allocate_func_names[allocate] = "create_tensor_from_sizes_" +
          std::to_string(reinterpret_cast<uintptr_t>(allocate));
    }
  }

  // Add the module to the JIT
  ExitOnErr(jit->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx))));

  // Look up all functions and store their pointers
  for (const auto& [allocate, func_name] : allocate_func_names) {
    auto func_addr = ExitOnErr(jit->lookup(func_name));
    // Lookup and reinterpret the function pointer to store in the map
    allocate_funcs_[allocate] = allocate_fn(reinterpret_cast<void*(*)(int64_t*, int64_t, void*)>(func_addr.getValue()));
  }
}

// Constructor implementation
HostIrJit::HostIrJit(hir::HostIrContainer* container, int num_threads) : pimpl_(new LlvmJitImpl) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  pimpl_->jit = ExitOnErr(
      llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());
  llvm::orc::JITDylib& dest_dynamic_lib = pimpl_->jit->getMainJITDylib();
  auto mangler = llvm::orc::MangleAndInterner(
      dest_dynamic_lib.getExecutionSession(), pimpl_->jit->getDataLayout());
  dest_dynamic_lib.addGenerator(
      ExitOnErr(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          pimpl_->jit->getDataLayout().getGlobalPrefix())));

  // Create wrapper function pointers to at::empty_strided and at::empty
  void* empty_strided_func_ptr = reinterpret_cast<void*>(
      +[](int64_t* sizes, int64_t ndim, int64_t* strides, void* options) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::IntArrayRef aten_strides(strides, ndim);
        at::TensorOptions opts = options
            ? *reinterpret_cast<at::TensorOptions*>(options)
            : at::TensorOptions();
        return new at::Tensor(
            at::empty_strided(aten_sizes, aten_strides, opts));
      });

  void* empty_func_ptr =
      reinterpret_cast<void*>(+[](int64_t* sizes, int64_t ndim, void* options) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::TensorOptions opts = options
            ? *reinterpret_cast<at::TensorOptions*>(options)
            : at::TensorOptions();
        return new at::Tensor(at::empty(aten_sizes, opts));
      });

  // Register at::empty_strided and at::empty functions in LLVM
  auto empty_strided_addr =
      llvm::orc::ExecutorAddr::fromPtr(empty_strided_func_ptr);
  auto empty_addr = llvm::orc::ExecutorAddr::fromPtr(empty_func_ptr);
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler("at::empty_strided")] = llvm::orc::ExecutorSymbolDef(
      empty_strided_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("at::empty")] =
      llvm::orc::ExecutorSymbolDef(empty_addr, llvm::JITSymbolFlags::Exported);

  ExitOnErr(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));
  compile(container, pimpl_->jit.get(), pimpl_->allocate_funcs_);
}

HostIrJit::~HostIrJit() = default;

at::Tensor HostIrJit::allocate(
    const kir::Allocate* allocate,
    const std::vector<int64_t>& input_sizes) {
  if (pimpl_->allocate_funcs_.find(allocate) == pimpl_->allocate_funcs_.end()) {
    NVF_ERROR(false, "allocate function not found for ", allocate);
  }
  auto func_ptr = pimpl_->allocate_funcs_[allocate];
  at::TensorOptions opts = at::TensorOptions().device(at::kCUDA);
  void* result = func_ptr(
      const_cast<int64_t*>(input_sizes.data()),
      input_sizes.size(),
      reinterpret_cast<void*>(&opts));
  return *reinterpret_cast<at::Tensor*>(result);
}

} // namespace nvfuser
