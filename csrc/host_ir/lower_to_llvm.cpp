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
#include <host_ir/container.h>
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
#include <host_ir/lower_to_llvm.h>

namespace nvfuser {

using at_empty_strided_func = at::Tensor(*)(const std::vector<int64_t>&, const std::vector<int64_t>&, const at::TensorOptions&);
using at_empty_func = at::Tensor(*)(const std::vector<int64_t>&);
using JitFuncPtr = void* (*)(int64_t*, int64_t, void*);

// PIMPL implementation for HostIrLlvmJit
struct HostIrLlvmJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unordered_map<const kir::Allocate*, JitFuncPtr> allocate_funcs_;
};

// Helper function to exit on error on LLVM JIT initialization
template <typename T>
T ExitOnErr(llvm::Expected<T>&& E) {
  if (!E) {
    NVF_ERROR(
        false,
        "LLVM JIT Initialization Error: " + llvm::toString(E.takeError()));
    llvm::errs() << llvm::toString(E.takeError()) << "\n";
    exit(1);
  }
  return std::move(*E);
}

inline void ExitOnErr(llvm::Error&& Err) {
  if (Err) {
    NVF_ERROR(
        false,
        "LLVM JIT Initialization Error: " + llvm::toString(std::move(Err)));
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
    exit(1);
  }
}

llvm::orc::ThreadSafeModule generateAllocateFunc(const kir::Allocate* allocate) {
  // Create LLVM context and module
  auto ctx = std::make_unique<llvm::LLVMContext>();
  llvm::LLVMContext& context = *ctx;
  std::unique_ptr<llvm::Module> mod = std::make_unique<llvm::Module>("allocate_func", context);
  llvm::IRBuilder<> builder(context);
  
  // Define function signature: at::Tensor func(const std::vector<int64_t>& strides)
  llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  llvm::Type* int64_ptr_type = int64_type->getPointerTo();
  
  // Create function type: at::Tensor (*)(const std::vector<int64_t>&)
  // For simplicity, we'll use void* for at::Tensor return type
  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  llvm::FunctionType* func_type = llvm::FunctionType::get(
      void_ptr_type, {int64_ptr_type, int64_type, void_ptr_type}, false);
  
  // Create the function
  llvm::Function* func = llvm::Function::Create(
      func_type, 
      llvm::Function::ExternalLinkage, 
      "create_tensor_from_sizes", 
      mod.get()
  );
  
  // Create basic block
  llvm::BasicBlock* entry_block = llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_block);
  
  // Get function arguments
  llvm::Value* sizes_arg = func->getArg(0); // int64_t* (buffer)
  llvm::Value* ndim_arg = func->getArg(1);  // int64_t   (length)
  llvm::Value* options_arg = func->getArg(2); // void*    (TensorOptions*)
  
  // Get the at::empty function pointer (registered in the JIT)
  llvm::Function* empty_func = mod->getFunction("at::empty");
  if (!empty_func) {
    llvm::FunctionType* empty_type = llvm::FunctionType::get(
        void_ptr_type, {int64_ptr_type, int64_type, void_ptr_type}, false
    );
    empty_func = llvm::Function::Create(
        empty_type,
        llvm::Function::ExternalLinkage,
        "at::empty",
        mod.get()
    );
  }
  
  // Call at::empty
  llvm::Value* result = builder.CreateCall(
      empty_func,
      {sizes_arg, ndim_arg, options_arg}
  );
  
  // Return the result
  builder.CreateRet(result);
  
  mod->print(llvm::errs(), nullptr);
  // Verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  if (llvm::verifyModule(*mod, &error_stream)) {
    NVF_ERROR(false, "LLVM module verification failed: " + error);
  }
  llvm::outs() << "=== LLVM IR ===\n";
  mod->print(llvm::outs(), nullptr);

  return llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx));
}



// Constructor implementation
HostIrLlvmJit::HostIrLlvmJit(int num_threads) : pimpl_(new LlvmJitImpl) {
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

  // Disambiguate the overload using a lambda:
  void* empty_strided_func_ptr = reinterpret_cast<void*>(
      +[](at::IntArrayRef a, at::IntArrayRef b, const at::TensorOptions& c) {
        return at::empty_strided(a, b, c);
      });

  void* empty_func_ptr = reinterpret_cast<void*>(
      +[](int64_t* data, int64_t len, void* options) {
        at::IntArrayRef sizes(data, len);
        at::TensorOptions opts = options ? *reinterpret_cast<at::TensorOptions*>(options) : at::TensorOptions();
        return new at::Tensor(at::empty(sizes, opts));
      });

  auto empty_strided_addr = llvm::orc::ExecutorAddr::fromPtr(empty_strided_func_ptr);
  auto empty_addr = llvm::orc::ExecutorAddr::fromPtr(empty_func_ptr);
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler("at::empty_strided")] =
      llvm::orc::ExecutorSymbolDef(empty_strided_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("at::empty")] =
      llvm::orc::ExecutorSymbolDef(empty_addr, llvm::JITSymbolFlags::Exported);


  ExitOnErr(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));
}

// The destructor must be defined here where LlvmJitImpl is a complete type.
HostIrLlvmJit::~HostIrLlvmJit() = default;

// Move constructor and assignment operator
HostIrLlvmJit::HostIrLlvmJit(HostIrLlvmJit&&) noexcept = default;
HostIrLlvmJit& HostIrLlvmJit::operator=(HostIrLlvmJit&&) noexcept = default;

void HostIrLlvmJit::compile(const hir::HostIrContainer* container) {
  FUSER_PERF_SCOPE("HostIrLlvmJit::compile");
  for (auto expr : container->topLevelExprs()) {
    if (auto allocate = dynamic_cast<const kir::Allocate*>(expr)) {
      llvm::orc::ThreadSafeModule mod = generateAllocateFunc(allocate);
      ExitOnErr(pimpl_->jit->addIRModule(std::move(mod)));
      auto func_addr = ExitOnErr(pimpl_->jit->lookup("create_tensor_from_sizes"));
      auto func_ptr = func_addr.toPtr<JitFuncPtr>();
      pimpl_->allocate_funcs_[allocate] = func_ptr;
    }
  }
}

at::Tensor HostIrLlvmJit::allocate(const kir::Allocate* allocate, const std::vector<int64_t>& input_sizes) {
  if (!pimpl_->allocate_funcs_[allocate]) {
    NVF_ERROR(false, "create_tensor_from_sizes function not found");
  }
  auto func_ptr = pimpl_->allocate_funcs_[allocate];
  at::TensorOptions opts = at::TensorOptions().device(at::kCUDA);
  void* result = func_ptr(const_cast<int64_t*>(input_sizes.data()), input_sizes.size(), reinterpret_cast<void*>(&opts));
  return *reinterpret_cast<at::Tensor*>(result);
}

HostIrLlvmJit& HostIrLlvmJit::getInstance(int num_threads) {
  static HostIrLlvmJit instance(num_threads);
  return instance;
}

} // namespace nvfuser
