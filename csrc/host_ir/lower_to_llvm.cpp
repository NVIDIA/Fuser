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

// PIMPL implementation for HostIrLlvmJit
struct HostIrLlvmJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
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
  void* func_ptr = reinterpret_cast<void*>(
      +[](at::IntArrayRef a, at::IntArrayRef b, const at::TensorOptions& c) {
        return at::empty_strided(a, b, c);
      });

  auto addr = llvm::orc::ExecutorAddr::fromPtr(func_ptr);
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler("at::empty_strided")] =
      llvm::orc::ExecutorSymbolDef(addr, llvm::JITSymbolFlags::Exported);
  ExitOnErr(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));
}

// The destructor must be defined here where LlvmJitImpl is a complete type.
HostIrLlvmJit::~HostIrLlvmJit() = default;

// Move constructor and assignment operator
HostIrLlvmJit::HostIrLlvmJit(HostIrLlvmJit&&) noexcept = default;
HostIrLlvmJit& HostIrLlvmJit::operator=(HostIrLlvmJit&&) noexcept = default;

void HostIrLlvmJit::compile(const hir::HostIrContainer* container) {
  FUSER_PERF_SCOPE("HostIrLlvmJit::compile");
}

HostIrLlvmJit& HostIrLlvmJit::getInstance(int num_threads) {
  static HostIrLlvmJit instance(num_threads);
  return instance;
}

} // namespace nvfuser
