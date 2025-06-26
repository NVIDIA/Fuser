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
#include <memory>

// Include the actual header files instead of forward declarations
#include <multidevice/communicator.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

using allocate_fn = std::function<
    void(const int64_t*, int64_t, const int64_t*, int64_t, at::Tensor&)>;

class HostIrJitParams {
 private:
  Communicator* communicator_;

 public:
  HostIrJitParams(hir::HostIrContainer* container)
      : communicator_(&Communicator::getInstance()) {}

  Communicator* getCommunicator() const {
    return communicator_;
  }
};

// PIMPL implementation for HostIrJit
struct HostIrJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unordered_map<const kir::Allocate*, allocate_fn> allocate_funcs_;
  std::unique_ptr<HostIrJitParams> host_ir_jit_params_;
  LlvmJitImpl() = default;
  ~LlvmJitImpl() = default; // unique_ptr handles cleanup automatically
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

// Generate kir::Allocate runtime function
void generateAllocateFunc(
    const kir::Allocate* allocate,
    const HostIrJitParams& host_ir_jit_params,
    llvm::Module* mod) {
  llvm::LLVMContext& context = mod->getContext();

  // Define function signature: void(i64*, i64, i64*, i64, void*)
  // The last void* is for at::Tensor&
  llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  llvm::Type* int64_ptr_type = int64_type->getPointerTo();
  llvm::Type* int32_type = llvm::Type::getInt32Ty(context);
  llvm::PointerType* void_ptr_type =
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  llvm::FunctionType* func_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context),
      {int64_ptr_type, int64_type, int64_ptr_type, int64_type, void_ptr_type},
      false);

  std::string func_name = ir_utils::varName(allocate->buffer()->as<Val>());
  llvm::Function* func = llvm::Function::Create(
      func_type,
      llvm::Function::ExternalLinkage,
      func_name, // Use the generated unique name
      mod);

  // Set argument names for better readability in IR
  func->getArg(0)->setName("sizes");
  func->getArg(1)->setName("ndim");
  func->getArg(2)->setName("strides");
  func->getArg(3)->setName("strides_ndim");
  func->getArg(4)->setName("out_tensor");

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", func);
  llvm::IRBuilder<> builder(entry);

  // Get arguments from the function
  llvm::Value* sizes_arg = func->getArg(0); // const int64_t*
  llvm::Value* ndim_arg = func->getArg(1); // int64_t
  llvm::Value* strides_arg = func->getArg(2); // const int64_t*
  llvm::Value* strides_ndim_arg = func->getArg(3); // int64_t (strides_ndim)
  llvm::Value* out_tensor_arg = func->getArg(4); // at::Tensor&

  // Bounds checking for ndim
  auto logical_domain = TensorDomain::noReductions(
      allocate->buffer()->as<TensorView>()->getLogicalDomain());
  size_t compile_time_ndim = logical_domain.size();
  llvm::Value* compile_time_ndim_val =
      llvm::ConstantInt::get(int64_type, compile_time_ndim);
  llvm::Value* cmp =
      builder.CreateICmpEQ(ndim_arg, compile_time_ndim_val, "ndim_check");
  llvm::Function* parent_func = builder.GetInsertBlock()->getParent();
  llvm::BasicBlock* then_bb =
      llvm::BasicBlock::Create(context, "if.then", parent_func);
  llvm::BasicBlock* else_bb =
      llvm::BasicBlock::Create(context, "if.else", parent_func);
  builder.CreateCondBr(cmp, then_bb, else_bb);
  builder.SetInsertPoint(else_bb);
  llvm::Function* trap_func =
      llvm::Intrinsic::getDeclaration(mod, llvm::Intrinsic::trap);
  builder.CreateCall(trap_func, {});
  builder.CreateUnreachable();
  builder.SetInsertPoint(then_bb);

  // Create constants for type and device from params
  at::ScalarType data_type = data_type_to_aten(
      allocate->buffer()->dtype() == DataType::Index
          ? PrimDataType::Int
          : allocate->buffer()->dtype());
  llvm::Value* dtype_constant =
      llvm::ConstantInt::get(int32_type, static_cast<int32_t>(data_type));
  llvm::Value* device_index_constant = llvm::ConstantInt::get(
      int64_type, host_ir_jit_params.getCommunicator()->deviceId());

  // Get the at::native::empty_strided_cuda function pointer (registered in the
  // JIT)
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

  // Call at::native::empty_strided_cuda with constants
  builder.CreateCall(
      at_empty_strided_cuda_func,
      {sizes_arg,
       ndim_arg,
       strides_arg,
       strides_ndim_arg,
       dtype_constant,
       device_index_constant,
       out_tensor_arg});

  builder.CreateRetVoid();

  // Verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(
      !llvm::verifyModule(*mod, &error_stream),
      "LLVM module verification failed: " + error);

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR ===\n";
    mod->print(llvm::outs(), nullptr);
  }
}

void compile(
    const hir::HostIrContainer* container,
    llvm::orc::LLJIT* jit,
    std::unordered_map<const kir::Allocate*, allocate_fn>& allocate_funcs_,
    const HostIrJitParams& host_ir_jit_params) {
  FUSER_PERF_SCOPE("HostIrJit::compile");
  // If the JIT is already compiled, return
  if (allocate_funcs_.size() > 0) {
    return;
  }
  NVF_ERROR(
      container != nullptr,
      "container is nullptr during host ir JIT compilation");
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("host_ir_jit_module", *ctx);
  std::vector<kir::Allocate*> allocate_exprs;
  for (auto* expr : container->topLevelExprs()) {
    if (auto* allocate = dynamic_cast<kir::Allocate*>(expr)) {
      generateAllocateFunc(allocate, host_ir_jit_params, mod.get());
      allocate_exprs.push_back(allocate);
    } else if (auto* for_loop = dynamic_cast<ForLoop*>(expr)) {
      for (auto* expr : for_loop->body().exprs()) {
        if (auto* allocate = dynamic_cast<kir::Allocate*>(expr)) {
          generateAllocateFunc(allocate, host_ir_jit_params, mod.get());
          allocate_exprs.push_back(allocate);
        }
      }
    }
  }

  // Add the module to the JIT
  throwIfError(jit->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx))));

  // Look up all functions and store their pointers
  for (auto* allocate : allocate_exprs) {
    auto func_name = ir_utils::varName(allocate->buffer()->as<Val>());
    auto func_addr = throwIfError(jit->lookup(func_name));
    allocate_funcs_[allocate] = reinterpret_cast<void (*)(
        const int64_t*, int64_t, const int64_t*, int64_t, at::Tensor&)>(
        func_addr.getValue());
  }
}

HostIrJit::HostIrJit(hir::HostIrContainer* container, int num_threads)
    : pimpl_(new LlvmJitImpl) {
  // Initialize params with passed parameters
  pimpl_->host_ir_jit_params_ = std::make_unique<HostIrJitParams>(container);

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

  // Create wrapper function for at::native::empty_strided_cuda
  // TODO: Remove this wrapper in the future
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

  // Register wrapper functions in JIT
  auto empty_strided_cuda_addr =
      llvm::orc::ExecutorAddr::fromPtr(empty_strided_cuda_func_ptr);
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler(kHostIrJitEmptyStridedCudaFuncName)] =
      llvm::orc::ExecutorSymbolDef(
          empty_strided_cuda_addr, llvm::JITSymbolFlags::Exported);

  throwIfError(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));

  // Compile the module
  compile(
      container,
      pimpl_->jit.get(),
      pimpl_->allocate_funcs_,
      *pimpl_->host_ir_jit_params_);
}

HostIrJit::~HostIrJit() = default;

at::Tensor HostIrJit::allocate(
    const kir::Allocate* allocate,
    const std::vector<int64_t>& input_sizes,
    const std::vector<int64_t>& input_strides) {
  FUSER_PERF_SCOPE("HostIrJit::allocate");
  auto allocate_func_iter = pimpl_->allocate_funcs_.find(allocate);
  NVF_ERROR(
      allocate_func_iter != pimpl_->allocate_funcs_.end(),
      "allocate function not found for ",
      allocate);
  auto& func_ptr = allocate_func_iter->second;

  at::Tensor tensor;
  func_ptr(
      input_sizes.data(),
      input_sizes.size(),
      input_strides.data(),
      input_strides.size(),
      tensor);

  return tensor;
}

} // namespace nvfuser
