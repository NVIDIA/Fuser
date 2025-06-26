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

using allocate_fn =
    void (*)(const int64_t*, int64_t, const int64_t*, int64_t, at::Tensor&);

class HostIrJitParams {
  private:
  Communicator* communicator_;
  hir::HostIrEvaluatorParams evaluator_params_;
  

  public:
  HostIrJitParams(hir::HostIrContainer* container, Communicator* communicator, hir::HostIrEvaluatorParams evaluator_params) : 
  communicator_(communicator), 
  evaluator_params_(evaluator_params){}
  
  // Getters for accessing the parameters
  Communicator* getCommunicator() const { return communicator_; }
  const hir::HostIrEvaluatorParams& getEvaluatorParams() const { return evaluator_params_; }
};

// PIMPL implementation for HostIrJit
struct HostIrJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unordered_map<const kir::Allocate*, allocate_fn> allocate_funcs_;
  std::unique_ptr<HostIrJitParams> host_ir_jit_params_;
  LlvmJitImpl() = default;
  ~LlvmJitImpl() = default; // unique_ptr handles cleanup automatically
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

// Generate a function that calls at::native::empty_strided_cuda
void generateAllocateFunc(
    const kir::Allocate* allocate,
    llvm::Module* mod,
    const HostIrJitParams& host_ir_jit_params) {
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
      {int64_ptr_type,
       int64_type,
       int64_ptr_type,
       int64_type,
       void_ptr_type},
      false);

  std::string func_name = hostIrJitAllocateFuncName + "_" +
      std::to_string(reinterpret_cast<uintptr_t>(allocate));
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
  llvm::Value* device_index_constant =
      llvm::ConstantInt::get(int64_type, host_ir_jit_params.getCommunicator()->deviceId());

  // Get the at::native::empty_strided_cuda function pointer (registered in the
  // JIT)
  llvm::Function* at_empty_strided_cuda_func =
      mod->getFunction("at::native::empty_strided_cuda");
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
        "at::native::empty_strided_cuda",
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

  // Return the result
  builder.CreateRetVoid();

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

void compile(const hir::HostIrContainer* container, llvm::orc::LLJIT* jit, std::unordered_map<const kir::Allocate*, allocate_fn>& allocate_funcs_, const HostIrJitParams& host_ir_jit_params) {
  FUSER_PERF_SCOPE("HostIrJit::compile");
  if (allocate_funcs_.size() > 0) {
    return;
  }
  
  if (container == nullptr) {
    NVF_ERROR(false, "container is nullptr during host ir JIT compilation");
    return;
  }
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>(
      "host_ir_container_" +
          std::to_string(reinterpret_cast<uintptr_t>(container)),
      *ctx);
  
  std::unordered_map<const kir::Allocate*, std::string> allocate_func_names;
  for (auto expr : container->topLevelExprs()) {
    if (auto allocate = dynamic_cast<const kir::Allocate*>(expr)) {
      // Generate a unique function name for this allocate
      generateAllocateFunc(allocate, mod.get(), host_ir_jit_params);
      // Store the mapping from allocate to function name
      allocate_func_names[allocate] = hostIrJitAllocateFuncName + "_" +
          std::to_string(reinterpret_cast<uintptr_t>(allocate));
    }
    else if (auto for_loop = dynamic_cast<const ForLoop*>(expr)) {
      for (auto expr : for_loop->body().exprs()) {
        if (auto allocate = dynamic_cast<const kir::Allocate*>(expr)) {
          generateAllocateFunc(allocate, mod.get(), host_ir_jit_params);
          allocate_func_names[allocate] = hostIrJitAllocateFuncName + "_" +
          std::to_string(reinterpret_cast<uintptr_t>(allocate));
        }
      }
    }
  }

  // Add the module to the JIT
  ExitOnErr(jit->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx))));

  // Look up all functions and store their pointers
  for (const auto& [allocate, func_name] : allocate_func_names) {
    auto func_addr = ExitOnErr(jit->lookup(func_name));
    // Lookup and reinterpret the function pointer to store in the map
    allocate_funcs_[allocate] = (allocate_fn)reinterpret_cast<void (*)(
        const int64_t*, int64_t, const int64_t*, int64_t, at::Tensor&)>(
        func_addr.getValue());
  }
}

HostIrJit::HostIrJit(
    hir::HostIrContainer* container,
    Communicator* communicator,
    const hir::HostIrEvaluatorParams& evaluator_params,
    int num_threads)
    : pimpl_(new LlvmJitImpl) {
  // Initialize params with passed parameters
  pimpl_->host_ir_jit_params_ =
      std::make_unique<HostIrJitParams>(container, communicator, evaluator_params);

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

  // Create wrapper function pointer to at::native::empty_strided_cuda
  // TODO: Remove this wrapper in the future
  void* empty_strided_cuda_func_ptr = reinterpret_cast<void*>(
      +[](const int64_t* sizes,
          int64_t ndim,
          const int64_t* strides,
          int64_t strides_ndim,
          int32_t dtype,
          int64_t device_index,
          at::Tensor& out_tensor) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::IntArrayRef aten_strides(strides, strides_ndim);
        // Use the type and device passed as parameters
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

  // Register at::native::empty_strided_cuda function in LLVM
  auto empty_strided_cuda_addr =
      llvm::orc::ExecutorAddr::fromPtr(empty_strided_cuda_func_ptr);
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler("at::native::empty_strided_cuda")] = llvm::orc::ExecutorSymbolDef(
      empty_strided_cuda_addr, llvm::JITSymbolFlags::Exported);

  ExitOnErr(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));
  
  // Only compile if container is provided
  if (container) {
    compile(container, pimpl_->jit.get(), pimpl_->allocate_funcs_, *pimpl_->host_ir_jit_params_);
  }
}

HostIrJit::~HostIrJit() = default;

at::Tensor HostIrJit::allocate(
    const kir::Allocate* allocate,
    const std::vector<int64_t>& input_sizes,
    const std::vector<int64_t>& input_strides) {
  if (pimpl_->allocate_funcs_.find(allocate) == pimpl_->allocate_funcs_.end()) {
    NVF_ERROR(false, "allocate function not found for ", allocate);
  }
  FUSER_PERF_SCOPE("HostIrJit::allocate");
  auto func_ptr = pimpl_->allocate_funcs_[allocate];

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
