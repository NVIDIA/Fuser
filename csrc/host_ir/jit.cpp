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
    at::Tensor* (*)(const int64_t*, int64_t, const int64_t*, int64_t);

class HostIrJitParams {
  private:
  Communicator* communicator_;
  hir::HostIrEvaluatorParams params_;
  // Cache Fusions, KernelExecutors
  std::unordered_map<hir::HostUnit*, std::unique_ptr<ExecutorAbstract>> executors_;
  std::unordered_map<hir::HostUnit*, FusionExecutorCache> fec_;
  using StreamKey = std::variant<int64_t, hir::Stream*>;
  std::unordered_map<StreamKey, c10::cuda::CUDAStream> streams_;
  std::unordered_map<Expr*, c10::intrusive_ptr<c10d::Work>> works_;
  const int64_t my_local_device_index_;
  // IpcHandleCache ipc_handle_cache_;

  public:
  HostIrJitParams(hir::HostIrContainer* container, Communicator* communicator, hir::HostIrEvaluatorParams params) : 
  communicator_(communicator), 
  params_(params),
  my_local_device_index_(communicator_ ? communicator_->local_rank() : 0){
    const DeviceIdxType device_index = (communicator_ != nullptr && communicator_->is_available())
      ? communicator_->deviceId() : 0;
    streams_.insert(
        {container->getDefaultStream(),
        c10::cuda::getDefaultCUDAStream(
            static_cast<c10::DeviceIndex>(device_index))});
  }
  
  // Getters for accessing the parameters
  Communicator* getCommunicator() const { return communicator_; }
  const hir::HostIrEvaluatorParams& getParams() const { return params_; }
  int64_t getLocalDeviceIndex() const { return my_local_device_index_; }
  int64_t getDeviceId() const { 
    return (communicator_ != nullptr && communicator_->is_available()) 
      ? communicator_->deviceId() : 0; 
  }
  c10::Device getDevice() const { 
    return communicator_ ? communicator_->device() : at::Device("cuda:0"); 
  }
  
  // Method to update global variables in LLVM module (if needed at runtime)
  void updateGlobalVariables(llvm::Module* mod) const {
    llvm::LLVMContext& context = mod->getContext();
    llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
    llvm::Type* bool_type = llvm::Type::getInt1Ty(context);
    
    // Update device index
    if (auto* device_index_global = mod->getGlobalVariable("device_index")) {
      device_index_global->setInitializer(llvm::ConstantInt::get(int64_type, getDeviceId()));
    }
    
    // Update local device index
    if (auto* local_device_index_global = mod->getGlobalVariable("local_device_index")) {
      local_device_index_global->setInitializer(llvm::ConstantInt::get(int64_type, getLocalDeviceIndex()));
    }
    
    // Update device availability
    if (auto* device_available_global = mod->getGlobalVariable("device_available")) {
      bool is_available = (communicator_ != nullptr && communicator_->is_available());
      device_available_global->setInitializer(llvm::ConstantInt::get(bool_type, is_available));
    }
  }
};

// PIMPL implementation for HostIrJit
struct HostIrJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unordered_map<const kir::Allocate*, allocate_fn> allocate_funcs_;
  std::unique_ptr<HostIrJitParams> params_;
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

// Helper function to create global variables for shared parameters
void createGlobalVariables(llvm::Module* mod, const HostIrJitParams& params) {
  // // Note: We're now using constants instead of global variables for better performance
  // // This function is kept for potential future use if runtime-modifiable globals are needed
  
  // llvm::LLVMContext& context = mod->getContext();
  // llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  // llvm::Type* int32_type = llvm::Type::getInt32Ty(context);
  // llvm::Type* bool_type = llvm::Type::getInt1Ty(context);
  // llvm::Type* ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  
  // // Example: Global memory pool size (if needed for runtime configuration)
  // if (!mod->getGlobalVariable("memory_pool_size")) {
  //   new llvm::GlobalVariable(
  //       *mod,
  //       int64_type,
  //       false,
  //       llvm::GlobalValue::InternalLinkage,
  //       llvm::ConstantInt::get(int64_type, 1024 * 1024 * 1024), // 1GB default
  //       "memory_pool_size");
  // }
  
  // // Example: Global stream ID (if needed for runtime configuration)
  // if (!mod->getGlobalVariable("default_stream_id")) {
  //   new llvm::GlobalVariable(
  //       *mod,
  //       int32_type,
  //       false,
  //       llvm::GlobalValue::InternalLinkage,
  //       llvm::ConstantInt::get(int32_type, 0), // Default stream
  //       "default_stream_id");
  // }
  
}

// Generate a function that calls at::native::empty_strided_cuda
void generateAllocateFunc(
    const kir::Allocate* allocate,
    llvm::Module* mod,
    const HostIrJitParams& params) {

  at::ScalarType type = data_type_to_aten(allocate->buffer()->dtype() == DataType::Index ? PrimDataType::Int : allocate->buffer()->dtype());

  llvm::LLVMContext& context = mod->getContext();
  llvm::IRBuilder<> builder(context);

  // Create constants directly from params (more efficient than loading from globals)
  llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  
  std::string func_name = "create_tensor_from_sizes_" +
          std::to_string(reinterpret_cast<uintptr_t>(allocate));
  
  // Define function signature: std::shared_ptr<at::Tensor> func(int64_t*, int64_t, int64_t*, int64_t)
  // The type and device are embedded as constants in the function
  llvm::Type* int64_ptr_type = int64_type->getPointerTo();
  llvm::Type* int32_type = llvm::Type::getInt32Ty(context);
  llvm::PointerType* void_ptr_type =
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  llvm::FunctionType* func_type = llvm::FunctionType::get(
      void_ptr_type, {int64_ptr_type, int64_type, int64_ptr_type, int64_type}, false);

  // Create the function with the custom name
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, func_name, mod);

  // Create basic block
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(context, "entry", func);
  builder.SetInsertPoint(entry_block);

  // Get function arguments
  llvm::Value* sizes_arg = func->getArg(0);     // int64_t* (sizes)
  llvm::Value* ndim_arg = func->getArg(1);      // int64_t   (ndim)
  llvm::Value* strides_arg = func->getArg(2);   // int64_t* (strides)
  llvm::Value* strides_ndim_arg = func->getArg(3); // int64_t (strides_ndim)

  // Create constants for type and device from params
  llvm::Value* dtype_constant = llvm::ConstantInt::get(int32_type, static_cast<int32_t>(type));
  llvm::Value* device_index_constant = llvm::ConstantInt::get(int64_type, params.getDeviceId());

  // Get the at::native::empty_strided_cuda function pointer (registered in the JIT)
  llvm::Function* at_empty_strided_cuda_func = mod->getFunction("at::native::empty_strided_cuda");
  if (at_empty_strided_cuda_func == nullptr) {
    llvm::FunctionType* at_empty_strided_cuda_func_type = llvm::FunctionType::get(
        void_ptr_type, {int64_ptr_type, int64_type, int64_ptr_type, int64_type, int32_type, int64_type}, false);
    at_empty_strided_cuda_func = llvm::Function::Create(
        at_empty_strided_cuda_func_type, llvm::Function::ExternalLinkage, "at::native::empty_strided_cuda", mod);
  }

  // Call at::native::empty_strided_cuda with constants
  llvm::Value* result =
      builder.CreateCall(at_empty_strided_cuda_func, {sizes_arg, ndim_arg, strides_arg, strides_ndim_arg, dtype_constant, device_index_constant});

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

void compile(const hir::HostIrContainer* container, llvm::orc::LLJIT* jit, std::unordered_map<const kir::Allocate*, allocate_fn>& allocate_funcs_, const HostIrJitParams& params) {
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
  
  // Create global variables once for the entire module
  createGlobalVariables(mod.get(), params);
  
  std::unordered_map<const kir::Allocate*, std::string> allocate_func_names;
  for (auto expr : container->topLevelExprs()) {
    if (auto allocate = dynamic_cast<const kir::Allocate*>(expr)) {
      // Generate a unique function name for this allocate
      generateAllocateFunc(allocate, mod.get(), params);
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
    allocate_funcs_[allocate] =
        (allocate_fn)reinterpret_cast<at::Tensor* (*)(
            const int64_t*, int64_t, const int64_t*, int64_t)>(func_addr.getValue());
  }
}

// Constructor implementation
HostIrJit::HostIrJit(
    hir::HostIrContainer* container,
    Communicator* communicator,
    int num_threads)
    : HostIrJit(
          container,
          communicator,
          hir::HostIrEvaluatorParams{},
          num_threads) {}

HostIrJit::HostIrJit(
    hir::HostIrContainer* container,
    Communicator* communicator,
    const hir::HostIrEvaluatorParams& params,
    int num_threads)
    : pimpl_(new LlvmJitImpl) {
  // Initialize params with passed parameters
  pimpl_->params_ =
      std::make_unique<HostIrJitParams>(container, communicator, params);

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
  void* empty_strided_cuda_func_ptr = reinterpret_cast<void*>(
      +[](const int64_t* sizes,
          int64_t ndim,
          const int64_t* strides,
          int64_t strides_ndim,
          int32_t dtype,
          int64_t device_index) -> at::Tensor* {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::IntArrayRef aten_strides(strides, strides_ndim);
        // Use the type and device passed as parameters
        at::ScalarType scalar_type = static_cast<at::ScalarType>(dtype);
        at::Device device =
            at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index));
        return new at::Tensor(at::native::empty_strided_cuda(
            aten_sizes,
            aten_strides,
            scalar_type,
            c10::nullopt,
            device,
            c10::nullopt));
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
    compile(container, pimpl_->jit.get(), pimpl_->allocate_funcs_, *pimpl_->params_);
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
  
  auto result = func_ptr(
      input_sizes.data(),
      input_sizes.size(),
      input_strides.data(),
      input_strides.size());
  
  at::Tensor tensor = *result;
  delete result;
  return tensor;
}

} // namespace nvfuser
