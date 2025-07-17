// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bfs.h>
#include <unordered_map>
#include <functional>
#include <memory>

#include <instrumentation.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include "llvm/ADT/SmallVector.h"
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
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ops/all_ops.h>
#include <val_graph_visitor.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

using main_func_t = void(*)(const void**, void**);
constexpr std::string_view kMainFuncName = "main";
constexpr std::string_view kTensorSizeFuncName = "tensor_size";
constexpr std::string_view kNewTensorFuncName = "new_tensor";
constexpr std::string_view kDeleteTensorFuncName = "delete_tensor";
constexpr std::string_view kSetTensorFuncName = "set_tensor";
constexpr std::string_view kAtEmptyStridedCudaWrapper = "at_empty_strided_cuda";
constexpr std::string_view kAtTensorType = "at.Tensor";
constexpr size_t kMaxTensorDim = 8;

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
  // Implementation methods
  void compile();
  void registerExternalFunctions();

  // data members
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

llvm::ArrayType* getInt64StaticArrayType(llvm::LLVMContext& context, size_t size) {
  return llvm::ArrayType::get(llvm::Type::getInt64Ty(context), size);
}

llvm::Type* getInt64PtrType(llvm::LLVMContext& context) {
  return llvm::Type::getInt64Ty(context)->getPointerTo();
}

llvm::Type* getVoidType(llvm::LLVMContext& context) {
  return llvm::Type::getVoidTy(context);
}

// Helper function to generate LLVM IR that extracts tensor size for a given
// dimension
llvm::Value* generateTensorSizeExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();

  // Look up the tensor_size wrapper function
  llvm::Function* tensor_size_func = module->getFunction(kTensorSizeFuncName);
  llvm::Value* dim_val = builder.getInt64(dim);

  return builder.CreateCall(tensor_size_func, {tensor_ptr, dim_val});
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

// Helper function to infer tensor shapes and strides
// Currently, we support only tensors with a constant shape and strides. This
// is to demonstrate a aten tensor is able to be allocated and deallocated
// properly, we will support more complex tensor shapes and strides in future
// PRs.
void inferTensorShapesAndStrides(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder,
    llvm::SmallVectorImpl<llvm::Value*>& sizes,
    llvm::SmallVectorImpl<llvm::Value*>& strides) {
  llvm::Value* stride = builder.getInt64(1);
  for (const auto& id : TensorDomain::noReductions(tv->getLogicalDomain())) {
    NVF_ERROR(id->extent()->isConst(), "Extent is not constant", id->extent());
    llvm::Value* extent_value =
        builder.getInt64(id->extent()->evaluate().as<int64_t>());
    val_to_value[id->extent()] = extent_value;
    sizes.push_back(val_to_value[id->extent()]);
    strides.push_back(stride);
    stride = builder.CreateMul(stride, extent_value);
  }
}

void unpackInputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();

  // Get the current function (main) and its first argument
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* aten_tensor_array_ptr = func->getArg(0);

  llvm::Type* aten_tensor_array_type = getInt8PtrDynamicArrayType(context);
  llvm::Type* tensor_ptr_type = getTensorPtrType(context);

  // bind input aten tensor sizes to val_to_value
  for (const auto [i, input] : enumerate(container->inputs())) {
    auto* tv = dynamic_cast<TensorView*>(input);
    NVF_ERROR(tv != nullptr, "Unsupported expression type: ", input);
    llvm::Value* tensor_addr = builder.CreateGEP(
        aten_tensor_array_type, aten_tensor_array_ptr, {builder.getInt64(i)});
    tensor_addr->setName("input_aten_tensor_addr");
    // Load the actual tensor pointer from the array
    llvm::Value* tensor = builder.CreateLoad(tensor_ptr_type, tensor_addr);
    tensor->setName("input_aten_tensor");
    // bind input aten tensor sizes to val_to_value
    const std::vector<IterDomain*> logical_domain =
        TensorDomain::noReductions(tv->getLogicalDomain());
    // TODO: We should validate const size and strides here, ie. dim check
    for (const auto& [dim_idx, id] : enumerate(logical_domain)) {
      if (id->isBroadcast()) {
        val_to_value[id->extent()] = builder.getInt64(1);
        if (id->hasExpandedExtent()) {
          val_to_value[id->expandedExtent()] =
              generateTensorSizeExtraction(tensor, dim_idx, builder);
        }
      } else {
        val_to_value[id->extent()] =
            generateTensorSizeExtraction(tensor, dim_idx, builder);
      }
    }
    // bind input aten tensor to val_to_value
    val_to_value[tv] = tensor;
  }

  if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
    printLlvmIr(func, "Main Function Inputs");
  }
}

void packOutputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();

  // Get the current function (main) and its second argument
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* aten_tensor_array_ptr = func->getArg(1);

  llvm::Type* aten_tensor_array_type = getInt8PtrDynamicArrayType(context);
  // Store output tensor pointers from val_to_value into the output array
  for (const auto [i, output] : enumerate(container->outputs())) {
    auto* tv = dynamic_cast<TensorView*>(output);
    NVF_ERROR(tv != nullptr, "Unsupported expression type: ", output);
    llvm::Value* tensor_addr = builder.CreateGEP(
        aten_tensor_array_type, aten_tensor_array_ptr, {builder.getInt64(i)});
    tensor_addr->setName("output_aten_tensor_addr");

    // Get the tensor pointer from val_to_value and store it in the output
    // array
    llvm::Value* tensor_from_val_to_value = val_to_value[tv];
    builder.CreateStore(tensor_from_val_to_value, tensor_addr);
  }
  builder.CreateRetVoid();
  if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
    printLlvmIr(func, "Main Function Outputs");
  }
}

void compileFunctionDeclarations(llvm::Module* module, llvm::LLVMContext& context) {
  // get the types
  auto* void_type = getVoidType(context);
  auto* void_array_ptr_type = getInt8PtrDynamicArrayType(context);
  auto* int64_type = llvm::Type::getInt64Ty(context);
  auto* int64_ptr_type = getInt64PtrType(context);
  auto* int32_type = llvm::Type::getInt32Ty(context);
  auto* tensor_ptr_type = getTensorPtrType(context);

  // tensor_size function: int64_t tensor_size(at::Tensor* tensor, int64_t dim)
  auto* tensor_size_type = llvm::FunctionType::get(
      int64_type, {tensor_ptr_type, int64_type}, false);
  llvm::Function::Create(
      tensor_size_type,
      llvm::Function::ExternalLinkage,
      kTensorSizeFuncName,
      module);

  // new_tensor function: at::Tensor* new_tensor()
  auto* new_tensor_type =
      llvm::FunctionType::get(tensor_ptr_type, {}, false);
  llvm::Function::Create(
      new_tensor_type,
      llvm::Function::ExternalLinkage,
      kNewTensorFuncName,
      module);

  // set_tensor function: void set_tensor(at::Tensor* tensor, at::Tensor*
  // other_tensor)
  auto* set_tensor_type = llvm::FunctionType::get(
      void_type, {tensor_ptr_type, tensor_ptr_type}, false);
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
       tensor_ptr_type},
      false);
  llvm::Function::Create(
      empty_strided_cuda_type,
      llvm::Function::ExternalLinkage,
      kAtEmptyStridedCudaWrapper,
      module);

  // delete_tensor function: void delete_tensor(at::Tensor* tensor)
  auto* delete_tensor_type =
      llvm::FunctionType::get(void_type, {tensor_ptr_type}, false);
  llvm::Function::Create(
      delete_tensor_type,
      llvm::Function::ExternalLinkage,
      kDeleteTensorFuncName,
      module);

  // main function: void main(void** input_tensors, void** output_tensors)
  auto* main_type = llvm::FunctionType::get(
      void_type, {void_array_ptr_type, void_array_ptr_type}, false);
  llvm::Function::Create(
      main_type, llvm::Function::ExternalLinkage, kMainFuncName, module);
}

class HostIrCompileDispatcher : public OptInDispatch {
 public:
  HostIrCompileDispatcher(
      llvm::IRBuilder<>& builder,
      std::unordered_map<Val*, llvm::Value*>& val_to_value)
      : builder_(builder), val_to_value_(val_to_value) {}
  using OptInDispatch::handle;

  // NOTE: this is just a simple example of allocate a output tensor and set it
  // to input tensor. The whole concept is to demonstrate llvm jit works, we
  // will change this in the future LoadStoreOp Function LLVM IR Generation
  void handle(LoadStoreOp* load_store_op) final {
    NVF_ERROR(
        load_store_op->opType() == LoadStoreOpType::Set ||
        load_store_op->opType() == LoadStoreOpType::SegmenterSet);
    NVF_ERROR(
        load_store_op->out()->isA<TensorView>(), "out must be a TensorView");
    auto* in_tv = load_store_op->in()->as<Val>();
    auto* out_tv = load_store_op->out()->as<Val>();
    auto it = val_to_value_.find(in_tv);
    NVF_ERROR(
        it != val_to_value_.end(), "input tensor is not found in val_to_value");
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::Value* in_tensor = it->second;
    // allocate a new tensor
    llvm::Function* new_tensor_func = module->getFunction(kNewTensorFuncName);
    llvm::Value* out_tensor =
        builder_.CreateCall(new_tensor_func, {}, "out_tensor");

    // set the output tensor to the input tensor
    llvm::Function* set_tensor_func = module->getFunction(kSetTensorFuncName);
    builder_.CreateCall(set_tensor_func, {out_tensor, in_tensor});

    // bind the output tensor to val_to_value
    val_to_value_[out_tv] = out_tensor;

    if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
      printLlvmIr(builder_.GetInsertBlock()->getParent(), "LoadStoreOp");
    }
  }

  // Allocate Function LLVM IR Generation
  void handle(kir::Allocate* allocate) final {
    llvm::LLVMContext& context = builder_.getContext();
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();

    // Define LLVM types
    llvm::Type* int64_ptr_type = getInt64PtrType(context);

    // Get tensor sizes and strides using the inference function
    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_sizes;
    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_strides;
    inferTensorShapesAndStrides(
        allocate->buffer()->as<TensorView>(),
        val_to_value_,
        builder_,
        tensor_sizes,
        tensor_strides);

    // Bounds checking for ndim
    auto logical_domain = TensorDomain::noReductions(
        allocate->buffer()->as<TensorView>()->getLogicalDomain());

    NVF_ERROR_EQ(tensor_sizes.size(), logical_domain.size());

    // Create arrays for sizes and strides
    llvm::ArrayType* sizes_type =
        getInt64StaticArrayType(context, tensor_sizes.size());
    llvm::ArrayType* strides_type =
        getInt64StaticArrayType(context, tensor_strides.size());

    llvm::Value* sizes =
        builder_.CreateAlloca(sizes_type, nullptr, "sizes");
    llvm::Value* strides =
        builder_.CreateAlloca(strides_type, nullptr, "strides");

    // Populate sizes array
    for (const auto [i, size] : enumerate(tensor_sizes)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          sizes_type,
          sizes,
          {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(size, gep);
    }

    // Populate strides array
    for (const auto [i, stride] : enumerate(tensor_strides)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          strides_type,
          strides,
          {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(stride, gep);
    }

    // Convert arrays to pointers
    llvm::Value* sizes_arg =
        builder_.CreateBitCast(sizes, int64_ptr_type);
    llvm::Value* strides_arg =
        builder_.CreateBitCast(strides, int64_ptr_type);

    // Create array size arguments
    llvm::Value* shape_ndim_arg = builder_.getInt64(tensor_sizes.size());
    llvm::Value* strides_ndim_arg = builder_.getInt64(tensor_strides.size());

    // Create output tensor
    llvm::Value* raw_tensor_ptr = builder_.CreateCall(
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

    // Configure output tensor
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
         raw_tensor_ptr});
    val_to_value_[allocate->buffer()->as<Val>()] = raw_tensor_ptr;

    if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
      printLlvmIr(builder_.GetInsertBlock()->getParent(), "Allocate Function");
    }
  }

  // Deallocation Function LLVM IR Generation
  void handle(hir::Deallocate* deallocate) final {
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::Function* delete_tensor_func =
        module->getFunction(kDeleteTensorFuncName);
    builder_.CreateCall(
        delete_tensor_func, {val_to_value_.at(deallocate->buffer()->as<Val>())});
    if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
      printLlvmIr(builder_.GetInsertBlock()->getParent(), "Deallocate Function");
    }
  }
  // Not handled instructions automatically trigger an error.
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
  llvm::BasicBlock* entry =
      llvm::BasicBlock::Create(*context, "entry", module->getFunction(kMainFuncName));
  builder.SetInsertPoint(entry);

  // compile inputs in llvm ir
  unpackInputs(container_.get(), builder, val_to_value);
  HostIrCompileDispatcher dispatcher(builder, val_to_value);
  // compile all top level expressions in host ir container
  for (auto* expr : container_->topLevelExprs()) {
    dispatcher.dispatch(expr);
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

  // tensor size and stride extraction functions
  void* extract_tensor_size_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* tensor, int64_t dim) -> int64_t {
        NVF_ERROR(tensor != nullptr, kTensorSizeFuncName, " tensor is nullptr");
        return tensor->size(dim);
      });

  // raw tensor allocation, we only allocate a wrapper here
  void* new_tensor_func_ptr = reinterpret_cast<void*>(
      +[]() -> at::Tensor* { return new at::Tensor(); });

  // in place tensor update
  void* set_tensor_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* out, at::Tensor* in) -> void {
        NVF_ERROR(out != nullptr, kSetTensorFuncName, " out is nullptr");
        NVF_ERROR(in != nullptr, kSetTensorFuncName, " in is nullptr");
        *out = in->clone(); // Clone the input tensor
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

  // Register wrapper functions in JIT
  llvm::orc::SymbolMap name_to_symbol;
  registerExternalFunction(
      extract_tensor_size_func_ptr,
      name_to_symbol,
      mangler,
      kTensorSizeFuncName);
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
    NVF_ERROR(arg.is<at::Tensor>(), "Unsupported argument type: ", arg, " for input ", in_val);
    input_aten_tensors.push_back(&arg.as<at::Tensor>());
  }

  // Run the main function
  std::vector<void*> output_aten_tensors(container_->outputs().size());
  main_func_(input_aten_tensors.data(), output_aten_tensors.data());

  // Collect the outputs
  KernelArgumentHolder outputs;
  for (const auto [output, tensor_ptr] : zip(container_->outputs(), output_aten_tensors)) {
    NVF_ERROR(output->isA<TensorView>(), "Unsupported output type: ", output, " for output ", output);
    // Cast void* to at::Tensor* first, then dereference
    at::Tensor* aten_tensor_ptr = static_cast<at::Tensor*>(tensor_ptr);
    outputs.push(*aten_tensor_ptr);
    // Clean up the individual tensor object (not the array)
    delete aten_tensor_ptr;
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
