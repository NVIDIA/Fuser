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
#include <functional>
#include <memory>

#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

using main_func_t = std::function<void(const void**, void**)>;
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
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unique_ptr<hir::HostIrContainer> container;
  main_func_t main_func;
  HostIrJitImpl() = default;
  ~HostIrJitImpl() = default;
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
llvm::Type* getInt8PtrType(llvm::LLVMContext& ctx) {
  return llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(ctx));
}

llvm::Type* getInt8PtrStaticArrayType(llvm::LLVMContext& ctx, size_t size) {
  return llvm::ArrayType::get(getInt8PtrType(ctx), size);
}

llvm::Type* getInt8PtrDynamicArrayType(llvm::LLVMContext& ctx) {
  return llvm::PointerType::getUnqual(getInt8PtrType(ctx));
}

// Helper function to get opaque at::Tensor type for better type safety
llvm::Type* getAtTensorPtrType(llvm::LLVMContext& ctx) {
  // Create an opaque struct type for at::Tensor
  // This provides better type safety than using void* for tensor pointers
  // while still being compatible with LLVM's type system
  llvm::StructType* tensor_type = llvm::StructType::create(ctx, kAtTensorType);
  return llvm::PointerType::getUnqual(tensor_type);
}

llvm::ArrayType* getInt64StaticArrayType(llvm::LLVMContext& ctx, size_t size) {
  return llvm::ArrayType::get(llvm::Type::getInt64Ty(ctx), size);
}

llvm::Type* getInt64PtrType(llvm::LLVMContext& ctx) {
  return llvm::Type::getInt64Ty(ctx)->getPointerTo();
}

llvm::Type* getVoidType(llvm::LLVMContext& ctx) {
  return llvm::Type::getVoidTy(ctx);
}

// Helper function to generate LLVM IR that extracts tensor size for a given
// dimension
llvm::Value* generateTensorSizeExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::IRBuilder<>& builder) {
  auto* mod = builder.GetInsertBlock()->getParent()->getParent();

  // Look up the tensor_size wrapper function
  llvm::Function* tensor_size_func = mod->getFunction(kTensorSizeFuncName);
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

// Helper function to infer tensor shapes and strides
// NOTE: This is only for constant known shape and stride tensor, the whole idea
// is to demonstrate a aten tensor is able to be allocated and deallocated
// properly, we will support more complex tensor shapes and strides in future
// PRs

void inferTensorShapesAndStrides(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val_to_value,
    llvm::IRBuilder<>& builder,
    llvm::SmallVectorImpl<llvm::Value*>& sizes,
    llvm::SmallVectorImpl<llvm::Value*>& strides) {
  llvm::Value* stride = builder.getInt64(1);
  for (const auto& id : TensorDomain::noReductions(tv->getLogicalDomain())) {
    NVF_ERROR(id->extent()->isConst(), "Extent is not constant");
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
  llvm::LLVMContext& ctx = builder.getContext();

  // Get the current function (main) and its first argument
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* aten_tensor_array_ptr = func->getArg(0);

  llvm::Type* aten_tensor_array_type = getInt8PtrDynamicArrayType(ctx);
  llvm::Type* tensor_ptr_type = getAtTensorPtrType(ctx);

  // bind input aten tensor sizes to val_to_value
  for (size_t i = 0; i < container->inputs().size(); ++i) {
    auto* input = container->inputs()[i];
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
    llvm::outs() << "=== LLVM IR After Generating Main Function Inputs ===\n";
    func->getParent()->print(llvm::outs(), nullptr);
  }
}

void packOutputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& ctx = builder.getContext();

  // Get the current function (main) and its second argument
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* aten_tensor_array_ptr = func->getArg(1);

  llvm::Type* aten_tensor_array_type = getInt8PtrDynamicArrayType(ctx);
  // Store output tensor pointers from val_to_value into the output array
  for (size_t i = 0; i < container->outputs().size(); ++i) {
    auto* output = container->outputs()[i];
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
    llvm::outs() << "=== LLVM IR After Generating Main Function Outputs ===\n";
    llvm::Function* func = builder.GetInsertBlock()->getParent();
    func->getParent()->print(llvm::outs(), nullptr);
  }
}

void compileFunctionDeclarations(llvm::Module* mod, llvm::LLVMContext& ctx) {
  // get the types
  auto* void_type = getVoidType(ctx);
  auto* void_array_ptr_type = getInt8PtrDynamicArrayType(ctx);
  auto* int64_t_type = llvm::Type::getInt64Ty(ctx);
  auto* int64_ptr_type = getInt64PtrType(ctx);
  auto* int32_t_type = llvm::Type::getInt32Ty(ctx);
  auto* tensor_ptr_type = getAtTensorPtrType(ctx);

  // tensor_size function: int64_t tensor_size(at::Tensor* tensor, int64_t dim)
  llvm::FunctionType* tensor_size_type = llvm::FunctionType::get(
      int64_t_type, {tensor_ptr_type, int64_t_type}, false);
  llvm::Function::Create(
      tensor_size_type,
      llvm::Function::ExternalLinkage,
      kTensorSizeFuncName,
      mod);

  // new_tensor function: at::Tensor* new_tensor()
  llvm::FunctionType* new_tensor_type =
      llvm::FunctionType::get(tensor_ptr_type, {}, false);
  llvm::Function::Create(
      new_tensor_type,
      llvm::Function::ExternalLinkage,
      kNewTensorFuncName,
      mod);

  // set_tensor function: void set_tensor(at::Tensor* tensor, at::Tensor*
  // other_tensor)
  llvm::FunctionType* set_tensor_type = llvm::FunctionType::get(
      void_type, {tensor_ptr_type, tensor_ptr_type}, false);
  llvm::Function::Create(
      set_tensor_type,
      llvm::Function::ExternalLinkage,
      kSetTensorFuncName,
      mod);

  // at::native::empty_strided_cuda function: void at_empty_strided_cuda(const
  // int64_t* sizes, int64_t ndim, const int64_t* strides, int64_t strides_ndim,
  // int32_t dtype, int64_t device_index, at::Tensor* out_tensor)
  llvm::FunctionType* empty_strided_cuda_type = llvm::FunctionType::get(
      void_type,
      {int64_ptr_type,
       int64_t_type,
       int64_ptr_type,
       int64_t_type,
       int32_t_type,
       int64_t_type,
       tensor_ptr_type},
      false);
  llvm::Function::Create(
      empty_strided_cuda_type,
      llvm::Function::ExternalLinkage,
      kAtEmptyStridedCudaWrapper,
      mod);

  // delete_tensor function: void delete_tensor(at::Tensor* tensor)
  llvm::FunctionType* delete_tensor_type =
      llvm::FunctionType::get(void_type, {tensor_ptr_type}, false);
  llvm::Function::Create(
      delete_tensor_type,
      llvm::Function::ExternalLinkage,
      kDeleteTensorFuncName,
      mod);

  // main function: void main(void** input_tensors, void** output_tensors)
  llvm::FunctionType* main_type = llvm::FunctionType::get(
      void_type, {void_array_ptr_type, void_array_ptr_type}, false);
  llvm::Function::Create(
      main_type, llvm::Function::ExternalLinkage, kMainFuncName, mod);
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
    llvm::Module* mod = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::Value* in_tensor = it->second;
    // allocate a new tensor
    llvm::Function* new_tensor_func = mod->getFunction(kNewTensorFuncName);
    llvm::Value* out_tensor =
        builder_.CreateCall(new_tensor_func, {}, "out_tensor_raw");

    // set the output tensor to the input tensor
    llvm::Function* set_tensor_func = mod->getFunction(kSetTensorFuncName);
    builder_.CreateCall(set_tensor_func, {out_tensor, in_tensor});

    // bind the output tensor to val_to_value
    val_to_value_[out_tv] = out_tensor;

    if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
      auto* func = builder_.GetInsertBlock()->getParent();
      llvm::outs() << "=== LLVM IR After Generating LoadStoreOp ===\n";
      func->print(llvm::outs(), nullptr);
    }
  }

  // Allocate Function LLVM IR Generation
  void handle(kir::Allocate* allocate) final {
    llvm::LLVMContext& context = builder_.getContext();
    auto mod = builder_.GetInsertBlock()->getParent()->getParent();

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

    NVF_ERROR(tensor_sizes.size() == logical_domain.size());

    // Create arrays for sizes and strides
    llvm::ArrayType* sizes_array_type =
        getInt64StaticArrayType(context, tensor_sizes.size());
    llvm::ArrayType* strides_array_type =
        getInt64StaticArrayType(context, tensor_strides.size());

    llvm::Value* sizes_array =
        builder_.CreateAlloca(sizes_array_type, nullptr, "sizes_array");
    llvm::Value* strides_array =
        builder_.CreateAlloca(strides_array_type, nullptr, "strides_array");

    // Populate sizes array
    for (size_t i = 0; i < tensor_sizes.size(); ++i) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          sizes_array_type,
          sizes_array,
          {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(tensor_sizes[i], gep);
    }

    // Populate strides array
    for (size_t i = 0; i < tensor_strides.size(); ++i) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          strides_array_type,
          strides_array,
          {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(tensor_strides[i], gep);
    }

    // Convert arrays to pointers
    llvm::Value* sizes_arg =
        builder_.CreateBitCast(sizes_array, int64_ptr_type);
    llvm::Value* strides_arg =
        builder_.CreateBitCast(strides_array, int64_ptr_type);

    // Create array size arguments
    llvm::Value* shape_ndim_arg = builder_.getInt64(tensor_sizes.size());
    llvm::Value* strides_ndim_arg = builder_.getInt64(tensor_strides.size());

    // Create output tensor
    llvm::Value* raw_tensor_ptr = builder_.CreateCall(
        mod->getFunction(kNewTensorFuncName), {}, "out_tensor");

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
        mod->getFunction(kAtEmptyStridedCudaWrapper);

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
      llvm::outs() << "=== LLVM IR After Generating Allocate Function ===\n";
      mod->print(llvm::outs(), nullptr);
    }
  }

  // Deallocation Function LLVM IR Generation
  void handle(hir::Deallocate* deallocate) final {
    auto mod = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::Function* delete_tensor_func =
        mod->getFunction(kDeleteTensorFuncName);
    builder_.CreateCall(
        delete_tensor_func, {val_to_value_[deallocate->buffer()->as<Val>()]});
    if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
      auto* func = builder_.GetInsertBlock()->getParent();
      llvm::outs() << "=== LLVM IR After Generating Deallocate Function ===\n";
      func->print(llvm::outs(), nullptr);
    }
  }
  // Not handled instructions automatically trigger an error.
 private:
  llvm::IRBuilder<>& builder_;
  std::unordered_map<Val*, llvm::Value*>& val_to_value_;
};

void compile(HostIrJitImpl* pimpl) {
  NVF_ERROR(
      pimpl->container != nullptr,
      "container is nullptr during host ir JIT compilation");
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("host_ir_jit_module", *ctx);
  llvm::IRBuilder<> builder(*ctx);
  std::unordered_map<Val*, llvm::Value*> val_to_value;

  // compile external functions and main function declarations
  compileFunctionDeclarations(mod.get(), *ctx);

  // Create entry block and set insertion point
  llvm::BasicBlock* entry =
      llvm::BasicBlock::Create(*ctx, "entry", mod->getFunction(kMainFuncName));
  builder.SetInsertPoint(entry);

  // compile inputs in llvm ir
  unpackInputs(pimpl->container.get(), builder, val_to_value);
  HostIrCompileDispatcher dispatcher(builder, val_to_value);
  // compile all top level expressions in host ir container
  for (auto* expr : pimpl->container->topLevelExprs()) {
    dispatcher.dispatch(expr);
  }

  // compile outputs in llvm ir
  packOutputs(pimpl->container.get(), builder, val_to_value);

  // verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(
      !llvm::verifyModule(*mod, &error_stream),
      "LLVM module verification failed: ",
      error);

  // Add the module to the JIT
  throwIfError(pimpl->jit->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx))));

  // Look up the main function
  auto main_func_addr = throwIfError(pimpl->jit->lookup(kMainFuncName));
  using main_func_ptr_t = void (*)(const void**, void**);
  auto main_func_ptr =
      reinterpret_cast<main_func_ptr_t>(main_func_addr.getValue());
  pimpl->main_func = main_func_ptr;
}

// NOTE: We have to keep the destructor here, otherwise the unique_ptr can't
// find complete type of LlvmJitImpl
HostIrJit::~HostIrJit() = default;

HostIrJit::HostIrJit(
    std::unique_ptr<hir::HostIrContainer> container,
    int num_threads)
    : pimpl_(new HostIrJitImpl) {
  FUSER_PERF_SCOPE("HostIrJit::HostIrJit");
  // Initialize params with passed parameters
  pimpl_->container = std::move(container);

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
  // Compile the module
  compile(pimpl_.get());
}

KernelArgumentHolder HostIrJit::runWithInputs(
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("HostIrJit::runWithInputs");
  // Bind cache id to llvm global variable or align with main function inputs
  NVF_ERROR(args.getCacheId().has_value(), "Cache ID is not set");
  NVF_ERROR_EQ(std::ssize(pimpl_->container->inputs()), args.size());

  std::vector<const void*> input_aten_tensors;
  // Bind the inputs to the tensor map
  for (auto&& [in_val, arg] : zip(pimpl_->container->inputs(), args)) {
    NVF_ERROR(arg.is<at::Tensor>(), "Unsupported argument type: ", arg);
    input_aten_tensors.push_back(&arg.as<at::Tensor>());
  }

  // Run the main function
  std::vector<void*> output_aten_tensors(pimpl_->container->outputs().size());
  pimpl_->main_func(input_aten_tensors.data(), output_aten_tensors.data());

  // Collect the outputs
  KernelArgumentHolder outputs;
  for (size_t i = 0; i < pimpl_->container->outputs().size(); ++i) {
    auto* output = pimpl_->container->outputs()[i];
    NVF_ERROR(output->isA<TensorView>(), "Unsupported output type: ", output);
    // Cast void* to at::Tensor* first, then dereference
    at::Tensor* tensor_ptr = static_cast<at::Tensor*>(output_aten_tensors[i]);
    outputs.push(*tensor_ptr);
    // Clean up the individual tensor object (not the array)
    delete tensor_ptr;
  }
  // Note: output_aten_tensors points to a global array managed by JIT, don't
  // delete the array itself
  return outputs;
}

const std::vector<Val*>& HostIrJit::inputs() const {
  return pimpl_->container->inputs();
}

const std::vector<Val*>& HostIrJit::outputs() const {
  return pimpl_->container->outputs();
}

const hir::HostIrContainer& HostIrJit::container() const {
  return *pimpl_->container;
}

} // namespace nvfuser
