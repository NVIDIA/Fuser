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

#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

using main_func_t = std::function<void(const void**, void**)>;
constexpr std::string_view kMainFuncName = "main";
constexpr std::string_view kTensorSizeFuncName = "tensor_size";
constexpr std::string_view kAllocateTensorFuncName = "allocate_tensor";
constexpr std::string_view kSetTensorFuncName = "set_tensor";

llvm::Value* generateTensorSizeExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::IRBuilder<>& builder);

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

llvm::Type* getInt64Type(llvm::LLVMContext& ctx) {
  return llvm::Type::getInt64Ty(ctx);
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

// NOTE: this is just a simple example of allocate a output tensor and set it
// to input tensor. The whole concept is to demonstrate llvm jit works, we will
// change this in the future
void compileLoadStoreOp(
    LoadStoreOp* load_store_op,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  NVF_ERROR(
      load_store_op->opType() == LoadStoreOpType::Set ||
      load_store_op->opType() == LoadStoreOpType::SegmenterSet);
  NVF_ERROR(
      load_store_op->out()->isA<TensorView>(), "out must be a TensorView");
  auto* in_tv = load_store_op->in()->as<Val>();
  auto* out_tv = load_store_op->out()->as<Val>();
  auto it = val_to_value.find(in_tv);
  NVF_ERROR(
      it != val_to_value.end(), "input tensor is not found in val_to_value");
  llvm::Module* mod = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Value* in_tensor = it->second;

  // allocate a new tensor
  llvm::Function* allocate_tensor_func =
      mod->getFunction(kAllocateTensorFuncName);
  llvm::Value* out_tensor =
      builder.CreateCall(allocate_tensor_func, {}, "out_tensor_raw");

  // set the output tensor to the input tensor
  llvm::Function* set_tensor_func = mod->getFunction(kSetTensorFuncName);
  builder.CreateCall(set_tensor_func, {out_tensor, in_tensor});

  // bind the output tensor to val_to_value
  val_to_value[out_tv] = out_tensor;

  if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
    auto* func = builder.GetInsertBlock()->getParent();
    llvm::outs() << "=== LLVM IR After Generating LoadStoreOp ===\n";
    func->print(llvm::outs(), nullptr);
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
  // bind input aten tensor sizes to val_to_value
  for (size_t i = 0; i < container->inputs().size(); ++i) {
    auto* input = container->inputs()[i];
    auto* tv = dynamic_cast<TensorView*>(input);
    NVF_ERROR(tv != nullptr, "Unsupported expression type: ", input);
    llvm::Value* tensor_addr = builder.CreateGEP(
        aten_tensor_array_type, aten_tensor_array_ptr, {builder.getInt64(i)});
    tensor_addr->setName("input_aten_tensor_addr");
    // Load the actual tensor pointer from the array
    llvm::Value* tensor = builder.CreateLoad(getInt8PtrType(ctx), tensor_addr);
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
  auto* void_ptr_type = getInt8PtrType(ctx);
  auto* void_array_ptr_type = getInt8PtrDynamicArrayType(ctx);
  auto* int64_t_type = getInt64Type(ctx);

  // tensor_size function: int64_t tensor_size(at::Tensor* tensor, int64_t dim)
  llvm::FunctionType* tensor_size_type = llvm::FunctionType::get(
      int64_t_type, {void_ptr_type, int64_t_type}, false);
  llvm::Function::Create(
      tensor_size_type,
      llvm::Function::ExternalLinkage,
      kTensorSizeFuncName,
      mod);

  llvm::FunctionType* allocate_tensor_type =
      llvm::FunctionType::get(void_ptr_type, {}, false);
  llvm::Function::Create(
      allocate_tensor_type,
      llvm::Function::ExternalLinkage,
      kAllocateTensorFuncName,
      mod);

  // set_tensor function: void set_tensor(at::Tensor* tensor, at::Tensor*
  // other_tensor)
  llvm::FunctionType* set_tensor_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx), {void_ptr_type, void_ptr_type}, false);
  llvm::Function::Create(
      set_tensor_type,
      llvm::Function::ExternalLinkage,
      kSetTensorFuncName,
      mod);

  // main function: void main(void** input_tensors, void** output_tensors)
  llvm::FunctionType* main_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx),
      {void_array_ptr_type, void_array_ptr_type},
      false);
  llvm::Function::Create(
      main_type, llvm::Function::ExternalLinkage, kMainFuncName, mod);
}

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

  // compile all top level expressions in host ir container
  for (auto* expr : pimpl->container->topLevelExprs()) {
    // TODO: support more expression types
    if (expr->isA<LoadStoreOp>()) {
      compileLoadStoreOp(expr->as<LoadStoreOp>(), builder, val_to_value);
    } else {
      NVF_THROW("Unsupported expression type: ", expr);
    }
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
  void* allocate_tensor_func_ptr = reinterpret_cast<void*>(
      +[]() -> at::Tensor* { return new at::Tensor(); });

  // in place tensor update
  void* set_tensor_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* out, at::Tensor* in) -> void {
        NVF_ERROR(out != nullptr, kSetTensorFuncName, " out is nullptr");
        NVF_ERROR(in != nullptr, kSetTensorFuncName, " in is nullptr");
        *out = in->clone(); // Clone the input tensor
      });

  // Register wrapper functions in JIT
  llvm::orc::SymbolMap name_to_symbol;
  registerExternalFunction(
      extract_tensor_size_func_ptr,
      name_to_symbol,
      mangler,
      kTensorSizeFuncName);
  registerExternalFunction(
      allocate_tensor_func_ptr,
      name_to_symbol,
      mangler,
      kAllocateTensorFuncName);
  registerExternalFunction(
      set_tensor_func_ptr, name_to_symbol, mangler, kSetTensorFuncName);
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
