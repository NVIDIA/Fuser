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

using main_func_t = std::function<at::Tensor**(const at::Tensor**)>;

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

enum class DataType {  
  void_ptr,
  void_array_ptr,
  int64_t
};

llvm::Type* getType(PtrType type, llvm::LLVMContext& ctx) {
  switch(type) {
    case PtrType::void_ptr:
      return llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(ctx));
    case PtrType::void_array_ptr:
      return llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(ctx)));
    case DataType::int64_t:
      return llvm::Type::getInt64Ty(ctx);
    default:
      NVF_ERROR("Invalid pointer type");
  }
  return nullptr;
}

void compileLoadStoreOp(
    LoadStoreOp* load_store_op,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    std::unordered_map<TensorView*, llvm::Value*>& tv2atenMap) {
    llvm::LLVMContext& ctx = builder.getContext();
      
    // bind input aten tensor sizes to val2llvmMap
}


void compileMainFuncInputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    std::unordered_map<TensorView*, llvm::Value*>& tv2atenMap) {
    llvm::LLVMContext& ctx = builder.getContext();
    
    // Get the current function (main) and its first argument
    llvm::Function* func = builder.GetInsertBlock()->getParent();
    llvm::Value* aten_tensor_array_ptr = func->getArg(0);
    
    llvm::Type* aten_tensor_array_type = getPtrType(PtrType::void_array_ptr, ctx);
  // bind input aten tensor sizes to val2llvmMap
  for(size_t i = 0; i < container->inputs().size(); ++i) {
    auto* input = container->inputs()[i];
    if(TensorView* tv = dynamic_cast<TensorView*>(input)) {
      llvm::Value* aten_tensor_ptr = builder.CreateGEP(aten_tensor_array_type, aten_tensor_array_ptr, {builder.getInt64(0), builder.getInt64(i)});
      aten_tensor_ptr->setName("input_aten_tensor_" + std::to_string(i));
      // bind input aten tensor sizes to val2llvmMap
      auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
      for (size_t dim = 0; dim < logical_domain.size(); ++dim) {
        if (logical_domain[dim]->isBroadcast()) {
          val2llvmMap[logical_domain[dim]->extent()] = builder.getInt64(1);
          if (logical_domain[dim]->hasExpandedExtent()) {
            val2llvmMap[logical_domain[dim]->expandedExtent()] = generateTensorSizeExtraction(aten_tensor_ptr, dim, builder);
          }
        } else {
          val2llvmMap[logical_domain[dim]->extent()] = generateTensorSizeExtraction(aten_tensor_ptr, dim, builder);
        }
      }
      // bind input aten tensor to tv2atenMap
      tv2atenMap[tv] = aten_tensor_ptr;
    }
  }

  const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
  if (debug_print) {
    llvm::outs() << "=== LLVM IR Generation for Main Function Inputs ===\n";
    func->getParent()->print(llvm::outs(), nullptr);
  }
}

void compileMainFuncOutputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    std::unordered_map<TensorView*, llvm::Value*>& tv2atenMap) {
    int num_outputs = container->outputs().size();
    llvm::Module* mod = builder.GetInsertBlock()->getParent()->getParent();
    llvm::LLVMContext& ctx = builder.getContext();
    llvm::Type* aten_tensor_array_type = llvm::ArrayType::get(getPtrType(PtrType::void_ptr, ctx), num_outputs);

    llvm::GlobalVariable* aten_tensor_array_ptr = new llvm::GlobalVariable(
        *mod, 
        aten_tensor_array_type, 
        false,
        llvm::GlobalValue::InternalLinkage,
        llvm::ConstantAggregateZero::get(aten_tensor_array_type),
        "output_array"
    );

    for(size_t i = 0; i < container->outputs().size(); ++i) {
      auto* output = container->outputs()[i];
      if(TensorView* tv = dynamic_cast<TensorView*>(output)) {
        llvm::Value* aten_tensor_ptr = builder.CreateGEP(aten_tensor_array_type, aten_tensor_array_ptr, 
        {builder.getInt64(0), builder.getInt64(i)});
        aten_tensor_ptr->setName("output_aten_tensor_" + std::to_string(i)); 
        builder.CreateStore(tv2atenMap[tv], aten_tensor_ptr);
      }
    }
    llvm::Value* result = builder.CreateBitCast(aten_tensor_array_ptr, getPtrType(PtrType::void_array_ptr, ctx));
    builder.CreateRet(result);

    const bool debug_print = isDebugDumpEnabled(DebugDumpOption::HostIrJit);
    if (debug_print) {
      llvm::outs() << "=== LLVM IR Generation for Main Function Outputs ===\n";
      mod->print(llvm::outs(), nullptr);
    }
}


void compileFunctionDeclarations(
    llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& ctx = builder.getContext();
  llvm::Module* mod = builder.GetInsertBlock()->getParent()->getParent();

  // tensor_size function: int64_t tensor_size(at::Tensor* tensor, int64_t dim)
  llvm::FunctionType* tensor_size_type = llvm::FunctionType::get(getType(DataType::int64_t, ctx), {getType(DataType::void_ptr, ctx), getType(DataType::int64_t, ctx)}, false);
  llvm::Function::Create(tensor_size_type, llvm::Function::ExternalLinkage, "tensor_size", mod);

  // main function: at::Tensor** main(at::Tensor** input_tensors)
  llvm::FunctionType* main_type = llvm::FunctionType::get(getType(DataType::void_array_ptr, ctx), {getType(DataType::void_array_ptr, ctx)}, false);
  llvm::Function::Create(main_type, llvm::Function::ExternalLinkage, "main", mod);

}

void compile(HostIrJitImpl* pimpl) {
  FUSER_PERF_SCOPE("HostIrJit::compile");
  NVF_ERROR(
      pimpl->container != nullptr,
      "container is nullptr during host ir JIT compilation");
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("host_ir_jit_module", *ctx);
  llvm::IRBuilder<> builder(*ctx);
  std::unordered_map<Val*, llvm::Value*> val2llvmMap;
  std::unordered_map<TensorView*, llvm::Value*> tv2atenMap;

  // compile external functions and main function declarations
  compileFunctionDeclarations(builder);

  // Create entry block and set insertion point
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(*ctx, "entry", mod->getFunction("main"));
  builder.SetInsertPoint(entry);
     
  // bind the constants
  compileMainFuncInputs(pimpl->container.get(), builder, val2llvmMap, tv2atenMap);
  
  // Collect output tensors and garbage collect intermediate tensors
  compileMainFuncOutputs(pimpl->container.get(), builder, val2llvmMap, tv2atenMap);


  // verify the module
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  NVF_ERROR(
      !llvm::verifyModule(*mod, &error_stream),
      "LLVM module verification failed: " + error);

  // Add the module to the JIT
  throwIfError(pimpl->jit->addIRModule(
      llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx))));
 
  // Look up the main function
  auto main_func_addr = throwIfError(pimpl->jit->lookup("main"));
  using main_func_ptr_t = at::Tensor**(*)(const at::Tensor**);
  pimpl->main_func = reinterpret_cast<main_func_ptr_t>(main_func_addr.getValue());
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

  // TODO: move all declartions to the top of the file, beginning of the module
  if (!tensor_size_func) {
    // Create function declaration
    llvm::FunctionType* func_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context),
      {getPtrType(PtrType::void_ptr, context), llvm::Type::getInt64Ty(context)},
      false
    );
    tensor_size_func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "tensor_size", mod
    );
  }
  
  llvm::Value* dim_val = builder.getInt64(dim);
  return builder.CreateCall(tensor_size_func, {tensor_ptr, dim_val});
}



// NOTE: We have to keep the destructor here, otherwise the unique_ptr can't
// find complete type of LlvmJitImpl
HostIrJit::~HostIrJit() = default;

HostIrJit::HostIrJit(std::unique_ptr<hir::HostIrContainer> container, int num_threads)
    : pimpl_(new HostIrJitImpl) {
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
  void* extract_tensor_size_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* tensor_ptr, int64_t dim) -> int64_t {
        if (tensor_ptr == nullptr) {
          return 0;
        }
        return tensor_ptr->size(dim);
      });
  // Register wrapper functions in JIT
  auto extract_tensor_size_addr = llvm::orc::ExecutorAddr::fromPtr(extract_tensor_size_func_ptr);
  // Register wrapper functions in JIT
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler("tensor_size")] = 
      llvm::orc::ExecutorSymbolDef(extract_tensor_size_addr, llvm::JITSymbolFlags::Exported);
  throwIfError(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));

  // Compile the module
  compile(pimpl_.get());
}

KernelArgumentHolder HostIrJit::runWithInputs(
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("HostIrJit::runWithInputs");
  // Bind cache id to llvm global variable or align with main function inputs
  NVF_ERROR(args.getCacheId().has_value(), "Cache ID is not set");
  NVF_ERROR_EQ(std::ssize(pimpl_->container->inputs()), args.size());

  std::vector<const at::Tensor*> input_aten_tensors;
  // Bind the inputs to the tensor map
  for (auto&& [in_val, arg] : zip(pimpl_->container->inputs(), args)) {
    if (arg.is<at::Tensor>()) {
      input_aten_tensors.push_back(&arg.as<at::Tensor>());
    }
    else{
      // TODO: handle other primitive types so we can just align them to input of main function
    }
  }

  // Run the main function
  at::Tensor** output_aten_tensors = pimpl_->main_func(input_aten_tensors.data());

  // Collect the outputs
  KernelArgumentHolder outputs;
  for(size_t i = 0; i < pimpl_->container->outputs().size(); ++i) {
    auto* output = pimpl_->container->outputs()[i];
    if(output->isA<TensorView>()) {
      outputs.push(at::Tensor(*output_aten_tensors[i]));
    }
  }
  // free the output_aten_tensors
  delete[] output_aten_tensors;
  return outputs;
}

KernelArgumentHolder HostIrJit::runWithInput(
    const std::unordered_map<Val*, PolymorphicValue>& val_to_PValue) {
  FUSER_PERF_SCOPE("HostIrJit::runWithInput");
  // Bind the inputs to the tensor map
  std::vector<const at::Tensor*> input_aten_tensors;
  for (auto&& [in_val, arg] : val_to_PValue) {
    if (arg.is<at::Tensor>()) {
      input_aten_tensors.push_back(&arg.as<at::Tensor>());
    }
    else{
      // TODO: handle other primitive types so we can just align them to input of main function
    }
  }

  // Run the main function
  at::Tensor** output_aten_tensors = pimpl_->main_func(input_aten_tensors.data());

  // Collect the outputs
  KernelArgumentHolder outputs;
  for(size_t i = 0; i < pimpl_->container->outputs().size(); ++i) {
    auto* output = pimpl_->container->outputs()[i];
    if(output->isA<TensorView>()) {
      outputs.push(at::Tensor(*output_aten_tensors[i]));
    }
  }
  // free the output_aten_tensors
  delete[] output_aten_tensors;
  return outputs;
}

const std::vector<Val*>& HostIrJit::inputs() const {
  return pimpl_->container->inputs();
}

const std::vector<Val*>& HostIrJit::outputs() const {
  return pimpl_->container->outputs();
}

auto* HostIrJit::container() const {
  return pimpl_->container.get();
}

const hir::HostIrContainer& HostIrJit::getHostIrContainer() const {
  return *pimpl_->container;
}

std::ostream& HostIrJit::print(std::ostream& os) const {
  return pimpl_->container->print(os);
}



} // namespace nvfuser
