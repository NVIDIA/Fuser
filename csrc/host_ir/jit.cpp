// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "host_ir/jit.h"

#include <cstdint>
#include <memory>
#include <ranges>
#include <unordered_map>

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/llvm_jit_strings.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/MemoryFormat.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "driver_api.h"
#include "runtime/compiled_kernel.h"
#include "runtime/executor.h"

#include "bfs.h"
#include "expr_evaluator.h"
#include "fusion_profiler.h"
#include "host_ir/evaluator.h"
#include "host_ir/jit_constants.h"
#include "host_ir/jit_external.h"
#include "host_ir/jit_tensor_utils.h"
#include "instrumentation.h"
#include "ir/all_nodes.h"
#include "ir/iostream.h"
#include "linked_hash_map.h"
#include "ops/all_ops.h"
#include "polymorphic_value.h"
#include "runtime/executor_kernel_arg.h"
#include "runtime/fusion_executor_cache.h"
#include "runtime/fusion_kernel_runtime.h"
#include "tensor_metadata.h"
#include "val_graph_visitor.h"

namespace nvfuser {

// cacheId, inputTensors, outputTensors
using main_func_t = void (*)(int64_t, const void**, void**);

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
  void compile();
  void registerExternalFunctions();

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

llvm::Type* getInt8PtrStaticArrayType(
    llvm::LLVMContext& context,
    int64_t size) {
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
  return llvm::PointerType::getUnqual(
      llvm::StructType::create(context, kAtTensorType));
}

llvm::ArrayType* getInt64StaticArrayType(
    llvm::LLVMContext& context,
    int64_t size) {
  return llvm::ArrayType::get(llvm::Type::getInt64Ty(context), size);
}

llvm::Type* getInt64PtrType(llvm::LLVMContext& context) {
  return llvm::PointerType::getUnqual(llvm::Type::getInt64Ty(context));
}

// Helper function to insert nvtxRangePush call
void insertNvtxRangePush(const char* op_name, llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Function* nvtx_range_push_func =
      module->getFunction(kNvtxRangePushFuncName);

  llvm::Value* op_name_ptr = builder.CreateGlobalString(op_name);
  builder.CreateCall(nvtx_range_push_func, {op_name_ptr});
}

void insertNvtxRangePop(llvm::IRBuilder<>& builder) {
  llvm::Module* module = builder.GetInsertBlock()->getParent()->getParent();
  llvm::Function* nvtx_range_pop_func =
      module->getFunction(kNvtxRangePopFuncName);

  // Call nvtxRangePop function
  builder.CreateCall(nvtx_range_pop_func, {});
}

// Helper function to print generated LLVM IR after each node is processed
void printLlvmIr(llvm::Function* func, std::string_view msg) {
  llvm::outs() << "=== LLVM IR After Generating " << msg << " ===\n";
  func->print(llvm::outs(), nullptr);
  llvm::outs() << "\n\n";
}

void checkMemoryLeak(llvm::Module& module) {
  auto& context = module.getContext();
  auto* main_func = module.getFunction(kMainFuncName);
  std::unordered_set<llvm::Value*> allocated_tensors;
  for (auto& bb : *main_func) {
    for (auto& inst : bb) {
      if (auto* call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
        auto* called_func = call->getCalledFunction();
        NVF_ERROR(
            called_func != nullptr,
            "LLVM Lowering Error: called an indirect function");

        auto* return_type = called_func->getReturnType();
        auto func_name = called_func->getName().str();

        // Note: In opaque pointer mode (default since LLVM 15), all pointers
        // are evaluated with the same (ptr) type. We need to exclude helper
        // functions that return data pointers but don't allocate tensors.
        if (return_type == getTensorPtrType(context) &&
            func_name != kTensorDataPtrFuncName) {
          // (new_tensor, set_tensor, reshape, permute, etc.)
          allocated_tensors.insert(call);
          continue;
        }

        // Remove if we have a corresponding delete
        if (called_func->getName().str() == kDeleteTensorFuncName) {
          auto* tensor_to_delete = call->getOperand(0);
          auto it = allocated_tensors.find(tensor_to_delete);
          NVF_ERROR(
              it != allocated_tensors.end(),
              "Extra tensor deallocation detected: tensor was never allocated");
          allocated_tensors.erase(it);
          continue;
        }
      }

      if (auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
        auto* dest = store->getPointerOperand();
        auto* src = store->getValueOperand();
        if (dest->getName().str().find(kMainFuncOutputTensorName) !=
            std::string::npos) {
          auto it = allocated_tensors.find(src);
          if (it != allocated_tensors.end()) {
            allocated_tensors.erase(it);
          }
        }
      }
    }
  }
  NVF_ERROR(
      allocated_tensors.empty(),
      "Memory leak detected: ",
      allocated_tensors.size(),
      " tensors allocated but not deallocated");
}

void unpackInputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();

  insertNvtxRangePush("unpackInputs", builder);

  llvm::Function* func = builder.GetInsertBlock()->getParent();

  // Get the cacheId from the main function's first argument
  llvm::Value* cache_id = func->getArg(0);
  cache_id->setName("cacheId");
  // NOTE: Currently we can only grab cacheId by traversing all vals
  // In the future we should add a cacheId to the host ir container or fusion
  bool found_cache_id = false;
  Val* cache_id_val = nullptr;
  for (Val* val : container->deterministic_vals()) {
    if (auto* named_scalar = dynamic_cast<NamedScalar*>(val)) {
      if (named_scalar->name() == "cacheId") {
        if (found_cache_id) {
          NVF_ERROR(
              named_scalar != cache_id_val,
              "cacheId is not the first deterministic val");
        }
        cache_id_val = named_scalar;
        val_to_value[cache_id_val] = cache_id;
        found_cache_id = true;
      }
    }
  }

  // Get the current function (main) and its input tensor array
  llvm::Value* main_func_input_array = func->getArg(1);
  main_func_input_array->setName("KernelInputArgs");

  llvm::Type* input_args_type = getInt8PtrDynamicArrayType(context);
  llvm::Type* tensor_type = getTensorPtrType(context);

  // bind input aten tensor sizes to val_to_value
  for (const auto [i, input] : enumerate(container->inputs())) {
    if (auto* tv = dynamic_cast<TensorView*>(input)) {
      llvm::Value* tensor_addr = builder.CreateGEP(
          input_args_type, main_func_input_array, {builder.getInt64(i)});
      // Load the actual tensor pointer from the array
      llvm::Value* tensor = builder.CreateLoad(tensor_type, tensor_addr);
      tensor->setName(ir_utils::varName(tv));
      // bind input aten tensor sizes to val_to_value
      // TODO: We should validate const size and strides here, ie. dim check
      for (const auto [dim_idx, id] :
           enumerate(TensorDomain::noReductions(tv->getLogicalDomain()))) {
        if (id->isBroadcast()) {
          val_to_value[id->extent()] = builder.getInt64(1);
          if (id->hasExpandedExtent()) {
            val_to_value[id->expandedExtent()] =
                createTensorSize(tensor, dim_idx, builder);
          }
        } else {
          val_to_value[id->extent()] =
              createTensorSize(tensor, dim_idx, builder);
        }
      }
      // bind input aten tensor to val_to_value
      val_to_value[tv] = tensor;
    } else if (input->dtype() == DataType::Index) {
      // NOTE: we currently only support index scalar inputs, we need to support
      // other scalar types in the future
      llvm::Value* scalar_addr = builder.CreateGEP(
          input_args_type, main_func_input_array, {builder.getInt64(i)});
      llvm::Value* int64_ptr =
          builder.CreateBitCast(scalar_addr, getInt64PtrType(context));
      llvm::Value* scalar =
          builder.CreateLoad(llvm::Type::getInt64Ty(context), int64_ptr);
      scalar->setName(ir_utils::varName(input));
      val_to_value[input] = scalar;
    } else {
      NVF_THROW("Unsupported expression type: ", input);
    }
  }
  insertNvtxRangePop(builder);
  if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
    printLlvmIr(func, "Main Function Inputs");
  }
}

void packOutputs(
    const hir::HostIrContainer* container,
    llvm::IRBuilder<>& builder,
    std::unordered_map<Val*, llvm::Value*>& val_to_value) {
  llvm::LLVMContext& context = builder.getContext();
  insertNvtxRangePush("packOutputs", builder);
  // Get the current function (main) and its output tensor array
  llvm::Function* func = builder.GetInsertBlock()->getParent();
  llvm::Value* output_args = func->getArg(2);
  output_args->setName("KernelOutputArgs");

  llvm::Type* output_args_type = getInt8PtrDynamicArrayType(context);
  // Store output tensor pointers from val_to_value into the output array
  for (const auto [i, output] : enumerate(container->outputs())) {
    if (auto* tv = dynamic_cast<TensorView*>(output)) {
      llvm::Value* tensor_addr = builder.CreateGEP(
          output_args_type, output_args, {builder.getInt64(i)});
      tensor_addr->setName(kMainFuncOutputTensorName);

      // Get the tensor pointer from val_to_value and store it in the output
      // array
      llvm::Value* tensor = getOrDefault(val_to_value, tv);
      NVF_ERROR(tensor != nullptr)
      builder.CreateStore(tensor, tensor_addr);
    } else if (auto* named_scalar = dynamic_cast<NamedScalar*>(output)) {
      llvm::Value* scalar_addr = builder.CreateGEP(
          output_args_type, output_args, {builder.getInt64(i)});
      llvm::Value* scalar = getOrDefault(val_to_value, named_scalar);
      NVF_ERROR(scalar != nullptr)
      builder.CreateStore(scalar, scalar_addr);
    } else {
      NVF_THROW("Unsupported expression type: ", output);
    }
  }
  insertNvtxRangePop(builder);
  builder.CreateRetVoid();
  if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
    printLlvmIr(func, "Main Function Outputs");
  }
}

void compileFunctionDeclarations(
    llvm::Module* module,
    llvm::LLVMContext& context) {
  // Get the types
  auto* void_type = llvm::Type::getVoidTy(context);
  auto* void_ptr_type = getInt8PtrType(context);
  auto* void_array_ptr_type = getInt8PtrDynamicArrayType(context);
  auto* int64_type = llvm::Type::getInt64Ty(context);
  auto* int64_ptr_type = getInt64PtrType(context);
  auto* int32_type = llvm::Type::getInt32Ty(context);
  auto* tensor_type = getTensorPtrType(context);

  // tensor_size function: int64_t tensor_size(at::Tensor* tensor, int64_t dim)
  auto* tensor_size_type =
      llvm::FunctionType::get(int64_type, {tensor_type, int64_type}, false);
  llvm::Function::Create(
      tensor_size_type,
      llvm::Function::ExternalLinkage,
      kTensorSizeFuncName,
      module);

  // tensor_stride function: int64_t tensor_stride(at::Tensor* tensor, int64_t
  // dim)
  llvm::Function::Create(
      tensor_size_type, // Same signature as tensor_size
      llvm::Function::ExternalLinkage,
      kTensorStrideFuncName,
      module);

  // tensor_data_ptr function: void* tensor_data_ptr(at::Tensor* tensor)
  auto* tensor_data_ptr_type =
      llvm::FunctionType::get(void_ptr_type, {tensor_type}, false);
  llvm::Function::Create(
      tensor_data_ptr_type,
      llvm::Function::ExternalLinkage,
      kTensorDataPtrFuncName,
      module);

  // new_tensor function: at::Tensor* new_tensor()
  auto* new_tensor_type = llvm::FunctionType::get(tensor_type, {}, false);
  llvm::Function::Create(
      new_tensor_type,
      llvm::Function::ExternalLinkage,
      kNewTensorFuncName,
      module);

  // set_tensor function: at::Tensor* set_tensor(at::Tensor* tensor)
  auto* set_tensor_type =
      llvm::FunctionType::get(tensor_type, {tensor_type}, false);
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
       tensor_type},
      false);
  llvm::Function::Create(
      empty_strided_cuda_type,
      llvm::Function::ExternalLinkage,
      kAtEmptyStridedCudaWrapper,
      module);

  // delete_tensor function: void delete_tensor(at::Tensor* tensor)
  auto* delete_tensor_type =
      llvm::FunctionType::get(void_type, {tensor_type}, false);
  llvm::Function::Create(
      delete_tensor_type,
      llvm::Function::ExternalLinkage,
      kDeleteTensorFuncName,
      module);

  // nvtx_range_push function: void nvtx_range_push(const char* name)
  auto* nvtx_range_push_type =
      llvm::FunctionType::get(void_type, {void_ptr_type}, false);
  llvm::Function::Create(
      nvtx_range_push_type,
      llvm::Function::ExternalLinkage,
      kNvtxRangePushFuncName,
      module);

  // nvtx_range_pop function: void nvtx_range_pop()
  auto* nvtx_range_pop_type = llvm::FunctionType::get(void_type, {}, false);
  llvm::Function::Create(
      nvtx_range_pop_type,
      llvm::Function::ExternalLinkage,
      kNvtxRangePopFuncName,
      module);

  // launch_kernel_direct function: void launch_kernel_direct(
  //   void** kernel_args, void* cuda_function_ptr,
  //   int64_t gdimx, int64_t gdimy, int64_t gdimz,
  //   int64_t bdimx, int64_t bdimy, int64_t bdimz, int64_t smem)
  auto* launch_kernel_direct_type = llvm::FunctionType::get(
      void_type,
      {void_array_ptr_type, // void** kernel_args
       void_ptr_type, // cuda_function_ptr
       int64_type, // gdimx
       int64_type, // gdimy
       int64_type, // gdimz
       int64_type, // bdimx
       int64_type, // bdimy
       int64_type, // bdimz
       int64_type}, // smem
      false);
  llvm::Function::Create(
      launch_kernel_direct_type,
      llvm::Function::ExternalLinkage,
      kLaunchKernelDirectFuncName,
      module);

  // matmul_out function: void matmul_out(at::Tensor* out, at::Tensor* a,
  // at::Tensor* b)
  auto* matmul_out_type = llvm::FunctionType::get(
      void_type, {tensor_type, tensor_type, tensor_type}, false);
  llvm::Function::Create(
      matmul_out_type,
      llvm::Function::ExternalLinkage,
      kMatmulOutFuncName,
      module);

  // linear_out function: void linear_out(at::Tensor* out, at::Tensor* in,
  // at::Tensor* weight, at::Tensor* bias = nullptr)
  auto* linear_out_type = llvm::FunctionType::get(
      void_type, {tensor_type, tensor_type, tensor_type, tensor_type}, false);
  llvm::Function::Create(
      linear_out_type,
      llvm::Function::ExternalLinkage,
      kLinearOutFuncName,
      module);

  // permute function: at::Tensor* permute(at::Tensor* in, const int64_t*
  // permutation, int64_t perm_size)
  auto* permute_type = llvm::FunctionType::get(
      tensor_type, {tensor_type, int64_ptr_type, int64_type}, false);
  llvm::Function::Create(
      permute_type, llvm::Function::ExternalLinkage, kPermuteFuncName, module);

  // reshape function: at::Tensor* reshape(at::Tensor* in, const int64_t* shape,
  // int64_t shape_size)
  auto* reshape_type = llvm::FunctionType::get(
      tensor_type, {tensor_type, int64_ptr_type, int64_type}, false);
  llvm::Function::Create(
      reshape_type, llvm::Function::ExternalLinkage, kReshapeFuncName, module);

  // main function: void main(void** input_tensors, void** output_tensors)
  auto* main_type = llvm::FunctionType::get(
      void_type, {int64_type, void_array_ptr_type, void_array_ptr_type}, false);
  llvm::Function::Create(
      main_type, llvm::Function::ExternalLinkage, kMainFuncName, module);
}

// Not handled instructions automatically trigger an error.
class HostIrCompileDispatcher : public OptInDispatch {
 public:
  HostIrCompileDispatcher(
      llvm::IRBuilder<>& builder,
      std::unordered_map<Val*, llvm::Value*>& val_to_value)
      : builder_(builder), val_to_value_(val_to_value) {}
  using OptInDispatch::handle;

  void handle(ReshapeOp* vop) final {
    auto* in_tv = vop->in()->as<TensorView>();
    auto* out_tv = vop->out()->as<TensorView>();
    llvm::Value* in_tensor = getOrDefault(val_to_value_, in_tv);
    NVF_ERROR(in_tensor != nullptr)
    llvm::Value* out_tensor = getOrDefault(val_to_value_, out_tv);
    NVF_ERROR(out_tensor == nullptr)

    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::LLVMContext& context = builder_.getContext();

    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_sizes;
    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_strides;
    inferTensorShapesAndStrides(
        out_tv, val_to_value_, builder_, tensor_sizes, tensor_strides);

    const std::vector<IterDomain*>& logical_domain =
        TensorDomain::noReductions(out_tv->getLogicalDomain());

    NVF_ERROR_EQ(tensor_sizes.size(), logical_domain.size());

    llvm::ArrayType* sizes_type =
        getInt64StaticArrayType(context, tensor_sizes.size());
    llvm::Value* sizes_array =
        builder_.CreateAlloca(sizes_type, nullptr, "sizes");
    for (auto [i, tensor_size] : enumerate(tensor_sizes)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          sizes_type,
          sizes_array,
          {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(tensor_size, gep);
    }

    llvm::Value* sizes_ptr =
        builder_.CreateBitCast(sizes_array, getInt64PtrType(context));
    out_tensor = builder_.CreateCall(
        module->getFunction(kReshapeFuncName),
        {in_tensor, sizes_ptr, builder_.getInt64(tensor_sizes.size())});
    val_to_value_[out_tv] = out_tensor;
  }

  void handle(LoadStoreOp* load_store_op) final {
    NVF_ERROR(
        load_store_op->opType() == LoadStoreOpType::Set ||
        load_store_op->opType() == LoadStoreOpType::SegmenterSet);
    NVF_ERROR(
        load_store_op->out()->isA<TensorView>(), "out must be a TensorView");
    auto* in_tv = load_store_op->in()->as<TensorView>();
    auto* out_tv = load_store_op->out()->as<TensorView>();
    llvm::Value* in_tensor = getOrDefault(val_to_value_, in_tv);
    NVF_ERROR(in_tensor != nullptr)
    // we assume all output tensors are already created, either through new or
    // allocated
    llvm::Value* out_tensor = getOrDefault(val_to_value_, out_tv);
    NVF_ERROR(out_tensor == nullptr)

    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::LLVMContext& context = builder_.getContext();

    if (out_tv->hasRoot()) {
      std::optional<std::vector<int64_t>> permutation =
          ir_utils::computePermutation(
              out_tv->getRootDomain(), out_tv->getLogicalDomain());
      NVF_ERROR(
          permutation.has_value(),
          "The logical domain of a Set.Permute is supposed to be a permutation"
          " of the root domain: ",
          out_tv);

      // Create array of permutation values
      llvm::ArrayType* perm_array_type =
          getInt64StaticArrayType(context, permutation.value().size());
      llvm::Value* perm_array =
          builder_.CreateAlloca(perm_array_type, nullptr, "permutation");

      for (auto [i, extent] : enumerate(permutation.value())) {
        llvm::Value* gep = builder_.CreateInBoundsGEP(
            perm_array_type,
            perm_array,
            {builder_.getInt32(0), builder_.getInt32(i)});
        builder_.CreateStore(builder_.getInt64(extent), gep);
      }

      llvm::Type* int64_ptr_type = getInt64PtrType(context);
      llvm::Value* perm_ptr =
          builder_.CreateBitCast(perm_array, int64_ptr_type);
      llvm::Value* perm_size = builder_.getInt64(permutation.value().size());
      out_tensor = builder_.CreateCall(
          module->getFunction(kPermuteFuncName),
          {in_tensor, perm_ptr, perm_size},
          "permute");
      val_to_value_[out_tv] = out_tensor;
      return;
    }
    out_tensor = builder_.CreateCall(
        module->getFunction(kSetTensorFuncName), {in_tensor}, "set");
    val_to_value_[out_tv] = out_tensor;
  }

  void handle(MatmulOp* matmul_op) final {
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();

    llvm::Value* a = getOrDefault(val_to_value_, matmul_op->inA());
    llvm::Value* b = getOrDefault(val_to_value_, matmul_op->inB());
    llvm::Value* out = getOrDefault(val_to_value_, matmul_op->out());
    NVF_ERROR(out != nullptr);
    builder_.CreateCall(module->getFunction(kMatmulOutFuncName), {out, a, b});
  }

  void handle(LinearOp* linear_op) final {
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::LLVMContext& context = builder_.getContext();

    llvm::Value* in = getOrDefault(val_to_value_, linear_op->inA());
    NVF_ERROR(in != nullptr)
    llvm::Value* weight = getOrDefault(val_to_value_, linear_op->inB());
    NVF_ERROR(weight != nullptr)
    llvm::Value* out = getOrDefault(val_to_value_, linear_op->out());

    llvm::Value* bias = nullptr;
    if (linear_op->hasBias()) {
      bias = getOrDefault(val_to_value_, linear_op->bias());
      NVF_ERROR(bias != nullptr)
    } else {
      // Create a proper null pointer for LLVM
      auto* tensor_type = getTensorPtrType(context);
      bias = llvm::ConstantPointerNull::get(
          llvm::cast<llvm::PointerType>(tensor_type));
    }
    NVF_ERROR(out != nullptr);
    builder_.CreateCall(
        module->getFunction(kLinearOutFuncName), {out, in, weight, bias});
  }

  void handle(hir::LaunchKernel* launch_kernel) final {
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::LLVMContext& context = builder_.getContext();
    auto* void_ptr_type = getInt8PtrType(context);
    auto* void_array_ptr_type = getInt8PtrDynamicArrayType(context);

    // Get index type from CompiledKernel
    PrimDataType index_type =
        launch_kernel->compiledKernel()->kernel()->indexType();

    // Pack each input/output argument using LLVM IR
    llvm::SmallVector<llvm::Value*, 16> packed_buffers;

    // Helper lambda to pack a single Val (tensor or scalar)
    auto packArgument = [&](Val* val) {
      if (auto* tv = dynamic_cast<TensorView*>(val)) {
        // Pack tensor argument
        llvm::Value* tensor = getOrDefault(val_to_value_, tv);
        NVF_ERROR(
            tensor != nullptr, "Tensor not found in val_to_value map: ", val);
        packed_buffers.push_back(packTensorArgument(
            tensor, tv, index_type, val_to_value_, builder_));
      } else {
        // Pack scalar argument
        llvm::Value* scalar = getOrDefault(val_to_value_, val);
        NVF_ERROR(
            scalar != nullptr, "Scalar not found in val_to_value map: ", val);

        // For scalars, we need to create a stack allocation and get its pointer
        // The scalar value is already an LLVM value (e.g., i64)
        // We need to store it in memory and pass a pointer to that memory
        llvm::Value* scalar_alloca = builder_.CreateAlloca(scalar->getType());
        builder_.CreateStore(scalar, scalar_alloca);

        // Cast to i8* (void*)
        llvm::Value* scalar_ptr =
            builder_.CreateBitCast(scalar_alloca, void_ptr_type);
        packed_buffers.push_back(scalar_ptr);
      }
    };

    // Pack inputs
    for (auto* in : launch_kernel->inputs()) {
      packArgument(in);
    }

    // Pack outputs
    for (auto* out : launch_kernel->outputs()) {
      packArgument(out);
    }

    // Create kernel_args array (void**)
    auto* args_array_type =
        llvm::ArrayType::get(void_ptr_type, packed_buffers.size());
    llvm::Value* args_array =
        builder_.CreateAlloca(args_array_type, nullptr, "kernel_args_array");

    for (auto [i, packed_buffer] : enumerate(packed_buffers)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          args_array_type,
          args_array,
          {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(packed_buffer, gep);
    }

    // Cast to void**
    llvm::Value* args_array_ptr =
        builder_.CreateBitCast(args_array, void_array_ptr_type);

    // Get launch parameters from LaunchParams (compile-time constants)
    const LaunchParams& lp = launch_kernel->launchParams();

    llvm::Value* gdimx = builder_.getInt64(lp.gdimx());
    llvm::Value* gdimy = builder_.getInt64(lp.gdimy());
    llvm::Value* gdimz = builder_.getInt64(lp.gdimz());
    llvm::Value* bdimx = builder_.getInt64(lp.bdimx());
    llvm::Value* bdimy = builder_.getInt64(lp.bdimy());
    llvm::Value* bdimz = builder_.getInt64(lp.bdimz());
    llvm::Value* smem = builder_.getInt64(lp.smem());

    // Get CUDA function pointer from CompiledKernel
    CUfunction cuda_function =
        launch_kernel->compiledKernel()->cudaExecutable()->function;
    llvm::Value* function_ptr = builder_.CreateIntToPtr(
        builder_.getInt64(reinterpret_cast<uintptr_t>(cuda_function)),
        void_ptr_type);

    // Call launch_kernel_direct with all parameters
    builder_.CreateCall(
        module->getFunction(kLaunchKernelDirectFuncName),
        {args_array_ptr,
         function_ptr,
         gdimx,
         gdimy,
         gdimz,
         bdimx,
         bdimy,
         bdimz,
         smem});
  }

  void handle(kir::Allocate* allocate) final {
    llvm::LLVMContext& context = builder_.getContext();
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();

    llvm::Type* int64_ptr_type = getInt64PtrType(context);

    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_sizes;
    llvm::SmallVector<llvm::Value*, kMaxTensorDim> tensor_strides;
    inferTensorShapesAndStrides(
        allocate->buffer()->as<TensorView>(),
        val_to_value_,
        builder_,
        tensor_sizes,
        tensor_strides);

    const std::vector<IterDomain*>& logical_domain = TensorDomain::noReductions(
        allocate->buffer()->as<TensorView>()->getLogicalDomain());

    NVF_ERROR_EQ(tensor_sizes.size(), logical_domain.size());

    llvm::ArrayType* sizes_type =
        getInt64StaticArrayType(context, tensor_sizes.size());
    llvm::ArrayType* strides_type =
        getInt64StaticArrayType(context, tensor_strides.size());

    llvm::Value* sizes = builder_.CreateAlloca(sizes_type, nullptr, "sizes");
    llvm::Value* strides =
        builder_.CreateAlloca(strides_type, nullptr, "strides");

    for (const auto [i, size] : enumerate(tensor_sizes)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          sizes_type, sizes, {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(size, gep);
    }

    for (const auto [i, stride] : enumerate(tensor_strides)) {
      llvm::Value* gep = builder_.CreateInBoundsGEP(
          strides_type, strides, {builder_.getInt32(0), builder_.getInt32(i)});
      builder_.CreateStore(stride, gep);
    }

    // Convert arrays to pointers
    llvm::Value* sizes_arg = builder_.CreateBitCast(sizes, int64_ptr_type);
    llvm::Value* strides_arg = builder_.CreateBitCast(strides, int64_ptr_type);

    // Create array size arguments
    llvm::Value* shape_ndim_arg = builder_.getInt64(tensor_sizes.size());
    llvm::Value* strides_ndim_arg = builder_.getInt64(tensor_strides.size());

    // Create output tensor
    llvm::Value* out_tensor = builder_.CreateCall(
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
         out_tensor});
    val_to_value_[allocate->buffer()] = out_tensor;
  }

  void handle(hir::Deallocate* deallocate) final {
    llvm::Module* module = builder_.GetInsertBlock()->getParent()->getParent();
    llvm::Function* delete_tensor_func =
        module->getFunction(kDeleteTensorFuncName);
    builder_.CreateCall(
        delete_tensor_func, {val_to_value_.at(deallocate->buffer())});
  }

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
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(
      *context, "entry", module->getFunction(kMainFuncName));
  builder.SetInsertPoint(entry);

  // compile inputs in llvm ir
  unpackInputs(container_.get(), builder, val_to_value);
  HostIrCompileDispatcher dispatcher(builder, val_to_value);
  // compile all top level expressions in host ir container
  for (auto* expr : container_->topLevelExprs()) {
    insertNvtxRangePush(expr->getOpString(), builder);
    dispatcher.dispatch(expr);
    insertNvtxRangePop(builder);
    if (isDebugDumpEnabled(DebugDumpOption::HostIrJit)) {
      printLlvmIr(builder.GetInsertBlock()->getParent(), expr->getOpString());
    }
  }

  // compile outputs in llvm ir
  packOutputs(container_.get(), builder, val_to_value);

  // check memory leak
  checkMemoryLeak(*module);

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
  dest_dynamic_lib.addGenerator(throwIfError(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit_->getDataLayout().getGlobalPrefix())));

  // Call implementation in jit_external.cpp
  registerExternalFunctionsImpl(jit_.get(), dest_dynamic_lib);
}

// NOTE: we delete output tensors created in llvm main function here
KernelArgumentHolder HostIrJitImpl::runWithInputs(
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("HostIrJitImpl::runWithInputs");
  // Bind cache id to llvm global variable or align with main function inputs
  NVF_ERROR(args.getCacheId().has_value(), "Cache ID is not set");
  NVF_ERROR_EQ(std::ssize(container_->inputs()), args.size());

  std::unordered_set<const at::Tensor*> preserved_tensors;
  std::vector<const void*> input_aten_tensors;
  // Bind the inputs to the tensor map
  for (auto [in_val, arg] : zip(container_->inputs(), args)) {
    if (arg.is<at::Tensor>()) {
      const auto* aten_tensor = &arg.as<at::Tensor>();
      preserved_tensors.insert(aten_tensor);
      input_aten_tensors.push_back(aten_tensor);
    }
    // NOTE: we currently only support index scalar inputs, we need to support
    // other scalar types in the future
    else if (in_val->dtype() == DataType::Index) {
      // Cast int64_t to void* for the mixed array
      auto scalar_value = arg.as<int64_t>();
      input_aten_tensors.push_back(reinterpret_cast<const void*>(scalar_value));
    } else {
      NVF_THROW("Unsupported argument type: ", arg, " for input ", in_val);
    }
  }

  // Run the main function
  std::vector<void*> output_aten_tensors(container_->outputs().size());
  main_func_(
      args.getCacheId().value(),
      input_aten_tensors.data(),
      output_aten_tensors.data());

  // Collect the outputs
  KernelArgumentHolder outputs;
  for (const auto [output, tensor] :
       zip(container_->outputs(), output_aten_tensors)) {
    // NOTE: we currently only support tensor outputs, we need to support other
    // types in the future
    NVF_ERROR(
        output->isA<TensorView>(),
        "Unsupported output type: ",
        output,
        " for output ",
        output);
    at::Tensor* aten_tensor = static_cast<at::Tensor*>(tensor);
    outputs.push(*aten_tensor);
    // Clean up the individual tensor object (not the array)
    if (preserved_tensors.find(aten_tensor) == preserved_tensors.end()) {
      delete aten_tensor;
    }
  }
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
