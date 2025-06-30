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
#include <c10/core/MemoryFormat.h>
#include <host_ir/jit.h>
#include <functional>
#include <memory>

#include <multidevice/communicator.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

using allocate_fn = std::function<
    void(const int64_t*, int64_t, const int64_t*, int64_t, at::Tensor&)>;

/*
input: cache id
output: KernelArgumentHolder
*/ 
struct KernelArgumentHolderPair {
  void* args;      // KernelArgumentHolder* for inputs
  void* outputs;   // KernelArgumentHolder* for outputs
};

using launch_kernel_fn = std::function<KernelArgumentHolderPair(int64_t, at::Tensor**, at::Tensor**)>;

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

using main_func_fn = std::function<void(at::Tensor**, int64_t, at::Tensor**)>;

// PIMPL implementation for HostIrJit
struct HostIrJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unordered_map<const kir::Allocate*, allocate_fn> allocate_funcs_;
  std::unordered_map<const hir::LaunchKernel*, launch_kernel_fn> launch_kernel_funcs_;
  main_func_fn main_func_;
  std::unique_ptr<HostIrJitParams> host_ir_jit_params_;
  std::unordered_map<Val*, llvm::Value*> val2llvmMap;
  std::unordered_map<const kir::Allocate*, at::Tensor> allocate_tensors_;
  LlvmJitImpl() = default;
  ~LlvmJitImpl() = default;
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

// Generate a function for LaunchKernel runtime
void generateLaunchKernelFunc(
    const hir::LaunchKernel* launch_kernel,
    llvm::Module* mod) {
  llvm::LLVMContext& context = mod->getContext();
  llvm::IRBuilder<> builder(context);

  std::string func_name = launch_kernel->toString();
  
  // Since we registered the wrapper functions with these exact names,
  // we can look them up directly without mangling
  std::string constructor_name = "KernelArgumentHolder::KernelArgumentHolder";
  std::string set_cache_name = "KernelArgumentHolder::setCacheId";
  std::string set_device_name = "KernelArgumentHolder::setDeviceIndex";
  std::string push_name = "KernelArgumentHolder::push";
  
  // Look up functions using the registered names
  llvm::Function* constructor_func = mod->getFunction(constructor_name);
  if (!constructor_func) {
    // Create function declaration for constructor
    llvm::FunctionType* ctor_type = llvm::FunctionType::get(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // return KernelArgumentHolder*
      {}, // no parameters for default constructor
      false
    );
    constructor_func = llvm::Function::Create(
      ctor_type, llvm::Function::ExternalLinkage, constructor_name, mod
    );
  }

  llvm::Function* push_func = mod->getFunction(push_name);
  if (!push_func) {
    // Create function declaration for member function
    std::vector<llvm::Type*> param_types = {
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // this pointer
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))    // at::Tensor object (passed as pointer for simplicity)
    };
    llvm::FunctionType* push_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), param_types, false
    );
    push_func = llvm::Function::Create(
      push_type, llvm::Function::ExternalLinkage, push_name, mod
    );
  }
  
  llvm::Function* set_cache_func = mod->getFunction(set_cache_name);
  if (!set_cache_func) {
    // Create function declaration for member function
    std::vector<llvm::Type*> param_types = {
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // this pointer
      llvm::Type::getInt64Ty(context)    // size_t parameter
    };
    llvm::FunctionType* set_cache_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), param_types, false
    );
    set_cache_func = llvm::Function::Create(
      set_cache_type, llvm::Function::ExternalLinkage, set_cache_name, mod
    );
  }
  
  llvm::Function* set_device_func = mod->getFunction(set_device_name);
  if (!set_device_func) {
    // Create function declaration for setDeviceIndex
    std::vector<llvm::Type*> param_types = {
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)) // this pointer only
    };
    llvm::FunctionType* set_device_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), param_types, false
    );
    set_device_func = llvm::Function::Create(
      set_device_type, llvm::Function::ExternalLinkage, set_device_name, mod
    );
  }
  
  // Create the main function
  // Parameters: cache_id, input_tensors_ptr, output_tensors_ptr
  std::vector<llvm::Type*> param_types = {
    llvm::Type::getInt64Ty(context),  // cache_id
    llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))), // input_tensors_ptr (at::Tensor**)
    llvm::PointerType::getUnqual(llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)))  // output_tensors_ptr (at::Tensor**)
  };
  
  // Create struct type for return value
  std::vector<llvm::Type*> struct_elements = {
    llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), // args pointer
    llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))  // outputs pointer
  };
  llvm::StructType* return_struct_type = llvm::StructType::create(context, struct_elements, "KernelArgumentHolderPair");
  
  llvm::FunctionType* main_func_type = llvm::FunctionType::get(
    return_struct_type, // return KernelArgumentHolderPair
    param_types,
    false
  );
  
  llvm::Function* main_func = llvm::Function::Create(
    main_func_type, llvm::Function::ExternalLinkage, func_name, mod
  );
  
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", main_func);
  builder.SetInsertPoint(entry);
  
  // Get function arguments
  llvm::Value* cache_id_arg = main_func->getArg(0);
  llvm::Value* input_tensors_ptr = main_func->getArg(1);
  llvm::Value* output_tensors_ptr = main_func->getArg(2);
  
  // Create KernelArgumentHolder args (for inputs)
  llvm::Value* args_ptr = builder.CreateCall(constructor_func, {});
  
  // Set cache ID if not monostate (cache_id != -1)
  llvm::Value* monostate_check = builder.CreateICmpNE(cache_id_arg, llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), -1));
  llvm::BasicBlock* set_cache_block = llvm::BasicBlock::Create(context, "set_cache", main_func);
  llvm::BasicBlock* skip_cache_block = llvm::BasicBlock::Create(context, "skip_cache", main_func);
  builder.CreateCondBr(monostate_check, set_cache_block, skip_cache_block);
  
  builder.SetInsertPoint(set_cache_block);
  builder.CreateCall(set_cache_func, {args_ptr, cache_id_arg});
  builder.CreateBr(skip_cache_block);
  
  builder.SetInsertPoint(skip_cache_block);
  
  // Push all input tensors to args
  for (size_t i = 0; i < launch_kernel->inputs().size(); ++i) {
    llvm::Value* tensor_ptr = builder.CreateGEP(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      input_tensors_ptr, 
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), i)
    );
    llvm::Value* input_tensor = builder.CreateLoad(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      tensor_ptr
    );
    
    // Push tensor to args - pass the tensor pointer to our wrapper
    builder.CreateCall(push_func, {args_ptr, input_tensor});
  }
  
  // Create KernelArgumentHolder outputs (for outputs)
  llvm::Value* outputs_ptr = builder.CreateCall(constructor_func, {});
  
  // Push all output tensors to outputs
  for (size_t i = 0; i < launch_kernel->outputs().size(); ++i) {
    llvm::Value* tensor_ptr = builder.CreateGEP(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      output_tensors_ptr, 
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), i)
    );
    llvm::Value* output_tensor = builder.CreateLoad(
      llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), 
      tensor_ptr
    );
    
    // Push tensor to outputs - pass the tensor pointer to our wrapper
    builder.CreateCall(push_func, {outputs_ptr, output_tensor});
  }
  
  // Set device index on args
  builder.CreateCall(set_device_func, {args_ptr});
  
  // Create the return struct
  llvm::Value* return_struct = llvm::UndefValue::get(return_struct_type);
  return_struct = builder.CreateInsertValue(return_struct, args_ptr, 0);
  return_struct = builder.CreateInsertValue(return_struct, outputs_ptr, 1);
  
  // Return the struct containing both args and outputs
  builder.CreateRet(return_struct);

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


llvm::Value* traverseExtentDFS(Val* val, std::unordered_map<Val*, llvm::Value*>& val2llvmMap, llvm::LLVMContext& context, llvm::IRBuilder<>& builder) {
  if (val2llvmMap.find(val) != val2llvmMap.end()) {
    return val2llvmMap[val];
  }
  if (val->definition() != nullptr) {
    auto* def = val->definition();
    if(auto* binary_op = def->as<BinaryOp>()) {
      auto* left = binary_op->lhs()->as<Val>();
      auto* right = binary_op->rhs()->as<Val>();
      if(left->isConst() && val2llvmMap.find(left) == val2llvmMap.end()) {
        val2llvmMap[left] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), left->value().as<int64_t>());
      }
      else if(!left->isConst() && val2llvmMap.find(left) == val2llvmMap.end()) {
        traverseExtentDFS(left, val2llvmMap, context, builder);
      }
      if(right->isConst() && val2llvmMap.find(right) == val2llvmMap.end()) {
        val2llvmMap[right] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), right->value().as<int64_t>());
      }
      else if(!right->isConst() && val2llvmMap.find(right) == val2llvmMap.end()) {
        traverseExtentDFS(right, val2llvmMap, context, builder);
      }
      if(binary_op->getBinaryOpType() == BinaryOpType::Add) {
        val2llvmMap[val] = builder.CreateAdd(val2llvmMap[left], val2llvmMap[right]);
      }
      else if(binary_op->getBinaryOpType() == BinaryOpType::Sub) {
        val2llvmMap[val] = builder.CreateSub(val2llvmMap[left], val2llvmMap[right]);
      }
      else if(binary_op->getBinaryOpType() == BinaryOpType::Mul) {
        val2llvmMap[val] = builder.CreateMul(val2llvmMap[left], val2llvmMap[right]);
      }
    }
  }
  else if(val->isConst()) {
    val2llvmMap[val] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), val->value().as<int64_t>());
  }
  else{
    std::cout << "val: " << val->toString() << " is not a binary op or constant" << std::endl;
  }
  return val2llvmMap[val];
}

std::vector<llvm::Value*> getContiguousStrides(
    const std::vector<llvm::Value*>& sizes,
    const std::vector<bool>& expand_flags,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  NVF_ERROR(sizes.size() == expand_flags.size());

  std::vector<llvm::Value*> strides(sizes.size());
  llvm::Value* cur_stride = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1);
  for (auto i = sizes.size(); i > 0; --i) {
    llvm::Value* size = sizes.at(i - 1);
    llvm::Value* stride = cur_stride;

    // If expanded, stride is 0
    if (expand_flags.at(i - 1)) {
      stride = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0);
    } else {
      // Create comparison: size == 0
      llvm::Value* is_zero = builder.CreateICmpEQ(size, llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0));
      
      // Get current function for creating basic blocks
      llvm::Function* current_function = builder.GetInsertBlock()->getParent();
      
      // Create basic blocks
      llvm::BasicBlock* zero_block = llvm::BasicBlock::Create(context, "size_zero", current_function);
      llvm::BasicBlock* nonzero_block = llvm::BasicBlock::Create(context, "size_nonzero", current_function);
      llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(context, "stride_merge", current_function);
      
      // Conditional branch
      builder.CreateCondBr(is_zero, zero_block, nonzero_block);
      
      // Handle size == 0 case
      builder.SetInsertPoint(zero_block);
      llvm::Value* stride_if_zero = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1);
      builder.CreateBr(merge_block);
      
      // Handle size != 0 case
      builder.SetInsertPoint(nonzero_block);
      llvm::Value* new_cur_stride = builder.CreateMul(cur_stride, size);
      builder.CreateBr(merge_block);
      
      // Merge the results
      builder.SetInsertPoint(merge_block);
      llvm::PHINode* stride_phi = builder.CreatePHI(llvm::Type::getInt64Ty(context), 2);
      stride_phi->addIncoming(stride_if_zero, zero_block);
      stride_phi->addIncoming(cur_stride, nonzero_block);
      stride = stride_phi;
      
      // Update cur_stride for next iteration
      llvm::PHINode* cur_stride_phi = builder.CreatePHI(llvm::Type::getInt64Ty(context), 2);
      cur_stride_phi->addIncoming(cur_stride, zero_block);  // Don't update if size is 0
      cur_stride_phi->addIncoming(new_cur_stride, nonzero_block);  // Update if size != 0
      cur_stride = cur_stride_phi;
    }

    strides.at(i - 1) = stride;
  }

  return strides;
}

// Infer the size and stride of each dimension
std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferShape(
    const TensorView* tv,
    std::vector<Val*> symbolic_sizes,
    std::vector<bool> expand_flags,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferShape");

  std::vector<llvm::Value*> concrete_sizes(symbolic_sizes.size(), nullptr);

  for (const auto i : arange(symbolic_sizes.size())) {
    auto symbolic_size = symbolic_sizes.at(i);
    traverseExtentDFS(symbolic_size, val2llvmMap, context, builder);
    auto* inferred_val = val2llvmMap[symbolic_size];
    if(inferred_val == nullptr) {
      std::cout << "inferred_val is nullptr for " << symbolic_size->toString() << std::endl;
    }
    concrete_sizes.at(i) = inferred_val;
  }

  auto strides = getContiguousStrides(concrete_sizes, expand_flags, context, builder);
  return {concrete_sizes, strides};
}

std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferAllocationShape(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  std::vector<Val*> symbolic_sizes;
  std::vector<bool> expand_flags;

  // Allocate the allocation domain
  for (const auto id : tv->getMaybeAllocationDomain()) {
    if (id->isReduction() || id->isStride()) {
      continue;
    }

    if (id->isDeviceDim()) {
      symbolic_sizes.push_back(id->container()->oneVal());
    } else {
      symbolic_sizes.push_back(id->getMaybeExpandedExtent());
    }
    if (id->hasExpandedExtent()) {
      NVF_ERROR(
          id->isBroadcast(),
          "Non-broadcast domain should not have an expanded extent: ",
          id->toString());
      expand_flags.push_back(true);
    } else {
      expand_flags.push_back(false);
    }
  }
  return inferShape(tv, symbolic_sizes, expand_flags, val2llvmMap, context, builder);
}

Val* mapToInputDomain(
    Val* currentDomain,
    std::unordered_map<Val*, bool>& boundaryVals
) {
  for(auto* val : boundaryVals) { 
    auto* domain = val->as<IterDomain>();
    if(currentDomain->as<IterDomain>()->extent().sameAs(domain->extent())) {
      return val;
    }
  }
  return nullptr;
}

void generate_stride_llvm_ir(
    Val* current_val,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::IRBuilder<>& builder,
    llvm::Value*& running_stride_product,
    std::unordered_map<Val*, bool>& boundary_vals,
    std::vector<llvm::Value*>& strides
) {

    // Check if the current val is nullptr
    if (current_val == nullptr) {
        NVF_ERROR(false, "LLVM Lowering Error: generate_stride_llvm_ir called with nullptr Val.");
        return;
    }
    auto* def_expr = current_val->definition();
    // Check if the current val is missing
    if (def_expr == nullptr) {
      // Check if the current val is a boundary val
      Val* original_val = mapToInputDomain(current_val, boundary_vals);
      if(original_val != nullptr){
        // TODO: If the iter domain is a broadcast domain, then we have multiple inputs values pointing to the same valgroup
        // NVF_ERROR(!original_val->as<IterDomain>()->isBroadcast(), "LLVM Lowering Error: Broadcast domain is not supported in stride inference");
        if(boundary_vals[original_val] == false){
          boundary_vals[original_val] = true;
          strides.push_back(running_stride_product);
          running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[original_val], "mapped_stride");
        }
      }
      return;
    }

    // For each merge op, we need to check if it is valid split, we don't want to merge two values that has gaps in between
    if (def_expr->isA<Merge>()) {
        auto* merge_expr = def_expr->as<Merge>();
        auto* input_inner_val = merge_expr->inner()->as<Val>();
        auto* input_outer_val = merge_expr->outer()->as<Val>();
        auto* inner_mapped_val = mapToInputDomain(input_inner_val, boundary_vals);
        auto* outer_mapped_val = mapToInputDomain(input_outer_val, boundary_vals);
        // Check if the inner val is a boundary val
        if(inner_mapped_val != nullptr){
          if(boundary_vals[inner_mapped_val] == false){
            boundary_vals[inner_mapped_val] = true;
            strides.push_back(running_stride_product);
            running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[inner_mapped_val->as<IterDomain>()->extent()], "mapped_stride");
            return;
          }
        }
        else{
          generate_stride_llvm_ir(input_inner_val, val2llvmMap, builder, running_stride_product, boundary_vals, strides);
        }

        // Check if the outer val is a boundary val
        if(outer_mapped_val != nullptr){
          if(boundary_vals[outer_mapped_val] == false){
            boundary_vals[outer_mapped_val] = true;
            strides.push_back(running_stride_product);
            running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[outer_mapped_val->as<IterDomain>()->extent()], "mapped_stride");
            return;
          }
        }
        else{
          generate_stride_llvm_ir(input_outer_val, val2llvmMap, builder, running_stride_product, boundary_vals, strides);
        }
        
        // Extent of merged domain
        if(val2llvmMap[input_outer_val] == nullptr || val2llvmMap[input_inner_val] == nullptr || val2llvmMap[current_val] != nullptr){
          return;
        }
        else{
          val2llvmMap[current_val->as<IterDomain>()->extent()] = builder.CreateMul(
              val2llvmMap[input_outer_val->as<IterDomain>()->extent()],
              val2llvmMap[input_inner_val->as<IterDomain>()->extent()],
              current_val->toString() + "mapped_extent"
          );
        }

    } else if (def_expr->isA<Split>()) {
        auto* split_expr = def_expr->as<Split>();
        auto* input_val = split_expr->in()->as<Val>();
        auto* output_inner_val = split_expr->inner()->as<Val>();
        auto* output_outer_val = split_expr->outer()->as<Val>();
        auto* input_mapped_val = mapToInputDomain(input_val, boundary_vals);
        auto* output_inner_mapped_val = mapToInputDomain(output_inner_val, boundary_vals);
        auto* output_outer_mapped_val = mapToInputDomain(output_outer_val, boundary_vals);

        if(input_mapped_val != nullptr){
          if(boundary_vals[input_mapped_val] == false){
            boundary_vals[input_mapped_val] = true;
            strides.push_back(running_stride_product);
            running_stride_product = builder.CreateMul(running_stride_product, val2llvmMap[input_mapped_val->as<IterDomain>()->extent()], "mapped_stride");
            return;
          }
        }
        else{
          generate_stride_llvm_ir(input_val, val2llvmMap, builder, running_stride_product, boundary_vals, strides);
        }

        auto* split_factor = split_expr->factor()->as<Val>();
        if(split_expr->innerSplit()){
          if(split_factor->isConstInt()){
            val2llvmMap[output_inner_val->as<IterDomain>()->extent()] = builder.getInt64(split_factor->as<Int>()->value());
          }
          else{
            if(val2llvmMap.find(split_factor) != val2llvmMap.end()){
              val2llvmMap[output_inner_val->as<IterDomain>()->extent()] = val2llvmMap[split_factor];
            }
            else{
              NVF_ERROR(false, "LLVM Lowering Error: Inner split factor is not a constant and not found in val2stride_map");
              return;
            }
          }
          if(val2llvmMap[input_val] == nullptr || val2llvmMap[output_inner_val] == nullptr || val2llvmMap[output_outer_val] != nullptr){
            return;
          }
          val2llvmMap[output_outer_val->as<IterDomain>()->extent()] = builder.CreateUDiv(
            val2llvmMap[input_val->as<IterDomain>()->extent()],
            val2llvmMap[output_inner_val->as<IterDomain>()->extent()],
            output_outer_val->toString() + "mapped_stride"
          );
        }
        else{
          if(split_expr->factor()->isConstInt()){
            val2llvmMap[output_outer_val->as<IterDomain>()->extent()] = builder.getInt64(split_factor->as<Int>()->value());
          }
          else{
            if(val2llvmMap.find(split_factor) != val2llvmMap.end()){
              val2llvmMap[output_outer_val->as<IterDomain>()->extent()] = val2llvmMap[split_factor];
            }
            else{
              NVF_ERROR(false, "LLVM Lowering Error: Outer split factor is not a constant and not found in val2stride_map");
              return;
            }
          }
          if(val2llvmMap[input_val] == nullptr || val2llvmMap[output_inner_val] == nullptr || val2llvmMap[output_outer_val] != nullptr){
            return;
          }
          val2llvmMap[output_inner_val->as<IterDomain>()->extent()] = builder.CreateUDiv(
            val2llvmMap[input_val->as<IterDomain>()->extent()],
            val2llvmMap[output_outer_val->as<IterDomain>()->extent()],
            output_inner_val->toString() + "mapped_stride"
          );
        }

    } else { // Fallback for other ops (e.g., simple unary pass-through)
        NVF_ERROR(false, "LLVM Lowering Error: Unhandled op_type '" + def_expr->toString() + "' for Val " + current_val->toString());
    }
}
std::vector<llvm::Value*> inferTensorStrides(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  std::vector<llvm::Value*> strides;
   llvm::Value* running_stride = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1);
   std::unordered_map<Val*, bool> boundaryValVisited;
   auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
   for(auto* val : logical_domain) {
    boundaryValVisited[val] = false;
   }
   for(auto it = tv->getMaybeAllocationDomain().rbegin(); it != tv->getMaybeAllocationDomain().rend(); ++it) {
      auto iter_domain = *it;
      if(iter_domain->getParallelType() == ParallelType::DIDx) {
          continue;
      }
      generate_stride_llvm_ir(iter_domain->as<Val>(), val2llvmMap, builder, running_stride, boundaryValVisited, strides);
    }
  return strides;
}

std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> inferTensorShapesNonAlias(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  // Non-alias handling:
  auto allocation_size_stride = inferAllocationShape(tv, val2llvmMap, context, builder);
  if (!tv->hasAllocation()) {
    return {allocation_size_stride.first, allocation_size_stride.second};
  }
  // otherwise we want return the reordered size and stride
  return {allocation_size_stride.first, inferTensorStrides(tv, val2llvmMap, context, builder)};
}

std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>> compileOutputTensorView(
    const TensorView* tv,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  // Alias handling, just return empty vector for now:
  auto alias_info = tv->fusion()->getOutputAlias(tv);
  if (alias_info.type != AllocationType::New) {
    return {std::vector<llvm::Value*>(), std::vector<llvm::Value*>()};
  }

  return inferTensorShapesNonAlias(tv, val2llvmMap, context, builder);
}




void compileNamedScalar(
    const NamedScalar* named_scalar,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  return;
}

void compileInputTensorView(
    const TensorView* tv,
    const std::vector<llvm::Value*>& logical_sizes,
    const std::vector<llvm::Value*>& logical_strides,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    llvm::LLVMContext& context,
    llvm::IRBuilder<>& builder) {
  auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
  
  NVF_ERROR(
      logical_sizes.size() == logical_domain.size(),
      "Size mismatch: logical_sizes has ",
      logical_sizes.size(),
      " elements but logical_domain has ",
      logical_domain.size(),
      " elements");
  
  NVF_ERROR(
      logical_strides.size() == logical_domain.size(),
      "Size mismatch: logical_strides has ",
      logical_strides.size(),
      " elements but logical_domain has ",
      logical_domain.size(),
      " elements");
  
  for (auto i : arange(logical_domain.size())) {
    auto id = logical_domain[i];
    
    // Map dimension extents to runtime LLVM values
    if (id->isBroadcast()) {
      val2llvmMap[id->extent()] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1);
      if (id->hasExpandedExtent()) {
        val2llvmMap[id->expandedExtent()] = logical_sizes[i];
      }
    } else {
      val2llvmMap[id->extent()] = logical_sizes[i];
    }
  }
}

// Helper function to generate LLVM IR that extracts tensor size for a given dimension
llvm::Value* generateTensorSizeExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::Module* mod,
    llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& context = mod->getContext();
  
  // Look up the tensor_size wrapper function
  llvm::Function* tensor_size_func = mod->getFunction("tensor_size");
  if (!tensor_size_func) {
    // Create function declaration
    llvm::FunctionType* func_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context),
      {llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), llvm::Type::getInt64Ty(context)},
      false
    );
    tensor_size_func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "tensor_size", mod
    );
  }
  
  llvm::Value* dim_val = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), dim);
  return builder.CreateCall(tensor_size_func, {tensor_ptr, dim_val});
}

// Helper function to generate LLVM IR that extracts tensor stride for a given dimension
llvm::Value* generateTensorStrideExtraction(
    llvm::Value* tensor_ptr,
    int64_t dim,
    llvm::Module* mod,
    llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& context = mod->getContext();
  
  // Look up the tensor_stride wrapper function
  llvm::Function* tensor_stride_func = mod->getFunction("tensor_stride");
  if (!tensor_stride_func) {
    // Create function declaration
    llvm::FunctionType* func_type = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(context),
      {llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)), llvm::Type::getInt64Ty(context)},
      false
    );
    tensor_stride_func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "tensor_stride", mod
    );
  }
  
  llvm::Value* dim_val = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), dim);
  return builder.CreateCall(tensor_stride_func, {tensor_ptr, dim_val});
}

void processTensorViewsLLVM(
    const hir::HostIrContainer* container,
    llvm::Module* mod,
    llvm::Function* func,
    std::unordered_map<const kir::Allocate*, std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>>>& tensorShapeMap,
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    const HostIrJitParams& host_ir_jit_params,
    llvm::IRBuilder<>& builder,
    size_t num_inputs) {
  llvm::LLVMContext& context = mod->getContext();
  
  // Get function parameters
  llvm::Value* input_tensor_array = func->getArg(0);
  // llvm::Value* num_inputs_val = func->getArg(1);
  llvm::Value* output_tensor_array = func->getArg(2);
  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  // std::cout << "num_inputs val: " << num_inputs_val << std::endl;
  // setup input tensor views
  size_t tensor_idx = 0;
  for(auto* input : container->inputs()) {
    if(auto* tv = dynamic_cast<const TensorView*>(input)) {
      // Get tensor from input array
      llvm::Value* tensor_slot = builder.CreateGEP(
          void_ptr_type, 
          input_tensor_array, 
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), tensor_idx)
      );
      llvm::Value* tensor_ptr = builder.CreateLoad(void_ptr_type, tensor_slot);
      
      auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
      
      // Extract sizes and strides as LLVM values
      std::vector<llvm::Value*> logical_sizes;
      std::vector<llvm::Value*> logical_strides;
      
      for (size_t dim = 0; dim < logical_domain.size(); ++dim) {
        logical_sizes.push_back(generateTensorSizeExtraction(tensor_ptr, dim, mod, builder));
        logical_strides.push_back(generateTensorStrideExtraction(tensor_ptr, dim, mod, builder));
      }
      
      compileInputTensorView(tv, logical_sizes, logical_strides, val2llvmMap, context, builder);
      tensor_idx++;
    }
  }
  
  // resolve each output tensor shape and allocate tensors
  size_t output_idx = 0;
  for(auto* output : container->topLevelExprs()) {
    if(auto* allocate = dynamic_cast<const kir::Allocate*>(output)) {
      auto* tv = allocate->buffer()->as<TensorView>();
      auto size_stride_pair = compileOutputTensorView(tv, val2llvmMap, context, builder);
      tensorShapeMap[allocate] = size_stride_pair;
      
      // Get the computed sizes and strides
      const auto& sizes = size_stride_pair.first;
      const auto& strides = size_stride_pair.second;
      
      if (!sizes.empty() && !strides.empty()) {
        // Create array allocation for sizes
        llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
        llvm::Type* int64_ptr_type = int64_type->getPointerTo();
        llvm::Type* int32_type = llvm::Type::getInt32Ty(context);
        llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
        
        // Allocate memory for sizes array
        llvm::Value* sizes_array = builder.CreateAlloca(int64_type, llvm::ConstantInt::get(int64_type, sizes.size()));
        for (size_t i = 0; i < sizes.size(); ++i) {
          llvm::Value* gep = builder.CreateGEP(int64_type, sizes_array, llvm::ConstantInt::get(int64_type, i));
          builder.CreateStore(sizes[i], gep);
        }
        
        // Allocate memory for strides array  
        llvm::Value* strides_array = builder.CreateAlloca(int64_type, llvm::ConstantInt::get(int64_type, strides.size()));
        for (size_t i = 0; i < strides.size(); ++i) {
          llvm::Value* gep = builder.CreateGEP(int64_type, strides_array, llvm::ConstantInt::get(int64_type, i));
          builder.CreateStore(strides[i], gep);
        }
        
        // Create constants for other parameters
        llvm::Value* ndim_val = llvm::ConstantInt::get(int64_type, sizes.size());
        llvm::Value* strides_ndim_val = llvm::ConstantInt::get(int64_type, strides.size());
        
        at::ScalarType data_type = data_type_to_aten(
            allocate->buffer()->dtype() == DataType::Index
                ? PrimDataType::Int
                : allocate->buffer()->dtype());
        llvm::Value* dtype_constant = llvm::ConstantInt::get(int32_type, static_cast<int32_t>(data_type));
        llvm::Value* device_index_constant = llvm::ConstantInt::get(int64_type, host_ir_jit_params.getCommunicator()->deviceId());
        
        // Get the pre-allocated tensor from the output array
        llvm::Value* output_slot = builder.CreateGEP(
            void_ptr_type, 
            output_tensor_array, 
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), output_idx)
        );
        // Load the tensor pointer - this points to a pre-allocated tensor from runFullGraph
        llvm::Value* tensor_ptr = builder.CreateLoad(void_ptr_type, output_slot);
        
        // Get the at::native::empty_strided_cuda function
        llvm::Function* at_empty_strided_cuda_func = mod->getFunction(kHostIrJitEmptyStridedCudaFuncName);
        if (at_empty_strided_cuda_func == nullptr) {
          llvm::FunctionType* at_empty_strided_cuda_func_type = llvm::FunctionType::get(
              builder.getVoidTy(),
              {int64_ptr_type, int64_type, int64_ptr_type, int64_type, int32_type, int64_type, void_ptr_type},
              false);
          at_empty_strided_cuda_func = llvm::Function::Create(
              at_empty_strided_cuda_func_type,
              llvm::Function::ExternalLinkage,
              kHostIrJitEmptyStridedCudaFuncName,
              mod);
        }
        
        // Call at::native::empty_strided_cuda with the pre-allocated tensor
        builder.CreateCall(
            at_empty_strided_cuda_func,
            {sizes_array,
             ndim_val,
             strides_array,
             strides_ndim_val,
             dtype_constant,
             device_index_constant,
             tensor_ptr});
        
        // No need to store back - we modified the tensor in place
        
        output_idx++;
      }
    }
  }
}

void generateMainFunc(
    const hir::HostIrContainer* container,
    llvm::Module* mod,  
    std::unordered_map<Val*, llvm::Value*>& val2llvmMap,
    const HostIrJitParams& host_ir_jit_params) {
  std::unordered_map<const kir::Allocate*, std::pair<std::vector<llvm::Value*>, std::vector<llvm::Value*>>> tensorShapeMap;
  llvm::LLVMContext& context = mod->getContext();

  // Count input and output tensors
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  
  for (auto* input : container->inputs()) {
    if (dynamic_cast<const TensorView*>(input)) {
      num_inputs++;
    }
  }
  
  for (auto* output : container->topLevelExprs()) {
    if (dynamic_cast<const kir::Allocate*>(output)) {
      num_outputs++;
    }
  }

  // Create function signature with input tensor array, count, and output tensor array parameters
  llvm::PointerType* void_ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
  llvm::PointerType* tensor_array_type = llvm::PointerType::getUnqual(void_ptr_type);
  llvm::Type* int64_type = llvm::Type::getInt64Ty(context);
  
  std::vector<llvm::Type*> param_types = {
    tensor_array_type, // at::Tensor** input_tensors
    int64_type,        // int64_t num_inputs
    tensor_array_type  // at::Tensor** output_tensors
  };
  
  llvm::FunctionType* func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(context), param_types, false);
  llvm::Function* func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, "full_graph_induction", mod);
  
  // Set parameter names
  func->getArg(0)->setName("input_tensors");
  func->getArg(1)->setName("num_inputs");
  func->getArg(2)->setName("output_tensors");

  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", func);
  llvm::IRBuilder<> builder(entry);

  processTensorViewsLLVM(container, mod, func, tensorShapeMap, val2llvmMap, host_ir_jit_params, builder, num_inputs);

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
    std::unordered_map<const hir::LaunchKernel*, launch_kernel_fn>& launch_kernel_funcs_,
    main_func_fn& main_func_,
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

  // Generate the allocate functions
  std::vector<kir::Allocate*> allocate_exprs;
  std::vector<hir::LaunchKernel*> launch_kernel_exprs;
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
    } else if (auto* launch_kernel = dynamic_cast<hir::LaunchKernel*>(expr)) {
      generateLaunchKernelFunc(launch_kernel, mod.get());
      launch_kernel_exprs.push_back(launch_kernel);
    }
  }

  // Generate the main function that processes all tensor views and outputs
  std::unordered_map<Val*, llvm::Value*> val2llvmMap;
  generateMainFunc(container, mod.get(), val2llvmMap, host_ir_jit_params);

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
  for (auto* launch_kernel : launch_kernel_exprs) {
    auto func_name = launch_kernel->toString();
    auto func_addr = throwIfError(jit->lookup(func_name));
    launch_kernel_funcs_[launch_kernel] = reinterpret_cast<KernelArgumentHolderPair(*)(int64_t, at::Tensor**, at::Tensor**)>(
        func_addr.getValue());
  }
  
  // Look up the main function
  auto main_func_addr = throwIfError(jit->lookup("full_graph_induction"));
  main_func_ = reinterpret_cast<void(*)(at::Tensor**, int64_t, at::Tensor**)>(main_func_addr.getValue());
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

  // Register KernelArgumentHolder functions
  void* kernel_argument_holder_constructor_func_ptr = reinterpret_cast<void*>(
      +[]() -> KernelArgumentHolder* {
        return new KernelArgumentHolder();
      });

  void* kernel_argument_holder_set_cache_id_func_ptr = reinterpret_cast<void*>(
      +[](KernelArgumentHolder* self, size_t id) {
        self->setCacheId(id);
      });

  void* kernel_argument_holder_set_device_index_func_ptr = reinterpret_cast<void*>(
      +[](KernelArgumentHolder* self) {
        self->setDeviceIndex();
      });

   void* kernel_argument_holder_push_func_ptr = reinterpret_cast<void*>(
      +[](KernelArgumentHolder* self, at::Tensor* tensor_ptr) {
        // std::cout << "Wrapper function called with tensor_ptr: " << tensor_ptr << std::endl;
        if (tensor_ptr == nullptr) {
          // std::cout << "ERROR: tensor_ptr is null!" << std::endl;
          return;
        }
        // std::cout << "About to call self->push(*tensor_ptr)" << std::endl;
        self->push(*tensor_ptr);
        // std::cout << "Successfully called push" << std::endl;
      });

  // Register tensor size and stride extraction functions
  void* tensor_size_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* tensor_ptr, int64_t dim) -> int64_t {
        // std::cout << "tensor_size_func_ptr called with tensor_ptr: " 
        // << tensor_ptr << ", dim: " << dim 
        // << " tensor_ptr->size(dim): " << tensor_ptr->size(dim)
        // << std::endl;
        if (tensor_ptr == nullptr) {
          return 0;
        }
        return tensor_ptr->size(dim);
      });

  void* tensor_stride_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* tensor_ptr, int64_t dim) -> int64_t {
        // std::cout << "tensor_stride_func_ptr called with tensor_ptr: " 
        // << tensor_ptr << ", dim: " << dim 
        // << " tensor_ptr->stride(dim): " << tensor_ptr->stride(dim)
        // << std::endl;
        if (tensor_ptr == nullptr) {
          return 0;
        }
        return tensor_ptr->stride(dim);
      });

  // Register wrapper functions in JIT
  auto empty_strided_cuda_addr = llvm::orc::ExecutorAddr::fromPtr(empty_strided_cuda_func_ptr);
  auto kernel_argument_holder_constructor_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_constructor_func_ptr);
  auto kernel_argument_holder_set_cache_id_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_set_cache_id_func_ptr);
  auto kernel_argument_holder_set_device_index_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_set_device_index_func_ptr);
  auto kernel_argument_holder_push_addr = llvm::orc::ExecutorAddr::fromPtr(kernel_argument_holder_push_func_ptr); 
  auto tensor_size_addr = llvm::orc::ExecutorAddr::fromPtr(tensor_size_func_ptr);
  auto tensor_stride_addr = llvm::orc::ExecutorAddr::fromPtr(tensor_stride_func_ptr);

  // Register wrapper functions in JIT
  llvm::orc::SymbolMap symbolMap;
  symbolMap[mangler(kHostIrJitEmptyStridedCudaFuncName)] =
      llvm::orc::ExecutorSymbolDef(
          empty_strided_cuda_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("KernelArgumentHolder::KernelArgumentHolder")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_constructor_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("KernelArgumentHolder::setCacheId")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_set_cache_id_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("KernelArgumentHolder::setDeviceIndex")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_set_device_index_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("KernelArgumentHolder::push")] = 
      llvm::orc::ExecutorSymbolDef(kernel_argument_holder_push_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("tensor_size")] = 
      llvm::orc::ExecutorSymbolDef(tensor_size_addr, llvm::JITSymbolFlags::Exported);
  symbolMap[mangler("tensor_stride")] = 
      llvm::orc::ExecutorSymbolDef(tensor_stride_addr, llvm::JITSymbolFlags::Exported);

  throwIfError(dest_dynamic_lib.define(llvm::orc::absoluteSymbols(symbolMap)));

  // Compile the module
  compile(
      container,
      pimpl_->jit.get(),
      pimpl_->allocate_funcs_,
      pimpl_->launch_kernel_funcs_,
      pimpl_->main_func_,
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


// Allocate with fullgraph mode
at::Tensor HostIrJit::allocate(
    const kir::Allocate* allocate) {
  FUSER_PERF_SCOPE("HostIrJit::allocate");
  auto allocate_tensor_iter = pimpl_->allocate_tensors_.find(allocate);
  if (allocate_tensor_iter != pimpl_->allocate_tensors_.end()) {
    at::Tensor tensor = allocate_tensor_iter->second;
    // Remove from map to prevent memory leak
    pimpl_->allocate_tensors_.erase(allocate_tensor_iter);
    return tensor;
  }
  return at::empty({0}, at::kFloat);
}

std::vector<at::Tensor> HostIrJit::runFullGraph(
    const hir::HostIrContainer* container,
    const std::unordered_map<Val*, PolymorphicValue>& val_to_PValue) {
  FUSER_PERF_SCOPE("HostIrJit::runFullGraph");
  std::vector<at::Tensor> inputs;
  // process input values, converting IValue to PolymorphicValue
  for(auto* input : container->inputs()) {
    if (input->isA<TensorView>()) {
      auto tensor = val_to_PValue.at(input).as<at::Tensor>();
      inputs.push_back(tensor);
    }
  }
  // Count the number of output tensors
  size_t num_outputs = 0;
  for (auto* output : container->topLevelExprs()) {
    if (dynamic_cast<const kir::Allocate*>(output)) {
      num_outputs++;
    }
  }
  
  // Create array to hold output tensor pointers - use properly initialized empty tensors
  std::vector<at::Tensor> result_tensors;
  result_tensors.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    // Create a properly initialized empty tensor instead of default-constructed
    result_tensors.emplace_back(at::empty({0}, at::kCUDA));
  }
  
  std::vector<at::Tensor*> output_ptrs;
  output_ptrs.reserve(num_outputs);
  for (auto& tensor : result_tensors) {
    output_ptrs.push_back(&tensor);
  }
  
  // Convert input tensors to pointers
  std::vector<at::Tensor*> input_ptrs;
  input_ptrs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    input_ptrs.push_back(const_cast<at::Tensor*>(&tensor));
  }
  
  // Call the main function with input array, count, and outputs
  pimpl_->main_func_(input_ptrs.data(), static_cast<int64_t>(inputs.size()), output_ptrs.data());
  
  size_t i = 0;
  for (auto* expr : container->topLevelExprs()) {
    if (auto* allocate = dynamic_cast<kir::Allocate*>(expr)) {
      pimpl_->allocate_tensors_[allocate] = result_tensors[i];
      i++; 
    }
  }
  
  return result_tensors;
}

HostIrJit::LaunchKernelResult HostIrJit::launchKernel(
    const hir::LaunchKernel* launch_kernel,
    int64_t cache_id,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  FUSER_PERF_SCOPE("HostIrJit::launchKernel");
  if (pimpl_->launch_kernel_funcs_.find(launch_kernel) == pimpl_->launch_kernel_funcs_.end()) {
    NVF_ERROR(false, "launch kernel function not found for ", launch_kernel);
  }
  
  auto func_ptr = pimpl_->launch_kernel_funcs_[launch_kernel];
  
  // std::cout << "Calling LLVM function with:" << std::endl;
  // std::cout << "  cache_id: " << cache_id << std::endl;
  // std::cout << "  inputs.size(): " << inputs.size() << std::endl;
  // std::cout << "  outputs.size(): " << outputs.size() << std::endl;
  
  // Convert const std::vector<at::Tensor>& to at::Tensor** arrays
  std::vector<at::Tensor*> input_ptrs;
  input_ptrs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    input_ptrs.push_back(const_cast<at::Tensor*>(&tensor));
  }
  
  std::vector<at::Tensor*> output_ptrs;
  output_ptrs.reserve(outputs.size());
  for (const auto& tensor : outputs) {
    output_ptrs.push_back(const_cast<at::Tensor*>(&tensor));
  }
  
  // Get raw pointer arrays
  at::Tensor** input_array = input_ptrs.data();
  at::Tensor** output_array = output_ptrs.data();
  
  KernelArgumentHolderPair result = func_ptr(cache_id, input_array, output_array);
  
  // Use unique_ptr to manage memory automatically
  std::unique_ptr<KernelArgumentHolder> args_ptr(reinterpret_cast<KernelArgumentHolder*>(result.args));
  std::unique_ptr<KernelArgumentHolder> outputs_ptr(reinterpret_cast<KernelArgumentHolder*>(result.outputs));
  
  // Move the objects out of the unique_ptrs to return by value
  KernelArgumentHolder args = std::move(*args_ptr);
  KernelArgumentHolder outputs_holder = std::move(*outputs_ptr);
  
  return LaunchKernelResult{std::move(args), std::move(outputs_holder)};
}

} // namespace nvfuser
