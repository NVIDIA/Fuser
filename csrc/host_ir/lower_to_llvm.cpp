// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <val_graph_visitor.h>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/LLVMContext.h"
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include "llvm/Support/Error.h"

#include <unordered_map>
#include <queue>
#include <chrono>

#include <host_ir/lower_to_llvm.h>

namespace nvfuser {
/*

Helper Data Structures & Functions

*/
using FuncType = void (*)(const int64_t* input, int64_t input_len, int64_t* output, int64_t output_len);

// Dependency graph entry for the stride inference
struct StrideInfo {
public:
    llvm::Value* llvm_extent = nullptr;   // LLVM Value for the extent of this IterDomain
    llvm::Value* llvm_stride = nullptr;   // LLVM Value for the calculated stride of this IterDomain
};

// Helper function to exit on error on LLVM JIT initialization
template <typename T>
T ExitOnErr(llvm::Expected<T> &&E) {
    if (!E) {
        llvm::errs() << llvm::toString(E.takeError()) << "\n";
        exit(1);
    }
    return std::move(*E);
}

inline void ExitOnErr(llvm::Error &&Err) {
    if (Err) {
        llvm::errs() << llvm::toString(std::move(Err)) << "\n";
        exit(1);
    }
}

// Helper function to cast iter domains to vals
std::vector<Val*> domain2vals(const std::vector<IterDomain*>& domain){
  std::vector<Val*> vals;
  for(auto* id : domain){
    vals.push_back(dynamic_cast<Val*>(id));
  }
  return vals;
}

// Helper function to map the current domain to the input domain strictly
int isSameToInputDomain(std::unordered_map<int, Val*>& boundary_vals, Val* current_domain){
  for(auto boundary_val : boundary_vals){
    if(boundary_val.second == current_domain){
      return boundary_val.first;
    }
  }
  return -1;
}

// Helper function to check if the current iter domain is alias to the input iter domain
int mapToInputDomain(std::unordered_map<int, Val*>& boundary_vals, Val* current_domain, const ValGraph& exact_graph){
  int input_domain_index = isSameToInputDomain(boundary_vals, current_domain);
  if(input_domain_index != -1){
    return input_domain_index;
  }
  for(auto boundary_val : boundary_vals){ 
    if(exact_graph.disjointValSets().strictAreMapped(boundary_val.second, current_domain)){
      return boundary_val.first;
    }
  } 
  return -1;
}

/*

Generate LLVM IR for a dependency graph
By default, we assume it is in typological order, which means input values are ready to use

*/
void generate_shape_llvm_ir(Expr* expr, llvm::IRBuilder<>& builder, std::unordered_map<ValGroup,llvm::Value*>& val2llvm, std::unordered_map<int, Val*>& boundary_vals, const ValGraph& graph) {
  std::string op_string = std::string(expr->getOpString());

  // Perform the merge -> mul transformation
  if(op_string == "Merge"){
    llvm::Value* result = nullptr;
    auto* merge_expr = expr->as<Merge>();
    auto* merge_input_outer_val = merge_expr->outer()->as<Val>();
    auto* merge_input_inner_val = merge_expr->inner()->as<Val>();
    auto* merge_output_val = merge_expr->outputs()[0]->as<Val>();

    int input_outer_potential_index = mapToInputDomain(boundary_vals, merge_input_outer_val, graph);
    int input_inner_potential_index = mapToInputDomain(boundary_vals, merge_input_inner_val, graph);
    llvm::Value* input_outer_llvm_val = nullptr;
    llvm::Value* input_inner_llvm_val = nullptr;

    if(input_outer_potential_index != -1){
      input_outer_llvm_val = val2llvm[graph.toGroup(boundary_vals[input_outer_potential_index])];
    }
    else{
      input_outer_llvm_val = val2llvm[graph.toGroup(merge_input_outer_val)];
    }

    if(input_inner_potential_index != -1){
      input_inner_llvm_val = val2llvm[graph.toGroup(boundary_vals[input_inner_potential_index])];
    }
    else{
      input_inner_llvm_val = val2llvm[graph.toGroup(merge_input_inner_val)];
    }

    result = builder.CreateMul(input_outer_llvm_val, input_inner_llvm_val, merge_output_val->toString());

    val2llvm[graph.toGroup(merge_output_val)] = result;
  }
  else if(op_string == "Split"){
    auto* split_expr = expr->as<Split>();
    auto* split_input_val = split_expr->in()->as<Val>();
    auto* split_output_outer_val = split_expr->outer()->as<Val>();
    auto* split_output_inner_val = split_expr->inner()->as<Val>();

    int input_potential_index = mapToInputDomain(boundary_vals, split_input_val, graph);
    llvm::Value* input_llvm_val = nullptr;
    if(input_potential_index != -1){
      input_llvm_val = val2llvm[graph.toGroup(boundary_vals[input_potential_index])]; 
    }
    else{
      input_llvm_val = val2llvm[graph.toGroup(split_input_val)];
    }

    // Perform the split -> ceildiv transformation
    if(split_expr->innerSplit()){
      // inner = factor
      if(split_expr->factor()->isConstInt()){
        val2llvm[graph.toGroup(split_output_inner_val)] = builder.getInt64(std::stoi(split_expr->factor()->toString()));
      }
      else{
        if(val2llvm.find(graph.toGroup(split_expr->factor())) != val2llvm.end()){
          val2llvm[graph.toGroup(split_output_inner_val)] = val2llvm[graph.toGroup(split_expr->factor())];
        }
        else{
          std::cerr << "Missing factor val: " << split_expr->factor()->toString() << std::endl;
          exit(1);
        }
      }
      // outer = input + 1
      llvm::Value* minus_1 = builder.CreateSub(input_llvm_val, builder.getInt64(1), "minus_1");
      // outer = (input + 1) + inner
      llvm::Value* sum_ab = builder.CreateAdd(minus_1, val2llvm[graph.toGroup(split_output_inner_val)], "sum_ab");
      // outer = (input + 1 + inner) / inner
      val2llvm[graph.toGroup(split_output_outer_val)] = builder.CreateUDiv(sum_ab, val2llvm[graph.toGroup(split_output_inner_val)], split_output_outer_val->as<IterDomain>()->extent()->toString());
    }
    else{
      // outer = factor
      if(split_expr->factor()->isConstInt()){
        val2llvm[graph.toGroup(split_output_outer_val)] = builder.getInt64(std::stoi(split_expr->factor()->toString()));
      }
      else{
        if(val2llvm.find(graph.toGroup(split_expr->factor())) != val2llvm.end()){
          val2llvm[graph.toGroup(split_output_outer_val)] = val2llvm[graph.toGroup(split_expr->factor())];
        }
        else{
          std::cerr << "Missing factor val: " << split_expr->factor()->toString() << std::endl;
          exit(1);
        }
      }
      // inner = input - 1
      llvm::Value* minus_1 = builder.CreateSub(input_llvm_val, builder.getInt64(1), "minus_1");
      // inner = (input - 1) + outer
      llvm::Value* sum_ab = builder.CreateAdd(minus_1, val2llvm[graph.toGroup(split_output_outer_val)], "sum_ab");
      // inner = (input - 1 + outer) / outer
      val2llvm[graph.toGroup(split_output_inner_val)] = builder.CreateUDiv(sum_ab, val2llvm[graph.toGroup(split_output_outer_val)], split_output_inner_val->as<IterDomain>()->extent()->toString());
    }
  }
  else{
    std::cerr << "Unsupported op: " << op_string << std::endl;
    exit(1);
  }
}

/*

Dumping all exprs between input and output domain, currently this is only used for shape inference

*/

void generate_all_shape_llvm_ir(const ValGraph& graph, std::vector<IterDomain*>& input_domain, std::vector<IterDomain*>& output_domain, 
std::unordered_map<ValGroup, llvm::Value*>& val2llvm_val, std::unordered_map<int, Val*>& boundary_vals, llvm::IRBuilder<>& builder){
  ValGroups tv0_loop_groups = graph.toGroups(input_domain);
  ValGroups tv1_loop_groups = graph.toGroups(output_domain);
  auto result = getAllExprGroupsBetween(graph, tv0_loop_groups, tv1_loop_groups).first;
  for(auto expr_group : result){
    for(auto expr : *expr_group.first){
      generate_shape_llvm_ir(expr, builder, val2llvm_val, boundary_vals, graph);
    }
  }
}

/*

Verify if the merge is legit by traversing the exprs between output and input domain

TODO: Need to implement this function
*/

int findMostUpmostParent(Val* val, bool is_inner_path, std::unordered_map<int, Val*>& boundary_vals, const ValGraph& graph) {
  int potential_index = mapToInputDomain(boundary_vals, val, graph);
  if(potential_index != -1){
    return potential_index;
  }
  auto* def = val->definition();
  if(def == nullptr){
    return -1;
  }
  if(auto* split = def->as<Split>()){
    return findMostUpmostParent(split->in(), is_inner_path, boundary_vals, graph);
  }
  else if(auto* merge = def->as<Merge>()){
    if(is_inner_path){
      return findMostUpmostParent(merge->inner(), is_inner_path, boundary_vals, graph);
    }
    else{
      return findMostUpmostParent(merge->outer(), is_inner_path, boundary_vals, graph);
    }
  }
  return -1;
}

Val* findLowestCommonAncestor(Val* left_val, Val* right_val) {
  if (!left_val || !right_val) {
    return nullptr;
  }

  // Collect all ancestors of left_val using BFS
  std::unordered_set<Val*> left_ancestors;
  std::queue<Val*> q;
  q.push(left_val);
  left_ancestors.insert(left_val);

  while (!q.empty()) {
    Val* current = q.front();
    q.pop();

    if (auto* def = current->definition()) {
      for (auto* input : def->inputs()) {
        if (left_ancestors.find(input) == left_ancestors.end()) {
          left_ancestors.insert(input);
          q.push(input);
        }
      }
    }
  }

  // Find the first ancestor of right_val that is also an ancestor of left_val
  std::queue<Val*> q2;
  q2.push(right_val);
  std::unordered_set<Val*> visited_right;
  visited_right.insert(right_val);

  while (!q2.empty()) {
    Val* current = q2.front();
    q2.pop();

    if (left_ancestors.count(current)) {
      return current;
    }

    if (auto* def = current->definition()) {
      for (auto* input : def->inputs()) {
        if (visited_right.find(input) == visited_right.end()) {
          visited_right.insert(input);
          q2.push(input);
        }
      }
    }
  }

  return nullptr;
}

// Helper function to recursively trace the path from a value up to a given
// ancestor, ensuring the path integrity is maintained (inner vs. outer).
bool tracePathToAncestor(Val* current, Val* ancestor, bool must_be_inner_path) {
  // Base case: We've successfully reached the ancestor.
  if (current == ancestor) {
    return true;
  }

  // If there's no defining expression, we've hit a root, which is not the
  // ancestor, so the path is invalid.
  Expr* def = current->definition();
  if (!def) {
    return false;
  }

  // Handle Split expressions: check if we are on the correct side.
  if (auto* split = def->as<Split>()) {
    // Continue from the split's input
    if(split->in() == ancestor){
      if(split->outer() == current && must_be_inner_path){
        return true;
      }
      else if(split->inner() == current && !must_be_inner_path){
        return true;
      }
      else{
        return false;
      }
    }
    if (must_be_inner_path) {
      if (current != split->inner()){
        return false;
      }
    } else {
      if (current != split->outer()){
        return false;
      }
    }
    return tracePathToAncestor(split->in(), ancestor, must_be_inner_path);
  }
  // Handle Merge expressions: continue up the corresponding path.
  else if (auto* merge = def->as<Merge>()) {
    if (must_be_inner_path) {
      return tracePathToAncestor(merge->inner(), ancestor, must_be_inner_path);
    } else {
      return tracePathToAncestor(merge->outer(), ancestor, must_be_inner_path);
    }
  }
  // Handle other expressions that are simple pass-throughs.
  else if (def->inputs().size() == 1) {
    return tracePathToAncestor(def->inputs()[0], ancestor, must_be_inner_path);
  }

  // Any other expression type breaks the verifiable path.
  return false;
}

bool verify(Expr* expr, std::unordered_map<int, Val*>& boundary_vals, const ValGraph& graph) {
  // This function only verifies Merge expressions.
  auto merge = dynamic_cast<Merge*>(expr);
  if (!merge) {
    return true;
  }

  Val* inner_val = merge->inner(); // Rightmost input
  Val* outer_val = merge->outer(); // Leftmost input

  // Find the lowest common ancestor.
  Val* ancestor = findLowestCommonAncestor(inner_val, outer_val);
  if (!ancestor) {
    // case 1:
    int outer_parent_rightmost = findMostUpmostParent(outer_val, true, boundary_vals, graph);
    int inner_parent_leftmost = findMostUpmostParent(inner_val, false, boundary_vals, graph);

    if(outer_parent_rightmost != -1 && inner_parent_leftmost != -1){
      return std::abs(outer_parent_rightmost - inner_parent_leftmost) == 1;
    }

    // case 2:
    int outer_parent_leftmost = findMostUpmostParent(outer_val, false, boundary_vals, graph);
    int inner_parent_rightmost = findMostUpmostParent(inner_val, true, boundary_vals, graph);

    if (outer_parent_leftmost != -1 && inner_parent_rightmost != -1){
      return std::abs(outer_parent_leftmost - inner_parent_rightmost) == 1;
    }
    
    std::cerr << "No common ancestor found for merge: " << merge->toString() << std::endl;
    return false;
  }

  // One of the inputs must be on an inner path, the other on an outer path.
  // We check both combinations.

  // Combination 1: inner_val is on inner path, outer_val is on outer path.
  if (tracePathToAncestor(inner_val, ancestor, true) &&
      tracePathToAncestor(outer_val, ancestor, false)) {
    return true;
  }

  // Combination 2: inner_val is on outer path, outer_val is on inner path.
  if (tracePathToAncestor(inner_val, ancestor, false) &&
      tracePathToAncestor(outer_val, ancestor, true)) {
    return true;
  }

  std::cerr
      << "Merge validation failed: inputs are not on valid inner/outer paths."
      << std::endl;
  return false;
}

/*

Generate LLVM IR for stride inference

*/
void generate_stride_llvm_ir(
    Val* current_val_to_process,
    std::unordered_map<ValGroup, StrideInfo>& val2stride_map,
    llvm::IRBuilder<>& builder,
    std::unordered_map<int, Val*>& boundary_vals,
    llvm::Value*& running_stride_product,
    const ValGraph& graph
    ) {

    // Check if the current val is nullptr
    if (current_val_to_process == nullptr) {
        std::cerr << "Error: generate_stride_llvm_ir called with nullptr Val." << std::endl;
        return;
    }

    // Check if the current val is a boundary val
    int cur_val_potential_index = mapToInputDomain(boundary_vals, current_val_to_process, graph);
    if(cur_val_potential_index != -1){
      if(val2stride_map[graph.toGroup(boundary_vals[cur_val_potential_index])].llvm_stride == nullptr){
        val2stride_map[graph.toGroup(boundary_vals[cur_val_potential_index])].llvm_stride = running_stride_product;
        running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[graph.toGroup(boundary_vals[cur_val_potential_index])].llvm_extent, "stride_root_val");
      }
      return;
    }

    // Memoization: Already processed
    if (val2stride_map.find(graph.toGroup(current_val_to_process)) != val2stride_map.end() && val2stride_map[graph.toGroup(current_val_to_process)].llvm_stride != nullptr) {
        return;
    }

    auto* def_expr = current_val_to_process->definition();

    // Check if the current val is missing
    if (def_expr == nullptr) {
        if (val2stride_map.find(graph.toGroup(current_val_to_process)) == val2stride_map.end() || val2stride_map[graph.toGroup(current_val_to_process)].llvm_stride == nullptr) {
            std::cerr << "Warning: StrideInfo not pre-populated for root Val: "
                      << current_val_to_process->toString() << ". Its stride will be unknown." << std::endl;
        }
        return;
    }

    std::string op_type = def_expr->getOpString();
    // For each merge op, we need to check if it is valid split, we don't want to merge two values that has gaps in between
    if (op_type == "Merge") {
        auto* merge_expr = def_expr->as<Merge>();
        auto* input_inner_val = merge_expr->inner()->as<Val>();
        auto* input_outer_val = merge_expr->outer()->as<Val>();
        int input_inner_potential_index = mapToInputDomain(boundary_vals, input_inner_val, graph);
        int input_outer_potential_index = mapToInputDomain(boundary_vals, input_outer_val, graph);
        if(!verify(merge_expr->as<Expr>(), boundary_vals, graph)){
          std::cerr << "Invalid merge expr: " << merge_expr->toString() << std::endl;
          exit(1);
        }
        // Check if the inner val is a boundary val
        if(input_inner_potential_index != -1 && val2stride_map[graph.toGroup(boundary_vals[input_inner_potential_index])].llvm_stride == nullptr){
          val2stride_map[graph.toGroup(boundary_vals[input_inner_potential_index])].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[graph.toGroup(boundary_vals[input_inner_potential_index])].llvm_extent, "stride_merge_inner_val");
        }
        else if(input_inner_potential_index != -1 && val2stride_map[graph.toGroup(boundary_vals[input_inner_potential_index])].llvm_stride != nullptr){
          return;
        }
        else{
          generate_stride_llvm_ir(input_inner_val, val2stride_map, builder, boundary_vals, running_stride_product, graph);
        }

        // Check if the outer val is a boundary val
        if(input_outer_potential_index != -1 && val2stride_map[graph.toGroup(boundary_vals[input_outer_potential_index])].llvm_stride == nullptr){
          val2stride_map[graph.toGroup(boundary_vals[input_outer_potential_index])].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[graph.toGroup(boundary_vals[input_outer_potential_index])].llvm_extent, "stride_merge_outer_val");
        }
        else if(input_outer_potential_index != -1 && val2stride_map[graph.toGroup(boundary_vals[input_outer_potential_index])].llvm_stride != nullptr){
          // case where the outer val is already computed in previous dfs calls
          return;
        }
        else{
          generate_stride_llvm_ir(input_outer_val, val2stride_map, builder, boundary_vals, running_stride_product, graph);
        }
        
        // Extent of merged domain
        if(val2stride_map[graph.toGroup(input_outer_val)].llvm_extent == nullptr || val2stride_map[graph.toGroup(input_inner_val)].llvm_extent == nullptr || val2stride_map[graph.toGroup(current_val_to_process)].llvm_extent != nullptr){
          return;
        }
        else{
          val2stride_map[graph.toGroup(current_val_to_process)].llvm_extent = builder.CreateMul(
              val2stride_map[graph.toGroup(input_outer_val)].llvm_extent,
              val2stride_map[graph.toGroup(input_inner_val)].llvm_extent,
              current_val_to_process->toString() + "_merged_extent"
          );
        }

    } else if (op_type == "Split") {
        auto* split_expr = def_expr->as<Split>();
        auto* input_val = split_expr->in()->as<Val>();
        auto* output_inner_val = split_expr->inner()->as<Val>();
        auto* output_outer_val = split_expr->outer()->as<Val>();
        int input_val_potential_index = mapToInputDomain(boundary_vals, input_val, graph);

        if(input_val_potential_index != -1 && val2stride_map[graph.toGroup(boundary_vals[input_val_potential_index])].llvm_stride == nullptr){
          val2stride_map[graph.toGroup(boundary_vals[input_val_potential_index])].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[graph.toGroup(boundary_vals[input_val_potential_index])].llvm_extent, "stride_split_input_val");
          return;
        }
        else{
          generate_stride_llvm_ir(input_val, val2stride_map, builder, boundary_vals, running_stride_product, graph);
        }

        int64_t split_factor = stoi(split_expr->factor()->toString());
        if(split_expr->innerSplit()){
          if(split_expr->factor()->isConstInt()){
            val2stride_map[graph.toGroup(output_inner_val)].llvm_extent = builder.getInt64(split_factor);
          }
          else{
            if(val2stride_map.find(graph.toGroup(split_expr->factor())) != val2stride_map.end()){
              val2stride_map[graph.toGroup(output_inner_val)].llvm_extent = val2stride_map[graph.toGroup(split_expr->factor())].llvm_extent;
            }
            else{
              std::cerr << "Error: Inner split factor is not a constant and not found in val2stride_map" << std::endl;
              return;
            }
          }
          if(val2stride_map[graph.toGroup(input_val)].llvm_extent == nullptr || val2stride_map[graph.toGroup(output_inner_val)].llvm_extent == nullptr || val2stride_map[graph.toGroup(output_outer_val)].llvm_extent != nullptr){
            return;
          }
          val2stride_map[graph.toGroup(output_outer_val)].llvm_extent = builder.CreateUDiv(
            val2stride_map[graph.toGroup(input_val)].llvm_extent,
            val2stride_map[graph.toGroup(output_inner_val)].llvm_extent,
            output_outer_val->toString() + "_split_extent"
          );
        }
        else{
          if(split_expr->factor()->isConstInt()){
            val2stride_map[graph.toGroup(output_outer_val)].llvm_extent = builder.getInt64(split_factor);
          }
          else{
            if(val2stride_map.find(graph.toGroup(split_expr->factor())) != val2stride_map.end()){
              val2stride_map[graph.toGroup(output_outer_val)].llvm_extent = val2stride_map[graph.toGroup(split_expr->factor())].llvm_extent;
            }
            else{
              std::cerr << "Error: Outer split factor is not a constant and not found in val2stride_map" << std::endl;
              return;
            }
          }
          if(val2stride_map[graph.toGroup(input_val)].llvm_extent == nullptr || val2stride_map[graph.toGroup(output_inner_val)].llvm_extent == nullptr || val2stride_map[graph.toGroup(output_outer_val)].llvm_extent != nullptr){
            return;
          }
          val2stride_map[graph.toGroup(output_inner_val)].llvm_extent = builder.CreateUDiv(
            val2stride_map[graph.toGroup(input_val)].llvm_extent,
            val2stride_map[graph.toGroup(output_outer_val)].llvm_extent,
            output_inner_val->toString() + "_split_extent"
          );
        }

    } else { // Fallback for other ops (e.g., simple unary pass-through)
        std::cerr << "Warning: Unhandled op_type '" << op_type << "' for Val ";
    }
}

/*

Generate infer stride module

*/
llvm::orc::ThreadSafeModule generate_infer_stride_module(std::vector<IterDomain*>& logical_domain, std::vector<IterDomain*>& allocation_domain, Fusion& fusion, std::string name) {
  auto Context = std::make_unique<llvm::LLVMContext>();
  auto* ctx = Context.get();
  auto Module = std::make_unique<llvm::Module>(name, *ctx);
  llvm::IRBuilder<> builder(*ctx);
  auto* int64Ty = llvm::Type::getInt64Ty(*ctx);
  auto* ptrTy = llvm::PointerType::getUnqual(int64Ty);

  auto* funcTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*ctx), {ptrTy, int64Ty, ptrTy, int64Ty}, false);
  auto* func = llvm::Function::Create(funcTy, llvm::Function::ExternalLinkage, "infer_stride", Module.get());
  auto* entry = llvm::BasicBlock::Create(*ctx, "entry", func);
  builder.SetInsertPoint(entry);

  std::vector<Val*> input_vals = domain2vals(logical_domain);
  std::vector<Val*> output_vals = domain2vals(allocation_domain);
  auto arg_it = func->arg_begin();
  llvm::Value* input_ptr = &*arg_it;
  llvm::Value* output_ptr = &*arg_it+2;

  std::unordered_map<ValGroup, StrideInfo> val2stride;
  std::unordered_map<int, Val*> boundary_vals;
  for(size_t i = 0; i < input_vals.size(); i++){
    boundary_vals[i] = input_vals[i];
  }

  IdModel id_model(&fusion);
  const ValGraph& graph = id_model.buildExactGraph();
  
  for(long unsigned int i = 0; i < input_vals.size(); i++){
    auto* zero = builder.getInt64(i);
    auto* input_i_ptr = builder.CreateGEP(int64Ty, input_ptr, zero, "ptr");
    auto* input_i_val = builder.CreateLoad(int64Ty, input_i_ptr, "val");
    val2stride[graph.toGroup(input_vals[i])] = StrideInfo();
    val2stride[graph.toGroup(input_vals[i])].llvm_extent = input_i_val;
  }

  for(auto* val : output_vals){
    auto index = mapToInputDomain(boundary_vals, val, graph);
    if(index != -1){
      val2stride[graph.toGroup(val)] = val2stride[graph.toGroup(boundary_vals[index])];
    }
  }

  llvm::Value* running_stride_product = builder.getInt64(1);
  for(auto it = allocation_domain.rbegin(); it != allocation_domain.rend(); ++it){
    generate_stride_llvm_ir(*it, val2stride, builder, boundary_vals, running_stride_product, graph);
  }

  for(long unsigned int i = 0; i < logical_domain.size(); i++){
    if(val2stride[graph.toGroup(input_vals[i])].llvm_stride == nullptr){
      continue;
    }
    auto* output_i_ptr = builder.CreateGEP(int64Ty, output_ptr, builder.getInt64(i), "ptr");
    builder.CreateStore(val2stride[graph.toGroup(input_vals[i])].llvm_stride, output_i_ptr);
  }

  builder.CreateRetVoid();
  // llvm::outs() << "=== LLVM IR ===\n";
  // Module->print(llvm::outs(), nullptr);
  return llvm::orc::ThreadSafeModule(std::move(Module), std::move(Context));
}

/*

Generate infer shape module

*/
llvm::orc::ThreadSafeModule generate_infer_shape_module(std::vector<IterDomain*>& input_domain, std::vector<IterDomain*>& output_domain, Fusion& fusion, std::string name) {
  auto Context = std::make_unique<llvm::LLVMContext>();
  auto* ctx = Context.get();
  auto Module = std::make_unique<llvm::Module>(name, *ctx);
  llvm::IRBuilder<> builder(*ctx);
  std::vector<llvm::Type*> output_types;

  // Initialize the output types, linking with llvm outputs
  for(size_t i = 0; i < output_domain.size(); i++){
    output_types.push_back(builder.getInt64Ty());
  }

  // Initialize the input types, linking with llvm inputs
  std::vector<llvm::Type*> input_types;
  for(size_t i = 0; i < input_domain.size(); i++){
    input_types.push_back(builder.getInt64Ty());
  }

  // Initialize the function type, input and output types
  auto* int64Ty = llvm::Type::getInt64Ty(*ctx);
  auto* ptrTy = llvm::PointerType::getUnqual(int64Ty);
  auto* funcTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*ctx), {ptrTy, int64Ty, ptrTy, int64Ty}, false);
  auto* func = llvm::Function::Create(funcTy, llvm::Function::ExternalLinkage, "infer_shape", Module.get());
  auto* entry = llvm::BasicBlock::Create(*ctx, "entry", func);
  builder.SetInsertPoint(entry);

  // Cast input and output domains to vals
  std::vector<Val*> input_values = domain2vals(input_domain);
  std::vector<Val*> output_values = domain2vals(output_domain);

  // Get the function arguments
  auto arg_it = func->arg_begin();
  llvm::Value* input_ptr = &*arg_it;
  llvm::Value* output_ptr = &*arg_it+2;

  // Initialize the id model and the val graph, and Val to llvm value map
  IdModel id_model(&fusion);
  const ValGraph& graph = id_model.buildExactGraph();
  std::unordered_map<int, Val*> boundary_vals;
  std::unordered_map<ValGroup, llvm::Value*> val2llvm_val;

  // Initialize the input values, linking with llvm inputs
  for(size_t i = 0; i < input_domain.size(); i++){
    boundary_vals[i] = input_values[i];
    auto* zero = builder.getInt64(i);
    auto* input_i_ptr = builder.CreateGEP(int64Ty, input_ptr, zero, "ptr");
    auto* input_i_val = builder.CreateLoad(int64Ty, input_i_ptr, "val");
    val2llvm_val[graph.toGroup(input_values[i])] = input_i_val;
  }

  // Generate the shape llvm ir for all the exprs between input and output domain
  generate_all_shape_llvm_ir(graph, input_domain, output_domain, val2llvm_val, boundary_vals, builder);

  // Map the output values to the input values if they are the same
  for(auto* val : output_values){
    auto index = mapToInputDomain(boundary_vals, val, graph);
    if(index != -1){
      val2llvm_val[graph.toGroup(val)] = val2llvm_val[graph.toGroup(boundary_vals[index])];
    }
  }

  // Store the output values to the preallocated output buffer
  for(size_t i = 0; i < output_values.size(); i++){
    auto* output_i_ptr = builder.CreateGEP(int64Ty, output_ptr, builder.getInt64(i), "ptr");
    builder.CreateStore(val2llvm_val[graph.toGroup(output_values[i])], output_i_ptr);
  }

  builder.CreateRetVoid();
  // llvm::outs() << "=== LLVM IR ===\n";
  // Module->print(llvm::outs(), nullptr);
  return llvm::orc::ThreadSafeModule(std::move(Module), std::move(Context));
}

// JIT add module wrapper function
std::unique_ptr<llvm::orc::LLJIT> llvm_jit_init(int num_threads){
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto JIT = ExitOnErr(llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());
  return JIT;
}

// Allocate the output tensor based on the shape and stride inference
at::Tensor aten_output_allocation(FuncType shape_infer_func, FuncType stride_infer_func, const at::Tensor& input, int64_t output_tensor_dim) { 
  std::vector<int64_t> shape_result(output_tensor_dim);
  auto start_time = std::chrono::high_resolution_clock::now();
  shape_infer_func(input.sizes().data(), input.sizes().size(), shape_result.data(), shape_result.size());
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> shape_infer_time = end_time - start_time;
  std::cout << "Shape infer time: " << std::chrono::duration_cast<std::chrono::microseconds>(shape_infer_time).count() << " microseconds" << std::endl;
  std::vector<int64_t> stride_result(output_tensor_dim);
  start_time = std::chrono::high_resolution_clock::now();
  stride_infer_func(shape_result.data(), shape_result.size(), stride_result.data(), stride_result.size());
  end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> stride_infer_time = end_time - start_time;
  std::cout << "Stride infer time: " << std::chrono::duration_cast<std::chrono::microseconds>(stride_infer_time).count() << " microseconds" << std::endl;
  at::Tensor output_tensor = at::empty_strided(shape_result, stride_result, input.options());
  return output_tensor;
}

template llvm::orc::ExecutorAddr nvfuser::ExitOnErr<llvm::orc::ExecutorAddr>(llvm::Expected<llvm::orc::ExecutorAddr> &&E);

} // namespace nvfuser


namespace nvfuser {

// PIMPL implementation for HostIrLlvmJit
struct HostIrLlvmJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  FuncType logical_shape_infer_fn = nullptr;
  FuncType logical_stride_infer_fn = nullptr;
  TensorView* output_tv = nullptr;
};

// Constructor implementation
HostIrLlvmJit::HostIrLlvmJit(int num_threads) : pimpl_(new LlvmJitImpl) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  pimpl_->jit = ExitOnErr(
      llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());
}

// The destructor must be defined here where LlvmJitImpl is a complete type.
HostIrLlvmJit::~HostIrLlvmJit() = default;

// Move constructor and assignment operator
HostIrLlvmJit::HostIrLlvmJit(HostIrLlvmJit&&) noexcept = default;
HostIrLlvmJit& HostIrLlvmJit::operator=(HostIrLlvmJit&&) noexcept = default;

void HostIrLlvmJit::compile(TensorView* output_tv) {
  pimpl_->output_tv = output_tv;
  Fusion* fusion = output_tv->fusion();
  NVF_ERROR(fusion != nullptr, "Output TensorView must belong to a fusion.");

  // This simplified API assumes a single input TensorView.
  // This can be extended to handle multiple inputs.
  std::vector<TensorView*> input_tvs;
  TensorView* input_tv = nullptr;
  for (auto inp : fusion->inputs()) {
    if (auto tv = dynamic_cast<TensorView*>(inp)) {
      NVF_ERROR(
          input_tv == nullptr,
          "Multiple input TensorViews not yet supported in this simplified API");
      input_tvs.push_back(tv);
    }
  }
  NVF_ERROR(input_tvs.size() > 0, "No input TensorView found in fusion");

  std::vector<IterDomain*> input_logical_domains;
  for (auto input_tv : input_tvs) {
    input_logical_domains.insert(
        input_logical_domains.end(),
        input_tv->getLogicalDomain().begin(),
        input_tv->getLogicalDomain().end());
  }

  auto output_logical_domain = output_tv->getLogicalDomain();
  auto output_allocation_domain = output_tv->getMaybeAllocationDomain();

  auto TSM_logical_shape =
      generate_infer_shape_module(input_logical_domains, output_logical_domain, *fusion, "infer_logical_shape_module");
  if (auto Err = pimpl_->jit->addIRModule(std::move(TSM_logical_shape))) {
    llvm::errs() << "Error adding shape infer module to JIT: "
                 << llvm::toString(std::move(Err)) << "\n";
  }

  // JIT compile stride inference module
  auto TSM_logical_stride =
      generate_infer_stride_module(output_logical_domain, output_allocation_domain, *fusion, "infer_logical_stride_module");
  if (auto Err = pimpl_->jit->addIRModule(std::move(TSM_logical_stride))) {
    llvm::errs() << "Error adding stride infer module to JIT: "
                 << llvm::toString(std::move(Err)) << "\n";
  }
  // Look up the function pointers and store them
  pimpl_->logical_shape_infer_fn =
      ExitOnErr(pimpl_->jit->lookup("infer_shape")).toPtr<FuncType>();
  pimpl_->logical_stride_infer_fn =
      ExitOnErr(pimpl_->jit->lookup("infer_stride")).toPtr<FuncType>();
}

at::Tensor HostIrLlvmJit::allocateOutputTensor(const std::vector<at::Tensor>& input_tensors) {
  NVF_ERROR(
      pimpl_->logical_shape_infer_fn != nullptr && pimpl_->logical_stride_infer_fn != nullptr
      && pimpl_->output_tv != nullptr,
      "JIT must be compiled before running.");

  // Allocate memory for shape result
  std::vector<int64_t> logical_shape_result(pimpl_->output_tv->getLogicalDomain().size());
  std::vector<int64_t> input_sizes;
  for(auto& input_tensor : input_tensors){
    input_sizes.insert(input_sizes.end(), input_tensor.sizes().begin(), input_tensor.sizes().end());
  }

  // Run output tensor logical shape inference
  pimpl_->logical_shape_infer_fn(
      input_sizes.data(),
      input_sizes.size(),
      logical_shape_result.data(),
      logical_shape_result.size());

  // Allocate memory for stride result
  std::vector<int64_t> logical_stride_result(pimpl_->output_tv->getLogicalDomain().size());

  // Run output tensor logical stride inference
  pimpl_->logical_stride_infer_fn(
      logical_shape_result.data(),
      logical_shape_result.size(),
      logical_stride_result.data(),
      logical_stride_result.size());

  // Create the output tensor with the computed shape and strides
  at::Tensor allocated_tensor = at::empty_strided(logical_shape_result, logical_stride_result, input_tensors[0].options());
  return allocated_tensor;
}

} // namespace nvfuser