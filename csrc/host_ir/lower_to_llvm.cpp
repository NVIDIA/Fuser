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
// Codegen type for the shape inference
enum class codegenType{
  Merge, // nvfuser merge -> llvm mul
  Split // nvfuser split -> llvm constant padding + udiv
};

// Dependency graph entry for the shape inference
class dependency_graph{
  public:
  codegenType op; // Codegen type for the shape inference
  std::vector<Val*> input_vals; // Vals that defined the current Val
  llvm::Value* llvm_val; // LLVM Value for the current Val
  bool is_left_val; // Whether the current Val is the left Val of the output of Split operation or input of Merge operation
  dependency_graph(){
    op = codegenType::Merge;
    llvm_val = nullptr;
    is_left_val = false;
  }
};

// Dependency graph entry for the stride inference
struct StrideInfo {
public:
    llvm::Value* llvm_extent = nullptr;   // LLVM Value for the extent of this IterDomain
    llvm::Value* llvm_stride = nullptr;   // LLVM Value for the calculated stride of this IterDomain
};


// Print all exprs between input and output domain
void print_getExprsBetween(const std::vector<IterDomain*>& input_domain, 
const std::vector<IterDomain*>& output_domain){
   auto path = getExprsBetween<IRBFS>(
                  {input_domain.begin(), input_domain.end()}, {output_domain.begin(), output_domain.end()}, false)
                  .first;
  for (const auto& [expr, direction] : path) {
    std::cout<< "Expr: " << expr->toString() << " " << std::endl;
  }
}

// Print all expr groups between input and output domain
void print_getAllExprGroupsBetween(Fusion& fusion, const std::vector<IterDomain*>& in_loop_domain, 
const std::vector<IterDomain*>& out_loop_domain){
  IdModel id_model(&fusion);
  const ValGraph& graph = id_model.buildExactGraph();
  ValGroups tv0_loop_groups = graph.toGroups(in_loop_domain);
  ValGroups tv1_loop_groups = graph.toGroups(out_loop_domain);
  auto result =
      getAllExprGroupsBetween(graph, tv0_loop_groups, tv1_loop_groups, true, Direction::Forward).first;
  for(auto expr_group : result){
    for(auto expr : *expr_group.first){
      std::cout<< "Expr: " << expr->toString() << " ";  
      auto val_inputs = expr->inputs();
      std::cout << "Op: " << expr->getOpString() << " " << std::endl;
      for(auto* val_input : val_inputs){
        std::cout<< "Val: " << val_input->toString() << " ";
      }
      if(std::string(expr->getOpString()) == "Split"){
        for(auto* output_value : expr->outputs()){
          auto* out_domain = dynamic_cast<IterDomain*>(output_value);
          if(out_domain->extent()->isConstInt()){
            std::cout<< "non symbolic output_value: " << out_domain->extent()->toString() << " " << std::endl;
          }
          else{
            std::cout<< "symbolic output_value: " << output_value->toString() << " " << std::endl;
          }
        }
      }  
      std::cout<<std::endl;
    }
  }
}

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

// Helper function to cast vals to iter domains
std::vector<IterDomain*> vals2domain(const std::vector<Val*>& domain){
  std::vector<IterDomain*> vals;
  for(auto* val : domain){
    vals.push_back(dynamic_cast<IterDomain*>(val));
  }
  return vals;
}

// Helper function to map the current domain to the input domain strictly
int mapToInputDomainStrict(std::unordered_map<int, Val*>& boundary_vals, Val* current_domain){
  for(auto boundary_val : boundary_vals){
    if(boundary_val.second == current_domain){
      return boundary_val.first;
    }
  }
  return -1;
}

// Helper function to check if the current iter domain is alias to the input iter domain
int mapToInputDomain(std::unordered_map<int, Val*>& boundary_vals, Val* current_domain, const ValGraph& exact_graph){
  int strict_map_index = mapToInputDomainStrict(boundary_vals, current_domain);
  if(strict_map_index != -1){
    return strict_map_index;
  }
  for(auto boundary_val : boundary_vals){ 
    if(exact_graph.disjointValSets().strictAreMapped(boundary_val.second, current_domain)){
      return boundary_val.first;
    }
  } 
  return -1;
}

/*

Build dependency graph for a given expression in shape inference before llvm codegen

TODO: Need more comments here to explain the logic

*/
void build_dep_graph(Expr* expr, std::unordered_map<Val*, dependency_graph>& val2graph, llvm::IRBuilder<>& builder){
  for(auto* output_value : expr->outputs()){
    if(val2graph.find(output_value) != val2graph.end()){
      return;
    }
  }
  if(std::string(expr->getOpString()) == "Split"){
    auto* output_value_1 = expr->outputs()[0];
    auto* output_value_2 = expr->outputs()[1];
    auto* out_domain_1 = dynamic_cast<IterDomain*>(output_value_1);
    auto* out_domain_2 = dynamic_cast<IterDomain*>(output_value_2);

    if(out_domain_1->extent()->isConstInt()){
      val2graph[output_value_2] = dependency_graph();
      val2graph[output_value_2].op = codegenType::Split;
      val2graph[output_value_2].input_vals.push_back(output_value_1);
      val2graph[output_value_2].input_vals.push_back(expr->inputs()[0]);
      val2graph[output_value_2].is_left_val = true;

      val2graph[output_value_1] = dependency_graph();
      val2graph[output_value_1].op = codegenType::Split;
      val2graph[output_value_1].llvm_val = builder.getInt64(std::stoi(out_domain_1->extent()->toString()));
    }
    else if(out_domain_2->extent()->isConstInt()){
      val2graph[output_value_1] = dependency_graph();
      val2graph[output_value_1].op = codegenType::Split;
      val2graph[output_value_1].input_vals.push_back(output_value_2);
      val2graph[output_value_1].input_vals.push_back(expr->inputs()[0]);
      val2graph[output_value_1].is_left_val = false;

      val2graph[output_value_2] = dependency_graph();
      val2graph[output_value_2].op = codegenType::Split;
      val2graph[output_value_2].llvm_val = builder.getInt64(std::stoi(out_domain_2->extent()->toString()));
    }
    else{
      val2graph[output_value_1] = dependency_graph();
      val2graph[output_value_1].op = codegenType::Split;
      val2graph[output_value_1].llvm_val = builder.getInt64(std::stoi(out_domain_1->extent()->toString()));

      val2graph[output_value_2] = dependency_graph();
      val2graph[output_value_2].op = codegenType::Split;
      val2graph[output_value_2].llvm_val = builder.getInt64(std::stoi(out_domain_2->extent()->toString()));
    }
  }
  else if(std::string(expr->getOpString()) == "Merge"){
    auto* output_value = expr->outputs()[0];

    val2graph[output_value] = dependency_graph();
    val2graph[output_value].op = codegenType::Merge;
    val2graph[output_value].input_vals.push_back(expr->inputs()[0]);

    val2graph[output_value].input_vals.push_back(expr->inputs()[1]);
  }
}


/*

Generate LLVM IR for a dependency graph
By default, we assume it is in typological order, which means input values are ready to use

*/
void generate_shape_llvm_ir(Expr* expr, llvm::IRBuilder<>& builder, std::unordered_map<Val*,llvm::Value*>& val2llvm, std::unordered_map<int, Val*>& boundary_vals) {
  std::string op_string = std::string(expr->getOpString());

  // Perform the merge -> mul transformation
  if(op_string == "Merge"){
    llvm::Value* result = nullptr;
    auto* merge_expr = expr->as<Merge*>();
    auto merge_input_outer_val = merge_expr->outer()->as<Val*>();
    auto merge_input_inner_val = merge_expr->inner()->as<Val*>();
    auto merge_output_val = merge_expr->outputs()[0]->as<Val*>();

    int input_outer_potential_index = mapToInputDomain(boundary_vals, merge_input_outer_val, graph);
    int input_inner_potential_index = mapToInputDomain(boundary_vals, merge_input_inner_val, graph);
    llvm::Value* input_outer_llvm_val = nullptr;
    llvm::Value* input_inner_llvm_val = nullptr;

    if(input_outer_potential_index != -1){
      input_outer_llvm_val = val2llvm[boundary_vals[input_outer_potential_index]];
    }
    else{
      input_outer_llvm_val = val2llvm[merge_input_outer_val];
    }

    if(input_inner_potential_index != -1){
      input_inner_llvm_val = val2llvm[boundary_vals[input_inner_potential_index]];
    }
    else{
      input_inner_llvm_val = val2llvm[merge_input_inner_val];
    }

    result = builder.CreateMul(input_outer_llvm_val, input_inner_llvm_val, merge_output_val->toString());

    val2llvm[merge_output_val] = result;
  }
  else if(op_string == "Split"){
    auto* split_expr = expr->as<Split*>();
    auto* split_input_val = split_expr->in()->as<Val*>();
    auto* split_output_outer_val = split_expr->outer()->as<Val*>();
    auto* split_output_inner_val = split_expr->inner()->as<Val*>();

    int input_potential_index = mapToInputDomain(boundary_vals, split_input_val, graph);
    llvm::Value* input_llvm_val = nullptr;
    if(input_potential_index != -1){
      input_llvm_val = val2llvm[boundary_vals[input_potential_index]]; 
    }
    else{
      input_llvm_val = val2llvm[split_input_val];
    }

    // Perform the split -> ceildiv transformation
    if(split_expr->innerSplit()){
      // inner = factor
      val2llvm[split_output_inner_val] = builder.getInt64(std::stoi(split_expr->factor()->toString()));
      // outer = input + 1
      llvm::Value* minus_1 = builder.CreateSub(input_llvm_val, builder.getInt64(1), "minus_1");
      // outer = (input + 1) + inner
      llvm::Value* sum_ab = builder.CreateAdd(minus_1, val2llvm[split_output_inner_val], "sum_ab");
      // outer = (input + 1 + inner) / inner
      val2llvm[split_output_outer_val] = builder.CreateUDiv(sum_ab, val2llvm[split_output_inner_val], split_output_outer_val->extent()->toString());
    }
    else{
      val2llvm[split_output_outer_val] = builder.getInt64(std::stoi(split_expr->factor()->toString()));
      // inner = input - 1
      llvm::Value* minus_1 = builder.CreateSub(input_llvm_val, builder.getInt64(1), "minus_1");
      // inner = (input - 1) + outer
      llvm::Value* sum_ab = builder.CreateAdd(minus_1, val2llvm[split_output_outer_val], "sum_ab");
      // inner = (input - 1 + outer) / outer
      val2llvm[split_output_inner_val] = builder.CreateUDiv(sum_ab, val2llvm[split_output_outer_val], split_output_inner_val->extent()->toString());
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
std::vector<Expr*> traverse_expr_group(const ValGraph& graph, std::vector<IterDomain*>& input_domain, std::vector<IterDomain*>& output_domain){
  ValGroups tv0_loop_groups = graph.toGroups(input_domain);
  ValGroups tv1_loop_groups = graph.toGroups(output_domain);
  std::unordered_map<Val*, llvm::Value*> val2llvm_val;
  auto result = getAllExprGroupsBetween(graph, tv0_loop_groups, tv1_loop_groups).first;
  for(auto expr_group : result){
    for(auto expr : *expr_group.first){
      for(auto* input : expr->inputs()){
        if(val2llvm_val.find(input) == val2llvm_val.end()){
          val2llvm_val[input] = builder.getInt64(std::stoi(input->extent()->toString()));
        }
      }
    }
  }
  return exprs; 
}

/*

Verify if the merge is legit by traversing the exprs between output and input domain

TODO: Need to implement this function
*/
Val* findLowestCommonAncestor(Val* left_val, Val* right_val, std::unordered_map<Val*, dependency_graph>& val2graph) {
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

    auto it = val2graph.find(current);
    if (it != val2graph.end()) {
      for (auto* input : it->second.input_vals) {
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

    auto it = val2graph.find(current);
    if (it != val2graph.end()) {
      for (auto* input : it->second.input_vals) {
        if (visited_right.find(input) == visited_right.end()) {
          visited_right.insert(input);
          q2.push(input);
        }
      }
    }
  }

  return nullptr;
}

bool verify(Val* left_val, Val* right_val, std::unordered_map<Val*, dependency_graph>& val2graph) {
  Val* ancestor = findLowestCommonAncestor(left_val, right_val, val2graph);
  if (!ancestor) {
    // If there is no common ancestor, it's not a valid merge.
    return false;
  }

  auto traverse = [&](Val* start_node, bool is_from_left_val) -> bool {
    Val* current = start_node;
    std::unordered_set<Val*> visited;

    while (current != ancestor) {
      if (!current || visited.count(current)) {
        return false; // Reached null or cycle without finding ancestor
      }
      visited.insert(current);

      auto it = val2graph.find(current);
      if (it == val2graph.end()) {
        return false; // Path incomplete in val2graph
      }

      auto& dep_node = it->second;
      if (dep_node.op == codegenType::Merge) {
        if (dep_node.input_vals.size() != 2) {
          return false;
        }
        if (is_from_left_val) {
          current =
              dep_node.input_vals[1]; // traverse rightmost for left_val path
        } else {
          current =
              dep_node.input_vals[0]; // traverse leftmost for right_val path
        }
      } else if (dep_node.op == codegenType::Split) {
        if (dep_node.input_vals.size() != 2) {
          return false;
        }
        if (is_from_left_val) {
          // for left_val, path should be on the right side, so is_left_val
          // should be true
          if (!dep_node.is_left_val) {
            return false;
          }
        } else {
          // for right_val, path should be on the left side, so is_left_val
          // should be false
          if (dep_node.is_left_val) {
            return false;
          }
        }
        // Traverse up to the original input of the split operation
        current = dep_node.input_vals[1];
      } else {
        // Other op types are not expected in this verification path.
        return false;
      }
    }
    return true;
  };

  if (!traverse(left_val, true)) {
    return false;
  }
  if (!traverse(right_val, false)) {
    return false;
  }

  return true;
}

bool verify_exprs(const ValGraph& graph, std::vector<IterDomain*>& input_domain, std::vector<IterDomain*>& output_domain, std::unordered_map<Val*, dependency_graph>& val2graph){
  auto exprs = traverse_expr_group(graph, input_domain, output_domain);
  for(auto* expr : exprs){
    if(std::string(expr->getOpString()) == "Merge"){
      if(verify(expr->inputs()[0], expr->inputs()[1], val2graph)){
        std::cerr << "Invalid merge expr: " << expr->toString() << std::endl;
        return false;
      }
    }
  }
  return true;
}

/*

Generate LLVM IR for stride inference

*/
void generate_stride_llvm_ir(
    Val* current_val_to_process,
    std::unordered_map<Val*, StrideInfo>& val2stride_map,
    llvm::IRBuilder<>& builder,
    std::unordered_map<int, Val*>& boundary_vals,
    llvm::Value*& running_stride_product,
    const ValGraph& graph
    ) {
    if (current_val_to_process == nullptr) {
        std::cerr << "Error: generate_stride_llvm_ir called with nullptr Val." << std::endl;
        return;
    }
    int cur_val_boundary_index = mapToInputDomain(boundary_vals, current_val_to_process, graph);
    if(cur_val_boundary_index != -1){
      if(val2stride_map[boundary_vals[cur_val_boundary_index]].llvm_stride == nullptr){
        val2stride_map[boundary_vals[cur_val_boundary_index]].llvm_stride = running_stride_product;
        running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[boundary_vals[cur_val_boundary_index]].llvm_extent, "stride_root_val");
      }
      return;
    }

    if (val2stride_map.count(current_val_to_process) && val2stride_map[current_val_to_process].llvm_stride != nullptr) {
        return; // Memoization: Already processed
    }

    auto def_expr = current_val_to_process->definition();
    if (def_expr == nullptr) { // Base Case: Root Val (e.g., physical input IterDomain)
        if (!val2stride_map.count(current_val_to_process) || val2stride_map[current_val_to_process].llvm_stride == nullptr) {
            std::cerr << "Warning: StrideInfo not pre-populated for root Val: "
                      << current_val_to_process->toString() << ". Its stride will be unknown." << std::endl;
        }
        return;
    }
    std::string op_type = def_expr->getOpString();
    if (op_type == "Merge") {
        Val* input_inner_val = def_expr->inputs()[1];
        Val* input_outer_val = def_expr->inputs()[0];
        int input_inner_val_boundary_index = mapToInputDomain(boundary_vals, input_inner_val, graph);
        int input_outer_val_boundary_index = mapToInputDomain(boundary_vals, input_outer_val, graph);
        if(input_inner_val_boundary_index != -1 && val2stride_map[boundary_vals[input_inner_val_boundary_index]].llvm_stride == nullptr){
          val2stride_map[boundary_vals[input_inner_val_boundary_index]].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[boundary_vals[input_inner_val_boundary_index]].llvm_extent, "stride_merge_inner_val");
        }
        else if(input_inner_val_boundary_index != -1 && val2stride_map[boundary_vals[input_inner_val_boundary_index]].llvm_stride != nullptr){
          // case where the inner val is already computed in previous dfs calls
          return;
        }
        else{
          generate_stride_llvm_ir(input_inner_val, val2stride_map, builder, boundary_vals, running_stride_product, graph);
        }
        if(input_outer_val_boundary_index != -1 && val2stride_map[boundary_vals[input_outer_val_boundary_index]].llvm_stride == nullptr){
          val2stride_map[boundary_vals[input_outer_val_boundary_index]].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[boundary_vals[input_outer_val_boundary_index]].llvm_extent, "stride_merge_outer_val");
        }
        else if(input_outer_val_boundary_index != -1 && val2stride_map[boundary_vals[input_outer_val_boundary_index]].llvm_stride != nullptr){
          // case where the outer val is already computed in previous dfs calls
          return;
        }
        else{
          generate_stride_llvm_ir(input_outer_val, val2stride_map, builder, boundary_vals, running_stride_product, graph);
        }
        
        // Extent of merged domain
        if(val2stride_map[input_outer_val].llvm_extent == nullptr || val2stride_map[input_inner_val].llvm_extent == nullptr){
          // std::cout << "outer and inner val extent not computed" << std::endl;
          return;
        }
        else if(val2stride_map[current_val_to_process].llvm_extent != nullptr){
          // std::cout << "current val extent already computed" << std::endl;
          return;
        }
        else{
          val2stride_map[current_val_to_process].llvm_extent = builder.CreateMul(
              val2stride_map[input_outer_val].llvm_extent,
              val2stride_map[input_inner_val].llvm_extent,
              current_val_to_process->toString() + "_merged_extent"
          );
        }

    } else if (op_type == "Split") {
        Val* input_val = def_expr->inputs()[0];
        Val* output_inner_val = def_expr->outputs()[1];
        Val* output_outer_val = def_expr->outputs()[0];
        auto output_inner_domain = dynamic_cast<IterDomain*>(output_inner_val);
        auto output_outer_domain = dynamic_cast<IterDomain*>(output_outer_val);
        int input_val_boundary_index = mapToInputDomain(boundary_vals, input_val, graph);
        if(input_val_boundary_index != -1 && val2stride_map[boundary_vals[input_val_boundary_index]].llvm_stride == nullptr){
          val2stride_map[boundary_vals[input_val_boundary_index]].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[boundary_vals[input_val_boundary_index]].llvm_extent, "stride_split_input_val");
        }
        else if(input_val_boundary_index != -1 && val2stride_map[boundary_vals[input_val_boundary_index]].llvm_stride != nullptr){
          return;
        }
        else{
          generate_stride_llvm_ir(input_val, val2stride_map, builder, boundary_vals, running_stride_product, graph);
        }
        bool is_inner_val_const = output_inner_domain->extent()->isConstInt();
        bool is_outer_val_const = output_outer_domain->extent()->isConstInt();
        if(is_inner_val_const && val2stride_map[output_outer_val].llvm_extent == nullptr){
          val2stride_map[output_inner_val].llvm_extent = builder.getInt64(stoi(output_inner_domain->extent()->toString()));
          val2stride_map[output_outer_val].llvm_extent = builder.CreateUDiv(
            val2stride_map[input_val].llvm_extent,
            val2stride_map[output_inner_val].llvm_extent,
            output_outer_val->toString() + "_split_extent"
          );
        }
        else if(is_outer_val_const && val2stride_map[output_inner_val].llvm_extent == nullptr){
          val2stride_map[output_outer_val].llvm_extent = builder.getInt64(stoi(output_outer_domain->extent()->toString()));
          val2stride_map[output_inner_val].llvm_extent = builder.CreateUDiv(
            val2stride_map[input_val].llvm_extent,
            val2stride_map[output_outer_val].llvm_extent,
            output_inner_val->toString() + "_split_extent"
          );
        }
        else{
          val2stride_map[output_inner_val].llvm_extent = builder.CreateUDiv(
            val2stride_map[input_val].llvm_extent,
            val2stride_map[output_outer_val].llvm_extent,
            output_inner_val->toString() + "_split_extent"
          );
          val2stride_map[output_outer_val].llvm_extent = builder.CreateUDiv(
            val2stride_map[input_val].llvm_extent,
            val2stride_map[output_inner_val].llvm_extent,
            output_outer_val->toString() + "_split_extent"
          );
        }
    } else { // Fallback for other ops (e.g., simple unary pass-through)
        std::cerr << "Warning: Unhandled op_type '" << op_type << "' for Val ";
    }
}

/*

Generate infer stride module

*/
llvm::orc::ThreadSafeModule generate_infer_stride_module(std::vector<IterDomain*>& allocation_domain, std::vector<IterDomain*>& logical_domain, Fusion& fusion) {
  auto Context = std::make_unique<llvm::LLVMContext>();
  auto* ctx = Context.get();
  auto Module = std::make_unique<llvm::Module>("infer_stride_module", *ctx);
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

  std::unordered_map<Val*, StrideInfo> val2stride;
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
    val2stride[input_vals[i]] = StrideInfo();
    val2stride[input_vals[i]].llvm_extent = input_i_val;
  }

  for(auto* val : output_vals){
    auto index = mapToInputDomain(boundary_vals, val, graph);
    if(index != -1){
      val2stride[val] = val2stride[boundary_vals[index]];
    }
  }

  llvm::Value* running_stride_product = builder.getInt64(1);
  for(auto it = allocation_domain.rbegin(); it != allocation_domain.rend(); ++it){
    generate_stride_llvm_ir(*it, val2stride, builder, boundary_vals, running_stride_product, graph);
  }

  for(long unsigned int i = 0; i < logical_domain.size(); i++){
    if(val2stride[input_vals[i]].llvm_stride == nullptr){
      continue;
    }
    auto* output_i_ptr = builder.CreateGEP(int64Ty, output_ptr, builder.getInt64(i), "ptr");
    builder.CreateStore(val2stride[input_vals[i]].llvm_stride, output_i_ptr);
  }

  builder.CreateRetVoid();
  // llvm::outs() << "=== LLVM IR ===\n";
  // Module->print(llvm::outs(), nullptr);
  return llvm::orc::ThreadSafeModule(std::move(Module), std::move(Context));
}


/*

Generate infer shape module

*/
llvm::orc::ThreadSafeModule generate_infer_shape_module(std::vector<IterDomain*>& input_logical_domain, std::vector<IterDomain*>& output_logical_domain, Fusion& fusion) {
  auto Context = std::make_unique<llvm::LLVMContext>();
  auto* ctx = Context.get();
  auto Module = std::make_unique<llvm::Module>("infer_shape_module", *ctx);
  llvm::IRBuilder<> builder(*ctx);
  std::vector<llvm::Type*> output_types;
  for(long unsigned int i = 0; i < output_logical_domain.size(); i++){
    output_types.push_back(builder.getInt64Ty());
  }
  std::vector<llvm::Type*> input_types;
  for(long unsigned int i = 0; i < input_logical_domain.size(); i++){
    input_types.push_back(builder.getInt64Ty());
  }
  auto* int64Ty = llvm::Type::getInt64Ty(*ctx);
  auto* ptrTy = llvm::PointerType::getUnqual(int64Ty);

  auto* funcTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*ctx), {ptrTy, int64Ty, ptrTy, int64Ty}, false);
  auto* func = llvm::Function::Create(funcTy, llvm::Function::ExternalLinkage, "infer_shape", Module.get());
  auto* entry = llvm::BasicBlock::Create(*ctx, "entry", func);
  builder.SetInsertPoint(entry);
  std::unordered_map<Val*, dependency_graph> val2graph;
  std::vector<Val*> input_values = domain2vals(input_logical_domain);
  std::vector<Val*> output_values = domain2vals(output_logical_domain);
  auto arg_it = func->arg_begin();
  llvm::Value* input_ptr = &*arg_it;
  llvm::Value* output_ptr = &*arg_it+2;
  IdModel id_model(&fusion);
  const ValGraph& graph = id_model.buildExactGraph();
  auto exprs = traverse_expr_group(graph, input_logical_domain, output_logical_domain);
  std::unordered_map<int, Val*> boundary_vals;
  for(long unsigned int i = 0; i < input_logical_domain.size(); i++){
    boundary_vals[i] = input_values[i];
    auto* zero = builder.getInt64(i);
    auto* input_i_ptr = builder.CreateGEP(int64Ty, input_ptr, zero, "ptr");
    auto* input_i_val = builder.CreateLoad(int64Ty, input_i_ptr, "val");
    val2graph[input_values[i]] = dependency_graph();
    val2graph[input_values[i]].llvm_val = input_i_val;
  }

  // Map the output values to the input values if they are the same
  for(auto* val : output_values){
    auto index = mapToInputDomain(boundary_vals, val, graph);
    if(index != -1){
      val2graph[val] = val2graph[boundary_vals[index]];
    }
  }

  for(auto* expr : exprs){
    std::cout << expr->toString() << std::endl;
  }

  for(auto* val : output_values){
    generate_shape_llvm_ir(val, val2graph, builder);
  }

  for(long unsigned int i = 0; i < output_values.size(); i++){
    auto* output_i_ptr = builder.CreateGEP(int64Ty, output_ptr, builder.getInt64(i), "ptr");
    builder.CreateStore(val2graph[output_values[i]].llvm_val, output_i_ptr);
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

// shape infer compile
void llvm_jit_compile_shape_infer(std::unique_ptr<llvm::orc::LLJIT>& JIT, Fusion& fusion, std::vector<IterDomain*>& input_domain, std::vector<IterDomain*>& output_domain){
  std::cout << "llvm_jit_compile shape infer" << std::endl;
  auto TSM_shape = generate_infer_shape_module(input_domain, output_domain, fusion);
  if (auto Err = JIT->addIRModule(std::move(TSM_shape))) {
    llvm::errs() << "Error adding module to JIT: " << llvm::toString(std::move(Err)) << "\n";
  }
}

// stride infer compile
void llvm_jit_compile_stride_infer(std::unique_ptr<llvm::orc::LLJIT>& JIT,Fusion& fusion, std::vector<IterDomain*>& allocation_domain, std::vector<IterDomain*>& logical_domain){
  std::cout << "llvm_jit_compile stride infer" << std::endl;
  auto TSM_stride = generate_infer_stride_module(allocation_domain, logical_domain, fusion);
  if (auto Err = JIT->addIRModule(std::move(TSM_stride))) {
    llvm::errs() << "Error adding module to JIT: " << llvm::toString(std::move(Err)) << "\n";
  }
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
  FuncType shape_infer_fn = nullptr;
  FuncType stride_infer_fn = nullptr;
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
  auto output_loop_domain = output_tv->getLoopDomain();
  auto output_allocation_domain = output_tv->getMaybeAllocationDomain();

  // Store the output dimension for the run method
  output_tensor_dim_ = output_loop_domain.size();

  // JIT compile shape inference module
  auto TSM_shape =
      generate_infer_shape_module(input_logical_domains, output_loop_domain, *fusion);
  if (auto Err = pimpl_->jit->addIRModule(std::move(TSM_shape))) {
    llvm::errs() << "Error adding shape infer module to JIT: "
                 << llvm::toString(std::move(Err)) << "\n";
  }

  // JIT compile stride inference module
  auto TSM_stride =
      generate_infer_stride_module(output_allocation_domain, output_loop_domain, *fusion);
  if (auto Err = pimpl_->jit->addIRModule(std::move(TSM_stride))) {
    llvm::errs() << "Error adding stride infer module to JIT: "
                 << llvm::toString(std::move(Err)) << "\n";
  }

  // Look up the function pointers and store them
  pimpl_->shape_infer_fn =
      ExitOnErr(pimpl_->jit->lookup("infer_shape")).toPtr<FuncType>();
  pimpl_->stride_infer_fn =
      ExitOnErr(pimpl_->jit->lookup("infer_stride")).toPtr<FuncType>();
}

at::Tensor HostIrLlvmJit::allocateOutputTensor(const std::vector<at::Tensor>& input_tensors) {
  NVF_ERROR(
      pimpl_->shape_infer_fn != nullptr && pimpl_->stride_infer_fn != nullptr,
      "JIT must be compiled before running.");

  // Allocate memory for shape result
  std::vector<int64_t> shape_result(output_tensor_dim_);
  std::vector<int64_t> input_sizes;
  for(auto& input_tensor : input_tensors){
    input_sizes.insert(input_sizes.end(), input_tensor.sizes().begin(), input_tensor.sizes().end());
  }
  // Run shape inference
  pimpl_->shape_infer_fn(
      input_sizes.data(),
      input_sizes.size(),
      shape_result.data(),
      shape_result.size());

  // Allocate memory for stride result
  std::vector<int64_t> stride_result(output_tensor_dim_);

  // Run stride inference
  pimpl_->stride_infer_fn(
      shape_result.data(),
      shape_result.size(),
      stride_result.data(),
      stride_result.size());

  // Create the output tensor with the computed shape and strides
  at::Tensor output_tensor =
      at::empty_strided(shape_result, stride_result, input_tensors[0].options());
  return output_tensor;
}

} // namespace nvfuser