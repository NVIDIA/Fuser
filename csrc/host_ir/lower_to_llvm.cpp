#include <host_ir/lower_to_llvm.h>

namespace nvfuser {

// Print tensor info
void print_compare_tensor(const at::Tensor& t0, const at::Tensor& t1){
  std::cout<< "t0" << std::endl;
  std::cout << "Tensor dtype: " << t0.dtype() << "\n";
  std::cout << "Shape: " << t0.sizes() << "\n"; 
  std::cout << "Strides: " << t0.strides() << "\n";
  std::cout << "Is Contiguous: " << t0.is_contiguous() << "\n";
  std::cout << "Device: " << t0.device() << "\n";
  std::cout << "Data ptr: " << t0.data_ptr() << "\n";
  std::cout<< "t1" << std::endl;
  std::cout << "Tensor dtype: " << t1.dtype() << "\n";
  std::cout << "Shape: " << t1.sizes() << "\n"; 
  std::cout << "Strides: " << t1.strides() << "\n";
  std::cout << "Is Contiguous: " << t1.is_contiguous() << "\n";
  std::cout << "Device: " << t1.device() << "\n";
  std::cout << "Data ptr: " << t1.data_ptr() << "\n";
  return;
}

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

// Helper function to check if the current iter domain is alias to the input iter domain
int mapToInputDomain(std::vector<Val*> input_domain, Val* current_domain, const ValGraph& exact_graph){
  std::vector<IterDomain*> input_domain_iter = vals2domain(input_domain);
  IterDomain* current_domain_iter = dynamic_cast<IterDomain*>(current_domain);
  int index = 0;
  for(auto* id : input_domain_iter){ 
    if(exact_graph.disjointValSets().strictAreMapped(id, current_domain_iter)){
      return index;
    }
    index++;
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

      val2graph[output_value_1] = dependency_graph();
      val2graph[output_value_1].op = codegenType::Split;
      val2graph[output_value_1].llvm_val = builder.getInt64(std::stoi(out_domain_1->extent()->toString()));
    }
    else if(out_domain_2->extent()->isConstInt()){
      val2graph[output_value_1] = dependency_graph();
      val2graph[output_value_1].op = codegenType::Split;
      val2graph[output_value_1].input_vals.push_back(output_value_2);
      val2graph[output_value_1].input_vals.push_back(expr->inputs()[0]);

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

*/
void generate_shape_llvm_ir(Val * node, std::unordered_map<Val*, dependency_graph>& val2graph, llvm::IRBuilder<>& builder) {
  if(val2graph.find(node) != val2graph.end() && val2graph[node].llvm_val != nullptr){
    return;
  }
  if(val2graph.find(node) == val2graph.end()){
    val2graph[node] = dependency_graph();
    val2graph[node].op = codegenType::Merge;
    bool is_constant_val =  dynamic_cast<IterDomain*>(node)->extent()->isConstInt();
    if(is_constant_val){
      val2graph[node].llvm_val = builder.getInt64(std::stoi(dynamic_cast<IterDomain*>(node)->extent()->toString()));
    }
    else{
      val2graph[node].llvm_val = builder.getInt64(1);
      std::cout<< "Missing node: " << node->toString() << " llvm_val: 1" << val2graph[node].llvm_val << std::endl;
    }
    return;
  }
  llvm::Value* result = nullptr;
  if(val2graph[node].input_vals.size() == 2){
    auto* input_val_1 = val2graph[node].input_vals[0];
    auto* input_val_2 = val2graph[node].input_vals[1];
    if(val2graph.find(input_val_1) == val2graph.end() || val2graph[input_val_1].llvm_val == nullptr){
      generate_shape_llvm_ir(input_val_1, val2graph, builder);
    }
    if(val2graph.find(input_val_2) == val2graph.end() || val2graph[input_val_2].llvm_val == nullptr){ 
      generate_shape_llvm_ir(input_val_2, val2graph, builder);
    }
    if(val2graph[node].op == codegenType::Merge){
      result = builder.CreateMul(val2graph[input_val_1].llvm_val, val2graph[input_val_2].llvm_val, node->toString());
    }
    else if(val2graph[node].op == codegenType::Split){
    bool is_input_1_const =  dynamic_cast<IterDomain*>(input_val_1)->extent()->isConstInt();
      if(is_input_1_const){
        llvm::Value* minus_1 = builder.CreateSub(val2graph[input_val_1].llvm_val, builder.getInt64(1), "minus_1");
        llvm::Value* sum_ab = builder.CreateAdd(minus_1, val2graph[input_val_2].llvm_val, "sum_ab");
        result = builder.CreateUDiv(sum_ab, val2graph[input_val_1].llvm_val, node->toString());
      }
      else{
        llvm::Value* minus_1 = builder.CreateSub(val2graph[input_val_2].llvm_val, builder.getInt64(1), "minus_1");  
        llvm::Value* sum_ab = builder.CreateAdd(minus_1, val2graph[input_val_1].llvm_val, "sum_ab");
        result = builder.CreateUDiv(sum_ab, val2graph[input_val_2].llvm_val, node->toString());
      }
    }
  }
  else{
    if(val2graph[node].llvm_val == nullptr){
      std::cout << "Missing result node: " << node->toString() << " llvm_val: 1" << std::endl;
      result = builder.getInt64(1);
    }
    else{
      return;
    }
  }
  val2graph[node].llvm_val = result;
}

/*

Dumping all exprs between input and output domain, currently this is only used for shape inference

*/
std::vector<Expr*> traverse_expr_group(const ValGraph& graph, std::vector<IterDomain*>& input_domain, std::vector<IterDomain*>& output_domain){
  ValGroups tv0_loop_groups = graph.toGroups(input_domain);
  ValGroups tv1_loop_groups = graph.toGroups(output_domain);
  auto result = getAllExprGroupsBetween(graph, tv0_loop_groups, tv1_loop_groups).first;
  std::vector<Expr*> exprs;
  for(auto expr_group : result){
    for(auto expr : *expr_group.first){
      std::cout << expr->toString() << std::endl;
      exprs.push_back(expr);
    }
  }
  return exprs; 
}

/*

Verify if the merge is legit by traversing the exprs between output and input domain

TODO: Need to implement this function
*/
std::vector<Expr*> verify_merge_expr(std::vector<IterDomain*>& output_domain, std::vector<IterDomain*>& input_domain){
  std::unordered_set<Expr*> visited;
  std::queue<Expr*> q;
  for(auto* id : output_domain){
    if(id->definition() == nullptr){
      continue;
    }
    q.push(id->definition());
  }
  std::vector<Expr*> exprs;
  while(!q.empty()){
    auto* current = q.front();
    if(current == nullptr){
      continue;
    }
    q.pop();
    if(visited.find(current) != visited.end()){
      continue;
    }
    visited.insert(current);
    auto val_inputs = current->inputs();
    for(auto* input_value : val_inputs){
      auto* input_expr = input_value->definition();
      if(input_expr == nullptr){
        continue;
      }
      if(visited.find(input_expr) == visited.end()){
        q.push(input_expr);
        exprs.push_back(input_expr);
      }
    }
  }
  return exprs;
}

/*

Generate LLVM IR for stride inference

*/
void generate_stride_llvm_ir(
    Val* current_val_to_process,
    std::unordered_map<Val*, StrideInfo>& val2stride_map,
    llvm::IRBuilder<>& builder,
    std::unordered_set<Val*>& boundary_vals,
    llvm::Value*& running_stride_product
    ) {
    if (current_val_to_process == nullptr) {
        std::cerr << "Error: generate_stride_llvm_ir called with nullptr Val." << std::endl;
        return;
    } 
    if (val2stride_map.count(current_val_to_process) && val2stride_map[current_val_to_process].llvm_stride != nullptr) {
        return; // Memoization: Already processed
    }
    else if (val2stride_map.count(current_val_to_process) && val2stride_map[current_val_to_process].llvm_stride == nullptr){
      val2stride_map[current_val_to_process].llvm_stride = running_stride_product;
      running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[current_val_to_process].llvm_extent, "stride_root_val");
      return;
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

        if(boundary_vals.find(input_inner_val) != boundary_vals.end()&&val2stride_map[input_inner_val].llvm_stride == nullptr){
          val2stride_map[input_inner_val].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[input_inner_val].llvm_extent, "stride_merge_inner_val");
        }
        else if(boundary_vals.find(input_inner_val) != boundary_vals.end()&&val2stride_map[input_inner_val].llvm_stride != nullptr){
          // case where the inner val is already computed in previous dfs calls
          return;
        }
        else{
          generate_stride_llvm_ir(input_inner_val, val2stride_map, builder, boundary_vals, running_stride_product);
        }
        if(boundary_vals.find(input_outer_val) != boundary_vals.end() && val2stride_map[input_outer_val].llvm_stride == nullptr){
          val2stride_map[input_outer_val].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[input_outer_val].llvm_extent, "stride_merge_outer_val");
        }
        else if(boundary_vals.find(input_outer_val) != boundary_vals.end() && val2stride_map[input_outer_val].llvm_stride != nullptr){
          // case where the outer val is already computed in previous dfs calls
          return;
        }
        else{
          generate_stride_llvm_ir(input_outer_val, val2stride_map, builder, boundary_vals, running_stride_product);
        }
        
        // Extent of merged domain
        if(val2stride_map[input_outer_val].llvm_extent == nullptr || val2stride_map[input_inner_val].llvm_extent == nullptr){
          std::cout << "outer and inner val extent not computed" << std::endl;
          return;
        }
        else if(val2stride_map[current_val_to_process].llvm_extent != nullptr){
          std::cout << "current val extent already computed" << std::endl;
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
        if(boundary_vals.find(input_val) != boundary_vals.end() && val2stride_map[input_val].llvm_stride == nullptr){
          val2stride_map[input_val].llvm_stride = running_stride_product;
          running_stride_product = builder.CreateMul(running_stride_product, val2stride_map[input_val].llvm_extent, "stride_split_input_val");
        }
        else if(boundary_vals.find(input_val) != boundary_vals.end() && val2stride_map[input_val].llvm_stride != nullptr){
          return;
        }
        else{
          generate_stride_llvm_ir(input_val, val2stride_map, builder, boundary_vals, running_stride_product);
        }
        bool is_inner_val_const = output_inner_domain->extent()->isConstInt();
        bool is_outer_val_const = output_outer_domain->extent()->isConstInt();
        if(is_inner_val_const && val2stride_map[output_outer_val].llvm_extent == nullptr){
          val2stride_map[output_outer_val].llvm_extent = builder.CreateUDiv(
            val2stride_map[input_val].llvm_extent,
            builder.getInt64(stoi(output_inner_domain->extent()->toString())),
            output_outer_val->toString() + "_split_extent"
          );
        }
        else if(is_outer_val_const && val2stride_map[input_val].llvm_extent == nullptr){
          val2stride_map[output_inner_val].llvm_extent = builder.CreateUDiv(
            val2stride_map[input_val].llvm_extent,
            builder.getInt64(stoi(output_outer_domain->extent()->toString())),
            output_inner_val->toString() + "_split_extent"
          );
        }
        else{
          std::cerr << "outer and inner val constant problem" << std::endl;
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
  std::unordered_set<Val*> boundary_vals;
  for(auto* val : input_vals){
    boundary_vals.insert(val);
  }

  IdModel id_model(&fusion);
  const ValGraph& graph = id_model.buildExactGraph();
  
  for(long unsigned int i = 0; i < logical_domain.size(); i++){
    auto* zero = builder.getInt64(i);
    auto* input_i_ptr = builder.CreateGEP(int64Ty, input_ptr, zero, "ptr");
    auto* input_i_val = builder.CreateLoad(int64Ty, input_i_ptr, "val");
    val2stride[input_vals[i]] = StrideInfo();
    val2stride[input_vals[i]].llvm_extent = input_i_val;
  }

  for(auto* val : output_vals){
    auto index = mapToInputDomain(input_vals, val, graph);
    if(index != -1){
      val2stride[val] = val2stride[input_vals[index]];
    }
  }

  llvm::Value* running_stride_product = builder.getInt64(1);
  for(auto it = allocation_domain.rbegin(); it != allocation_domain.rend(); ++it){
    generate_stride_llvm_ir(*it, val2stride, builder, boundary_vals, running_stride_product);
  }

  for(long unsigned int i = 0; i < logical_domain.size(); i++){
    auto* output_i_ptr = builder.CreateGEP(int64Ty, output_ptr, builder.getInt64(i), "ptr");
    builder.CreateStore(val2stride[input_vals[i]].llvm_stride, output_i_ptr);
  }

  builder.CreateRetVoid();
  llvm::outs() << "=== LLVM IR ===\n";
  Module->print(llvm::outs(), nullptr);
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
  std::vector<Val*> input_vals = domain2vals(input_logical_domain);
  std::vector<Val*> output_vals = domain2vals(output_logical_domain);
  auto arg_it = func->arg_begin();
  llvm::Value* input_ptr = &*arg_it;
  llvm::Value* output_ptr = &*arg_it+2;
  IdModel id_model(&fusion);
  const ValGraph& graph = id_model.buildExactGraph();
  auto exprs = traverse_expr_group(graph, input_logical_domain, output_logical_domain);
  for(long unsigned int i = 0; i < input_logical_domain.size(); i++){
    auto* zero = builder.getInt64(i);
    auto* input_i_ptr = builder.CreateGEP(int64Ty, input_ptr, zero, "ptr");
    auto* input_i_val = builder.CreateLoad(int64Ty, input_i_ptr, "val");
    val2graph[input_vals[i]] = dependency_graph();
    val2graph[input_vals[i]].llvm_val = input_i_val;
  }

  // Map the output values to the input values if they are the same
  for(auto* val : output_vals){
    auto index = mapToInputDomain(input_vals, val, graph);
    if(index != -1){
      val2graph[val] = val2graph[input_vals[index]];
    }
  }

  for(auto* expr : exprs){
    auto expr_input_domain = vals2domain(expr->inputs());
    build_dep_graph(expr, val2graph, builder);
    int current_index = 0;
    for(auto* expr_input : expr->inputs()){
      auto index = mapToInputDomain(input_vals, expr_input, graph);
      if(index != -1){
        val2graph[expr_input].input_vals[current_index] = input_vals[index];
      }
      current_index++;
    }
  }

  for(auto* val : output_vals){
    generate_shape_llvm_ir(val, val2graph, builder);
  }

  for(long unsigned int i = 0; i < output_vals.size(); i++){
    auto* output_i_ptr = builder.CreateGEP(int64Ty, output_ptr, builder.getInt64(i), "ptr");
    builder.CreateStore(val2graph[output_vals[i]].llvm_val, output_i_ptr);
  }

  builder.CreateRetVoid();
  llvm::outs() << "=== LLVM IR ===\n";
  Module->print(llvm::outs(), nullptr);
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
  shape_infer_func(input.sizes().data(), input.sizes().size(), shape_result.data(), shape_result.size());
  std::vector<int64_t> stride_result(output_tensor_dim);
  stride_infer_func(shape_result.data(), shape_result.size(), stride_result.data(), stride_result.size());
  at::Tensor output_tensor = at::empty_strided(shape_result, stride_result, input.options());
  return output_tensor;
}

template llvm::orc::ExecutorAddr nvfuser::ExitOnErr<llvm::orc::ExecutorAddr>(llvm::Expected<llvm::orc::ExecutorAddr> &&E);

} // namespace nvfuser