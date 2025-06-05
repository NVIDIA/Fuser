

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


namespace nvfuser {

/*

Helper Data Structures & Functions

*/

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
  dependency_graph(){
    op = codegenType::Merge;
    llvm_val = nullptr;
  }
};

// Dependency graph entry for the stride inference
struct StrideInfo {
public:
    llvm::Value* llvm_extent = nullptr;   // LLVM Value for the extent of this IterDomain
    llvm::Value* llvm_stride = nullptr;   // LLVM Value for the calculated stride of this IterDomain
};

/*

Helper functions

*/

// Helper functions to exit on error on LLVM JIT initialization
template <typename T>
T ExitOnErr(llvm::Expected<T> &&E);

inline void ExitOnErr(llvm::Error &&Err);

// Print compare tensor
void print_compare_tensor(const at::Tensor& t0, const at::Tensor& t1);

// Print all exprs between input and output domain
void print_getExprsBetween(const std::vector<IterDomain*>& input_domain, const std::vector<IterDomain*>& output_domain);

// Print all expr groups between input and output domain
void print_getAllExprGroupsBetween(Fusion& fusion, const std::vector<IterDomain*>& in_loop_domain, const std::vector<IterDomain*>& out_loop_domain);

// Helper function to map to input domain if current domain is in input domain
int mapToInputDomain(std::vector<Val*> input_domain, Val* current_domain, const ValGraph& exact_graph);

// Helper function to cast iter domains to vals
std::vector<Val*> domain2vals(const std::vector<IterDomain*>& domain);

// Helper function to cast vals to iter domains
std::vector<IterDomain*> vals2domain(const std::vector<Val*>& domain);


/*

LLVM IR Generation Functions

*/

// Build dependency graph for shape inference
void build_dep_graph(Expr* expr, std::unordered_map<Val*, dependency_graph>& val2graph, llvm::IRBuilder<>& builder);

// Generate shape infer llvm ir
void generate_shape_llvm_ir(Val* node, std::unordered_map<Val*, dependency_graph>& val2graph, llvm::IRBuilder<>& builder);

// Generate stride infer llvm ir
void generate_stride_llvm_ir(Val* current_val_to_process, std::unordered_map<Val*, StrideInfo>& val2stride_map, llvm::IRBuilder<>& builder, std::unordered_set<Val*>& boundary_vals, llvm::Value*& running_stride_product);

// Generate stride infer llvm module
llvm::orc::ThreadSafeModule generate_infer_stride_module(std::vector<IterDomain*>& allocation_domain, std::vector<IterDomain*>& logical_domain, Fusion& fusion);

// Generate shape infer llvm module
llvm::orc::ThreadSafeModule generate_infer_shape_module(std::vector<IterDomain*>& input_logical_domain, std::vector<IterDomain*>& output_logical_domain, Fusion& fusion);

/*

LLVM JIT Compile Functions

*/

// JIT compile shape infer
void llvm_jit_compile_shape_infer(std::unique_ptr<llvm::orc::LLJIT>& JIT, Fusion& fusion, std::vector<IterDomain*>& input_domain, std::vector<IterDomain*>& output_domain);

// JIT compile stride infer
void llvm_jit_compile_stride_infer(std::unique_ptr<llvm::orc::LLJIT>& JIT,Fusion& fusion, std::vector<IterDomain*>& allocation_domain, std::vector<IterDomain*>& logical_domain);

// JIT initialization
std::unique_ptr<llvm::orc::LLJIT> llvm_jit_init(int num_threads);

/*

LLVM JIT Runtime Interface Functions

*/

using FuncType = void (*)(const int64_t* input, int64_t input_len, int64_t* output, int64_t output_len);

at::Tensor aten_output_allocation(FuncType shape_infer_func, FuncType stride_infer_func, const at::Tensor& input, int64_t output_tensor_dim);

} // namespace nvfuser