// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <executor.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <kernel_cache.h>
#include <manager.h>
#include <parser.h>
#include <scheduler/all_schedulers.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/cuda_graph_fuser.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <type_inference.h>
#include <utils.h>

#include <ATen/DimVector.h>
#include <c10/core/DeviceType.h>
#include <c10/util/irange.h>

#include <unordered_map>

namespace nvfuser {

//! [ Note -- cache entry indexing ]
//!
//! CudaFusionManager holds the cache and handles interfacing to CudaFusionGroup
//! node, including selection, construction and execution of FusionExecutors.
//!
//! CudaFusionManager bridges PyTorch IR node CudaFusionGroup to GraphCache.
//! Therefore, we want to cache on stringified graph. But it is expensive to
//! stringify and hash on a computational graph, we cache the hash of a
//! stringified graph on node via cache_id.
//!
//! CudaFusionGroup node stores:
//!     i.  a PyTorch IR in `attr::Subgraph`
//!     ii. an int in `attr::cache_id`, (a cached hash value of
//!     `attr::Subgraph`)
//!
//! We have 2 unordered_map at CudaFusionGroup:
//!   std::unordered_map<std::string, int32_t> graph_cache_ids_;
//!   std::unordered_map<int64_t, std::unique_ptr<GraphCache>> graph_cache_;
//!
//! Mapping from std::string to graph_cache_id ensures that we assign the same
//! cache_id to CudaFusionGroup with identical computational grah, allowing
//! kernel reuse; Direct mapping from cache_id to GraphCache allows efficient
//! graph_cache indexing;

namespace {

// TODO remove this (75983):
//   we don't need this any more. I think we can use revertAliasCopyOps.
//   Similar refactor should be done infallback graph used by fusion guard.
//   implementation of xxxx_copy ops should be removed.
//
// Mark string attribute in alias-copy nodes to enable its implementation
// in the fallback path.
void enableAliasCopyNodes(
    const std::shared_ptr<torch::jit::Graph>& graph,
    torch::jit::Block* block) {
  static std::unordered_set<c10::Symbol> alias_copy_op(
      {at::aten::expand_copy,
       at::prim::expand_as_copy,
       at::prim::flatten_copy,
       at::aten::permute_copy,
       at::aten::_reshape_copy,
       at::aten::squeeze_copy,
       at::aten::t_copy,
       at::aten::transpose_copy,
       at::aten::unsqueeze_copy,
       at::aten::view_copy});

  for (torch::jit::Node* n : block->nodes()) {
    for (torch::jit::Block* b : n->blocks()) {
      enableAliasCopyNodes(graph, b);
    }
    if (alias_copy_op.find(n->kind()) != alias_copy_op.end()) {
      n->s_(at::attr::name, "CudaFusionGroup");
    }
  }
}

static std::unique_ptr<torch::jit::Code> createFallbackCode(
    const torch::jit::Node* fusion_node) {
  auto copied_graph = fusion_node->g(at::attr::Subgraph)->copy();
  EraseShapeInformation(copied_graph);
  enableAliasCopyNodes(copied_graph, copied_graph->block());
  auto code =
      std::make_unique<torch::jit::Code>(copied_graph, "fallback_cuda_fuser");
  return code;
}

// CudaFusionManager is not thread safe!
// TODO: we should make the tradeoff here to use thread_local instead of global
// singleton;
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class CudaFusionManager {
 public:
  static CudaFusionManager& getManager() {
    static CudaFusionManager cuda_fusion_manager_;
    return cuda_fusion_manager_;
  };

  // TODO: I'm assuming we have stride information in `graph->toString`
  //       We need to make sure stride information is in the final string, as we
  //       want to AVOID kernel reuse between different fusion_node, unless they
  //       have identical contiguity information! (So identical stride + shape
  //       is even more restricting in a good way)
  int32_t registerOrGetCacheId(std::shared_ptr<torch::jit::Graph>& graph) {
    // prepare graph for lowering;
    // We should not call `EraseShapeInformation(graph);`, graph representation
    // does not incorporate static sizes, but just rank of input tensors, which
    // is exactly what we wanted.
    auto canonical_graph = Canonicalize(graph, false);
    auto repr = canonical_graph->toString(false);

    std::lock_guard<std::mutex> guard(mutex_);
    // create new graph_cache_ids_ entry if none existed yet;
    if (graph_cache_ids_.count(repr) == 0) {
      int32_t kernel_id = getNextUniqueID();
      graph_cache_ids_[repr] = kernel_id;
      NVF_CHECK(
          graph_cache_.emplace(kernel_id, std::make_unique<GraphCache>(graph))
              .second);
    }
    return graph_cache_ids_[repr];
  };

  // get fallback kernel id
  int32_t getFallbackKernelId() {
    std::lock_guard<std::mutex> guard(mutex_);
    return getNextUniqueID();
  }

  void unregisterCacheId(std::shared_ptr<torch::jit::Graph>& graph) {
    auto canonical_graph = Canonicalize(graph, false);
    auto repr = canonical_graph->toString(false);

    // create new graph_cache_ids_ entry if none existed yet;
    if (graph_cache_ids_.count(repr) > 0) {
      int32_t kernel_id = graph_cache_ids_[repr];
      graph_cache_.erase(kernel_id);
      graph_cache_ids_.erase(repr);
    }
  }

  std::vector<at::Tensor> runFusionNode(
      int32_t kernel_id,
      const at::ArrayRef<c10::IValue> inputs) {
    std::lock_guard<std::mutex> guard(mutex_);
    NVF_ERROR(
        graph_cache_.count(kernel_id) > 0, "graph cache miss at run time");
    return graph_cache_[kernel_id]->runGraphWithInputs(inputs);
  }

  bool hasFallbackCode(int32_t kernel_id) {
    std::lock_guard<std::mutex> guard(mutex_);
    return fallback_cache_.count(kernel_id);
  }

  torch::jit::Code* getFallbackCode(
      int32_t kernel_id,
      const torch::jit::Node* fusion_node) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto it = fallback_cache_.find(kernel_id);
      if (it != fallback_cache_.end()) {
        return it->second.get();
      }
    }

    std::unique_ptr<torch::jit::Code> code = createFallbackCode(fusion_node);

    std::lock_guard<std::mutex> guard(mutex_);
    auto it = fallback_cache_.insert({kernel_id, std::move(code)}).first;
    return it->second.get();
  }

 private:
  // TODO: Dimension collapsing should be abstracted out and integrated into
  // graph caching.

  // Dimension collapsing only applicable to profiling executor at this moment
  bool graphHasReduction(const std::shared_ptr<torch::jit::Graph>& graph) {
    for (const auto& n : graph->nodes()) {
      if (isReductionNode(n)) {
        return true;
      }
    }
    return false;
  }

 private:
  std::mutex mutex_;

  void runCudaKernel(
      int32_t key,
      const std::vector<int>& contiguity_tag,
      const c10::Device){};

  int32_t getNextUniqueID() {
    return next_unique_id_++;
  };

  std::unordered_map<std::string, int32_t> graph_cache_ids_;
  std::unordered_map<int64_t, std::unique_ptr<GraphCache>> graph_cache_;
  std::unordered_map<int64_t, std::unique_ptr<torch::jit::Code>>
      fallback_cache_;

  int32_t next_unique_id_ = 0;
};

} // namespace

void compileCudaFusionGroup(torch::jit::Node* fusion_node) {
  FUSER_PERF_SCOPE("nvFuser::Manager::compileCudaFusionGroup");

  NVF_CHECK(
      fusion_node->kind() == at::prim::CudaFusionGroup,
      "Only prim::CudaFusionGroup can be compiled");
  if (fusion_node->hasAttribute(at::attr::cache_id)) {
    TORCH_WARN("Double registration of CudaFusionGroup on CudaFusionManager");
  }
  // This is not a critical code path, it's OK to do graph copy here;
  auto graph = fusion_node->g(at::attr::Subgraph)->copy();

  auto compile_fusion = [&]() {
    // type propagation is needed, as the protocol only requires scalar type on
    // input tensors.
    // Note that even for Profiling Executor, scalar type could still be
    // missing, especially for output tensor from a given node (as profiling
    // node only insert meta information after itself).
    PropagateShapesOnGraph(graph);
    TypePropagate(graph);

    int32_t fusion_cache_id =
        CudaFusionManager::getManager().registerOrGetCacheId(graph);
    fusion_node->i_(at::attr::cache_id, fusion_cache_id);
  };

  if (useFallback()) {
    try {
      compile_fusion();
    } catch (...) {
      TORCH_WARN(
          "FALLBACK path has been taken inside: ",
          __FUNCTION__,
          ". This is an indication that codegen Failed for some reason.\n"
          "To debug try disable codegen fallback path via setting the env"
          " variable `export NVFUSER_DISABLE=fallback`\n"
          "To report the issue, try enable logging via setting the env"
          "variable ` export PYTORCH_JIT_LOG_LEVEL=manager.cpp`\n");
      GRAPH_DUMP("`compile_fusion` hits fallback on graph\n", graph);
      CudaFusionManager::getManager().unregisterCacheId(graph);
    }
  } else {
    compile_fusion();
  }

  // Assigning a cache_id to facilitate graph execution and fallback
  if (!fusion_node->hasAttribute(at::attr::cache_id)) {
    int32_t fusion_cache_id =
        CudaFusionManager::getManager().getFallbackKernelId();
    fusion_node->i_(at::attr::cache_id, fusion_cache_id);
  }
}

void runCudaFusionGroup(
    const torch::jit::Node* fusion_node,
    torch::jit::Stack& stack) {
  FUSER_PERF_SCOPE("nvFuser::Manager::runCudaFusionGroup");
  NVF_CHECK(
      fusion_node->hasAttribute(at::attr::cache_id),
      "node prim::CudaFusionGroup has not been compiled yet");

  // Fallback to use if anything goes wrong
  auto take_fallback = [&](torch::jit::Stack& stack) {
    std::unique_ptr<torch::jit::Code> fallback_code_unique;
    torch::jit::Code* fallback_code = nullptr;
    int32_t kernel_id = (int32_t)fusion_node->i(at::attr::cache_id);
    fallback_code =
        CudaFusionManager::getManager().getFallbackCode(kernel_id, fusion_node);
    torch::jit::InterpreterState{*fallback_code}.run(stack);
  };

  std::optional<torch::jit::Stack> stack_copy;
  auto compare_callback = torch::jit::getCudaFuserComparisonCallback();
  if (compare_callback.run_fallback) {
    // make a copy of the stack
    int64_t inputs_size = static_cast<int64_t>(
        fusion_node->g(at::attr::Subgraph)->inputs().size());
    NVF_ERROR(int64_t(stack.size()) >= inputs_size);
    stack_copy = torch::jit::Stack();
    stack_copy->insert(
        stack_copy->end(), stack.begin(), stack.end() - inputs_size);
    // deepcopy the last (inputs_size) stack items
    std::transform(
        stack.end() - inputs_size,
        stack.end(),
        std::back_inserter(*stack_copy),
        [](const c10::IValue& ivalue) { return ivalue.deepcopy(); });
  }

  auto run_fusion = [&]() {
    NVF_CHECK(
        fusion_node->kind() == at::prim::CudaFusionGroup,
        "prim::CudaFusionGroup expected");
    int32_t kernel_id = (int32_t)fusion_node->i(at::attr::cache_id);
    // Currently we just construct I/O tensors for static graph;

    const auto nInputs = fusion_node->g(at::attr::Subgraph)->inputs().size();

    at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, nInputs);

    auto outputs =
        CudaFusionManager::getManager().runFusionNode(kernel_id, inputs);

    torch::jit::drop(stack, inputs.size());
    stack.insert(
        stack.end(),
        std::make_move_iterator(outputs.begin()),
        std::make_move_iterator(outputs.end()));
  };

  if (useFallback()) {
    try {
      // if fusion failed once, it's likely to fail again; and failures are
      // slow. So if the fusion fails, then record the failure and always use
      // the fallback instead
      int32_t kernel_id = (int32_t)fusion_node->i(at::attr::cache_id);
      bool force_fallback =
          CudaFusionManager::getManager().hasFallbackCode(kernel_id);
      if (force_fallback) {
        take_fallback(stack);
      } else {
        run_fusion();
      }
    } catch (...) {
      TORCH_WARN(
          "FALLBACK path has been taken inside: ",
          __FUNCTION__,
          ". This is an indication that codegen Failed for some reason.\n"
          "To debug try disable codegen fallback path via setting the env"
          " variable `export PYTORCH_NVFUSER_DISABLE=fallback`\n");
      take_fallback(stack);
    }
  } else {
    run_fusion();
  }

  if (compare_callback.callback != nullptr) {
    torch::jit::Stack fused_outputs;
    torch::jit::Stack fallback_outputs;
    int64_t output_count = static_cast<int64_t>(
        fusion_node->g(at::attr::Subgraph)->outputs().size());
    NVF_CHECK(
        output_count <= int64_t(stack.size()),
        "Expected ",
        output_count,
        " outputs but found only ",
        stack.size(),
        " items on the stack");

    fused_outputs.insert(
        fused_outputs.begin(), stack.end() - output_count, stack.end());

    if (stack_copy) {
      take_fallback(*stack_copy);
      NVF_CHECK(
          stack_copy->size() == stack.size(),
          "Fused graph returns stack with ",
          stack.size(),
          " items, compared to ",
          stack_copy->size(),
          " from unfused graph");
      fallback_outputs.insert(
          fallback_outputs.begin(),
          stack_copy->end() - output_count,
          stack_copy->end());
    }
    auto graph_str = fusion_node->g(at::attr::Subgraph)->toString();
    compare_callback.callback(fused_outputs, fallback_outputs, graph_str);
  }
}

} // namespace nvfuser
