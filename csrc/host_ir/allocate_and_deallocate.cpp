// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "host_ir/allocate_and_deallocate.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <list>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ir/builder.h"
#include "ir/iostream.h"
#include "ir/utils.h"

namespace nvfuser::hir {

namespace {

class DominatorTree {
 public:
  class Node {
   public:
    Node(Scope* scope, Scope::Iterator iterator)
        : scope_(scope), iterator_(iterator) {}
    Node(const Node& other) = delete;
    Node(Node&& other) = delete;
    Node& operator=(const Node& other) = delete;
    Node& operator=(Node&& other) = delete;

    const std::vector<Node*>& children() const {
      return children_;
    }

    void addChild(Node* child) {
      children_.push_back(child);
    }

    Scope* scope() const {
      return scope_;
    }

    Scope::Iterator iterator() const {
      return iterator_;
    }

    Expr* getExpr() const {
      return *iterator_;
    }

   private:
    // Consider putting `scope` and `iterator` into a separate Mutator class.
    // They are only needed when the user wants to modify the host IR.
    Scope* scope_;
    Scope::Iterator iterator_;

    std::vector<Node*> children_;
  };

  explicit DominatorTree(hir::HostIrContainer& hic) : hic_(hic) {
    build(hic_.topLevel(), /*parent=*/nullptr);
  }

  const Node* getRoot() const {
    const auto& top_level_exprs = hic_.topLevelExprs();
    NVF_ERROR(!top_level_exprs.empty());
    Expr* root = top_level_exprs.front();
    return &nodes_.at(root);
  }

  // `pre_fn` is called before traversing any child of a node.  `post_fn` is
  // called after traversing all children of a node.
  void depthFirstTraverse(
      const std::function<void(const Node*)>& pre_fn,
      const std::function<void(const Node*)>& post_fn) const {
    struct Frame {
      const Node* node;
      bool processed;
    };

    std::stack<Frame> stack;
    stack.emplace(getRoot(), /*processed=*/false);
    while (!stack.empty()) {
      Frame& top = stack.top();
      if (top.processed) {
        post_fn(top.node);
        stack.pop();
        continue;
      }

      pre_fn(top.node);
      top.processed = true;
      for (const Node* child : top.node->children()) {
        stack.emplace(child, /*processed=*/false);
      }
    }
  }

 private:
  void build(Scope& scope, Node* parent) {
    for (auto scope_it = scope.exprs().begin(); scope_it != scope.exprs().end();
         ++scope_it) {
      Expr* e = *scope_it;
      auto [node_it, inserted] = nodes_.try_emplace(e, &scope, scope_it);
      NVF_ERROR(inserted);
      Node& node = node_it->second;
      if (parent != nullptr) {
        parent->addChild(&node);
      }

      if (auto* loop = dynamic_cast<hir::ForLoop*>(e)) {
        // `e`, the ForLoop, dominates its body. However, the body doesn't
        // dominate the instruction after the loop, because the loop could be
        // executed zero times.
        build(loop->body(), &node);
      }

      if (auto* ite = dynamic_cast<kir::IfThenElse*>(e)) {
        build(ite->thenBody(), &node);
        build(ite->elseBody(), &node);
      }

      parent = &node;
    }
  }

  hir::HostIrContainer& hic_;
  std::unordered_map<const Expr*, Node> nodes_;
};

bool needsOutputPreallocation(Expr* e) {
  return e->isOneOf<MatmulOp, LinearOp>();
}

void insertAllocations(hir::HostIrContainer& hic) {
  // lowerSegmentedFusionToHostIr inserts **some** allocations. For example, it
  // inserts `Allocate` and `ShardByStream` for TVs whose loop is stream
  // parallelized but allocation is not.
  //
  // This function inserts more allocations for convenience. For example,
  // it ensures that after this pass, outputs of LinearOp and
  // MatmulOp are always preallocated. This allows HostIrEvaluator and
  // HostIrJit to uniformly handle outputs of LinearOp and MatmulOp, knowing
  // they will always be preallocated without needing to check for special
  // cases.
  //
  // This is done by traversing the dominator tree in depth-first order. If an
  // output TV of an Expr needs preallocation but doesn't have a **dominating**
  // definition, DFS will insert an allocation right before that Expr.
  DominatorTree dom_tree(hic);
  std::unordered_set<TensorView*> defined;

  dom_tree.depthFirstTraverse(
      /*pre_fn=*/
      [&](const DominatorTree::Node* node) {
        Expr* e = node->getExpr();
        // If `e`'s output needs preallocation but isn't defined, insert an
        // allocation right before `e`.
        for (auto* out : ir_utils::filterByType<TensorView>(e->outputs())) {
          if (defined.count(out) > 0) {
            continue;
          }

          if (needsOutputPreallocation(e)) {
            auto* allocate =
                IrBuilder::create<kir::Allocate>(out, out->getMemoryType());
            node->scope()->insert(node->iterator(), allocate);
          }

          defined.insert(out);
        }
      },
      /*post_fn=*/
      [&](const DominatorTree::Node* node) {
        Expr* e = node->getExpr();
        for (auto* out : ir_utils::filterByType<TensorView>(e->outputs())) {
          defined.erase(out);
        }
      });
}

void insertDeallocations(hir::HostIrContainer& hic) {
  const std::list<Expr*>& top_level_exprs = hic.topLevelExprs();
  std::for_each(top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
    NVF_ERROR(
        !expr->isA<hir::Deallocate>(),
        "Expected hostir container to not have deallocate, but found one "
        "anyways: ",
        expr);
  });

  // For each input in every expression in the container, find the position of
  // its last use and insert a deallocate directly after, except for fusion
  // inputs and outputs.
  std::unordered_set<TensorView*> last_use_found;
  for (auto insertion_point = top_level_exprs.end();
       insertion_point != top_level_exprs.begin();) {
    auto prev = std::prev(insertion_point);
    Expr* e = *prev;

    // Only tensors need to be allocated.
    for (auto* in : ir_utils::filterByType<TensorView>(e->inputs())) {
      // Fusion inputs are managed by the caller.
      if (in->isFusionInput()) {
        continue;
      }

      // Fusion outputs need to be kept alive for the caller.
      if (in->isFusionOutput()) {
        continue;
      }

      // Skip if `e` is not the last use.
      if (!last_use_found.insert(in).second) {
        continue;
      }

      auto* deallocate = IrBuilder::create<hir::Deallocate>(in);
      hic.insertExprBefore(insertion_point, deallocate);
    }

    // Don't `--insertion_point;` because we'd like to skip newly inserted
    // deallocations.
    insertion_point = prev;
  }
}

} // namespace

void AllocateAndDeallocate::runPass(Fusion* fusion) {
  auto* hic = fusion->as<HostIrContainer>();

  FusionGuard fg(hic);
  insertAllocations(*hic);
  insertDeallocations(*hic);
}

} // namespace nvfuser::hir
