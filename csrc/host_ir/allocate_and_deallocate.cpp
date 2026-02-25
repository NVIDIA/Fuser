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
#include <ranges>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fusion.h"
#include "host_ir/ir.h"
#include "ir/builder.h"
#include "ir/utils.h"

namespace nvfuser::hir {

namespace {

class Node {
 public:
  Node(Scope* scope, Expr* expr, const Node* parent)
      : scope_(scope),
        expr_(expr),
        parent_(parent),
        depth_(parent ? parent->depth() + 1 : 0) {}

  Scope* scope() const {
    return scope_;
  }

  Expr* getExpr() const {
    return expr_;
  }

  const Node* parent() const {
    return parent_;
  }

  int depth() const {
    return depth_;
  }

  const std::vector<Node*>& children() const {
    return children_;
  }

  void addChild(Node* child) {
    children_.push_back(child);
  }

 private:
  Scope* scope_;
  Expr* expr_;
  const Node* parent_;
  int depth_;
  std::vector<Node*> children_;
};

class DominatorTree {
 public:
  explicit DominatorTree(hir::HostIrContainer& hic) : hic_(&hic) {
    build(hic_->topLevel(), /*parent=*/nullptr);
  }

  const Node* getRoot() const {
    const auto& top_level_exprs = hic_->topLevelExprs();
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
    for (Expr* e : scope.exprs()) {
      auto [node_it, inserted] = nodes_.try_emplace(e, &scope, e, parent);
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

  hir::HostIrContainer* hic_;
  std::unordered_map<const Expr*, Node> nodes_;
};

// Post-dominator tree: node A post-dominates B if every path from B to exit
// goes through A. Built by traversing from exit toward entry.
class PostDominatorTree {
 public:
  explicit PostDominatorTree(
      hir::HostIrContainer& hic,
      std::unordered_map<TensorView*, const Node*>& lca)
      : hic_(&hic) {
    build(hic_->topLevel(), /*scope_exit_successor=*/nullptr, lca);
  }

  const Node* getNode(Expr* expr) const {
    auto it = nodes_.find(expr);
    return it != nodes_.end() ? &it->second : nullptr;
  }

 private:
  void build(
      Scope& scope,
      Node* parent,
      std::unordered_map<TensorView*, const Node*>& lca) {
    for (Expr* e : scope.exprs() | std::views::reverse) {
      auto [node_it, inserted] = nodes_.try_emplace(e, &scope, e, parent);
      NVF_ERROR(inserted);
      Node& node = node_it->second;

      if (auto* alloc = dynamic_cast<kir::Allocate*>(e)) {
        TensorView* tv = alloc->buffer()->as<TensorView>();
        lca[tv] = findLCA(lca[tv], &node);
      }
      for (auto* in : ir_utils::filterByType<TensorView>(e->inputs())) {
        lca[in] = findLCA(lca[in], &node);
      }

      if (auto* loop = dynamic_cast<hir::ForLoop*>(e)) {
        build(loop->body(), &node, lca);
      }
      if (auto* ite = dynamic_cast<kir::IfThenElse*>(e)) {
        build(ite->thenBody(), &node, lca);
        build(ite->elseBody(), &node, lca);
      }

      parent = &node;
    }
  }

  const Node* findLCA(const Node* a, const Node* b) const {
    if (a == nullptr) {
      return b;
    }
    if (b == nullptr) {
      return a;
    }
    while (a->depth() > b->depth()) {
      a = a->parent();
    }
    while (b->depth() > a->depth()) {
      b = b->parent();
    }
    while (a != b) {
      a = a->parent();
      b = b->parent();
    }
    return a;
  }

  hir::HostIrContainer* hic_;
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
      [&](const Node* node) {
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
            node->scope()->insert_before(node->getExpr(), allocate);
          }

          defined.insert(out);
        }
      },
      /*post_fn=*/
      [&](const Node* node) {
        Expr* e = node->getExpr();
        for (auto* out : ir_utils::filterByType<TensorView>(e->outputs())) {
          defined.erase(out);
        }
      });
}

bool needsDeallocation(TensorView* tv) {
  if (tv->isFusionInput()) {
    return false;
  }
  if (tv->isFusionOutput()) {
    return false;
  }
  if (tv->definition()->isA<ShardByStream>()) {
    return false;
  }
  const AliasInfo& alias_info = tv->container()->getOutputAlias(tv);
  if (alias_info.type == AllocationType::ReuseBuffer) {
    return false;
  }
  return true;
}

void insertDeallocations(hir::HostIrContainer& hic) {
  const std::list<Expr*>& top_level_exprs = hic.topLevelExprs();
  std::ranges::for_each(top_level_exprs, [](Expr* expr) {
    NVF_ERROR(
        !expr->isA<hir::Deallocate>(),
        "Expected hostir container to not have deallocate, but found one "
        "anyways: ",
        expr);
  });

  std::unordered_map<TensorView*, const Node*> lca;
  PostDominatorTree post_dom_tree(hic, lca);

  // Insert deallocate at LCA for each TV that needs deallocation.
  for (const auto& [tv, lca_node] : lca) {
    if (!needsDeallocation(tv)) {
      continue;
    }
    NVF_ERROR(
        lca_node != nullptr, "Could not find post-dominator for tensor ", tv);
    auto* deallocate = IrBuilder::create<hir::Deallocate>(tv);
    lca_node->scope()->insert_after(lca_node->getExpr(), deallocate);
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
