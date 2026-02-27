// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "host_ir/allocate_and_deallocate.h"

#include <functional>
#include <iterator>
#include <list>
#include <ranges>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ir/builder.h"
#include "ir/utils.h"

namespace nvfuser::hir {

namespace {

class Node {
 public:
  Node(Scope* scope, Scope::Iterator iterator, const Node* parent)
      : scope_(scope), iterator_(iterator), parent_(parent) {}
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

  const Node* parent() const {
    return parent_;
  }

 private:
  Scope* scope_;
  Scope::Iterator iterator_;
  const Node* parent_;
  std::vector<Node*> children_;
};

// `pre_fn` is called before traversing any child of a node.  `post_fn` is
// called after traversing all children of a node.
void depthFirstTraverseFromRoot(
    const Node* root,
    const std::function<void(const Node*)>& pre_fn,
    const std::function<void(const Node*)>& post_fn) {
  struct Frame {
    const Node* node;
    bool processed;
  };

  std::stack<Frame> stack;
  stack.push({root, /*processed=*/false});
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
      stack.push({child, /*processed=*/false});
    }
  }
}

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

  void depthFirstTraverse(
      const std::function<void(const Node*)>& pre_fn,
      const std::function<void(const Node*)>& post_fn) const {
    depthFirstTraverseFromRoot(getRoot(), pre_fn, post_fn);
  }

 private:
  void build(Scope& scope, Node* parent) {
    for (auto scope_it = scope.exprs().begin(); scope_it != scope.exprs().end();
         ++scope_it) {
      Expr* e = *scope_it;
      auto [node_it, inserted] =
          nodes_.try_emplace(e, &scope, scope_it, parent);
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

class PostDominatorTree {
 public:
  explicit PostDominatorTree(hir::HostIrContainer& hic) : hic_(&hic) {
    build(hic_->topLevel(), /*parent=*/nullptr);
  }

  const Node* getRoot() const {
    const auto& top_level_exprs = hic_->topLevelExprs();
    NVF_ERROR(!top_level_exprs.empty());
    Expr* root = top_level_exprs.back();
    return &nodes_.at(root);
  }

  const Node* getNode(Expr* expr) const {
    auto it = nodes_.find(expr);
    return it != nodes_.end() ? &it->second : nullptr;
  }

  void depthFirstTraverse(
      const std::function<void(const Node*)>& pre_fn,
      const std::function<void(const Node*)>& post_fn) const {
    depthFirstTraverseFromRoot(getRoot(), pre_fn, post_fn);
  }

 private:
  void build(Scope& scope, Node* parent) {
    auto& exprs = scope.exprs();
    for (auto it = exprs.end(); it != exprs.begin();) {
      --it;
      Expr* e = *it;
      auto [node_it, inserted] = nodes_.try_emplace(e, &scope, it, parent);
      NVF_ERROR(inserted);
      Node& node = node_it->second;
      if (parent != nullptr) {
        parent->addChild(&node);
      }

      if (auto* loop = dynamic_cast<hir::ForLoop*>(e)) {
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
            node->scope()->insert(node->iterator(), allocate);
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

// For each TensorView that is allocated or used as an input, find its
// least common ancestor in the Post-dominator Tree — the latest point at which
// it can be deallocated.
std::unordered_map<TensorView*, const Node*> computeLeastCommonAncestor(
    const PostDominatorTree& post_dom_tree) {
  std::unordered_map<const Node*, int64_t> depth;

  auto findLCA = [&](const Node* a, const Node* b) -> const Node* {
    if (a == nullptr) {
      return b;
    }
    if (b == nullptr) {
      return a;
    }
    int64_t depth_a = depth.at(a);
    int64_t depth_b = depth.at(b);
    while (depth_a > depth_b) {
      a = a->parent();
      depth_a--;
    }
    while (depth_b > depth_a) {
      b = b->parent();
      depth_b--;
    }
    while (a != b) {
      a = a->parent();
      b = b->parent();
    }
    return a;
  };

  std::unordered_map<TensorView*, const Node*> lca;
  int64_t current_depth = -1;

  post_dom_tree.depthFirstTraverse(
      /*pre_fn=*/
      [&](const Node* node) {
        current_depth++;
        depth[node] = current_depth;
        Expr* e = node->getExpr();

        if (auto* alloc = dynamic_cast<kir::Allocate*>(e)) {
          TensorView* tv = alloc->buffer()->as<TensorView>();
          lca[tv] = findLCA(lca[tv], node);
        }
        for (auto* in : ir_utils::filterByType<TensorView>(e->inputs())) {
          lca[in] = findLCA(lca[in], node);
        }
      },
      /*post_fn=*/
      [&](const Node*) { --current_depth; });

  return lca;
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

  PostDominatorTree post_dominator_tree(hic);
  const std::unordered_map<TensorView*, const Node*>& lca_map =
      computeLeastCommonAncestor(post_dominator_tree);

  // Insert deallocate at LCA for each tensorview that is not a fusion input or
  // output.
  for (const auto& [tv, lca_node] : lca_map) {
    if (tv->isFusionInput() || tv->isFusionOutput()) {
      continue;
    }
    NVF_ERROR(
        lca_node != nullptr,
        "Could not find least common ancestor for all uses of ",
        tv);
    auto* deallocate = IrBuilder::create<hir::Deallocate>(tv);
    lca_node->scope()->insert(std::next(lca_node->iterator()), deallocate);
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
