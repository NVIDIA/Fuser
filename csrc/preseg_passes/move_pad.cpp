// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_pad.h>

#include <fusion.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>

namespace nvfuser::preseg_passes {

namespace {

struct Edge {
  Expr* expr_;
  size_t index_;

  Val* val() const {
    return expr_->input(index_);
  }
}

Val* propagatePadToProducer(PadOp* pad) {
  std::vector<Val*> pad_dependencies;

  auto candidate_check = [&pad_dependencies](Val* val) {
    if (!val->isA<TensorView>) {
      return false;
    }
    if (val->uses().size() > 1) {
      return false;
    }
    if (val->isFusionOutput()) {
      return false;
    }
    if (std::any_of(pad_dependencies.begin(), pad_dependencies.end(), [val](Val* pad_dependency) {
      return DependencyCheck::isDependencyOf(pad_dependency, val);
    })) {
      return false;
    }
    return true;
  }

  // NOTE: the optimization logic assumes a zero pad.
  // This is used for logic in handling binary operations, we should extend this later.
  if (!pad->value()->isZero()) {
    return false;
  }

  if (!candidate_check(pad->in())) {
    return nullptr;
  }

  for (Val* val : pad->inputs()) {
    if (val == pad->in() || val->isConst()) {
      continue;
    }
    pad_dependencies.push_back(val);
  }

  std::vector<Edge> frontier;
  // TODO: not sure if I need a stack if I need to keep a replay_sequence.
  std::stack<Edge> stack;
  std::vector<Expr*> replay_sequence;
  stack.emplace(pad->in()->as<TensorView>(), 0);


  // tvs in stack are:
  //   1. single use;
  //   2. not an output;
  //   3. cleared dependency of `pad_depdencies;
  //   4. maybe also check aliases?!
  while(!stack.empty()) {
    Edge edge = stack.top();
    Expr* def = edge->val()->definition();
    stack.pop();


    if (def->isA<UnaryOp>()) {
      auto* uop = def->as<UnaryOp>();
      // TODO: exception to break propagation.
      if (candidate_check(uop->in())) {
        stack.emplace(uop, 0);
        replay_stack.push_back(uop);
        continue;
      }
    // This will require us having `replayExprWithNewInput` to support binary ops.
    // } else if (def->isA<BinaryOp>()) {
    //   auto* bop = def->as<BinaryOp>();
    //   // TODO: exception to break propagation.
    //   // TODO: check for broadcast stuff.
    //   if (candidate_check(bop->lhs()) && candidate_check(bop->rhs())) {
    //     stack.emplace(uop, 0);
    //     continue;
    //   }


    // TODO: adding pad
    // } else if (def->isA<PadOp>()) {
    //   if (candidate_check(def->input(0))) {
    //     // NOTE: stopping propagation, we'll merge it with its consumer padOp
    //     frontier.emplace_back(def, 0);
    //     continue;
    //   }
    }

    if (edge->val() != pad->in()) {
      // propagation stopped, push entry to frontier
      frontier.push_back(edge);
    }
  }

  if (frontier.empty()) {
    return nullptr;
  }

  std::unordered_map<Val*, Val*> replacement_map;
  // modify pad on frontier
  for (const Edge& edge : frontier) {
    // insert pad
    // Note: operation with multiple operand would require us to support partial update in each iteration.
    auto width_size = pad->inputs().size() - 2;

    const std::vector<Val*> pad_width;
    pad_width.reserve(width_size);
    for (auto i : c10::irange(width_size)) {
      pad_width.push_back(pad->inputs(2+i));
    }
    replacement_map[edge->val()] = pad(edge->val()->as<TensorView>(), pad_width, pad->value());
    // TODO: modify existing pad, when its only consumer is a pad
  }

  // propagate to update TensorProxy
  // I think I can just follow the reverse order from earlier stack traversal.
  for (Expr* e : replay_sequence) {
    // TODO extend this for multiple inputs.
    Expr* padded_e = replayExprWithNewInput(e, replacement_map(e->input(0)));
    replacement_map[e->output(0)] = padded_e->output(0);
  }

  // return the replacement input to pad, since we have already padded everything out.
  return replacement_map.at(pad->in());
}

}

void MovePadPass::runPass(Fusion* fusion) {
  fusion->printMath();
  // TODO: verify that no dead branch is traversed in exprs.
  std::vector<Expr*> exprs = fusion->exprs();

  // NOTE: should we expand this optimization to general pad but not just pad within cat?

  // is this traversing in topo order?
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {

    std::unordered_map<Val*, Val*> replacement_map;
    for (Val* in : cat->inputs()) {
      auto* pad = in->definition()->as<PadOp>();
      if (Val* new_pad_out = propagatePadToProducer(pad)) {
        replacement_map.at(in) = new_pad_out;
      }
    }
    if (replacement_map.empty()) {
      continue;
    }
    // TODO: We won't have it now, but would replaceValue also replace the outputs of the fusion?
    // TODO: does this invalidate the downstream exprs?
    ir_utils::replaceValue(fusion, replacement_map);
    // Do we *have to* swap cat with pointwise add?
  }

  fusion->printMath();
}

} // namespace nvfuser::preseg_passes
