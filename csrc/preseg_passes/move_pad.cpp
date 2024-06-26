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
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

struct Edge {
  Expr* expr_ = nullptr;
  size_t index_ = 0;

  Edge(Expr* expr, size_t index) : expr_(expr), index_(index) {}

  Val* val() const {
    return expr_->input(index_);
  }
};

Val* propagatePadToProducer(PadOp* pad_op) {
  std::vector<Val*> pad_dependencies;

  auto candidate_check = [&pad_dependencies](Val* val) {
    if (!val->isA<TensorView>()) {
      return false;
    }
    if (val->uses().size() > 1) {
      return false;
    }
    if (val->isFusionOutput()) {
      return false;
    }
    if (std::any_of(
            pad_dependencies.begin(),
            pad_dependencies.end(),
            [val](Val* pad_dependency) {
              return DependencyCheck::isDependencyOf(pad_dependency, val);
            })) {
      return false;
    }
    return true;
  };

  // NOTE: the optimization logic assumes a zero pad_op.
  // This is used for logic in handling binary operations, we should extend this
  // later.
  if (!pad_op->value()->isZero()) {
    return nullptr;
  }

  if (!candidate_check(pad_op->in())) {
    return nullptr;
  }

  for (Val* val : pad_op->inputs()) {
    if (val == pad_op->in() || val->isConst()) {
      continue;
    }
    pad_dependencies.push_back(val);
  }

  std::vector<Edge> frontier;
  // TODO: not sure if I need a stack if I need to keep a replay_sequence.
  std::stack<Edge> stack;
  std::vector<Expr*> replay_sequence;
  stack.emplace(pad_op, 0);

  // tvs in stack are:
  //   1. single use;
  //   2. not an output;
  //   3. cleared dependency of `pad_depdencies;
  //   4. maybe also check aliases?!
  while (!stack.empty()) {
    Edge edge = stack.top();
    Expr* def = edge.val()->definition();
    stack.pop();

    if (def->isA<UnaryOp>()) {
      auto* uop = def->as<UnaryOp>();
      // TODO: exception to break propagation. i.e. check op type and exclude division by 0
      if (candidate_check(uop->in())) {
        stack.emplace(uop, 0);
        replay_sequence.push_back(uop);
        continue;
      }
      // This will require us having `replayExprWithNewInput` to support binary
      // ops. 
      // TODO: adding pad_op
      // } else if (def->isA<PadOp>()) {
      //   if (candidate_check(def->input(0))) {
      //     // NOTE: stopping propagation, we'll merge it with its consumer
      //     padOp frontier.emplace_back(def, 0); continue;
      //   }
    } else if (def->isA<BinaryOp>()) {
      auto* bop = def->as<BinaryOp>();
      // TODO: exception to break propagation. i.e. check op type and exclude division by 0; check for broadcast on padded axis.
      if (candidate_check(bop->lhs()) && candidate_check(bop->rhs())) {
        stack.emplace(bop, 0);
        stack.emplace(bop, 1);
        replay_sequence.push_back(bop);
        continue;
      }
    }

    if (edge.val() != pad_op->in()) {
      // propagation stopped, push entry to frontier
      frontier.push_back(edge);
    }
  }

  if (frontier.empty()) {
    return nullptr;
  }

  std::unordered_map<Val*, Val*> replacement_map;
  // modify pad_op on frontier
  for (const Edge& edge : frontier) {
    // insert pad_op
    // Note: operation with multiple operand would require us to support partial
    // update in each iteration.

    // const auto width_size = pad_op->inputs().size() - 2;
    // const auto num_padded_dims = width_size / 2;
    // std::vector<Val*> pad_width;
    // pad_width.reserve(width_size);
    // for (auto i : c10::irange(num_padded_dims)) {
    //   pad_width.push_back(pad_op->input((num_padded_dims - i)*2));
    //   pad_width.push_back(pad_op->input((num_padded_dims - i)*2 + 1));
    // }
    // cannot use `pad` op, because it would give us symolic iter domain
    // replacement_map[edge.val()] = pad(edge.val()->as<TensorView>(),
    // pad_width, pad_op->value());

    NVF_ERROR(
        edge.val()->getDataType().has_value(), "pad source dtype is missing");
    // NOTE: the old pad_op is going away from DCE, so it's ok to reuse its
    // domains
    TensorView* pad_out_tv = pad_op->out()->as<TensorView>();
    std::vector<IterDomain*> new_root = IterDomain::clone(pad_out_tv->getMaybeRootDomain(), true);
    auto new_out = IrBuilder::create<TensorView>(
        TransformReplay::fullSelfReplay(IrBuilder::create<TensorDomain>(new_root), pad_out_tv->domain()),
        edge.val()->getDataType().value());
    IrBuilder::create<PadOp>(
        new_out,
        edge.val()->as<TensorView>(),
        pad_op->getPadWidths(),
        pad_op->value());

    replacement_map[edge.val()] = new_out;

    // TODO: modify existing pad_op, when its only consumer is a pad_op
  }

  // propagate to update TensorProxy
  // need to follow the reverse order from earlier stack traversal.
  std::reverse(replay_sequence.begin(), replay_sequence.end());
  for (Expr* e : replay_sequence) {
    if (e->isA<UnaryOp>()) {
      // TODO extend this for multiple inputs.
      Expr* padded_e = replayExprWithNewInput(e, replacement_map.at(e->input(0)));
      replacement_map[e->output(0)] = padded_e->output(0);
    } else if (e->isA<BinaryOp>()) {
      // Expr* padded_e = replayExprWithNewInput(e, replacement_map.at(e->input(0)), replacement_map.at(e->input(1)));
      std::vector<Val*> vals = {replacement_map.at(e->input(0)), replacement_map.at(e->input(1))};
      Val* out = ops::newOutputTV(vals, e->output(0)->getDataType().value());
      Expr* padded_e = IrBuilder::create<BinaryOp>(e->as<BinaryOp>()->getBinaryOpType(), out, vals[0], vals[1]);
      replacement_map[e->output(0)] = padded_e->output(0);
    } else {
      NVF_ERROR(false, "expr type for propagation is not implemented");
    }
  }

  // return the replacement input to pad_op, since we have already padded
  // everything out.
  return replacement_map.at(pad_op->in());
}

} // namespace

void MovePadPass::runPass(Fusion* fusion) {
  // TODO: verify that no dead branch is traversed in exprs.
  std::vector<Expr*> exprs = fusion->exprs();

  // TODO: should we expand this optimization to general pad but not just pad
  // within cat?

  // is this traversing in topo order?
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    std::unordered_map<Val*, Val*> replacement_map;
    for (Val* in : cat->inputs()) {
      auto* pad_op = in->definition()->as<PadOp>();
      if (Val* new_pad_out = propagatePadToProducer(pad_op)) {
        replacement_map[in] = new_pad_out;
      }
    }
    if (replacement_map.empty()) {
      continue;
    }
    // NOTE: I'm hitting an index error with PadOp in
    // device_lower/pass/index.cpp:1944
    Val* res = nullptr;
    TensorView* cat_out_tv = cat->output(0)->as<TensorView>();
    bool is_boolean = isBooleanType(cat_out_tv->getDataType().value());
    for (Val* inp : cat->inputs()) {
      if (res == nullptr) {
        res = replacement_map.count(inp) == 0 ? inp : replacement_map.at(inp);
      } else {
        Val* rhs =
            replacement_map.count(inp) == 0 ? inp : replacement_map.at(inp);
        if (is_boolean) {
          res = bitwise_or(res, rhs);
        } else {
          res = add(res, rhs);
        }
      }
    }
    // restore data type if it's promoted by BinaryOp.
    res = maybeCastOp(cat_out_tv->getDataType().value(), res);

    // TODO: We won't have it in tests yet, but would replaceValue also replace
    // the outputs of the fusion?
    // TODO: does this invalidate the downstream exprs?
    // ir_utils::replaceValue(fusion, replacement_map);
    ir_utils::replaceValue(fusion, {{cat->output(0), res}});
    // Do we *have to* swap cat with pointwise add?
    if (cat->output(0)->isFusionOutput()) {
      fusion->replaceOutput(cat->output(0), res);
    }
  }
}

} // namespace nvfuser::preseg_passes
