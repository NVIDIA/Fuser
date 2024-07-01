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
#include <expr_simplifier.h>

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

bool shouldPropagate(Val* val, std::unordered_map<Val*, int64_t>& frontier) {
  if (val->isFusionOutput()) {
    frontier.emplace(val, -1);
  } else {
    int64_t current_use = frontier.count(val) == 0 ? 1 : frontier[val] + 1;

    // previous use == -1, blocking propagation
    if (current_use == 0) {
      return false;
    }

    if (current_use == static_cast<int64_t>(val->uses().size())) {
      // all uses of the entry has been encounter in this traversal, we can safely propagate it across.
      frontier.erase(val);
      return true;
    } else {
      // updating uses in frontier.
      frontier[val] = current_use;
    }
  }
  return false;
}

using VecPadWidths = std::vector<std::vector<Val*>>;

// NOTE: this assumes all vec_pad_widths are positive entries so we don't need
// to consider accumulating them changing the output iter_type.
TensorView* replayConcretePad(
    TensorView* pad_tv,
    Val* pad_value,
    const VecPadWidths& vec_pad_widths,
    std::vector<IterDomain*> ref_iter_type) {
  NVF_ERROR(pad_tv->getDataType().has_value(), "pad source dtype is missing");
  const std::vector<IterDomain*> inp_dom =
      TensorDomain::noReductions(pad_tv->getLogicalDomain());
  const auto rank = inp_dom.size();

  NVF_ERROR(
      rank == ref_iter_type.size(),
      "ref_iter_type does not have compatible size regarding pad_tv");

  std::vector<Val*> merged_pad_widths;
  merged_pad_widths.reserve(rank * 2);

  NVF_ERROR(!vec_pad_widths.empty(), "vec_pad_widths cannot be empty");
  if (vec_pad_widths.size() == 1) {
    merged_pad_widths = vec_pad_widths.at(0);
  } else {
    for (const auto i : c10::irange(2 * rank)) {
      Val* merged_pad_width = nullptr;
      for (const auto idx : c10::irange(vec_pad_widths.size())) {
        // skipping zero pad;
        Val* pad_width = vec_pad_widths[idx].at(i);
        if (pad_width->isZeroInt()) {
          continue;
        }
        merged_pad_width = merged_pad_width == nullptr
            ? pad_width
            : add(merged_pad_width, pad_width);
      }
      merged_pad_widths.push_back(
          merged_pad_width == nullptr ? FusionGuard::getCurFusion()->zeroVal()
                                      : merged_pad_width);
    }
  }

  std::vector<IterDomain*> merged_root_ids;
  std::vector<IterDomain*> merged_logical_ids;
  for (const auto i : c10::irange(rank)) {
    Val* left_pad = merged_pad_widths.at(i * 2);
    Val* right_pad = merged_pad_widths.at(i * 2 + 1);
    IterDomain* inp_id = inp_dom.at(i);
    if (left_pad->isZeroInt() && right_pad->isZeroInt()) {
      merged_root_ids.push_back(inp_id->cloneWithoutRFactor());
      merged_logical_ids.push_back(merged_root_ids.back());
      continue;
    }
    // NOTE: nvfuser pad doesn't support negative padding, so we don't have to
    // worry about it cancelling out.
    IterDomain* merged_root_id =
        IterDomainBuilder(inp_id).is_rfactor_domain(true).build();
    merged_root_ids.push_back(merged_root_id);
    merged_logical_ids.push_back(IterDomain::resize(
        merged_root_id,
        left_pad,
        right_pad,
        true,
        ref_iter_type.at(i)->getIterType()));
  }

  auto* new_out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          merged_root_ids, merged_logical_ids, merged_logical_ids),
      pad_tv->getDataType().value());
  IrBuilder::create<PadOp>(new_out, pad_tv, merged_pad_widths, pad_value);
  return new_out;
}

Val* propagatePadToProducer(PadOp* pad_op) {
  // establish pad dependencies, ensures that pad_value and pad_widths are live at the time of replay.
  std::vector<Val*> pad_dependencies;
  for (Val* val : pad_op->inputs()) {
    if (val == pad_op->in() || val->isConst()) {
      continue;
    }
    pad_dependencies.push_back(val);
  }
  auto pad_replay_check = [&pad_dependencies](Val* val) {
    if (!val->isA<TensorView>()) {
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

  // NOTE: We skip the propagation when any of the three conditions is true:
  // 1. The optimization logic assumes a zero pad_op. This is used for logic in handling binary operations;
  // 2. if `pad_op->in()` is used more than by the pad_op;
  // 3. if `pad_op->in()` is an output tv.
  if (!pad_op->value()->isZero() || pad_op->in()->uses().size() > 1 || pad_op->in()->isFusionOutput()) {
    return nullptr;
  }

  // frontier contains `Val`s that needs to replay pad_op on.
  std::unordered_map<Val*, int64_t> frontier;
  // replay_sequence is used later to create the updated branch with padded inputs after all `frontier` has been updated with padding.
  std::vector<Expr*> replay_sequence;

  std::stack<Edge> stack;
  stack.emplace(pad_op, 0);
  // tvs in stack are:
  //   1. single use;
  //   2. not an output;
  //   3. cleared dependency of `pad_depdencies`;
  //   4. maybe also check aliases?!
  while (!stack.empty()) {
    Edge edge = stack.top();
    Expr* def = edge.val()->definition();
    stack.pop();

    if (def->isA<UnaryOp>()) {
      auto* uop = def->as<UnaryOp>();
      // TODO: exception to break propagation. i.e. check op type and exclude
      // division by 0
      if (!pad_replay_check(uop->in())) {
        frontier.emplace(edge.val(), -1);
        continue;
      }
      // uop is replayable.
      replay_sequence.push_back(uop);
      if (shouldPropagate(uop->in(), frontier)) {
        stack.emplace(uop, 0);
      }
    } else if (def->isA<BinaryOp>()) {
      auto* bop = def->as<BinaryOp>();
      // TODO: exception to break propagation. i.e. check op type and exclude
      // division by 0; check for broadcast on padded axis.
      if (!pad_replay_check(bop->lhs()) || !pad_replay_check(bop->rhs())) {
        frontier.emplace(edge.val(), -1);
        continue;
      }

      // TODO: padding on broadcast dimensions is not supported in propagation yet.
      auto* lhs = bop->lhs()->as<TensorView>();
      auto* rhs = bop->rhs()->as<TensorView>();
      bool pad_on_broadcast = false;
      for (auto i : pad_op->getPaddedAxes()) {
        if (lhs->getLogicalDomain()[i]->isBroadcast() ||
            rhs->getLogicalDomain()[i]->isBroadcast()) {
          pad_on_broadcast = true;
          break;
        }
      }
      if (!pad_on_broadcast) {
        replay_sequence.push_back(bop);
        if (shouldPropagate(lhs, frontier)) {
          stack.emplace(bop, 0);
        }
        if (shouldPropagate(rhs, frontier)) {
          stack.emplace(bop, 1);
        }
      } else {
        frontier.emplace(edge.val(), -1);
      }
    } else {
      // Unrecognized operation stops propagation, push entry to frontier for replay
      frontier.emplace(edge.val(), -1);
    }
  }

  // NOTE: if we have not propagated pass the original pad_op input, that means no changes has been made at all.
  if (frontier.count(pad_op->in()) != 0) {
    return nullptr;
  }

  std::unordered_map<Val*, Val*> replacement_map;
  // replay pad_op on frontier
  for (const auto& [pad_val, _] : frontier) {
    auto pad_tv = pad_val->as<TensorView>();
    const std::vector<IterDomain*> out_ids = TensorDomain::noReductions(
        pad_op->out()->as<TensorView>()->getLogicalDomain());
    // replay pad_op on frontier TVs assuming its output iter_type wouldn't change from the final output.
    TensorView* new_out = replayConcretePad(
        pad_tv, pad_op->value(), {pad_op->getPadWidths()}, out_ids);
    replacement_map[pad_val] = new_out;
  }

  // reverse traversal the replay_sequence and update each input to use padded TVs.
  std::reverse(replay_sequence.begin(), replay_sequence.end());
  for (Expr* e : replay_sequence) {
    if (e->isA<UnaryOp>()) {
      Val* out = ops::newValLike(
          replacement_map.at(e->input(0)), e->output(0)->getDataType().value());
      Expr* padded_e = IrBuilder::create<UnaryOp>(
          e->as<UnaryOp>()->getUnaryOpType(),
          out,
          replacement_map.at(e->input(0)));
      replacement_map[e->output(0)] = padded_e->output(0);
    } else if (e->isA<BinaryOp>()) {
      std::vector<Val*> vals = {
          replacement_map.at(e->input(0)), replacement_map.at(e->input(1))};
      Val* out = ops::newOutputTV(vals, e->output(0)->getDataType().value());
      Expr* padded_e = IrBuilder::create<BinaryOp>(
          e->as<BinaryOp>()->getBinaryOpType(), out, vals[0], vals[1]);
      replacement_map[e->output(0)] = padded_e->output(0);
    } else {
      NVF_ERROR(false, "expr type for propagation is not implemented");
    }
  }

  // return the final replacement input to pad_op
  return replacement_map.at(pad_op->in());
}

void decomposeCatOp(Fusion* fusion) {
  // TODO: verify that no dead branch is traversed in exprs.
  std::vector<Expr*> exprs = fusion->exprs();

  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    std::unordered_map<Val*, Val*> replacement_map;
    // try to propagate each PadOp before CatOp through its producers.
    for (Val* in : cat->inputs()) {
      auto* pad_op = in->definition()->as<PadOp>();
      if (Val* new_pad_out = propagatePadToProducer(pad_op)) {
        replacement_map[in] = new_pad_out;
      }
    }
    // if propagation fails, there's no point in further graph mutation.
    if (replacement_map.empty()) {
      continue;
    }

    // replay `CatOp` with series of BinaryOp instead, since we might have pushed `PadOp` out and breaking the codegen if `CatOp` remains.
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

    // replace `CatOp` with the replay result.
    ir_utils::replaceValue(fusion, {{cat->output(0), res}});
    if (cat->output(0)->isFusionOutput()) {
      fusion->replaceOutput(cat->output(0), res);
    }
  }
}

void mergeNeighboringPad(Fusion* fusion) {
  std::vector<Expr*> exprs = fusion->exprs();
  // traverse in topo order. We'll merge current pad into its one and only
  // consumer pad so we don't have to worry about interfering the traversal.
  for (auto* producer : ir_utils::filterByType<PadOp>(exprs)) {
    Val* pad_out = producer->out();
    if (pad_out->uses().size() != 1 || !pad_out->uses()[0]->isA<PadOp>()) {
      continue;
    }
    auto* consumer = pad_out->uses()[0]->as<PadOp>();

    // only allow merge pad when pad value is the same.
    if (simplifyExpr(SimplifyingIrBuilder::eqExpr(producer->value(), consumer->value()))->isFalse()) {
      continue;
    }
    // if ((producer->value() != consumer->value()) &&
    //     (!producer->value()->isZero() || !consumer->value()->isZero())) {
    //   continue;
    // }

    const std::vector<Val*> p_pad_widths = producer->getPadWidths();
    const std::vector<Val*> c_pad_widths = consumer->getPadWidths();

    // I think this should always hold, otherwise we can relax it and continue
    // instead.
    NVF_ERROR(
        p_pad_widths.size() == c_pad_widths.size(),
        "expect consecutive PadOp to have the same length of pad widths");

    // replay merged pad on producer input
    TensorView* new_out = replayConcretePad(
        producer->in()->as<TensorView>(),
        producer->value(),
        {producer->getPadWidths(), consumer->getPadWidths()},
        TensorDomain::noReductions(
            consumer->out()->as<TensorView>()->getLogicalDomain()));

    // replace consumer pad with the merged pad.
    ir_utils::replaceValue(
        fusion, {{consumer->out(), static_cast<Val*>(new_out)}});
    if (consumer->out()->isFusionOutput()) {
      fusion->replaceOutput(consumer->out(), new_out);
    }
  }
}

} // namespace

void MovePadPass::runPass(Fusion* fusion) {
  std::cout << "=== input fusion ===" << std::endl;
  fusion->printMath();
  decomposeCatOp(fusion);
  std::cout << "=== after decompose cat fusion ===" << std::endl;
  fusion->printMath();
  mergeNeighboringPad(fusion);
  std::cout << "=== after merge pad fusion ===" << std::endl;
  fusion->printMath();
}

} // namespace nvfuser::preseg_passes
