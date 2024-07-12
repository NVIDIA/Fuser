// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_pad.h>

#include <expr_simplifier.h>
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

bool isSamePadOp(Expr* use, PadOp* p) {
  if (!use->isA<PadOp>()) {
    return false;
  }

  auto* use_pad = use->as<PadOp>();
  if (simplifyExpr(SimplifyingIrBuilder::eqExpr(use_pad->value(), p->value()))
          ->isFalse()) {
    return false;
  }

  if (use_pad->getPadWidths().size() != p->getPadWidths().size()) {
    return false;
  }

  std::vector<int64_t> padded_axes = p->getPaddedAxes();
  if (use_pad->getPaddedAxes() != padded_axes) {
    return false;
  }

  for (auto idx : padded_axes) {
    if (simplifyExpr(
            SimplifyingIrBuilder::eqExpr(
                use_pad->getPadWidths(idx).first, p->getPadWidths(idx).first))
            ->isFalse() ||
        simplifyExpr(
            SimplifyingIrBuilder::eqExpr(
                use_pad->getPadWidths(idx).second, p->getPadWidths(idx).second))
            ->isFalse()) {
      return false;
    }
  }

  return true;
}

Val* replayCatOpWithBinaryOp(
    const std::vector<Val*>& inputs,
    DataType data_type) {
  // replay `CatOp` with series of BinaryOp instead, since we might have
  // pushed `PadOp` out and breaking the codegen if `CatOp` remains.
  Val* res = nullptr;
  bool is_boolean = isBooleanType(data_type);
  for (Val* inp : inputs) {
    if (res == nullptr) {
      res = inp;
    } else {
      if (is_boolean) {
        res = bitwise_or(res, inp);
      } else {
        res = add(res, inp);
      }
    }
  }
  // restore data type if it's promoted by BinaryOp.
  return maybeCastOp(data_type, res);
}

// The pass assumes propagating PadOp with zero pad. The criteria here for
// return true is that `unaryOp(0) == 0`
bool padCompatibleUnaryOp(UnaryOpType t) {
  switch (t) {
    case UnaryOpType::Cast:
    case UnaryOpType::Abs:
    case UnaryOpType::Asin:
    case UnaryOpType::Asinh:
    case UnaryOpType::Atan:
    case UnaryOpType::Atanh:
    case UnaryOpType::Ceil:
    case UnaryOpType::Erf:
    case UnaryOpType::Erfinv:
    case UnaryOpType::Floor:
    case UnaryOpType::Gelu:
    case UnaryOpType::Imag:
    case UnaryOpType::Silu:
    case UnaryOpType::Neg:
    case UnaryOpType::Real:
    case UnaryOpType::Relu:
    case UnaryOpType::Round:
    case UnaryOpType::Sin:
    case UnaryOpType::Sinh:
    case UnaryOpType::Sqrt:
    case UnaryOpType::Tan:
    case UnaryOpType::Tanh:
    case UnaryOpType::Trunc:
      return true;
    default:
      return false;
  }
}

// The pass assumes propagating PadOp with zero pad. The criteria here for
// return true is that `binaryOp(0, x) == 0` & `binaryOp(x, 0) == 0`
bool padCompatibleBinaryOp(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
    case BinaryOpType::Mul:
    case BinaryOpType::Sub:
    case BinaryOpType::BitwiseAnd:
    case BinaryOpType::LogicalAnd:
      return true;
    default:
      return false;
  }
}

// This function checks if we could propagate the PadOp on `val` to its
// definition. i.e. changing the graph from
//   inp --> definition() --> val --> PadOp --> padded_val
// to
//   inp --> PadOp --> padded_inp --> definition() --> padded_val
//
// returns true if `PadOp` should be propagated pass val->definition().
//
// NOTE 0: For details on the propagation logic, see Note [ PadOp Propagation
// Rule ] NOTE 1: This function also update frontier accordingly, see note
// below.
bool shouldPropagatePad(Val* val, std::unordered_map<Val*, int64_t>& frontier) {
  // NOTE [ Handling TV with Multiple Uses via Frontier ]
  //
  // Frontier counts the encounter of each Val during the propagation. When we
  // reach a Val during the propagation, we add Val entry *temporarily* to the
  // frontier map and increment its encounter count. We cannot propagate the
  // PadOp over the producer of Val yet, because it still have other uses that
  // expects the original Val. Until we see all uses of Val during the
  // traversal, we know all uses can be replaced with the padded Val and hence
  // it's safe to propagate `PadOp` across. We remove Val fron `frontier` map
  // since we can push the replay to its producer. We return `true` to signal
  // that further propagation is allowed.
  if (val->isFusionOutput()) {
    // skipping propagation when target is an output.
    frontier.emplace(val, -1);
    return false;
  }

  int64_t current_use = frontier.count(val) == 0 ? 1 : frontier[val] + 1;

  // `current_use == 0` means `previous use == -1`, blocking propagation
  if (current_use == 0) {
    return false;
  }

  if (current_use == static_cast<int64_t>(val->uses().size())) {
    // all uses of the entry has been encounter in this traversal, we can
    // safely propagate it across. remove val from frontier map
    frontier.erase(val);
    return true;
  }

  // updating uses in frontier. return false since we are not yet ready to
  // propagate
  frontier[val] = current_use;
  return false;
}

// replayConcretePad tries to replay a concretized PadOp on pad_tv. This
// function allows merging multiple consecutive PadOp by stacking their pad
// widths together as `vec_pad_widths`. This function assumes it has already
// resolved the output iter_type for each IterDomain and provided as
// `ref_iter_type`. The function returns the output from the replay PadOp.
//
// NOTE: this assumes all vec_pad_widths are positive entries so we don't need
// to consider accumulating them changing the output iter_type.
TensorView* replayConcretePad(
    TensorView* pad_tv,
    Val* pad_value,
    const std::vector<std::vector<Val*>>& vec_pad_widths,
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
  // stack vec_pad_widths as `merged_pad_widths`.
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

  // construct TensorDomain for output TV.
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

// Note [ PadOp Propagation Rule ]
//
// Push PadOp to its producer to reduce segmentation caused by the resize node
// introduced in PadOp. The hope is that we can either push PadOp to inputs to
// the fusion, or far enough that the fusion segments before the PadOp would
// become a trivial no-op.
//
// The concept is that the following program:
//   tv1 = UnaryOp(tv0)
//   tv2 = PadOp(tv1)
// can be replayed as the program below:
//   tv0_padded = PadOp(tv0)
//   tv2_new = UnaryOp(tv0_padded)
// Given certain constraints on the UnaryOp, we can ensure that the
// computational output remains the same when we replace all uses of `tv2` with
// `tv2_new`.
//
// This function only propagates `zero-padding`, i.e. PadOp with pad value as
// constant zero. Based on which we have certain operations that can allow PadOp
// to propagated across. See `padCompatibleUnaryOp` and `padCompatibleBinaryOp`.
Val* propagatePadToProducer(PadOp* pad_op) {
  // establish pad dependencies, ensures that pad_value and pad_widths are live
  // at the time of replay.
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
  //   1. The optimization logic assumes a zero pad. This is used for logic in
  //   handling binary operations, see Note [ PadOp Propagation Rule] ; or 2.a
  //   if `pad_op->in()` is used more than by the pad_op; or 2.b if
  //   `pad_op->in()` is an output tv.
  // for case 2.a/b, because pad_op->in() needs to stay alive after the
  // propagation, we want to avoid adding another data flow branch.
  if (!pad_op->value()->isZero() || pad_op->in()->uses().size() > 1 ||
      pad_op->in()->isFusionOutput()) {
    return nullptr;
  }

  // frontier contains `Val`s that needs to replay pad_op on. The map here
  // stores the number of encounter for each `Val*` during traversal, the logic
  // is used to decide whether propagation can cross Val with multiple uses in
  // the graph. See NOTE [ Handling TV with Multiple Uses via Frontier ]
  std::unordered_map<Val*, int64_t> frontier;
  // replay_sequence is used later to create the updated branch with padded
  // inputs after all `frontier` has been updated with padding.
  std::vector<Expr*> replay_sequence;

  std::stack<Edge> stack;
  stack.emplace(pad_op, 0);
  while (!stack.empty()) {
    Edge edge = stack.top();
    Expr* def = edge.val()->definition();
    stack.pop();

    if (def->isA<UnaryOp>()) {
      auto* uop = def->as<UnaryOp>();
      // check if unary op type is compatible for zero pad propagation.
      if (!padCompatibleUnaryOp(uop->getUnaryOpType())) {
        frontier.emplace(edge.val(), -1);
        continue;
      }
      // check if PadOp can be replayed on input
      if (!pad_replay_check(uop->in())) {
        frontier.emplace(edge.val(), -1);
        continue;
      }
      // uop is replayable so add it to replay_sequence
      replay_sequence.push_back(uop);

      // check if we want to further propagate the replay
      if (shouldPropagatePad(uop->in(), std::ref(frontier))) {
        stack.emplace(uop, 0);
      }
    } else if (def->isA<BinaryOp>()) {
      auto* bop = def->as<BinaryOp>();
      // check if binary op type is compatible for zero pad propagation.
      if (!padCompatibleBinaryOp(bop->getBinaryOpType())) {
        frontier.emplace(edge.val(), -1);
        continue;
      }
      // check if PadOp can be replayed on both operands
      if (!pad_replay_check(bop->lhs()) || !pad_replay_check(bop->rhs())) {
        frontier.emplace(edge.val(), -1);
        continue;
      }

      // check for broadcast on padded axis.
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
      // padding on broadcast dimensions is not supported in propagation yet.
      if (pad_on_broadcast) {
        frontier.emplace(edge.val(), -1);
        continue;
      }
      // bop is replayable so add it to replay_sequence
      replay_sequence.push_back(bop);

      // check if we want to further propagate the replay on each operand
      // separately
      if (shouldPropagatePad(lhs, std::ref(frontier))) {
        stack.emplace(bop, 0);
      }
      if (shouldPropagatePad(rhs, std::ref(frontier))) {
        stack.emplace(bop, 1);
      }
    } else {
      // Unrecognized operation stops propagation, push entry to frontier for
      // replay
      frontier.emplace(edge.val(), -1);
    }
  }

  // NOTE: if we have not propagated pass the original pad_op input, that means
  // no changes has been made at all.
  if (frontier.count(pad_op->in()) != 0) {
    return nullptr;
  }

  std::unordered_map<Val*, Val*> replacement_map;
  // entries in frontier contains Val that we want to replay the PadOp on.
  for (const auto& [pad_val, _] : frontier) {
    auto pad_tv = pad_val->as<TensorView>();
    const std::vector<IterDomain*> out_ids = TensorDomain::noReductions(
        pad_op->out()->as<TensorView>()->getLogicalDomain());
    // replay pad_op on frontier TVs assuming its output iter_type wouldn't
    // change from the final output.
    TensorView* new_out = replayConcretePad(
        pad_tv, pad_op->value(), {pad_op->getPadWidths()}, out_ids);
    replacement_map[pad_val] = new_out;
  }

  // reverse traversal the replay_sequence and update each input to use padded
  // TVs.
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

// This pass tries to push `PadOp`, which is part of `cat` (composed of `PadOp`s
// + `CatOp`), further to its producers to avoid possible segmentation. As a
// side effect, it replaces the `CatOp` with a series of pointwise add if any
// mutation has been made for propagating `PadOp`
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

    // replay `CatOp` with series of BinaryOp instead, since we might have
    // pushed `PadOp` out and breaking the codegen if `CatOp` remains.
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

// This pass merges neighboring `PadOp` when possible (identical pad value).
void mergeNeighboringPad(Fusion* fusion) {
  std::vector<Expr*> exprs = fusion->exprs();
  // traverse in topo order. We'll merge neighboring pad and replace the uses of
  // consumer pad with the merged producer. So it would not interfere traversal.
  for (auto* producer : ir_utils::filterByType<PadOp>(exprs)) {
    while (producer) {
      Val* pad_out = producer->out();
      if (pad_out->uses().size() != 1 || !pad_out->uses()[0]->isA<PadOp>()) {
        break;
      }
      auto* consumer = pad_out->uses()[0]->as<PadOp>();

      // only allow merge pad when pad value is the same.
      if (simplifyExpr(SimplifyingIrBuilder::eqExpr(
                           producer->value(), consumer->value()))
              ->isFalse()) {
        break;
      }

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
      producer = new_out->definition()->as<PadOp>();
    }
  }
}

void propagatePad(Fusion* fusion) {
  // propagating PadOp
  auto exprs = fusion->exprs();
  auto filtered_pads = ir_utils::filterByType<PadOp>(exprs);
  std::vector<PadOp*> frontier;
  frontier.reserve(filtered_pads.size());
  // NOTE: we only consider zero pad as propagation frontier.
  std::copy_if(
      filtered_pads.begin(),
      filtered_pads.end(),
      std::back_inserter(frontier),
      [](PadOp* pad) { return simplifyExpr(pad->value())->isZero(); });

  std::unordered_set<PadOp*> merged_pad;
  while (!frontier.empty()) {
    PadOp* p = frontier.back();
    frontier.pop_back();

    // TODO: should I check for if uses lead to output instead?
    // if no uses, this has already been short-wired.
    if (p->out()->uses().empty() && !p->out()->isFusionOutput()) {
      continue;
    }

    // unify all consumer pad of tv;
    TensorView* tv = p->in()->as<TensorView>();
    for (Expr* use : tv->uses()) {
      if (use == p) {
        continue;
      }
      // check if use is the same pad operation (same pad value / width e.t.c.)
      if (isSamePadOp(use, p)) {
        // replace consumer of use->out() with p->out()
        ir_utils::replaceValue(fusion, {{use->output(0), p->out()}});
        if (use->output(0)->isFusionOutput()) {
          fusion->replaceOutput(use->output(0), p->out());
        }
        // NVF_ERROR(tv->removeUse(use), "remove uses failed");
        fusion->removeExpr(use);
      }
    }

    // check for pad_dependencies to verify that 'p' can be moved before 'def'.
    std::vector<Val*> pad_dependencies;
    for (Val* val : p->inputs()) {
      if (val == p->in() || val->isConst()) {
        continue;
      }
      pad_dependencies.push_back(val);
    }
    auto pad_replay_check = [&pad_dependencies](Expr* expr) {
      return std::all_of(
          expr->inputs().begin(),
          expr->inputs().end(),
          [&pad_dependencies = std::as_const(pad_dependencies)](Val* val) {
            if (!val->isA<TensorView>()) {
              return false;
            }
            if (std::any_of(
                    pad_dependencies.begin(),
                    pad_dependencies.end(),
                    [val](Val* pad_dependency) {
                      return DependencyCheck::isDependencyOf(
                          pad_dependency, val);
                    })) {
              return false;
            }
            return true;
          });
    };

    Expr* def = p->in()->definition();
    Val* new_out = nullptr;

    if (def->isA<UnaryOp>()) {
      // stop propagation:
      //   1. if current PadOp p isn't the only use of tv; or
      //   2. if tv is an output.
      // since both case requires tv to be live in the fusion.
      if (tv->uses().size() != 1 || tv->isFusionOutput()) {
        continue;
      }
      NVF_ERROR(
          tv->uses()[0] == p,
          "expect existing PadOp to be the only use of its input");
      auto* uop = def->as<UnaryOp>();
      // check if unary op type is compatible for zero pad propagation.
      if (!padCompatibleUnaryOp(uop->getUnaryOpType())) {
        continue;
      }
      // check if PadOp can be replayed on input(s)
      if (!pad_replay_check(uop)) {
        continue;
      }

      // replay pad on input(s)
      Val* new_pad_out = replayConcretePad(
          uop->in()->as<TensorView>(),
          p->value(),
          {p->getPadWidths()},
          TensorDomain::noReductions(
              p->out()->as<TensorView>()->getLogicalDomain()));

      new_out = ops::newValLike(new_pad_out, uop->out()->getDataType().value());
      IrBuilder::create<UnaryOp>(uop->getUnaryOpType(), new_out, new_pad_out);
      // insert new PadOp(s) to frontier;
      frontier.push_back(new_pad_out->definition()->as<PadOp>());
    } else if (def->isA<BinaryOp>()) {
      // stop propagation:
      //   1. if current PadOp p isn't the only use of tv; or
      //   2. if tv is an output.
      // since both case requires tv to be live in the fusion.
      if (tv->uses().size() != 1 || tv->isFusionOutput()) {
        continue;
      }
      NVF_ERROR(
          tv->uses()[0] == p,
          "expect existing PadOp to be the only use of its input");
      auto* bop = def->as<BinaryOp>();

      // check if unary op type is compatible for zero pad propagation.
      if (!padCompatibleBinaryOp(bop->getBinaryOpType())) {
        continue;
      }
      // check if PadOp can be replayed on input(s)
      if (!pad_replay_check(bop)) {
        continue;
      }

      // check for broadcast on padded axis.
      auto* lhs = bop->lhs()->as<TensorView>();
      auto* rhs = bop->rhs()->as<TensorView>();
      bool pad_on_broadcast = false;
      for (auto i : p->getPaddedAxes()) {
        if (lhs->getLogicalDomain()[i]->isBroadcast() ||
            rhs->getLogicalDomain()[i]->isBroadcast()) {
          pad_on_broadcast = true;
          break;
        }
      }
      // padding on broadcast dimensions is not supported in propagation yet.
      if (pad_on_broadcast) {
        continue;
      }

      // replay pad on input(s)
      std::vector<Val*> vals = {
          replayConcretePad(
              bop->lhs()->as<TensorView>(),
              p->value(),
              {p->getPadWidths()},
              TensorDomain::noReductions(
                  p->out()->as<TensorView>()->getLogicalDomain())),
          replayConcretePad(
              bop->rhs()->as<TensorView>(),
              p->value(),
              {p->getPadWidths()},
              TensorDomain::noReductions(
                  p->out()->as<TensorView>()->getLogicalDomain()))};

      new_out = ops::newOutputTV(vals, bop->out()->getDataType().value());
      IrBuilder::create<BinaryOp>(
          bop->getBinaryOpType(), new_out, vals[0], vals[1]);
      // insert new PadOp(s) to frontier;
      frontier.push_back(vals[0]->definition()->as<PadOp>());
      frontier.push_back(vals[1]->definition()->as<PadOp>());
    } else if (def->isA<PadOp>()) {
      auto* pop = def->as<PadOp>();
      // only allow merge pad when pad value is the same.
      if (simplifyExpr(SimplifyingIrBuilder::eqExpr(p->value(), pop->value()))
              ->isFalse()) {
        continue;
      }
      const std::vector<Val*> p_pad_widths = pop->getPadWidths();
      const std::vector<Val*> c_pad_widths = p->getPadWidths();
      // I think this should always hold, otherwise we can relax it and
      // continue instead.
      NVF_ERROR(
          p_pad_widths.size() == c_pad_widths.size(),
          "expect consecutive PadOp to have the same length of pad widths");

      // replay merged pad on pop->in()
      new_out = replayConcretePad(
          pop->in()->as<TensorView>(),
          pop->value(),
          {pop->getPadWidths(), p->getPadWidths()},
          TensorDomain::noReductions(
              p->out()->as<TensorView>()->getLogicalDomain()));

      // insert new PadOp(s) to frontier;
      frontier.push_back(new_out->definition()->as<PadOp>());
    } else if (def->isA<CatOp>()) {
      auto* cat = def->as<CatOp>();

      // TODO: can cat support broadcast on any non-cat dimensions? Otherwise we
      // need to ensure that we are not padding on broadcast dimensions like
      // binary op
      std::vector<Val*> vals;
      std::transform(
          cat->inputs().begin(),
          cat->inputs().end(),
          std::back_inserter(vals),
          [&p, &frontier](Val* val) {
            Val* pad_out = replayConcretePad(
                val->as<TensorView>(),
                p->value(),
                {p->getPadWidths()},
                TensorDomain::noReductions(
                    p->out()->as<TensorView>()->getLogicalDomain()));
            frontier.push_back(pad_out->definition()->as<PadOp>());
            return pad_out;
          });

      new_out =
          replayCatOpWithBinaryOp(vals, cat->output(0)->getDataType().value());
    }
    // replace old (->pad->) with (->pads_before_new_def->new_def->)
    if (new_out != nullptr) {
      ir_utils::replaceValue(fusion, {{p->out(), new_out}});
      if (p->out()->isFusionOutput()) {
        fusion->replaceOutput(p->out(), new_out);
      }
    }
  }
}

void replaceCat(Fusion* fusion) {
  // updating CatOp
  std::vector<Expr*> exprs = fusion->exprs();

  // sanitizing CatOp with series of binary add;
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    if (std::any_of(cat->inputs().begin(), cat->inputs().end(), [](Val* val) {
          return !val->definition()->isA<PadOp>();
        })) {
      Val* res = replayCatOpWithBinaryOp(
          cat->inputs(), cat->output(0)->getDataType().value());

      // replace `CatOp` with the replay result.
      ir_utils::replaceValue(fusion, {{cat->output(0), res}});
      if (cat->output(0)->isFusionOutput()) {
        fusion->replaceOutput(cat->output(0), res);
      }
    }
  }
}

} // namespace

void MovePadPass::runPass(Fusion* fusion) {
  // decomposeCatOp(fusion);
  // mergeNeighboringPad(fusion);

  propagatePad(fusion);
  replaceCat(fusion);
}

} // namespace nvfuser::preseg_passes
