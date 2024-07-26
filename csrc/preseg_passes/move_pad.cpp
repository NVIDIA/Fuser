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

// Simple PadOp is defined as a pad op with:
// 1. zero pad value; and
// 2. non-negative pad widths.
bool isSimplePadOp(PadOp* pad) {
  if (!simplifyExpr(pad->value())->isZero() &&
      !simplifyExpr(pad->value())->isFalse()) {
    return false;
  }
  // TODO: we cannot seem to always been able to prove that pad widths are >= 0.
  // But we know that PadOp + CatOp implies such condition. Adding this
  // short-cut to ensure that we recognized such pattern.
  if (pad->out()->uses().size() == 1 && pad->out()->uses()[0]->isA<CatOp>()) {
    return true;
  }
  std::vector<Val*> pad_widths = pad->getPadWidths();
  return std::all_of(pad_widths.begin(), pad_widths.end(), [](Val* pad_val) {
    return simplifyExpr(
               SimplifyingIrBuilder::geExpr(pad_val, pad->fusion()->zeroVal()))
        ->isTrue();
  });
}

// check if `use` and `p` are the same pad operatoin.
bool isSamePadOp(Expr* use, PadOp* p) {
  if (!use->isA<PadOp>()) {
    return false;
  }

  auto* use_pad = use->as<PadOp>();
  if (!simplifyExpr(SimplifyingIrBuilder::eqExpr(use_pad->value(), p->value()))
           ->isTrue()) {
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
    if (!simplifyExpr(
             SimplifyingIrBuilder::eqExpr(
                 use_pad->getPadWidths(idx).first, p->getPadWidths(idx).first))
             ->isTrue() ||
        !simplifyExpr(SimplifyingIrBuilder::eqExpr(
                          use_pad->getPadWidths(idx).second,
                          p->getPadWidths(idx).second))
             ->isTrue()) {
      return false;
    }
  }

  return true;
}

// This operation replaces:
//   CatOp(inputs)
// with:
//   BinaryOp(inputs[n-1], BinaryOp(inputs[n-2], BinaryOp(..., BinaryOp[0])...))
// For boolean inputs, we use `logical_or` while `add` for the other dtypes
Val* replaceCatOpWithBinaryOp(const std::vector<Val*>& inputs) {
  // replay `CatOp` with series of BinaryOp instead, since we might have
  // pushed `PadOp` out and breaking the codegen if `CatOp` remains.
  DataType data_type = inputs[0]->getDataType().value();
  NVF_ERROR(
      std::all_of(
          inputs.begin(),
          inputs.end(),
          [&data_type](Val* val) { val->getDataType().value() == data_type; }),
      "all inputs to cat should be of the same datatype");
  NVF_ERROR(!inputs.empty(), "replace cat op expects to have non-empty inputs");

  auto binary_op = isBooleanType(data_type) ? logical_or : add;
  Val* res = inputs[0];
  for (auto i : c10::irange(1, inputs.size())) {
    res = binary_op(res, inputs[i]);
  }
  // restore data type if it's promoted by BinaryOp.
  return maybeCastOp(data_type, res);
}

// The pass assumes propagating PadOp with zero pad. The criteria here for
// return true is that `unaryOp(0) == 0` or `unaryOp(0) == false`
bool zeroIsFixedPoint(UnaryOpType t) {
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
    case UnaryOpType::Expm1:
    case UnaryOpType::Floor:
    case UnaryOpType::Frac:
    case UnaryOpType::Gelu:
    case UnaryOpType::Imag:
    case UnaryOpType::Log1p:
    case UnaryOpType::Neg:
    case UnaryOpType::Real:
    case UnaryOpType::Relu:
    case UnaryOpType::Round:
    case UnaryOpType::Silu:
    case UnaryOpType::Signbit:
    case UnaryOpType::Sin:
    case UnaryOpType::Sinh:
    case UnaryOpType::Sqrt:
    case UnaryOpType::Tan:
    case UnaryOpType::Tanh:
    case UnaryOpType::Trunc:
    case UnaryOpType::IsInf:
    case UnaryOpType::IsNan:
    case UnaryOpType::IsNegInf:
    case UnaryOpType::IsPosInf:
      return true;
    default:
      return false;
  }
}

// The pass assumes propagating PadOp with zero pad. The criteria here for
// return true is that `binaryOp(0, x) == 0` & `binaryOp(x, 0) == 0`
bool zeroIsIdentity(BinaryOpType t) {
  switch (t) {
    case BinaryOpType::Add:
    case BinaryOpType::Mul:
    case BinaryOpType::BitwiseOr:
    case BinaryOpType::BitwiseXor:
    case BinaryOpType::LogicalOr:
      return true;
    default:
      return false;
  }
}

// replayConcretePad tries to replay a concretized PadOp on pad_tv. This
// function allows merging multiple consecutive PadOp by stacking their pad
// widths together as `vec_pad_widths`. This function assumes it has already
// resolved the output iter_type for each IterDomain and provided as
// `ref_iter_type`. The function returns the output from the replay PadOp.
//
// NOTE: this assumes all vec_pad_widths are non-negative entries so we don't
// need to consider accumulating them changing the output iter_type.
TensorView* replayConcretePad(
    Val* pad_val,
    Val* pad_value,
    const std::vector<std::vector<Val*>>& vec_pad_widths,
    std::vector<IterDomain*> ref_iter_type) {
  auto* pad_tv = pad_val->as<TensorView>();
  NVF_ERROR(pad_tv->getDataType().has_value(), "pad source dtype is missing");
  const std::vector<IterDomain*> inp_dom =
      TensorDomain::noReductions(pad_tv->getLogicalDomain());
  const auto rank = inp_dom.size();

  NVF_ERROR(
      rank == ref_iter_type.size(),
      "ref_iter_type does not have compatible size regarding pad_tv");
  NVF_ERROR(
      std::all_of(
          vec_pad_widths.begin(),
          vec_pad_widths.end(),
          [&rank](const std::vector<Val*>& pad_widths) {
            return pad_widths.size() == 2 * rank;
          }),
      "vec_pad_widths doesn't have compatible length for pad_tv");

  std::vector<Val*> merged_pad_widths;

  NVF_ERROR(!vec_pad_widths.empty(), "vec_pad_widths cannot be empty");
  // stack vec_pad_widths as `merged_pad_widths`.
  if (vec_pad_widths.size() == 1) {
    merged_pad_widths = vec_pad_widths.at(0);
  } else {
    merged_pad_widths.reserve(rank * 2);
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
            : SimplifyingIrBuilder::addExpr(merged_pad_width, pad_width);
      }
      merged_pad_widths.push_back(
          merged_pad_width == nullptr ? pad_tv->fusion()->zeroVal()
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
        /*mark_as_rfactor=*/true,
        ref_iter_type.at(i)->getIterType()));
  }

  auto* new_out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          merged_root_ids, merged_logical_ids, merged_logical_ids),
      pad_tv->getDataType().value());
  IrBuilder::create<PadOp>(
      new_out,
      pad_tv,
      merged_pad_widths,
      SimplifyingIrBuilder::maybeCastExpr(
          pad_tv->getDataType().value(), pad_value));
  return new_out;
}

// This function tries to move `pad` on tv to tv->definition()->inputs and
// returns padded inputs. When moving pad fails, this function returns an empty
// vector.
std::vector<Val*> maybeMovePadBeforeDefinition(
    TensorView* tv,
    const std::unordered_set<Val*>& pad_dependencies,
    std::vector<PadOp*>& stack,
    std::unordered_set<PadOp*> simple_pad_set, ) {
  std::vector<Val*> padded_inputs;
  // stop propagation if current PadOp p isn't the only use of tv, since
  // it requires tv to be live in the fusion.
  if (tv->uses().size() != 1) {
    return padded_inputs;
  }

  Expr* expr = tv->definition();
  // stop propagation if any of expr's inputs are not TensorView, which we
  // cannot pad.
  if (std::any_of(expr->inputs().begin(), expr->inputs().end(), [](Val* val) {
        return !val->isA<TensorView>();
      })) {
    return padded_inputs;
  }

  // stop propagation if moving pad before definition would create cycles
  NVF_ERROR(
      expr->outputs().size() == 1,
      "expects tv to be the only output from its definition")
  if (pad_dependencies.count(tv) > 0) {
    return padded_inputs;
  }

  PadOp* p = tv->uses()[0]->as<PadOp>();
  padded_inputs.reserve(expr->inputs().size());
  std::transform(
      expr->inputs().begin(),
      expr->inputs().end(),
      std::back_inserter(padded_inputs),
      [&p, &stack, &simple_pad_set](TensorView* val) {
    TensorView new_pad_in = replayConcretePad(
        val,
        p->value(),
        {p->getPadWidths()},
        TensorDomain::noReductions(
            p->out()->as<TensorView>()->getLogicalDomain()));
    new_pad_op = new_pad_in->definition()->as<PadOp>();
    stack.push_back(new_pad_op);
    simple_pad_set.insert(new_pad_op);
    return new_pad_in;
      }
  return padded_inputs;
}

} // namespace

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
// This function only propagates simple padding, where its pad value is `zero`
// (or `false` for boolean) and pad widths are non-negative. This allows us to
// unconditionally merge neighboring PadOps as a single op. We also restrict
// propagation on operations that can allow PadOp to propagated across. See
// `zeroIsFixedPoint` and `zeroIsIdentity`.
void propagatePads(Fusion* fusion) {
  // propagating PadOp
  auto exprs = fusion->exprs();
  auto filtered_pads = ir_utils::filterByType<PadOp>(exprs);
  std::vector<PadOp*> stack;
  stack.reserve(filtered_pads.size());

  // NOTE: we only consider simple padop as propagation stack.
  std::copy_if(
      filtered_pads.begin(),
      filtered_pads.end(),
      std::back_inserter(stack),
      isSimplePadOp);

  // NOTE: this is a WAR. We use a set of `simple_pad_set` to track all mergable
  // PadOps, this is to leverage the assumption that all PadOps before CatOps
  // are simple op, but might not be able to be evaluated as such during compile
  // time.
  std::unordered_set<PadOp*> simple_pad_set(stack.begin(), stack.end());

  while (!stack.empty()) {
    PadOp* p = stack.back();
    stack.pop_back();

    // if no uses, this has already been short-wired.
    if (p->out()->uses().empty() && !p->out()->isFusionOutput()) {
      continue;
    }

    // unify all consumer pad of tv;
    auto* tv = p->in()->as<TensorView>();
    for (Expr* use : tv->uses()) {
      if (use == p) {
        continue;
      }
      // check if use is the same pad operation (same pad value / width e.t.c.)
      if (isSamePadOp(use, p)) {
        // replace consumer of use->out() with p->out()
        ir_utils::replaceValInAllExprInputsAndFusionOutputs(
            use->output(0), p->out());
        fusion->removeExpr(use);
      }
    }

    // if tv is fusion output, we need to keep tv alive, it might render
    // propagating PadOp before tv->definition() being non-optimal.
    if (tv->isFusionOutput()) {
      continue;
    }

    // check for pad_dependencies to verify that 'p' can be moved before 'def'.
    std::unordered_set<Val*> pad_inputs;
    for (Val* val : p->inputs()) {
      if (val == p->in() || val->isConst()) {
        continue;
      }
      pad_inputs.insert(val);
    }
    std::unordered_set<Val*> pad_dependencies =
        DependencyCheck::getAllDependentVals(pad_inputs);
    auto pad_replay_check = [&pad_dependencies](TensorView* tv) {
      // stop propagation if current PadOp p isn't the only use of tv, since
      // it requires tv to be live in the fusion.
      if (tv->uses().size() != 1) {
        return false;
      }
      Expr* expr = tv->definition();
      return std::all_of(
                 expr->inputs().begin(),
                 expr->inputs().end(),
                 [](Val* val) { return val->isA<TensorView>(); }) &&
          std::none_of(
                 expr->outputs().begin(),
                 expr->outputs().end(),
                 [&pad_dependencies = std::as_const(pad_dependencies)](
                     Val* val) { return pad_dependencies.count(val) > 0; });
    };

    auto pad_replay = [&p, &stack, &simple_pad_set](std::vector<Val*> vals) {
      std::vector<TensorView*> res;
      res.reserve(vals.size());
      std::transform(
          vals.begin(),
          vals.end(),
          std::back_inserter(res),
          [&p](TensorView* val) {
        TensorView new_pad_out = replayConcretePad(
            val,
            p->value(),
            {p->getPadWidths()},
            TensorDomain::noReductions(
                p->out()->as<TensorView>()->getLogicalDomain()));
        new_pad_op = new_pad_out->definition()->as<PadOp>();
        stack.push_back(new_pad_op);
        simple_pad_set.insert(new_pad_op);
        return new_pad_out;
          }
      return res;
    };

    Expr* def = p->in()->definition();
    Val* new_out = nullptr;

    if (auto* uop = dynamic_cast<UnaryOp*>(def)) {
      // check if unary op type is compatible for zero pad propagation.
      if (!zeroIsFixedPoint(uop->getUnaryOpType())) {
        continue;
      }
      std::vector<Val*> new_pad_inputs = maybeMovePadBeforeDefinition(
          tv, pad_dependencies, std::ref(stack), std::ref(simple_pad_set));
      // stop when move pad fails.
      if (new_pad_inputs.empty()) {
        continue;
      }
      // update new outputs.
      new_out =
          ops::newValLike(new_pad_inputs[0], uop->out()->getDataType().value());
      IrBuilder::create<UnaryOp>(
          uop->getUnaryOpType(), new_out, new_pad_inputs[0]);
    } else if (auto* bop = dynamic_cast<BinaryOp*>(def)) {
      // check if unary op type is compatible for zero pad propagation.
      if (!zeroIsIdentity(bop->getBinaryOpType())) {
        continue;
      }
      // check for broadcast on padded axis.
      auto* lhs = bop->lhs()->as<TensorView>();
      auto* rhs = bop->rhs()->as<TensorView>();
      std::vector<int64_t> padded_axes = p->getPaddedAxes();
      // padding on broadcast dimensions stops pad propagation.
      if (std::any_of(
              padded_axes.begin(), padded_axes.end(), [&lhs, &rhs](int64_t i) {
        return lhs->getLogicalDomain()[i]->isBroadcast() || rhs->getLogicalDomain()[i]->isBroadcast());
              })) {
        continue;
      }

      std::vector<Val*> new_pad_inputs = maybeMovePadBeforeDefinition(
          tv, pad_dependencies, std::ref(stack), std::ref(simple_pad_set));
      // stop when move pad fails.
      if (new_pad_inputs.empty()) {
        continue;
      }

      new_out =
          ops::newOutputTV(new_pad_inputs, bop->out()->getDataType().value());
      IrBuilder::create<BinaryOp>(
          bop->getBinaryOpType(),
          new_out,
          new_pad_inputs[0],
          new_pad_inputs[1]);
      // insert new PadOp(s) to stack;
    } else if (auto* pop = dynamic_cast<PadOp*>(def)) {
      // stop propagation if PadOp `pop` isn't a simple PadOp, since we can
      // only merge simple PadOp together. Note that we don't need to check
      // the other uses of `tv` here, since we want to merge the consecutive
      // pads anyway and it won't interfere the other uses of `tv`.
      if (simple_pad_set.count(pop) == 0) {
        continue;
      }

      // replay merged pad on pop->in()
      new_out = replayConcretePad(
          pop->in()->as<TensorView>(),
          pop->value(),
          {pop->getPadWidths(), p->getPadWidths()},
          TensorDomain::noReductions(
              p->out()->as<TensorView>()->getLogicalDomain()));
      // insert new PadOp(s) to stack;
      stack.push_back(new_out->definition()->as<PadOp>());
      simple_pad_set.insert(new_out->definition()->as<PadOp>());
    } else if (auto* cat = dynamic_cast<CatOp*>(def)) {
      // TODO: can cat support broadcast on any non-cat dimensions? Otherwise
      // we need to ensure that we are not padding on broadcast dimensions
      // like binary op

      // check if PadOp can be replayed on input(s)
      std::vector<Val*> new_pad_inputs = maybeMovePadBeforeDefinition(
          tv, pad_dependencies, std::ref(stack), std::ref(simple_pad_set));
      // stop when move pad fails.
      if (new_pad_inputs.empty()) {
        continue;
      }

      new_out = replaceCatOpWithBinaryOp(new_pad_inputs);
    }
    // replace old (->pad->) with (->pads_before_new_def->new_def->)
    if (new_out != nullptr) {
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(p->out(), new_out);
    }
  }
}

// clean up to fix CatOp which could be invalid after its producer PadOp has
// been altered by the pass.
void replaceCat(Fusion* fusion) {
  // updating CatOp
  std::vector<Expr*> exprs = fusion->exprs();

  // sanitizing CatOp with series of binary add
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    if (std::any_of(cat->inputs().begin(), cat->inputs().end(), [](Val* val) {
          return !val->definition()->isA<PadOp>();
        })) {
      Val* res = replaceCatOpWithBinaryOp(cat->inputs())

          // replace `CatOp` with the replay result.
          ir_utils::replaceValInAllExprInputsAndFusionOutputs(
              cat->output(0), res);
    }
  }
}

} // namespace

void MovePadPass::runPass(Fusion* fusion) {
  propagatePads(fusion);
  replaceCat(fusion);
}

} // namespace nvfuser::preseg_passes
