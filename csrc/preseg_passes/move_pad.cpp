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
  if (!simplifyExpr(pad->value())->isZero() && !simplifyExpr(pad->value())->isFalse()) {
    return false;
  }
  // TODO: we cannot seem to always been able to prove that pad widths are >= 0. But we know that PadOp + CatOp implies such condition. Adding this short-cut to ensure that we recognized such pattern.
  if (pad->out()->uses().size() == 1 && pad->out()->uses()[0]->isA<CatOp>()) {
    return true;
  }
  for (Val* pad_val : pad->getPadWidths()) {
    if (!simplifyExpr(SimplifyingIrBuilder::geExpr(pad_val, pad->fusion()->zeroVal()))
            ->isTrue()) {
      return false;
    }
  }
  return true;
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
        !simplifyExpr(
            SimplifyingIrBuilder::eqExpr(
                use_pad->getPadWidths(idx).second, p->getPadWidths(idx).second))
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
bool zeroIsIdentity(BinaryOpType t) {
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
  NVF_ERROR(
      std::all_of(vec_pad_widths.begin(), vec_pad_widths.end(), [&rank](const std::vector<Val*>& pad_widths) { return pad_widths.size() == 2 * rank; }), "vec_pad_widths doesn't have compatible length for pad_tv");

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
            : add(merged_pad_width, pad_width);
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
        true,
        ref_iter_type.at(i)->getIterType()));
  }

  auto* new_out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          merged_root_ids, merged_logical_ids, merged_logical_ids),
      pad_tv->getDataType().value());
  IrBuilder::create<PadOp>(new_out, pad_tv, merged_pad_widths, SimplifyingIrBuilder::maybeCastExpr(pad_tv->getDataType().value(), pad_value));
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
// Given certain constraints on the UnaryOp, we can ensure that the computational output remains the same when we replace all uses of `tv2` with `tv2_new`.
//
// This function only propagates simple padding, where its pad value is `zero` (or `false` for boolean) and pad widths are non-negative. This allows us to unconditionally merge neighboring PadOps as a single op.
// We also restrict propagation on operatoins that can allow PadOp to propagated across. See `zeroIsFixedPoint` and `zeroIsIdentity`.
void propagatePad(Fusion* fusion) {
  // propagating PadOp
  auto exprs = fusion->exprs();
  auto filtered_pads = ir_utils::filterByType<PadOp>(exprs);
  std::vector<PadOp*> frontier;
  frontier.reserve(filtered_pads.size());

  // NOTE: we only consider simple padop as propagation frontier.
  std::copy_if(
      filtered_pads.begin(),
      filtered_pads.end(),
      std::back_inserter(frontier),
      isSimplePadOp);

  // NOTE: this is a WAR. We use a set of `simple_pad_set` to track all mergable PadOps, this is to leverage the assumption that all PadOps before CatOps are simple op, but might not be able to be evaluated as such during compile time.
  std::unordered_set<PadOp*> simple_pad_set(frontier.begin(), frontier.end());

  while (!frontier.empty()) {
    PadOp* p = frontier.back();
    frontier.pop_back();

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
        ir_utils::replaceValInAllExprInputsAndFusionOutputs(use->output(0), p->out());
        fusion->removeExpr(use);
      }
    }

    // if tv is fusion output, we need to keep tv alive, it might render propagating PadOp before tv->definition() being non-optimal.
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
    std::unordered_set<Val*> pad_dependencies = DependencyCheck::getAllDependentVals(pad_inputs);
    auto pad_replay_check = [&pad_dependencies](Expr* expr) {
      return std::all_of(
          expr->inputs().begin(),
          expr->inputs().end(),
          [&pad_dependencies = std::as_const(pad_dependencies)](Val* val) {
            if (!val->isA<TensorView>()) {
              return false;
            }
            if (pad_dependencies.count(val) > 0) {
              return false;
            }
            return true;
          });
    };

    Expr* def = p->in()->definition();
    Val* new_out = nullptr;

    if (auto* uop = dynamic_cast<UnaryOp*>(def)) {
      // stop propagation if current PadOp p isn't the only use of tv, since it requires tv to be live in the fusion.
      if (tv->uses().size() != 1) {
        continue;
      }
      NVF_ERROR(
          tv->uses()[0] == p,
          "expect existing PadOp to be the only use of its input");
      // check if unary op type is compatible for zero pad propagation.
      if (!zeroIsFixedPoint(uop->getUnaryOpType())) {
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
      simple_pad_set.insert(new_pad_out->definition()->as<PadOp>());
    } else if (auto* bop = dynamic_cast<BinaryOp*>(def)) {
      // stop propagation if current PadOp p isn't the only use of tv, since it requires tv to be live in the fusion.
      if (tv->uses().size() != 1) {
        continue;
      }
      NVF_ERROR(
          tv->uses()[0] == p,
          "expect existing PadOp to be the only use of its input");

      // check if unary op type is compatible for zero pad propagation.
      if (!zeroIsIdentity(bop->getBinaryOpType())) {
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
      simple_pad_set.insert(vals[0]->definition()->as<PadOp>());
      frontier.push_back(vals[1]->definition()->as<PadOp>());
      simple_pad_set.insert(vals[1]->definition()->as<PadOp>());
    } else if (auto* pop = dynamic_cast<PadOp*>(def)) {
      // stop propagation if PadOp `pop` isn't a simple PadOp, since we can only merge simple PadOp together.
      // Note that we don't need to check the other uses of `tv` here, since we want to merge the consecutive pads anyway and it won't interfere the other uses of `tv`.
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

      // insert new PadOp(s) to frontier;
      frontier.push_back(new_out->definition()->as<PadOp>());
      simple_pad_set.insert(new_out->definition()->as<PadOp>());
    } else if (auto* cat = dynamic_cast<CatOp*>(def)) {
      // stop propagation if current PadOp p isn't the only use of tv, since it requires tv to be live in the fusion.
      if (tv->uses().size() != 1) {
        continue;
      }
      // check if PadOp can be replayed on input(s)
      if (!pad_replay_check(uop)) {
        continue;
      }

      // TODO: can cat support broadcast on any non-cat dimensions? Otherwise we
      // need to ensure that we are not padding on broadcast dimensions like
      // binary op
      std::vector<Val*> vals;
      std::transform(
          cat->inputs().begin(),
          cat->inputs().end(),
          std::back_inserter(vals),
          [&p, &frontier, &simple_pad_set](Val* val) {
            Val* pad_out = replayConcretePad(
                val->as<TensorView>(),
                p->value(),
                {p->getPadWidths()},
                TensorDomain::noReductions(
                    p->out()->as<TensorView>()->getLogicalDomain()));
            frontier.push_back(pad_out->definition()->as<PadOp>());
            simple_pad_set.insert(pad_out->definition()->as<PadOp>());
            return pad_out;
          });

      new_out =
          replayCatOpWithBinaryOp(vals, cat->output(0)->getDataType().value());
    }
    // replace old (->pad->) with (->pads_before_new_def->new_def->)
    if (new_out != nullptr) {
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(p->out(), new_out);
    }
  }
}

// clean up to fix CatOp which could be invalid after its producer PadOp has been altered by the pass.
void replaceCat(Fusion* fusion) {
  // updating CatOp
  std::vector<Expr*> exprs = fusion->exprs();

  // sanitizing CatOp with series of binary add
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    if (std::any_of(cat->inputs().begin(), cat->inputs().end(), [](Val* val) {
          return !val->definition()->isA<PadOp>();
        })) {
      Val* res = replayCatOpWithBinaryOp(
          cat->inputs(), cat->output(0)->getDataType().value());

      // replace `CatOp` with the replay result.
      ir_utils::replaceValInAllExprInputsAndFusionOutputs(cat->output(0), res);
    }
  }
}

} // namespace

void MovePadPass::runPass(Fusion* fusion) {
  propagatePad(fusion);
  replaceCat(fusion);
}

} // namespace nvfuser::preseg_passes
