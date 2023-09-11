// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <expr_simplifier.h>
#include <fusion.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <ops/arith.h>

#include <limits>
#include <set>

namespace nvfuser::ir_utils {

std::vector<int64_t> normalizeNew2Old(
    const std::vector<int64_t>& new2old_in,
    size_t ndims) {
  NVF_CHECK(
      new2old_in.size() == ndims,
      "There must be a transpose mapping for each dimension in domain");

  // Canonicalize dimensions by wrapping each dim for the given ndims
  std::vector<int64_t> new2old;
  std::transform(
      new2old_in.begin(),
      new2old_in.end(),
      std::inserter(new2old, new2old.begin()),
      [ndims](int64_t entry) { return entry < 0 ? entry + ndims : entry; });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid
  NVF_CHECK(
      std::none_of(
          new2old.begin(),
          new2old.end(),
          [ndims](int64_t entry) {
            return entry < 0 || (unsigned int)entry >= ndims;
          }),
      "New2Old axes are not within the number of dimensions of the provided domain.\t",
      new2old);

  // Going to use sets, to see if any duplicate values are in the map.
  std::set<int64_t> old_pos_set;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](int64_t entry) { return entry; });

  // Error out if duplicate values are found.
  NVF_CHECK(
      new2old.size() == ndims && old_pos_set.size() == new2old.size(),
      "Duplicate entries in transformation map.");

  // END VALIDATION CHECKS
  return new2old;
}

std::vector<int> normalizeOld2New(
    const std::unordered_map<int, int>& old2new_in,
    size_t ndims) {
  // adjust based on negative values (any negative values gets nDims added to
  // it)
  std::unordered_map<int, int> old2new;
  std::transform(
      old2new_in.begin(),
      old2new_in.end(),
      std::inserter(old2new, old2new.begin()),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return std::unordered_map<int, int>::value_type({
            entry.first < 0 ? entry.first + ndims : entry.first,
            entry.second < 0 ? entry.second + ndims : entry.second,
        });
      });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid

  NVF_CHECK(
      std::none_of(
          old2new.begin(),
          old2new.end(),
          [ndims](std::unordered_map<int, int>::value_type entry) {
            return entry.first < 0 || (unsigned int)entry.first >= ndims ||
                entry.second < 0 || (unsigned int)entry.second >= ndims;
          }),
      "Reorder axes are not within the number of dimensions of the provided domain.");

  // Going to use sets, to see if any duplicate values are in the map.

  std::set<int> old_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  std::set<int> new_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(new_pos_set, new_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.second;
      });

  // Error out if duplicate values are found.
  NVF_CHECK(
      old_pos_set.size() == old2new.size() &&
          new_pos_set.size() == old2new.size(),
      "Duplicate entries in transformation map sent to TensorView reorder.");

  // END VALIDATION CHECKS

  std::vector<int> new2old(ndims, -1);

  // Go through each old and new position, make sure they're within [0, ndims)
  for (std::pair<int, int> elem : old2new) {
    int old_pos = elem.first;
    int new_pos = elem.second;
    new2old[new_pos] = old_pos;
  }

  // old_positions that already have a new position
  std::set<int> old_positions(new2old.begin(), new2old.end());
  old_positions.erase(-1);

  // All available new positions
  std::set<int> all_positions;
  for (decltype(ndims) i{0}; i < ndims; i++)
    all_positions.insert((int)i);

  // Check what positions haven't been specified.
  std::set<int> positions_left;
  std::set_difference(
      all_positions.begin(),
      all_positions.end(),
      old_positions.begin(),
      old_positions.end(),
      std::inserter(positions_left, positions_left.end()));

  // Fill in positions that weren't specified, in relative order,
  // in empty spots in the set of new positions.
  // new2old[new_position] = old_position
  auto it = positions_left.begin(); // old positions left
  std::transform(
      new2old.begin(), new2old.end(), new2old.begin(), [&it](int i) -> int {
        return i == -1 ? *it++ : i;
      });

  return new2old;
}

namespace ValReplacement {
// Create New Expr given producer - [an input for the expression]
// Creates a new Expr substituting current with producer
struct SubstituteInExpr : public OptOutMutator {
 public:
  static Expr* subsitute(Expr* expr, Val* reference, Val* substitute) {
    NVF_ERROR(
        expr != nullptr && reference != nullptr && substitute != nullptr,
        "Nullptr arg found.");
    SubstituteInExpr sie(reference, substitute);
    sie.mutate(expr);
    // if nothing substituted, then return the original expr
    return sie.expr_ == nullptr ? expr : sie.expr_;
  }

 protected:
  void removeExpr(IrContainer*, Expr*) const override {}

  void registerNewExpr(Expr* expr) override {
    expr_ = expr;
  }

 private:
  explicit SubstituteInExpr(Val* reference, Val* substitute) {
    mutations_[reference] = substitute;
  }

 private:
  Expr* expr_ = nullptr;
};

} // namespace ValReplacement

Expr* replaceValInExpr(Expr* expr, Val* reference, Val* substitute) {
  FusionGuard fg(expr->fusion());
  return ValReplacement::SubstituteInExpr::subsitute(
      expr, reference, substitute);
}

TensorView* rfactorHelper(
    TensorView* reduction_tv,
    const std::vector<int>& axes) {
  NVF_ERROR(reduction_tv->definition() != nullptr);
  const bool has_multiple_tvs = reduction_tv->definition()->inputs().size() > 1;
  if (!has_multiple_tvs) {
    return reduction_tv->rFactor(axes);
  }

  std::vector<TensorView*> out_tvs;
  std::transform(
      reduction_tv->definition()->outputs().begin(),
      reduction_tv->definition()->outputs().end(),
      std::back_inserter(out_tvs),
      [](Val* val) { return val->as<TensorView>(); });

  auto rf_tvs = reduction_tv->rFactor(axes, out_tvs);

  return rf_tvs.at(std::distance(
      out_tvs.begin(),
      std::find(out_tvs.begin(), out_tvs.end(), reduction_tv)));
}

namespace {

template <typename T>
std::vector<T*> uniqueEntries(const std::vector<T*>& tv_deuqe) {
  std::vector<T*> unique_entries;
  std::unordered_set<T*> inserted;
  for (auto tv_entry : tv_deuqe) {
    if (inserted.emplace(tv_entry).second) {
      unique_entries.emplace_back(tv_entry);
    }
  }
  return unique_entries;
}

} // namespace

// Return immediate producers of val
std::vector<Val*> producerValsOf(const Val* val) {
  if (val->definition() == nullptr) {
    return {};
  }
  auto producer_vals = val->definition()->inputs();
  return uniqueEntries<Val>({producer_vals.begin(), producer_vals.end()});
}

// Return immediate consumers of val
std::vector<Val*> consumerValsOf(const Val* val) {
  std::vector<Val*> consumer_vals;
  for (auto use_expr : val->uses()) {
    auto outputs = use_expr->outputs();
    consumer_vals.insert(consumer_vals.end(), outputs.begin(), outputs.end());
  }
  return uniqueEntries<Val>(consumer_vals);
}

// Return immediate siblings of val
std::vector<Val*> siblingValsOf(const Val* val) {
  std::vector<Val*> sibling_vals;
  auto def = val->definition();
  if (def != nullptr) {
    auto outs = def->outputs();
    for (auto sibling_val : outs) {
      if (sibling_val == val) {
        continue;
      }
      sibling_vals.emplace_back(sibling_val);
    }
  }
  return sibling_vals;
}

// Return immediate producers of val
std::vector<Val*> producerValsOf(const std::vector<Val*>& vals) {
  std::vector<Val*> all_producer_vals;
  for (auto val : vals) {
    auto producer_vals = producerValsOf(val);
    all_producer_vals.insert(
        all_producer_vals.end(), producer_vals.begin(), producer_vals.end());
  }

  return uniqueEntries<Val>(all_producer_vals);
}

// Return immediate consumers of val
std::vector<Val*> consumerValsOf(const std::vector<Val*>& vals) {
  std::vector<Val*> all_consumer_vals;
  for (auto val : vals) {
    auto consumer_vals = consumerValsOf(val);
    all_consumer_vals.insert(
        all_consumer_vals.end(), consumer_vals.begin(), consumer_vals.end());
  }

  return uniqueEntries<Val>(all_consumer_vals);
}

std::vector<TensorView*> producerTvsOf(const TensorView* tv) {
  auto producer_vals = producerValsOf(tv);
  auto producer_tvs = ir_utils::filterByType<TensorView>(producer_vals);
  return {producer_tvs.begin(), producer_tvs.end()};
}

std::vector<TensorView*> consumerTvsOf(const TensorView* tv) {
  auto consumer_vals = consumerValsOf(tv);
  auto consumer_tvs = ir_utils::filterByType<TensorView>(consumer_vals);
  return {consumer_tvs.begin(), consumer_tvs.end()};
}

std::vector<TensorView*> siblingTvsOf(const TensorView* tv) {
  auto sibling_vals = siblingValsOf(tv);
  auto sibling_tvs = ir_utils::filterByType<TensorView>(sibling_vals);
  return {sibling_tvs.begin(), sibling_tvs.end()};
}

std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_producer_tvs;
  for (auto tv : tvs) {
    auto producer_tvs = producerTvsOf(tv);
    all_producer_tvs.insert(
        all_producer_tvs.end(), producer_tvs.begin(), producer_tvs.end());
  }

  return uniqueEntries<TensorView>(all_producer_tvs);
}

std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs) {
  std::vector<TensorView*> all_consumer_tvs;
  for (auto tv : tvs) {
    auto consumer_tvs = consumerTvsOf(tv);
    all_consumer_tvs.insert(
        all_consumer_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
  }

  return uniqueEntries<TensorView>(all_consumer_tvs);
}

std::vector<TensorView*> inputTvsOf(TensorView* tv) {
  return inputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> outputTvsOf(TensorView* tv) {
  return outputTvsOf(std::vector<TensorView*>{tv});
}

std::vector<TensorView*> inputTvsOf(std::vector<TensorView*> tvs) {
  auto inp_vals = IterVisitor::getInputsTo({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(inp_vals);
  std::vector<TensorView*> inp_tvs(filtered.begin(), filtered.end());
  return uniqueEntries<TensorView>(inp_tvs);
}

std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs) {
  auto out_vals = DependencyCheck::getAllOutputsOf({tvs.begin(), tvs.end()});
  auto filtered = ir_utils::filterByType<TensorView>(out_vals);
  std::vector<TensorView*> out_tvs(filtered.begin(), filtered.end());
  return uniqueEntries<TensorView>(out_tvs);
}

std::vector<TensorView*> allTvs(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);

  // This shouldn't be necessary but FusionSegmentIoAlias_CUDA due to aliasing
  // is having an input disconnected from outputs, and these iter domains are
  // being checked in compute at maps in scheduling logic. This shouldn't hurt
  // AFAICT.
  auto tv_inputs = ir_utils::filterByType<TensorView>(fusion->inputs());

  std::vector<TensorView*> all_tvs({used_tvs.begin(), used_tvs.end()});
  // Sometimes inputs are not connected to outputs, however, we still include
  // them when returning allTvs because they are registered as an input.
  all_tvs.insert(all_tvs.end(), tv_inputs.begin(), tv_inputs.end());

  // all_tvs has duplicates, to deduplicate it and return
  return uniqueEntries<TensorView>(all_tvs);
}

std::vector<TensorView*> allTvsExcept(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& except) {
  auto all_tvs = allTvs(fusion);
  std::vector<TensorView*> result;
  for (auto tv : all_tvs) {
    if (except.count(tv) == 0) {
      result.emplace_back(tv);
    }
  }
  return result;
}

std::vector<Expr*> getReductionOps(Fusion* fusion) {
  std::vector<Expr*> red_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<ReductionOp>() || expr->isA<GroupedReductionOp>() ||
        expr->isA<WelfordOp>()) {
      red_ops.push_back(expr);
    }
  }

  return red_ops;
}

std::vector<IndexSelectOp*> getIndexSelectOps(Fusion* fusion) {
  std::vector<IndexSelectOp*> idx_sel_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<IndexSelectOp>()) {
      idx_sel_ops.push_back(expr->as<IndexSelectOp>());
    }
  }

  return idx_sel_ops;
}

std::vector<TorchGatherOp*> getTorchGatherOps(Fusion* fusion) {
  std::vector<TorchGatherOp*> torch_gather_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<TorchGatherOp>()) {
      torch_gather_ops.push_back(expr->as<TorchGatherOp>());
    }
  }

  return torch_gather_ops;
}

std::vector<SelectOp*> getSelectOps(Fusion* fusion) {
  std::vector<SelectOp*> select_ops;

  for (auto expr : fusion->exprs()) {
    if (expr->isA<SelectOp>()) {
      select_ops.push_back(expr->as<SelectOp>());
    }
  }

  return select_ops;
}

std::vector<MmaOp*> getMmaOps(Fusion* fusion) {
  std::vector<MmaOp*> mma_ops;
  for (auto expr : fusion->exprs()) {
    if (expr->isA<MmaOp>()) {
      mma_ops.push_back(expr->as<MmaOp>());
    }
  }

  return mma_ops;
}

namespace {

class ValReplacementMutator : private OptOutMutator {
 public:
  ValReplacementMutator(
      Fusion* fusion,
      const std::unordered_map<Val*, Val*>& replacement_map)
      : replacement_map_(replacement_map) {
    FusionGuard fg(fusion);

    // Welford makes this a little annoying since it holds a count which is
    // typically not used by anything else. If we don't grab that count, then it
    // would be a tensorview that doesn't get updated extents. Therefore, first
    // grab all leaves towards outputs and grab stmts from there.
    auto stmts = StmtSort::getStmtsTo(fusion, allLeafOuts(fusion), true, true);

    // Some fusions, such as standalone rand_like, can have disconnected DAG, so
    // we need some mechanism to make sure our replacement set is as complete as
    // possible
    // TODO: I think we need a more general mechanism to support disconnected
    // DAG
    std::vector<Val*> more;
    for (auto v : fusion->inputs()) {
      if (std::find(stmts.begin(), stmts.end(), v) == stmts.end()) {
        more.emplace_back(v);
      }
    }
    for (auto v : fusion->axioms()) {
      if (std::find(stmts.begin(), stmts.end(), v) == stmts.end()) {
        more.emplace_back(v);
      }
    }
    auto more_stmts = StmtSort::getStmtsTo(fusion, more, true, true);
    more_stmts.insert(more_stmts.end(), stmts.begin(), stmts.end());

    for (auto stmt : more_stmts) {
      dispatchMutate(stmt);
    }
  }

 private:
  using OptOutMutator::dispatchMutate;
  using OptOutMutator::mutate;

  void dispatchMutate(Val* val) final {
    if (replacement_map_.find(val) == replacement_map_.end()) {
      return OptOutMutator::dispatchMutate(val);
    }
    auto replaced_val = replacement_map_.at(val);
    registerMutation(val, replaced_val);
  }

  std::vector<Val*> allLeafOuts(Fusion* fusion) {
    auto exprs = StmtSort::getExprs(fusion, true);
    std::unordered_set<Val*> inputs;
    std::unordered_set<Val*> outputs;
    std::vector<Val*> ordered_outputs;
    for (auto expr : exprs) {
      inputs.insert(expr->inputs().begin(), expr->inputs().end());
      outputs.insert(expr->outputs().begin(), expr->outputs().end());
      ordered_outputs.insert(
          ordered_outputs.end(),
          expr->outputs().begin(),
          expr->outputs().end());
    }
    for (auto input : inputs) {
      outputs.erase(input);
    }

    std::vector<Val*> ordered_leaf_outs;
    for (auto out : ordered_outputs) {
      if (outputs.find(out) != outputs.end()) {
        ordered_leaf_outs.push_back(out);
      }
    }
    return ordered_leaf_outs;
  }

  const std::unordered_map<Val*, Val*>& replacement_map_;
};

} // namespace

void replaceValue(
    Fusion* fusion,
    const std::unordered_map<Val*, Val*>& replacement_map) {
  ValReplacementMutator(fusion, replacement_map);
}

Val* getReductionInitValOf(TensorView* tv) {
  auto def = tv->definition();
  if (def == nullptr) {
    return nullptr;
  }

  Val* init = nullptr;
  if (auto rop = dynamic_cast<ReductionOp*>(def)) {
    init = rop->init();
  } else if (auto grop = dynamic_cast<GroupedReductionOp*>(def)) {
    int output_idx = grop->getExprIndexOfOutput(tv);
    init = grop->initVal(output_idx);
  } else if (auto wop = dynamic_cast<WelfordOp*>(def)) {
    return wop->getInitValOfOutput(tv);
  } else if (auto gwop = dynamic_cast<GroupedWelfordOp*>(def)) {
    init = gwop->getInitValOfOutput(tv);
  } else if (auto mma = dynamic_cast<MmaOp*>(def)) {
    init = mma->init();
  }

  return init;
}

// TODO: Should mma be in here? Should we return true if it's a trivial
// reduction?
bool isReductionOp(const Expr* expr) {
  // Note that GridReduction inherits ReductionOp
  return expr->isOneOf<
      ReductionOp,
      GroupedReductionOp,
      WelfordOp,
      GroupedWelfordOp,
      kir::GridWelford,
      kir::GroupedGridWelford>();
}

bool isReductionTvOp(const Expr* expr) {
  return ir_utils::isTvOp(expr) && isReductionOp(expr);
}

bool isPointwiseTvOp(const Expr* expr) {
  // LoadStoreOp with rfactor domain means transpose, which is not
  // considered pointwise
  return isTvOp(expr) &&
      (expr->isOneOf<UnaryOp, BinaryOp, TernaryOp>() ||
       (expr->isA<LoadStoreOp>() &&
        !ir_utils::getTvOutput(expr)->hasRFactor()));
}

std::vector<ViewOp*> getViewOps(Fusion* fusion) {
  auto all_exprs = fusion->exprs();

  auto all_view_ops = ir_utils::filterByType<ViewOp>(all_exprs);

  std::vector<ViewOp*> view_ops;

  std::copy_if(
      all_view_ops.begin(),
      all_view_ops.end(),
      std::back_inserter(view_ops),
      [](ViewOp* view) {
        return std::any_of(
            view->outputs().begin(), view->outputs().end(), [](Val* v) {
              if (!v->isA<TensorView>()) {
                return false;
              }
              return v->as<TensorView>()->hasRFactor();
            });
      });

  return view_ops;
}

Val* replaceValRecursively(
    Val* val,
    const std::unordered_map<Val*, Val*>& replacement_map) {
  if (replacement_map.find(val) != replacement_map.end()) {
    return replacement_map.at(val);
  }

  auto def = val->definition();
  if (def == nullptr) {
    return val;
  }

  NVF_ERROR(def->outputs().size() == 1);

  bool mutated = false;

  std::vector<Val*> mutated_inputs;
  mutated_inputs.reserve(def->inputs().size());
  for (auto input : def->inputs()) {
    auto new_input = replaceValRecursively(input, replacement_map);
    if (new_input != input) {
      mutated = true;
    }
    mutated_inputs.emplace_back(new_input);
  }

  std::vector<Statement*> mutated_attrs;
  mutated_attrs.reserve(def->attributes().size());
  for (auto attr : def->attributes()) {
    if (auto attr_val = dynamic_cast<Val*>(attr)) {
      auto new_attr_val = replaceValRecursively(attr_val, replacement_map);
      if (new_attr_val != attr_val) {
        mutated = true;
      }
      mutated_attrs.emplace_back(new_attr_val);
    } else {
      mutated_attrs.emplace_back(attr);
    }
  }

  if (!mutated) {
    return val;
  }

  auto out = IrBuilder::create<Val>(val->dtype());
  auto newObjectFunc = def->newObjectFunc();
  newObjectFunc(def->container(), mutated_inputs, {out}, mutated_attrs);

  return out;
}

bool isSqueezeInput(const TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<SqueezeOp>()) {
      return true;
    }
  }
  return false;
}

bool isSqueezedID(const TensorView* tv, const IterDomain* id) {
  auto root_dom = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  auto squeezes = ir_utils::filterByType<SqueezeOp>(tv->uses());
  for (auto i : c10::irange(root_dom.size())) {
    if (root_dom[i] != id) {
      continue;
    }
    for (auto squeeze : squeezes) {
      if (squeeze->isSqueezeDim(i)) {
        return true;
      }
    }
  }
  return false;
}

bool isIndexedID(const TensorView* tv, const IterDomain* id) {
  return isIndexedProducerID(tv, id) || isIndexedConsumerID(tv, id);
}

bool isIndexedProducerID(const TensorView* tv, const IterDomain* id) {
  return std::any_of(tv->uses().begin(), tv->uses().end(), [&](Expr* expr) {
    return getIndexedProducerID(expr) == id;
  });
}

IterDomain* getIndexedProducerID(const Expr* expr) {
  if (auto select = dynamic_cast<const SelectOp*>(expr)) {
    return select->getIndexedID();
  } else if (auto index_select = dynamic_cast<const IndexSelectOp*>(expr)) {
    return index_select->getIndexedID();
  } else if (auto gather = dynamic_cast<const TorchGatherOp*>(expr)) {
    return gather->getIndexedID();
  } else {
    return nullptr;
  }
}

IterDomain* getConsumerOfIndexedProducerID(const Expr* expr) {
  if (auto index_select = dynamic_cast<const IndexSelectOp*>(expr)) {
    return index_select->getConsumerOfIndexedID();
  } else if (auto gather = dynamic_cast<const TorchGatherOp*>(expr)) {
    return gather->getConsumerOfIndexedID();
  } else {
    return nullptr;
  }
}

bool isIndexedConsumerID(const TensorView* tv, const IterDomain* id) {
  return tv->definition()->isA<ScatterOp>() &&
      tv->definition()->as<ScatterOp>()->getIndexedID() == id;
}

std::vector<IterDomain*> allIDsOf(const TensorView* tv) {
  const auto& root_domain = tv->getRootDomain();
  const auto& domain = tv->getLeafDomain();
  // Grab all values in the history of the tensor view's domain
  auto all_vals = DependencyCheck::getAllValsBetween(
      {root_domain.begin(), root_domain.end()}, {domain.begin(), domain.end()});

  // Filter so we only have iteration domains (ignore Ints used in split)
  auto all_ids = ir_utils::filterByType<IterDomain>(all_vals);
  return std::vector<IterDomain*>(all_ids.begin(), all_ids.end());
}

bool isSelectInput(TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<SelectOp>()) {
      return true;
    }
  }
  return false;
}

bool isIndexSelectLookupTv(const TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<IndexSelectOp>()) {
      auto idx_sel = expr->as<IndexSelectOp>();
      if (idx_sel->input(0) == tv) {
        return true;
      }
    }
  }
  return false;
}

bool isIndexSelectIndicesTv(const TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<IndexSelectOp>()) {
      auto idx_sel = expr->as<IndexSelectOp>();
      if (idx_sel->input(1) == tv) {
        return true;
      }
    }
  }
  return false;
}

bool isTorchGatherLookupTv(const Val* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isA<TorchGatherOp>()) {
      auto idx_sel = expr->as<TorchGatherOp>();
      if (idx_sel->lookupTv() == tv) {
        return true;
      }
    }
  }
  return false;
}

std::string varName(const Val* val) {
  if (val->isA<kir::TensorIndex>()) {
    return varName(val->as<kir::TensorIndex>()->view());
  }
  std::stringstream name;
  if (val->isA<TensorView>()) {
    name << "T";
  } else {
    name << typePrefix(val->dtype());
  }
  name << val->name();
  return name.str();
}

bool hasResizedRfactor(const TensorView* tv) {
  if (!tv->hasRFactor()) {
    return false;
  }
  auto root_to_rf_exprs = StmtSort::getExprsBetween(
      tv->fusion(),
      {tv->getRootDomain().begin(), tv->getRootDomain().end()},
      {tv->getRFactorDomain().begin(), tv->getRFactorDomain().end()});
  return std::any_of(
      root_to_rf_exprs.begin(), root_to_rf_exprs.end(), [](Expr* expr) {
        return expr->isA<Resize>();
      });
}

std::vector<TensorView*> getTVsWithDynamicTransform(Fusion* fusion) {
  const auto all_tvs = ir_utils::allTvs(fusion);
  std::vector<TensorView*> dynamic_tvs;
  std::copy_if(
      all_tvs.begin(),
      all_tvs.end(),
      std::back_inserter(dynamic_tvs),
      [](auto tv) { return tv->domain()->hasSymbolicAxis(); });
  return dynamic_tvs;
}

namespace {

class ValidateDomainEquivalence : private IterVisitor {
 public:
  ValidateDomainEquivalence(
      const std::vector<IterDomain*>& initial_domain,
      const std::vector<IterDomain*>& derived_domain)
      : initial_domain_({initial_domain.begin(), initial_domain.end()}),
        derived_domain_({derived_domain.begin(), derived_domain.end()}),
        frontier_({initial_domain.begin(), initial_domain.end()}) {
    NVF_ERROR(!initial_domain.empty());
    NVF_ERROR(!derived_domain.empty());
    // Make sure there's no duplicate in the parameter vectors
    NVF_ERROR(
        initial_domain.size() == initial_domain_.size(),
        "Duplicated entry is detected in inial_domain: ",
        toDelimitedString(initial_domain));
    NVF_ERROR(
        derived_domain.size() == derived_domain_.size(),
        "Duplicated entry is detected in derived_domain: ",
        toDelimitedString(derived_domain));

    traverseBetween(
        initial_domain.at(0)->fusion(),
        {initial_domain.begin(), initial_domain.end()},
        {derived_domain.begin(), derived_domain.end()});

    // At this point, the frontier set and the derived set should be
    // equal, except when there's a symbolic ID in the derived set,
    // where the traversal may be incomplete.
    if (std::any_of(derived_domain.begin(), derived_domain.end(), [](auto id) {
          return id->getIterType() == IterType::Symbolic;
        })) {
      // Make sure all non-symbolic IDs of the derived set are included
      // in the frontier set
      NVF_ERROR(
          std::all_of(
              derived_domain.begin(),
              derived_domain.end(),
              [&](auto id) {
                return id->getIterType() == IterType::Symbolic ||
                    frontier_.count(id);
              }),
          "Invalid derived domain. Initial domain: ",
          toDelimitedString(initial_domain),
          ". Derived domain: ",
          toDelimitedString(derived_domain));
      // Similarly, all frontier vals should be included in the
      // derived set. It is also possible that an ID in the initial
      // domain set still remains in the frontier set as there may be
      // no expr connecting to the derived set, e.g., dynamic reshape
      NVF_ERROR(
          std::all_of(
              frontier_.begin(),
              frontier_.end(),
              [&](Val* val) {
                NVF_ERROR(val->isA<IterDomain>());
                return derived_domain_.count(val->as<IterDomain>()) ||
                    initial_domain_.count(val);
              }),
          "Invalid derived domain. Initial domain: ",
          toDelimitedString(initial_domain),
          ". Derived domain: ",
          toDelimitedString(derived_domain));
    } else {
      NVF_ERROR(
          derived_domain_ == frontier_,
          "Invalid derived domain. Initial domain: ",
          toDelimitedString(initial_domain),
          ". Derived domain: ",
          toDelimitedString(derived_domain));
    }
  };

  void dispatch(Expr* expr) override {
    NVF_ERROR(
        std::all_of(expr->inputs().begin(), expr->inputs().end(), [](Val* v) {
          return v->isA<IterDomain>();
        }));
    NVF_ERROR(
        std::all_of(expr->outputs().begin(), expr->outputs().end(), [](Val* v) {
          return v->isA<IterDomain>();
        }));
    // If any of the inputs is included in derived_domain_, that means there's a
    // dependency within derived_domain_ and the dependent domains
    // redundantly cover the initial domain
    NVF_ERROR(
        std::none_of(
            expr->inputs().begin(),
            expr->inputs().end(),
            [&](Val* input_val) {
              return derived_domain_.find(input_val) != derived_domain_.end();
            }),
        "Invalid derived domain due to dependent expr: ",
        expr->toString(),
        ". Derived domain: ",
        toDelimitedString(derived_domain_));
    for (auto out : expr->outputs()) {
      // Make sure the output is not yet visited
      NVF_ERROR(
          frontier_.insert(out).second,
          "Invalid derived domain due to dependent expr: ",
          expr->toString(),
          ". Output should just show up once: ",
          out->toString());
    }
    for (auto inp : expr->inputs()) {
      NVF_ERROR(
          frontier_.erase(inp) == 1,
          "Invalid derived domain due to dependent expr: ",
          expr->toString(),
          ". Input not seen before: ",
          inp->toString());
    }
  }

 private:
  const std::unordered_set<Val*> initial_domain_;
  const std::unordered_set<Val*> derived_domain_;
  //! Traversal frontier vals
  std::unordered_set<Val*> frontier_;
};

std::vector<Statement*> next(Statement* stmt) {
  if (stmt->isVal()) {
    if (auto val = stmt->as<Val>()->definition()) {
      return {val};
    } else {
      return {};
    }
  } else {
    auto expr = stmt->as<Expr>();
    std::vector<Statement*> inputs{
        expr->inputs().begin(), expr->inputs().end()};
    return inputs;
  }
}

} // namespace

void validateDomainEquivalence(
    const std::vector<IterDomain*>& initial_domain,
    const std::vector<IterDomain*>& derived_domain) {
  ValidateDomainEquivalence(initial_domain, derived_domain);
}

std::vector<Statement*> checkCycle(
    Fusion* fusion,
    const std::unordered_set<Statement*>& from,
    const std::vector<Val*>& to) {
  std::unordered_set<Statement*> path;
  std::unordered_set<Statement*> visited;
  std::deque<Statement*> queue;
  queue.insert(queue.end(), to.begin(), to.end());

  while (!queue.empty()) {
    auto val = queue.front();

    // early termination if we have already reached boundary or hit a previously
    // visited node
    if (from.count(val) != 0 || visited.count(val) != 0) {
      queue.pop_front();
      continue;
    }

    auto next_stmts = next(val);

    // if val is a leaf node.
    if (next_stmts.empty()) {
      queue.pop_front();
      visited.insert(val);
      continue;
    }

    // if val is already in path, we are just cleaning up the stack here.
    auto iter = path.find(val);
    if (iter != path.end()) {
      queue.pop_front();
      path.erase(iter);
      visited.insert(val);
      continue;
    }

    // putting self on path
    path.insert(val);

    // check for cycles
    for (auto stmt : next_stmts) {
      if (path.count(stmt) != 0) {
        // find a cycle, return current path;
        std::vector<Statement*> ret;
        std::copy(path.begin(), path.end(), std::back_inserter(ret));
        return ret;
      }
      // adding statement to a queue;
      queue.push_front(stmt);
    }
  }

  // no cycle detected, return empty
  return {};
}

bool isAlignedScopeExpr(const Expr* expr) {
  NVF_ERROR(expr != nullptr);
  if (auto ite = dynamic_cast<const kir::IfThenElse*>(expr)) {
    if (ite->predicate()->hasValue() &&
        getRegisterType(ite->predicate()->value()) ==
            RegisterType::GeneralPurpose) {
      return false;
    }

  } else if (auto fl = dynamic_cast<const kir::ForLoop*>(expr)) {
    // If the start, stop, step are not thread dependent
    //  then this for loop should be thread independent.
    if (getRegisterType(fl->start()) == RegisterType::GeneralPurpose ||
        getRegisterType(fl->stop()) == RegisterType::GeneralPurpose ||
        getRegisterType(fl->step()) == RegisterType::GeneralPurpose) {
      return false;
    }
  } else {
    NVF_ERROR(false, "Invalid scope expr: ", expr->toString());
  }

  return true;
}

std::vector<Statement*> checkCycle(Fusion* fusion) {
  return checkCycle(fusion, {}, fusion->outputs());
}

namespace {

inline bool isTensorAttr(const Val* val, const std::string& attr_name) {
  NVF_ERROR(val != nullptr);
  auto getitem = dynamic_cast<GetItem*>(val->definition());
  if (getitem == nullptr) {
    return false;
  }
  auto getattr = dynamic_cast<GetAttr*>(getitem->array()->definition());
  if (getattr == nullptr) {
    return false;
  }
  if (getattr->attr() != attr_name) {
    return false;
  }
  auto metadata = dynamic_cast<GetMetaData*>(getattr->struct_()->definition());
  if (metadata == nullptr) {
    return false;
  }
  return metadata->in()->isA<TensorView>();
}

} // namespace

bool isTensorSize(const Val* val) {
  return isTensorAttr(val, "logical_size") || isTensorAttr(val, "alloc_size");
}

bool isTensorStride(const Val* val) {
  return isTensorAttr(val, "logical_stride") ||
      isTensorAttr(val, "alloc_stride");
}

} // namespace nvfuser::ir_utils
