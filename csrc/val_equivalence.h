// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/Exception.h>

#include <expr_simplifier.h>
#include <ir_all_nodes.h>
#include <kernel_ir.h>
#include <ops/arith.h>
#include <type.h>
#include <union_find.h>
#include <utils.h>

#include <deque>
#include <iostream>
#include <typeinfo>
#include <vector>

namespace nvfuser {

//! [Val equivalence classes and E-graphs]
//!
//! This class tracks a single notion of Val equivalence over all Vals in an
//! IrContainer. It also includes "extraction" utilities which form part of the
//! E-Graph machinery. Equality saturation is the missing component, which
//! generates new Vals and Exprs to feed the extraction phase. Currently, these
//! steps are left to the user; in the future, we could automate equality
//! saturation (at least for Scalar and IterDomain).
template <typename IndexType>
class ValEquivalence {
 public:
  ValEquivalence(IrContainer& container) : container_(container) {
    enlarge(container.valsVector().size());
  }

  //! Merge the sets containing a and b
  void merge(Val* a, Val* b) {
    TORCH_CHECK(
        selected_.empty(),
        "Cannot merge equivalence classes after selection/extraction has begun");
    IndexType anum = a->number();
    IndexType bnum = b->number();
    if (anum > bnum) { // sort numbers for ease of checking size
      std::swap(anum, bnum);
    }
    if (bnum >= uf_.size()) {
      enlarge(bnum + 1);
    }
    uf_.merge(anum, bnum);
    if (!selected_.empty()) {
      // Setting this flag will notify selection and extraction
      need_selection_ = true;
    }
  }

  //! Merge the matching outputs of two expressions
  void mergeOutputs(Expr* a, Expr* b) {
    TORCH_CHECK(
        a->outputs().size() == b->outputs().size(),
        "Number of outputs must match in mergeOutputs");
    for (size_t i = 0; i < a->outputs().size(); ++i) {
      merge(a->output(i), b->output(i));
    }
  }

  //! Simplify all given Statements (Exprs or Vals), recursively mapping Vals to
  //! selected vals, modifying Vals and Exprs in place when possible, and
  //! inserting new Statements when necessary.
  void extractInPlace(std::vector<Expr*>& exprs) {
    if (need_selection_) {
      makeSelections();
    }
    // maintain a stack of Statements to process
    std::deque<Statement*> stmt_stack(exprs.begin(), exprs.end());
    while (!stmt_stack.empty()) {
      auto stmt = stmt_stack.back();
      stmt_stack.pop_back();
      TORCH_CHECK(
          stmt != nullptr, "extractInPlace encountered null Statement pointer");

      if (stmt->isExpr()) {
        auto expr = stmt->as<Expr>();
        // Replace Statement members with selections and queue for processing
        for (size_t pos = 0; pos < expr->outputs().size(); ++pos) {
          auto outp = expr->output(pos);
          auto sel = selectedVal(outp);
          TORCH_CHECK(
              sel.has_value(),
              "Val ",
              outp->toString(),
              " must have a selection before extractInPlace is called");
          if (sel != outp) { // swap sel in, in place of outp
            expr->setOutput(pos, sel.value());
          }
          // DON'T mark selected val for further processing
          // We only move one direction (upstream), to avoid looping
          // indefinitely
          // stmt_stack.push_back(expr->output(pos));
        }
        for (size_t pos = 0; pos < expr->inputs().size(); ++pos) {
          auto inp = expr->input(pos);
          auto sel = selectedVal(inp);
          TORCH_CHECK(
              sel.has_value(),
              "Val ",
              inp->toString(),
              " must have a selection before extractInPlace is called");
          if (sel != inp) { // swap sel in, in place of inp
            expr->setInput(pos, sel.value());
          }
          // mark selected val for further processing
          stmt_stack.push_back(sel.value());
        }
        for (size_t pos = 0; pos < expr->attributes_.size(); ++pos) {
          auto attr = expr->attributes_[pos];
          if (!attr) {
            continue;
          }
          Statement* sel = attr;
          if (attr->isVal()) {
            auto valattr = attr->as<Val>();
            auto sel = selectedVal(valattr);
            TORCH_CHECK(
                sel.has_value(),
                "Val ",
                valattr->toString(),
                " must have a selection before extractInPlace is called");
            expr->attributes_[pos] = sel.value();
          } // Don't swap non-Val attributes, but still process them
          stmt_stack.push_back(sel);
        }
        if (container_.isA<kir::Kernel>()) {
          // Swap predicates and mark them for further processing
          auto pred = expr->predicate();
          if (pred) {
            auto maybe_selpred = selectedVal(pred);
            TORCH_CHECK(
                maybe_selpred.has_value(),
                "Predicate ",
                pred->toString(),
                " must have a selection before extractInPlace is called");
            auto selpred = maybe_selpred.value();
            TORCH_CHECK(
                selpred->vtype() == ValType::Predicate,
                "Selected Val (",
                selpred->toString(),
                ") meant to replace predicate (",
                expr->predicate()->toString(),
                ") in Expr (",
                expr->toString(),
                ") is not a kir::Predicate");
            expr->setPredicate((kir::Predicate*)selpred);
            stmt_stack.push_back(expr->predicate());
          }

          pred = expr->writePredicate();
          if (pred) {
            auto maybe_selpred = selectedVal(pred);
            TORCH_CHECK(
                maybe_selpred.has_value(),
                "Write-predicate ",
                pred->toString(),
                " must have a selection before extractInPlace is called");
            auto selpred = maybe_selpred.value();
            TORCH_CHECK(
                selpred->vtype() == ValType::Predicate,
                "Selected Val (",
                selpred->toString(),
                ") meant to replace write-predicate (",
                expr->predicate()->toString(),
                ") in Expr (",
                expr->toString(),
                ") is not a kir::Predicate");
            expr->setWritePredicate((kir::Predicate*)selpred);
            stmt_stack.push_back(expr->writePredicate());
          }
        }
      } else {
        TORCH_CHECK(
            stmt->isVal(),
            "extractInPlace encountered unknown statement of unknown type: ",
            stmt->toString());

        // For Vals, we only need to worry about updating its definition.
        auto val = stmt->as<Val>();

        if (val->definition()) {
          stmt_stack.push_back(val->definition());
        }
      }
    }
  }

  void clear() {
    uf_.clear();
    complexity_.clear();
    selected_.clear();
    need_selection_ = true;
  }

  //! This is like running a->sameAs(b) but uses equivalence classes for
  //! comparisons. If merge_if_true=true, then if this check is true, then
  //! corresponding leaf Vals in a and b will be merged. Note that, unlike
  //! sameAs, immediate Scalar constants will be evaluated to test for equality.
  bool same(Statement* a, Statement* b, bool merge_if_true = true) {
    if (a == b) {
      return true;
    }
    if (a->isVal()) {
      if (!b->isVal()) {
        return false;
      }
      auto aval = a->as<Val>();
      auto bval = b->as<Val>();
      if (getClassNumber(aval) == getClassNumber(bval)) {
        return true; // already equivalent; no need to merge
      }
      if (aval->sameAs(bval)) {
        if (merge_if_true) {
          merge(aval, bval);
        }
        return true;
      } else {
        if (typeid(*aval) != typeid(bval)) {
          return false;
        }
        // Check immediate Scalars for equality
        if (aval->isScalar() && aval->isConst()) {
          if (!bval->isConst()) {
            return false;
          }
          if (aval->isIntegralScalar()) {
            if ((aval->isIntegralScalar() &&
                 (aval->getInt().value() != bval->getInt().value())) ||
                (aval->isFloatingPointScalar() &&
                 (aval->getDouble().value() != bval->getDouble().value())) ||
                (aval->isABool() &&
                 (aval->getBool().value() != bval->getBool().value()))) {
              return false;
            }
            // immediate constants match
            merge(aval, bval);
            return true;
          }
        }
        if (!aval->definition() || !bval->definition()) {
          return false;
        }
        // If true, the following call will merge aval and bval as they are
        // outputs
        return same(aval->definition(), bval->definition(), merge_if_true);
      }
    }
    TORCH_CHECK(a->isExpr(), "Only Vals and Exprs may be compared in same()");
    if (!b->isExpr()) {
      return false;
    }
    auto aexpr = a->as<Expr>();
    auto bexpr = b->as<Expr>();

    if (typeid(*aexpr) != typeid(*bexpr)) {
      return false;
    }
    if (aexpr->inputs().size() != bexpr->inputs().size() ||
        aexpr->outputs().size() != bexpr->outputs().size() ||
        aexpr->attributes().size() != bexpr->attributes().size()) {
      return false;
    }
    for (const auto i : c10::irange(aexpr->inputs().size())) {
      if (!same(aexpr->input(i), bexpr->input(i), merge_if_true)) {
        return false;
      }
    }
    for (const auto i : c10::irange(aexpr->attributes().size())) {
      if (!same(aexpr->attribute(i), bexpr->attribute(i), merge_if_true)) {
        return false;
      }
    }
    if (merge_if_true) {
      mergeOutputs(aexpr, bexpr);
    }
    return true;
  }

  IrContainer& container() {
    return container_;
  }

 private:
  //! Return number() of a representative of the set containing val.
  //! Note that this number might change if this set is merged with another.
  IndexType getClassNumber(IndexType num) {
    // num might be beyond the current size of uf_, if we have added Vals to
    // container_ since the last time we enlarged uf_. In this case, we know
    // that nothing has been merged, so val is the representative.
    if (num >= uf_.size()) {
      return num;
    }
    return uf_.find(num);
  }

  //! Return number() of a representative of the set containing val.
  //! This is the Val* which is at the root of the tree having
  //! val as a node.
  IndexType getClassNumber(Val* val) {
    return getClassNumber(getNumberFor(val));
  }

  //! How many Vals are we currently tracking? This may differ from the number
  //! in the container
  size_t size() const {
    return uf_.size();
  }

  //! Grow this datastructure to a new size
  void enlarge(size_t new_size) {
    uf_.enlarge(new_size);
  }

  IndexType getNumberFor(Val* val) const {
    size_t rawnum = val->number();
    TORCH_CHECK(
        rawnum <= std::numeric_limits<IndexType>::max(),
        "Encountered val numbered ",
        rawnum,
        " (",
        val->toString(),
        ") which is greater than IndexType's largest representable number: ",
        std::to_string(std::numeric_limits<IndexType>::max()));
    return (IndexType)rawnum;
  }

  //! Tests if the indicated Val is the one selected for its equivalence class.
  bool isSelected(IndexType num) const {
    if (num >= selected_.size()) {
      return false;
    }
    return selected_[num] == num;
  }

  //! Tests if the indicated Val is the one selected for its equivalence class.
  bool isSelected(Val* val) const {
    return isSelected(getNumberFor(val));
  }

  //! Return number of currently selected Val corresponding to given equivalence
  //! class. This will return nullopt for eclasses without selections.
  std::optional<IndexType> selectedNumber(IndexType eclass) const {
    if (eclass >= selected_.size()) {
      return std::nullopt;
    }
    // Note that selection might not match original ValType
    return selected_[eclass];
  }

  //! Get val by number, with bounds checking
  Val* getVal(IndexType num) const {
    auto vals = container_.valsVector();
    TORCH_CHECK(
        num < vals.size(),
        "Requestion Val number ",
        std::to_string(num),
        " but vals.size()=",
        vals.size());
    return vals[num];
  }

  //! Return currently selected Val corresponding to given Val.
  //! Note that this Val does not necessarily have its history rewritten.
  //! This will return nullptr for unselected vals.
  //! Note that this method is non-const since it may modify the underlying
  //! UnionFind when finding the equivalence class of val.
  std::optional<Val*> selectedVal(Val* val) {
    // get representative (will match this ValType)
    auto maybe_selnum = selectedNumber(getClassNumber(val));
    if (!maybe_selnum.has_value()) {
      return std::nullopt;
    }
    return getVal(maybe_selnum.value());
  }

  //! Select a Val to replace any Vals in its equivalence class
  void select(IndexType num) {
    auto repnum = getClassNumber(num);
    if (repnum >= selected_.size()) {
      selected_.resize(repnum + 1);
    }
    selected_[repnum] = num;
  }

  void select(Val* val) {
    select(getNumberFor(val));
  }

  //! Get a numeric precedence for a val.
  //! These will be lower for more preferred values.
  size_t selectionPrecedence(Val* val) {
    if (!val) {
      return std::numeric_limits<size_t>::max();
    }
    if (val->isScalar() && val->isConst()) {
      return 0; // immediate scalars have highest priority
    }
    if (val->vtype() == ValType::NamedScalar) {
      return 1; // next highest priority are NamedScalars
    }
    // For other cases, estimate their complexity based on definition
    size_t base_complexity = 2;
    if (!val->definition()) {
      return base_complexity;
    }
  }

  void setComplexity(IndexType num, size_t complexity) {
    if (num >= complexity_.size()) {
      complexity_.resize(num + 1);
    }
    complexity_[num] = complexity;
  }

  std::optional<size_t> getComplexity(IndexType num) {
    if (num >= complexity_.size()) {
      return std::nullopt;
    }
    return complexity_[num];
  }

  std::optional<size_t> getComplexity(Val* val) {
    return getComplexity(getNumberFor(val));
  }

  //! Computes an estimate of complexity for each Val
  void computeComplexity() {
    auto vals = container_.valsVector();
    std::vector<bool> estimated(vals.size(), false);
    std::deque<IndexType> stack(vals.size());
    std::iota(stack.begin(), stack.end(), 0);
    std::cout << "initial stack:" << std::endl;
    for (auto i : stack) {
      std::cout << "  " << i;
      std::cout << "    " << vals[i]->vtype() << std::endl;
    }
    while (!stack.empty()) {
      auto i = stack.front();
      stack.pop_front();
      if (estimated[i]) {
        continue;
      }
      auto val = getVal(i);
      if (!val) {
        // basically infinite complexity for nullptrs
        setComplexity(i, std::numeric_limits<size_t>::max());
        estimated[i] = true;
        continue;
      }

      std::cout << "computing complexity for val number " << i;
      std::cout << ": " << (ssize_t)val->vtype();
      std::cout << ": " << val->vtype();
      std::cout << " = " << val->toString() << std::endl;

      if (val->isScalar() && val->isConst()) {
        setComplexity(i, 1); // immediate scalars have lowest cost
        estimated[i] = true;
      } else if (val->vtype() == ValType::NamedScalar) {
        setComplexity(i, 2); // next lowest cost are NamedScalars
        estimated[i] = true;
      } else {
        // For other cases, estimate the complexity based on sum of complexities
        // of their inputs. We could specialize here on the type of Expr in the
        // definition in order to prioritize cheaper ops. Note that even then,
        // this is only an upper bound since we cannot take into account re-use
        // of intermediate expressions in this calculation.
        auto def = val->definition();
        if (!def) {
          // Input scalars have similar complexity to NamedScalars, but we
          // prioritize NamedScalars slightly.
          setComplexity(i, 3);
          estimated[i] = true;
        } else {
          size_t input_complexity = 0;
          bool missing_input = false;
          std::cout << "def=" << def->toString()
                    << " inputs().size()=" << def->inputs().size() << std::endl;
          for (auto inp : def->inputs()) {
            auto inp_num = getNumberFor(inp);
            if (!estimated[inp_num]) {
              // Need to get input complexity before we process the current val
              if (!missing_input) { // only push current val once
                stack.push_front(i);
              }
              stack.push_front(inp_num);
              missing_input = true;
              continue;
            }
            input_complexity += complexity_[inp_num];
          }
          if (!missing_input) {
            // Here we assign each op the constant complexity of 10 (plus the
            // sum of complexities of its inputs). In this future, this could
            // reflect the type of op, the size of the inputs, etc.
            setComplexity(i, input_complexity + 10);
            estimated[i] = true;
          }
        }
      }
    }
  }

  //! Update the selected_ vector to indicate a Val for every class
  void makeSelections() {
    computeComplexity();

    for (IndexType i = 0; i < container_.valsVector().size(); ++i) {
      auto val = container_.valsVector()[i];
      if (!val) {
        continue; // skip removed Vals which will show up here as nullptrs
      }

      auto eclass = getClassNumber(i);
      auto selected_num = selectedNumber(eclass);
      if (!selected_num.has_value()) {
        select(i); // Make a default selection
        continue;
      }

      auto selected_val = getVal(selected_num.value());
      if (!selected_val) {
        // Selection has been removed
        std::cout << "Selection for eclass " << std::to_string(eclass)
                  << " has been removed" << std::endl;
        select(i);
        continue;
      }

      auto sel_comp = getComplexity(selected_num.value());
      auto current_comp = getComplexity(i);
      TORCH_CHECK(
          sel_comp.has_value() && current_comp.has_value(),
          "All vals should have complexity estimates by now");
      if (current_comp.value() < sel_comp.value()) {
        select(i);
      }

      if (current_comp.value() == 0 && sel_comp.value() == 0) {
        // Consistency check that we didn't merge incompatible constants
        TORCH_CHECK(
            simplifyExpr(eq(val, selected_val))->getBool() == true,
            "Unequal constant Vals ",
            val->toString(),
            " and ",
            selected_val->toString(),
            " are marked as equivalent.");
      }
    }

    // print selections and other members for each eclass
    std::unordered_set<IndexType> printed_classes;
    for (IndexType i = 0; i < size(); ++i) {
      auto c = getClassNumber(i);
      if (printed_classes.find(c) != printed_classes.end()) {
        continue;
      }
      printed_classes.insert(c);
      auto sel = selectedNumber(c);
      TORCH_CHECK(sel.has_value(), "Expected all classes to have a selection");
      auto selval = getVal(sel.value());
      if (!selval) {
        std::cout << "Found nullptr selection for eclass " << std::to_string(c)
                  << std::endl;
        continue;
      }
      std::cout << "Val class " << std::to_string(c)
                << ": selection=" << selval->toString() << std::endl;
      // Now print selection, followed by other equivalent Vals
      for (IndexType j = 0; j < size(); ++j) {
        auto valj = getVal(j);
        if (!valj) {
          continue; // skip printing removed Vals
        }
        if (j == sel.value()) {
          continue;
        }
        auto cj = getClassNumber(j);
        if (cj != c) {
          continue;
        }
        std::cout << "  " << valj->toString() << std::endl;
      }
    }
    need_selection_ = false;
  }

 private:
  IrContainer& container_;
  UnionFind<IndexType> uf_;
  // denotes a "selected" value for each representative
  std::vector<std::optional<IndexType>> selected_;
  // Expressions can have complexity larger than the number of vals in the
  // Fusion
  std::vector<size_t> complexity_;

  // Flag to trigger a scan of merged selections. This is necessary if we merge
  // during extraction, since we might merge two eclasses which each have a
  // selection. In that case, we will need to reselect in the merged class,
  // choosing one of the selections based on our precedence rules.
  bool need_selection_ = true;
};

} // namespace nvfuser
