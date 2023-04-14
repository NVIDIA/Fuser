// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/Exception.h>

#include <ir_all_nodes.h>
#include <kernel_ir.h>
#include <type.h>
#include <union_find.h>
#include <utils.h>

#include <deque>
#include <iostream>
#include <typeinfo>
#include <vector>

namespace nvfuser {

//! Val equivalence classes and E-graphs
//! This class tracks a single notion of Val equivalence over all Vals in an
//! IrContainer. It also includes "extraction" utilities which form part of the
//! E-Graph machinery. Equality saturation is the missing component, which
//! generates new Vals and Exprs to feed the extraction phase. Currently, these
//! steps are left to the user; in the future, we could automate equality
//! saturation (at least for Scalar and IterDomain).
template <typename IndexType>
class ValEquivalence {
 public:
  ValEquivalence(IrContainer& container)
      : container_(container), uf_(0), blacklist_(0) {}

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
      merged_since_selection_ = true;
    }
  }

  //! Mark val as ineligible for extraction
  void blacklist(Val* val) {
    auto num = val->number();
    if (num >= size()) {
      enlarge(num + 1);
    }
    blacklist_[num] = true;
  }

  //! Simplify all given Statements (Exprs or Vals), recursively mapping Vals to
  //! selected vals, modifying Vals and Exprs in place when possible, and
  //! inserting new Statements when necessary.
  void extractInPlace(std::vector<Expr*>& exprs) {
    // maintain a stack of Statements to process
    std::deque<Statement*> stmt_stack(exprs.begin(), exprs.end());
    stmt_stack.pop_back();

    while (!stmt_stack.empty()) {
      auto stmt = stmt_stack.back();
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
          // mark selected val for further processing
          stmt_stack.push_back(expr->output(pos));
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
    blacklist_.resize(new_size);
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
        num,
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

  //! Update the selected_ vector to indicate a Val for every class
  void makeSelections() {
    for (IndexType i = 0; i < container_.valsVector().size(); ++i) {
      auto val = container_.valsVector()[i];
      if (!val) {
        continue; // skip removed vals which will show here as nullptr
      }

      auto eclass = getClassNumber(i);
      auto selected_num = selectedNumber(eclass);
      if (!selected_num.has_value()) {
        select(i); // Make a default selection
        continue;
      }

      auto selected_val = getVal(selected_num);

      // Current selection is immediate constant
      if (selected_val->isScalar() && selected_val->isConst()) {
        // If we've already selected a constant scalar, we will keep this
        // selection.
        if (val->isScalar() && val->isConst()) {
          // If val is _also_ a constant scalar, we do a consistency check
          TORCH_CHECK(
              simplifyExpr(eq(val, selected_val))->getBool() == true,
              "Unequal constant Vals ",
              val->toString(),
              " and ",
              selected_val->toString(),
              " are marked as equivalent.");
        }
        continue;
      }
      if (selected_val->isNamedScalar()) {
      }
    }
  }

 private:
  IrContainer& container_;
  UnionFind<IndexType> uf_;
  std::vector<bool> blacklist_;
  // denotes a "selected" value for each representative
  std::vector<std::optional<IndexType>> selected_;
  std::vector<IndexType> complexity_;

  // Flag to trigger a scan of merged selections. This is necessary if we merge
  // during extraction, since we might merge two eclasses which each have a
  // selection. In that case, we will need to reselect in the merged class,
  // choosing one of the selections based on our precedence rules.
  bool merged_since_selection_ = false;
};

} // namespace nvfuser
