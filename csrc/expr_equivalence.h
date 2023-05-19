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
#include <val_equivalence.h>

#include <deque>
#include <iostream>
#include <typeinfo>
#include <vector>

namespace nvfuser {

template <typename IndexType>
class ExprEquivalence;

//! This holds the content of an Expr, without creating it and registering it in
//! the Fusion. This is useful for determining whether an Expr already exists in
//! the Fusion before we create a new one.
template <class ExprType>
struct ExprSketch {
  std::vector<Val*>& attributes;
  std::vector<Val*>& inputs;
};

//! These classes are used for hash tables that should do their best to avoid
//! inserting equivalent Exprs twice. They do so by defining "equality" as
//! equivalence under an `ExprEquivalence` class object.
template <typename IndexType>
class ExprsHash {
  using ExprWithEquiv = std::pair<ExprEquivalence<IndexType>&, Expr*>;
  constexpr size_t operator()(ExprWithEquiv& a) const {
    // Hash the Expr type
    // For inputs and attributes, hash the equivalence class (IndexType)
    // Hash inputs
    // Hash attributes
    return 0;
  }
};

template <typename IndexType>
class ExprsEquiv {
  using ExprWithEquiv = std::pair<ExprEquivalence<IndexType>&, Expr*>;
  constexpr bool operator()(ExprWithEquiv& a, ExprWithEquiv& b) const {
    // NOTE: we assume a.first == b.first
    auto ee = a.first;

    return ee.equiv(a.second, b.second);
  }
};

//! [Expr equivalence]
//!
//! This class tracks Expr equivalence over all Exprs in an
//! IrContainer. Expressions are considered equivalent if they are of the same
//! type of op, and if their inputs and attributes are equivalent under a given
//! ValEquivalence object.
//!
//! The primary use of this class is in augmentation or equality saturation,
//! where we will pattern match the existing Exprs and Vals in a Fusion and add
//! equivalent Exprs in order to encourage simplification. In order to limit the
//! number of created Exprs, we need a notion of equivalence. Luckily, we have a
//! notion of Val equivalence already encoded using a `ValEquivalence` object.
//! This class composes that class and uses it to define equivalence of Exprs.
template <typename IndexType>
class ExprEquivalence {
 public:
  ExprEquivalence(ValEquivalence<IndexType>& ve) : ve_(ve) {}

  ValEquivalence<IndexType>& valEquivalence() const noexcept {
    return ve_;
  }

  //! Return true only if types of a and b are same, and all inputs and
  //! attributes are equivalent Vals.
  bool equiv(Expr* a, Expr* b) {
    if (typeid(a) != typeid(b)) {
      return false;
    }
    if (a->attributes().size() != b->attributes().size() ||
        a->inputs().size() != b->inputs().size()) {
      return false;
    }
    for (auto i: c10::irange(a->attributes().size())) {
      if (!ve_.equiv(a->attribute(i), b->attribute(i))) {
        return false;
      }
    }
    for (auto i: c10::irange(a->inputs().size())) {
      if (!ve_.equiv(a->input(i), b->input(i))) {
        return false;
      }
    }
    return true;
  }

  void clear() {
    ve_.clear();
    complexity_.clear();
    selected_.clear();
    need_selection_ = true;
  }

  //! This hashes the input to guess whether there already exists an equivalent
  //! Expr and return it if so.
  template <typename OpType>
  std::optional<OpType*> getExpr(
      std::vector<Val*>& inputs,
      std::vector<Val*>& attributes) {
    return std::nullopt;
  }

  //! This checks whether there is already an Expr of the given type with inputs
  //! and attributes equivalent to those provided. If so, returns the existing
  //! Expr. Otherwise, create one and return the new one.
  template <typename OpType>
  OpType* getOrCreateExpr(std::vector<Val*>& inputs, std::vector<Val*>& attributes) {
    auto maybeExpr = getExpr<OpType>(inputs, attributes);
    if (maybeExpr.has_value()) {
      return maybeExpr.value();
    }
    // TODO: create this Expr as usual
  }

 private:


 private:
  // Determines whether Vals are equivalent
  ValEquivalence<IndexType>& ve_;

  // denotes a "selected" expression for each representative
  std::vector<std::optional<IndexType>> selected_;

  // Note size_t since Expressions can have complexity larger than the number
  // of vals or exprs in the Fusion
  std::vector<size_t> complexity_;

  // Flag to trigger a scan of merged selections. This is necessary if we merge
  // during extraction, since we might merge two eclasses which each have a
  // selection. In that case, we will need to reselect in the merged class,
  // choosing one of the selections based on our precedence rules.
  bool need_selection_ = true;
};

} // namespace nvfuser

