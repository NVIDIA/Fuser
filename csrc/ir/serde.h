// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class Statement;
class Expr;
class Val;
class IterDomain;
class TensorView;
class IrContainer;

class IrSerde {
 public:
  IrSerde(const IrContainer* container);

  const std::vector<Statement*>& topologicalSortedStatements() const {
    return toposorted_stmts_;
  }

  std::vector<int64_t> update(const std::vector<Statement*>& new_stmts);

  int64_t map(Statement* v) const;
  int64_t map(const Statement* v) const;

  int64_t map(Val* v) const;
  int64_t map(const Val* v) const;

  int64_t map(Expr* e) const;
  int64_t map(const Expr* e) const;

  std::vector<int64_t> map(const std::vector<Statement*>& stmts) const;
  std::vector<int64_t> map(const std::vector<Val*>& vals) const;
  std::vector<int64_t> map(const std::vector<Expr*>& exprs) const;

  std::vector<int64_t> map(const std::vector<IterDomain*>& vals) const;
  std::vector<int64_t> map(const std::vector<TensorView*>& vals) const;

  std::vector<int64_t> map(const std::unordered_set<Val*>& vals) const;

 private:
  std::vector<Statement*> topologicalSortStatements(
      const IrContainer* container);
  std::vector<Statement*> topologicalSortStatements(
      const std::deque<Val*>& values,
      const std::deque<Expr*>& exprs);
  std::unordered_map<Val*, int64_t> createToposortValuesMap() const noexcept;
  std::unordered_map<Expr*, int64_t> createToposortExpressionsMap()
      const noexcept;

  const IrContainer* container_;

  std::vector<Statement*> toposorted_stmts_;

  //! Return mapping from value to integer id in topological sorted order
  const std::unordered_map<Val*, int64_t> vals_to_id_map_;

  const std::unordered_map<Expr*, int64_t> exprs_to_id_map_;
};

} // namespace nvfuser
