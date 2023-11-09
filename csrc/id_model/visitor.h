// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <ir/all_nodes.h>
#include <val_graph.h>

namespace nvfuser {

// Iterates through an IterDomain Graph in topological order, calling handle on
// all Id and all Expr groups in a forward topological order.
//
// Warning: Expr groups that have an input and output in the same IdGroup are
// ignored.
//
// Warning: This is not a great iterator if there's a desire to minimize paths
// traveled to simply visit all IdGroups in order. See ExprsBetween to see how
// we might minimize paths.
class IdGraphVisitor {
 public:
  IdGraphVisitor() = delete;

  IdGraphVisitor& operator=(const IdGraphVisitor& other) = delete;

  IdGraphVisitor& operator=(IdGraphVisitor&& other) = delete;

  virtual ~IdGraphVisitor() = default;

 protected:
  // If sub_selection is assumed to be a set of iter domains by which form a
  // sub-regrion of the IdGraph provided. Only that sub-region will be visited.
  IdGraphVisitor(
      const ValGraph& id_graph,
      const VectorOfUniqueEntries<IterDomain*> sub_selection = {})
      : id_graph_(id_graph), sub_selection_(sub_selection) {}

  IdGraphVisitor(const IdGraphVisitor& other) = default;

  IdGraphVisitor(IdGraphVisitor&& other) = default;

  virtual void handle(ValGroup id_group) = 0;
  virtual void handle(ExprGroup expr_group) = 0;

  void traverse();

  const ValGraph& graph() {
    return id_graph_;
  };

 private:
  const ValGraph& id_graph_;
  const VectorOfUniqueEntries<IterDomain*> sub_selection_;
};

// Statement sorting based on IdGraphVisitor, see warnings to IdGraph Visitor.
class IdGraphStmtSort : public IdGraphVisitor {
 public:
  IdGraphStmtSort(
      const ValGraph& id_graph,
      const VectorOfUniqueEntries<IterDomain*> sub_selection = {})
      : IdGraphVisitor(id_graph, sub_selection) {
    IdGraphVisitor::traverse();
  }

  // Return non-reference so that code like below can work
  // for (auto expr_group: IdGraphStmtSort(graph).exprs())
  ExprGroups exprs() const {
    return sorted_exprs_;
  }

  ValGroups ids() const {
    return sorted_ids_;
  }

  ~IdGraphStmtSort() override = default;

 protected:
  using IdGraphVisitor::handle;
  void handle(ValGroup id_group) override {
    sorted_ids_.pushBack(id_group);
  }

  void handle(ExprGroup expr_group) override {
    sorted_exprs_.pushBack(expr_group);
  }

  ExprGroups sorted_exprs_;
  ValGroups sorted_ids_;
};

} // namespace nvfuser
