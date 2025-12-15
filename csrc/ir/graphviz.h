// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>
#include <exceptions.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

// Generates a DOT (https://www.graphviz.org) graph
// representation of a fuser IR
//
// Usage:
// 1) Add calls to IrGraphGenerator::print(), for example:
//  `IrGraphGenerator::print(&fusion, "ir.dot")`
//
// 2) Call IrGraphGenerator::print() from a debugger. Using gdb for example:
//  `call IrGraphGenerator::print(&fusion, "ir.dot",
//      IrGraphGenerator::DetailLevel::Explicit)`
//
// Notes:
//  - When called from the debugger, the detail_level must be
//    explicitly passed in (most debuggers don't support default arguments)
//
//  - The output dot file path can't include shell specific notations,
//    for example you can't use "~/temp/ir.dot" ("/home/user/temp/ir.dot"
//    must be used instead)
//
class IrGraphGenerator : private OptInConstDispatch {
 public:
  enum class DetailLevel {
    ComputeOnly, // Only dataflow (compute) nodes
    Basic, // Compute + schedule, with minimal details (default)
    Explicit, // Additional details (ex. symbolic names for scalar constants)
    Verbose, // Includes all values and dead definitions
  };

  using ExprColorMap = std::unordered_map<const Expr*, size_t>;

 public:
  static void print(
      const Fusion* fusion,
      const char* filename,
      DetailLevel detail_level = DetailLevel::Basic,
      ExprColorMap* expr_color_map = nullptr);

  NVF_API static std::string toGraphviz(
      const Fusion* fusion,
      DetailLevel detail_level,
      ExprColorMap* expr_color_map = nullptr);

  ~IrGraphGenerator() override = default;

 private:
  IrGraphGenerator(
      const Fusion* fusion,
      DetailLevel detail_level,
      ExprColorMap* expr_color_map = nullptr);

  std::string generate();

  void generateComputeGraph();
  void generateScheduleGraph();

  void dispatch(const Statement*) override;
  void dispatch(const Val*) override;
  void dispatch(const Expr*) override;

  void handle(const TensorDomain*) override;
  void handle(const TensorView*) override;
  void handle(const IterDomain*) override;
  void handle(const RaggedIterDomain*) override;

  void handle(const Val*) override;
  void handle(const NamedScalar*) override;

  // lookup the graph id, creating one if not found
  std::string getid(const Statement* stm);

  bool visited(const Statement* s) const {
    return visited_.find(s) != visited_.end();
  }

  void addArc(
      const Statement* src,
      const Statement* dst,
      const std::string& style = "");

  void printExpr(const Expr* expr, const std::string& label);
  void printValue(const Val* val, const std::string& label);

 private:
  const DetailLevel detail_level_;
  const Fusion* const fusion_;
  std::stringstream graph_def_;
  std::unordered_map<const Statement*, std::string> id_map_;
  std::unordered_set<const Statement*> visited_;
  std::unordered_set<const Val*> inputs_;
  std::unordered_set<const Val*> outputs_;
  std::vector<const TensorView*> tensor_views_;
  std::vector<std::string> arcs_;
  int next_id_ = 1;
  ExprColorMap* expr_color_map_ = nullptr;
};

// Generates a DOT graph representation of fusion transform
std::string irTransformToDot(Fusion* fusion);

} // namespace nvfuser
