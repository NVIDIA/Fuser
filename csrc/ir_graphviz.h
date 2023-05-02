// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <dispatch.h>

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
class TORCH_CUDA_CU_API IrGraphGenerator : private OptInConstDispatch {
 public:
  enum class DetailLevel {
    ComputeOnly, // Only dataflow (compute) nodes
    Basic, // Compute + schedule, with minimal details (default)
    Explicit, // Additional details (ex. symbolic names for scalar constants)
    Verbose, // Includes all values and dead definitions
  };

  struct Config {
    //! Whether to show the compute definition, i.e. dataflow of Fusion
    bool showComputeDef = true;

    //! Whether to show scheduling, i.e. transforms between leaf and root or
    //! rfactor domains.
    bool showSchedule = false;

    //! If true, for TensorViews with rfactor domains, show them separate from
    //! their root domains, and trace the transforms relating their IterDomains.
    bool showRFactorTransformsInCompute = true;

    //! Whether to show nodes that are unused due to dead code elimination.
    bool showUnreachableNodes = false; // true for Explicit

    //! Whether to show definitions of constant Scalars
    bool showDefsOfConstants = false; // true for Explicit

    //! Print to string, prepending line_prefix to each line of output
    std::string toString(std::string line_prefix = "") {
      std::stringstream ss;
      ss << line_prefix << "showComputeDef: " << showComputeDef << std::endl;
      ss << line_prefix << "showSchedule: " << showSchedule << std::endl;
      ss << line_prefix
         << "showRFactorTransformsInCompute: " << showRFactorTransformsInCompute
         << std::endl;
      ss << line_prefix << "showUnreachableNodes: " << showUnreachableNodes
         << std::endl;
      return ss.str();
    }
  };

  using ExprColorMap = std::unordered_map<const Expr*, size_t>;

 public:
  static void print(
      const Fusion* fusion,
      const char* filename,
      DetailLevel detail_level = DetailLevel::Basic,
      ExprColorMap* expr_color_map = nullptr);

  static std::string toGraphviz(
      const Fusion* fusion,
      DetailLevel detail_level,
      ExprColorMap* expr_color_map = nullptr);

  //! L-value reference to enable setting config options manually.
  Config& config() {
    return config_;
  }

 private:
  IrGraphGenerator(
      const Fusion* fusion,
      DetailLevel detail_level,
      ExprColorMap* expr_color_map = nullptr);
  ~IrGraphGenerator() override = default;

  std::string generate();

  void generateComputeGraph();
  void generateScheduleGraph();

  void handle(const Statement*) override;
  void handle(const Val*) override;
  void handle(const Expr*) override;

  void handle(const TensorDomain*) override;
  void handle(const TensorView*) override;
  void handle(const IterDomain*) override;

  void handle(const Bool*) override;
  void handle(const Double*) override;
  void handle(const Int*) override;
  void handle(const ComplexDouble*) override;
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
  Config config_;
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

} // namespace nvfuser
