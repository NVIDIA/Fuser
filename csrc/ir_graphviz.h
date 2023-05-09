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
  // A selection of colors from the X11 color scheme
  // https://graphviz.org/doc/info/colors.html#x11
  enum class Color {
    AZURE,
    PINK,
    GREEN,
    GREY,
    YELLOW,
    LAVENDER,
    CYAN,
    WHITE,
    MAGENTA,
    RED,
    // Only the cases above are included in colorCycle
    LIGHTGRAY,
    LIGHTGREEN,
    LIGHTSALMON
  };

  // Small color palette from the X11 theme
  static Color colorCycle(size_t index) {
    const size_t number_of_colors = 10;
    index = index % number_of_colors;
    return static_cast<Color>(index);
  }

  std::string colorToString(Color color) {
    switch (color) {
      case Color::AZURE:
        return "azure";
      case Color::PINK:
        return "pink";
      case Color::GREEN:
        return "green";
      case Color::YELLOW:
        return "yellow";
      case Color::LAVENDER:
        return "lavender";
      case Color::CYAN:
        return "cyan";
      case Color::WHITE:
        return "white";
      case Color::MAGENTA:
        return "magenta";
      case Color::RED:
        return "red";
      case Color::LIGHTGRAY:
        return "lightgray";
      case Color::LIGHTGREEN:
        return "lightgreen";
      case Color::LIGHTSALMON:
        return "lightsalmon";
      default:
        break;
    }
    return "";
  }

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

    //! Whether to show definitions of Scalars in their node labels
    bool showValuesOfConstants = true; // true for Explicit

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

  IrGraphGenerator(const Fusion* fusion, DetailLevel detail_level);

  using ExprColorMap = std::unordered_map<const Expr*, Color>;

  void setExprColor(Expr* expr, Color color) {
    expr_color_map_[expr] = color;
  }

  //! Return graph as dot-format string
  std::string generate();

  //! Write dot-format string to file
  void print(const char* filename);

 public:
  //! L-value reference to enable setting config options manually.
  Config& config() {
    return config_;
  }

 private:
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

  //! If color not specified for an expression, set it to color
  void maybeSetExprColor(const Expr* e, Color color) {
    if (expr_color_map_.find(e) == expr_color_map_.end()) {
      expr_color_map_.insert({e, color});
    }
  }

  void printExpr(
      const Expr* expr,
      const std::string& label,
      const std::string& border_color = "blue");
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
  ExprColorMap expr_color_map_;
};

} // namespace nvfuser
