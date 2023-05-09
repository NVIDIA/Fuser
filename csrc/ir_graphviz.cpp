// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir_graphviz.h>

#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <type.h>

#include <fstream>

namespace nvfuser {

namespace {

// Private helper, generating node labels for IrGraphGenerator
class IrNodeLabel final : private OptInConstDispatch {
  using DetailLevel = IrGraphGenerator::DetailLevel;

 public:
  static std::string gen(
      const Statement* node,
      const IrGraphGenerator::Config& config,
      DetailLevel detail_level = DetailLevel::Basic) {
    IrNodeLabel generator(config, detail_level);
    generator.OptInConstDispatch::handle(node);
    return generator.label_.str();
  }

 private:
  explicit IrNodeLabel(
      const IrGraphGenerator::Config& config,
      IrGraphGenerator::DetailLevel detail_level)
      : config_(config), detail_level_(detail_level) {}

  ~IrNodeLabel() final = default;

  void handle(const Bool* b) override {
    if (b->isSymbolic()) {
      label_ << "b" << b->name();
    } else {
      if (config_.showValuesOfConstants) {
        label_ << "b" << b->name() << "=";
      }
      label_ << *b->value();
    }
  }

  void handle(const Double* d) override {
    if (d->isSymbolic()) {
      label_ << typePrefix(d->getDataType().value()) << d->name();
    } else {
      if (config_.showValuesOfConstants) {
        label_ << typePrefix(d->getDataType().value()) << d->name() << "=";
      }
      label_ << *d->value();
    }
  }

  void handle(const Int* i) override {
    if (i->isSymbolic()) {
      label_ << "i" << i->name();
    } else {
      if (config_.showValuesOfConstants) {
        label_ << "i" << i->name() << "=";
      }
      label_ << *i->value();
    }
  }

  void handle(const NamedScalar* ns) override {
    label_ << ns->name();
  }

  void handle(const IterDomain* id) override {
    label_ << id->getIterType();
    label_ << id->getParallelType();
    label_ << id->name();

    label_ << "(";
    if (!id->start()->isZeroInt()) {
      label_ << IrNodeLabel::gen(id->start(), config_, detail_level_) << " : ";
    }
    label_ << IrNodeLabel::gen(id->extent(), config_, detail_level_);
    label_ << ")";
  }

  void handle(const Split* split) override {
    label_ << "Split(inner=" << (split->innerSplit() ? "true" : "false")
           << ", factor="
           << IrNodeLabel::gen(split->factor(), config_, detail_level_) << ")";
  }

  void handle(const Merge* merge) override {
    label_ << "Merge";
  }

 private:
  std::stringstream label_;
  const IrGraphGenerator::Config& config_;
  const DetailLevel detail_level_;
};

} // anonymous namespace

void IrGraphGenerator::print(const char* filename) {
  std::ofstream dot_file(filename);
  TORCH_CHECK(dot_file.good(), "Failed to open the IR graph file");
  dot_file << generate();
}

IrGraphGenerator::IrGraphGenerator(
    const Fusion* fusion,
    DetailLevel detail_level)
    : detail_level_(detail_level), fusion_(fusion) {
  // setup inputs & outputs
  // (indexes used to quickly check if a value is fusion input or output)
  for (const auto* input : fusion->inputs()) {
    TORCH_CHECK(inputs_.count(input) == 0);
    inputs_.insert(input);
  }
  for (const auto* output : fusion->outputs()) {
    TORCH_CHECK(outputs_.count(output) == 0);
    outputs_.insert(output);
  }
}

std::string IrGraphGenerator::getid(const Statement* stm) {
  const auto it = id_map_.find(stm);
  if (it == id_map_.end()) {
    // First reference, generate a new id
    std::stringstream new_id;
    new_id << "stm_" << next_id_++;
    id_map_.insert({stm, new_id.str()});
    return new_id.str();
  } else {
    return it->second;
  }
}

void IrGraphGenerator::addArc(
    const Statement* src,
    const Statement* dst,
    const std::string& style) {
  // We automatically visit (handle) the arc's source and destination
  handle(src);
  handle(dst);

  // generate and queue the arc definition
  std::stringstream arc_def;
  arc_def << getid(src) << " -> " << getid(dst) << " " << style;
  arcs_.push_back(arc_def.str());
}

void IrGraphGenerator::printExpr(
    const Expr* expr,
    const std::string& label,
    const std::string& border_color) {
  graph_def_ << "    " << getid(expr) << " "
             << "[label=\"" << label
             << "\", shape=Mrecord, color=" << border_color << ", "
             << "style=filled, fillcolor=";
  if (expr_color_map_.count(expr)) {
    graph_def_ << colorToString(expr_color_map_.at(expr));
  } else {
    graph_def_ << "azure";
  }
  graph_def_ << "];\n";
}

void IrGraphGenerator::printValue(const Val* val, const std::string& label) {
  graph_def_ << "    " << getid(val) << " [label=\"" << label
             << "\", shape=rect, color=green, fontsize=10];\n";
}

std::string IrGraphGenerator::generate() {
  // IrGraphGenerator instances are not reusable
  TORCH_CHECK(graph_def_.str().empty());
  TORCH_CHECK(visited_.empty());

  // Record config
  graph_def_ << "// config:" << std::endl;
  graph_def_ << config_.toString(/*line_prefix*/ "//   ");

  graph_def_ << "digraph fusion_ir {\n"
             << "  node [shape=circle, color=gray];\n"
             << "  edge [color=black];\n";

  // Compute graph
  if (config_.showComputeDef) {
    generateComputeGraph();
  }

  // Schedule graph
  if (config_.showSchedule) {
    generateScheduleGraph();
  }

  // All expressions & values
  // (These are otherwise unreacheable (dead) nodes)
  if (config_.showUnreachableNodes) {
    for (const auto* expr : fusion_->unordered_exprs()) {
      handle(expr);
    }
    for (const auto* val : fusion_->vals()) {
      handle(val);
    }
  }

  // Finally, print all arc definitions
  for (const auto& arc : arcs_) {
    graph_def_ << "  " << arc << ";\n";
  }

  graph_def_ << "}\n";

  // Make sure that all referenced nodes have been visited
  for (const auto& kv : id_map_) {
    TORCH_CHECK(
        visited(kv.first),
        kv.first->toString(),
        " = ",
        kv.second,
        " was referenced but is not yet visited");
  }

  return graph_def_.str();
}

void IrGraphGenerator::generateComputeGraph() {
  graph_def_ << "  subgraph cluster_compute {\n"
             << "    label=\"Compute Definition\";\n"
             << "    style=dashed;\n";

  // Inputs
  for (const auto* input : fusion_->inputs()) {
    handle(input);
  }

  // Outputs
  for (const auto* output : fusion_->outputs()) {
    handle(output);
  }

  graph_def_ << "  }\n";
}

void IrGraphGenerator::generateScheduleGraph() {
  graph_def_ << "  subgraph cluster_schedule {\n"
             << "    label=\"Schedule\";\n"
             << "    style=dashed;\n";

  // Connect TensorView with their TensorDomain
  // (this will trigger the traversal of the schedule graph)

  for (auto tv : tensor_views_) {
    addArc(tv->domain(), tv, "[style=dashed, arrowhead=none]");
    if (detail_level_ >= DetailLevel::Explicit) {
      // Maybe not the best way to handle the root domain, but should be okay
      addArc(
          tv,
          IrBuilder::create<TensorDomain>(tv->getRootDomain()),
          "[style=dashed, color=green, arrowhead=none]");

      if (tv->domain()->hasRFactor())
        addArc(
            tv,
            IrBuilder::create<TensorDomain>(tv->getRFactorDomain()),
            "[style=dashed, color=green, arrowhead=none]");
    }
  }

  graph_def_ << "  }\n";
}

void IrGraphGenerator::handle(const Statement* s) {
  OptInConstDispatch::handle(s);
}

void IrGraphGenerator::handle(const Val* v) {
  if (!visited(v)) {
    visited_.insert(v);
    if (const auto* def = v->definition()) {
      handle(def);
    }
    OptInConstDispatch::handle(v);
  }
}

void IrGraphGenerator::handle(const Expr* e) {
  if (!visited(e)) {
    visited_.insert(e);

    // Determine whether this is an IterDomain expression
    // If so, we may want to skip some inputs
    bool is_id_expr = false;
    bool is_scalar_expr = true;
    for (auto v : e->inputs()) {
      if (v->isA<IterDomain>()) {
        is_id_expr = true;
      }
      if (!v->isScalar()) {
        is_scalar_expr = false;
      }
    }

    // Default colors for non-IterDomain and non-Scalar expressions
    std::string arc_style;
    std::string border_color("blue");

    // draw node for e
    if (is_id_expr) {
      arc_style = "[color=lightgray]";
      border_color = "gray";
      maybeSetExprColor(e, Color::LIGHTGRAY);
    } else if (is_scalar_expr) {
      arc_style = "[color=green]";
      border_color = "green";
      maybeSetExprColor(e, Color::WHITE);
    }

    printExpr(e, e->getGraphvizLabel(), border_color);

    // inputs & outputs
    for (auto v : e->inputs()) {
      if (!config_.showSchedule && is_id_expr && !v->isA<IterDomain>()) {
        continue;
      }
      addArc(v, e, arc_style);
    }
    for (auto v : e->outputs()) {
      if (v->isA<TensorView>() && v->as<TensorView>()->domain()->hasRFactor()) {
        handle(v); // handle(v) will also create getid(v) + "_root"

        std::stringstream arc_def;
        arc_def << getid(e) << " -> " << getid(v) << "_root";
        arcs_.push_back(arc_def.str());
      } else {
        addArc(e, v, arc_style);
      }
    }
  }
}

void IrGraphGenerator::handle(const TensorDomain* td) {
  graph_def_ << "    " << getid(td) << " [label=\"TensorDomain\", "
             << "shape=note, color=gray, "
             << "style=filled, fillcolor=gray90, fontsize=10];\n";
  for (auto iter_domain : td->leaf()) {
    addArc(iter_domain, td, "[color=gray]");
  }
}

void IrGraphGenerator::handle(const IterDomain* id) {
  graph_def_ << "    " << getid(id) << " [label=\""
             << IrNodeLabel::gen(id, config_, detail_level_)
             << "\", shape=cds, color=gray, fontsize=10];\n";

  // Don't show starts or extents as nodes in high-level modes
  if (config_.showSchedule) {
    if (!id->start()->isZeroInt()) {
      addArc(id->start(), id, "[color=gray]");
    }
    addArc(id->extent(), id, "[color=green]");
  } else {
    visited_.insert(id);
  }
}

void IrGraphGenerator::handle(const Bool* b) {
  printValue(b, IrNodeLabel::gen(b, config_, detail_level_));
}

void IrGraphGenerator::handle(const Double* d) {
  printValue(d, IrNodeLabel::gen(d, config_, detail_level_));
}

void IrGraphGenerator::handle(const Int* i) {
  printValue(i, IrNodeLabel::gen(i, config_, detail_level_));
}

void IrGraphGenerator::handle(const ComplexDouble* i) {
  printValue(i, IrNodeLabel::gen(i, config_, detail_level_));
}

void IrGraphGenerator::handle(const NamedScalar* i) {
  printValue(i, IrNodeLabel::gen(i, config_, detail_level_));
}

void IrGraphGenerator::handle(const TensorView* tv) {
  auto has_rfactor = tv->domain()->hasRFactor();
  std::stringstream label;
  label << "{T" << tv->name() << (has_rfactor ? " (r-factor domain)" : "")
        << "|{";
  int axis = 0;
  for (auto iter_domain : tv->getMaybeRFactorDomain()) {
    if (axis != 0) {
      label << "|";
    }
    label << "<" << axis++ << "> "
          << IrNodeLabel::gen(iter_domain, config_, detail_level_);
  }
  label << "}}";

  const bool is_input = inputs_.find(tv) != inputs_.end();
  const bool is_output = outputs_.find(tv) != outputs_.end();

  std::string root_color("beige");
  std::string rfactor_color("pink");
  std::string input_color("palegreen");
  std::string output_color("lightblue");
  std::string this_color = is_input ? input_color
      : is_output                   ? output_color
      : has_rfactor                 ? rfactor_color
                                    : root_color;
  std::string style = "style=filled, fillcolor=" + this_color;

  graph_def_ << "    " << getid(tv) << " [label=\"" << label.str()
             << "\", shape=record, color=brown, " << style << "];\n";

  tensor_views_.push_back(tv);

  // If the rfactor domain differs from root domain, then we show the root
  // domain nearby and link it
  if (tv->domain()->hasRFactor()) {
    // TensorDomain contains multiple std::vector<IterDomain*> objects, and we
    // don't currently have an IR for those vectors. If they were Statements,
    // then we could handle them like other objects here. Instead, we'll just
    // place the handling code for printing here.

    // NOTE: since these are not statements, we also can't use getid, so we
    // derive an object id from that of tv.
    auto rootd_id = getid(tv) + "_root";

    std::stringstream rootd_label;
    std::string root_link_style("[color=lightgray]");
    rootd_label << "{T" << tv->name() << " (root domain)|";
    rootd_label << "{";
    axis = 0;
    for (auto iter_domain : tv->domain()->getRootDomain()) {
      handle(iter_domain);
      if (axis != 0) {
        rootd_label << "|";
      }
      rootd_label << "<" << axis << "> "
                  << IrNodeLabel::gen(iter_domain, config_, detail_level_);

      std::stringstream arc_def;
      arc_def << rootd_id << ":" << axis << " -> " << getid(iter_domain) << " "
              << root_link_style;
      arcs_.push_back(arc_def.str());

      axis++;
    }
    rootd_label << "}}";

    // Add arcs from root domain to its IterDomains
    axis = 0;
    for (auto iter_domain : tv->domain()->getRFactorDomain()) {
      handle(iter_domain);

      if (const auto* def = iter_domain->definition()) {
        handle(def);
      }

      std::stringstream arc_def;
      arc_def << getid(iter_domain) << " -> " << getid(tv) << ":" << axis << " "
              << root_link_style;
      arcs_.push_back(arc_def.str());
      axis++;
    }

    graph_def_ << "    " << rootd_id << " [label=\"" << rootd_label.str()
               << "\", shape=record, color=brown, style=filled, fillcolor="
               << root_color << "];\n";
  }
}

} // namespace nvfuser
