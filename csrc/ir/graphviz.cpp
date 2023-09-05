// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/graphviz.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <type.h>

#include <fstream>

namespace nvfuser {

namespace {

// Private helper, generating node labels for IrGraphGenerator
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class IrNodeLabel final : private OptInConstDispatch {
  using DetailLevel = IrGraphGenerator::DetailLevel;

 public:
  static std::string gen(
      const Statement* node,
      DetailLevel detail_level = DetailLevel::Basic) {
    IrNodeLabel generator(detail_level);
    generator.OptInConstDispatch::dispatch(node);
    return generator.label_.str();
  }

 private:
  explicit IrNodeLabel(DetailLevel detail_level)
      : detail_level_(detail_level) {}

  ~IrNodeLabel() final = default;

  void handle(const Val* s) override {
    if (s->isSymbolic()) {
      label_ << ir_utils::varName(s);
    }
    if (s->isConst()) {
      if (detail_level_ >= DetailLevel::Explicit) {
        label_ << ir_utils::varName(s) << "=";
      }
      label_ << s->value();
    }
  }

  void handle(const NamedScalar* ns) override {
    label_ << ns->name();
  }

  void handle(const IterDomain* id) override {
    label_ << id->getIterType();
    label_ << id->getParallelType();

    label_ << "(";
    if (!id->start()->isZeroInt()) {
      label_ << IrNodeLabel::gen(id->start()) << " : ";
    }
    label_ << IrNodeLabel::gen(id->extent());
    label_ << ")";
  }

  void handle(const Split* split) override {
    label_ << "Split(inner=" << (split->innerSplit() ? "true" : "false")
           << ", factor=" << IrNodeLabel::gen(split->factor()) << ")";
  }

  void handle(const Merge* merge) override {
    label_ << "Merge";
  }

 private:
  std::stringstream label_;
  const DetailLevel detail_level_;
};

// Small color palette from the X11 theme
static const char* getColorFromIndex(size_t index) {
  const size_t number_of_colors = 10;
  index = index % number_of_colors;
  switch (index) {
    case 0: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "azure";
    case 1: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "pink";
    case 2: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "green";
    case 3: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "grey";
    case 4: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "yellow";
    case 5: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "lavender";
    case 6: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "cyan";
    case 7: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "white";
    case 8: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "magenta";
    case 9: // NOLINT(cppcoreguidelines-avoid-magic-numbers)
      return "red";
    default:
      break;
  }
  return "";
}

} // anonymous namespace

void IrGraphGenerator::print(
    const Fusion* fusion,
    const char* filename,
    DetailLevel detail_level,
    ExprColorMap* expr_color_map) {
  std::ofstream dot_file(filename);
  NVF_CHECK(dot_file.good(), "Failed to open the IR graph file");
  dot_file << toGraphviz(fusion, detail_level, expr_color_map);
}

std::string IrGraphGenerator::toGraphviz(
    const Fusion* fusion,
    DetailLevel detail_level,
    ExprColorMap* expr_color_map) {
  IrGraphGenerator ir_graph(fusion, detail_level, expr_color_map);
  return ir_graph.generate();
}

IrGraphGenerator::IrGraphGenerator(
    const Fusion* fusion,
    DetailLevel detail_level,
    ExprColorMap* expr_color_map)
    : detail_level_(detail_level),
      fusion_(fusion),
      expr_color_map_(expr_color_map) {
  // setup inputs & outputs
  // (indexes used to quickly check if a value is fusion input or output)
  for (const auto* input : fusion->inputs()) {
    NVF_CHECK(inputs_.count(input) == 0);
    inputs_.insert(input);
  }
  for (const auto* output : fusion->outputs()) {
    NVF_CHECK(outputs_.count(output) == 0);
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
  // We automatically visit (dispatch) the arc's source and destination
  dispatch(src);
  dispatch(dst);

  // generate and queue the arc definition
  std::stringstream arc_def;
  arc_def << getid(src) << " -> " << getid(dst) << " " << style;
  arcs_.push_back(arc_def.str());
}

void IrGraphGenerator::printExpr(const Expr* expr, const std::string& label) {
  graph_def_ << "    " << getid(expr) << " "
             << "[label=\"" << label << "\", shape=Mrecord, color=blue, "
             << "style=filled, fillcolor=";
  if (expr_color_map_ != nullptr && expr_color_map_->count(expr)) {
    graph_def_ << getColorFromIndex(expr_color_map_->at(expr));
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
  NVF_CHECK(graph_def_.str().empty());
  NVF_CHECK(visited_.empty());

  // record detail level
  graph_def_ << "// detail level: ";
  switch (detail_level_) {
    case DetailLevel::ComputeOnly:
      graph_def_ << "compute only\n";
      break;
    case DetailLevel::Basic:
      graph_def_ << "minimal\n";
      break;
    case DetailLevel::Explicit:
      graph_def_ << "explicit\n";
      break;
    case DetailLevel::Verbose:
      graph_def_ << "verbose\n";
      break;
    default:
      NVF_CHECK(!"Unexpected detail level");
  }

  graph_def_ << "digraph fusion_ir {\n"
             << "  node [shape=circle, color=gray];\n"
             << "  edge [color=black];\n";

  // Compute graph
  generateComputeGraph();

  // Schedule graph
  if (detail_level_ > DetailLevel::ComputeOnly) {
    generateScheduleGraph();
  }

  // All expressions & values
  // (These are otherwise unreacheable (dead) nodes)
  if (detail_level_ >= DetailLevel::Verbose) {
    for (const auto* expr : fusion_->unordered_exprs()) {
      dispatch(expr);
    }
    for (const auto* val : fusion_->vals()) {
      dispatch(val);
    }
  }

  // Finally, print all arc definitions
  for (const auto& arc : arcs_) {
    graph_def_ << "  " << arc << ";\n";
  }

  graph_def_ << "}\n";

  // Make sure that all referenced nodes have been visited
  for (const auto& kv : id_map_) {
    NVF_CHECK(visited(kv.first));
  }

  return graph_def_.str();
}

void IrGraphGenerator::generateComputeGraph() {
  graph_def_ << "  subgraph cluster_compute {\n"
             << "    label=\"compute\";\n"
             << "    style=dashed;\n";

  // Inputs
  for (const auto* input : fusion_->inputs()) {
    dispatch(input);
  }

  // Outputs
  for (const auto* output : fusion_->outputs()) {
    dispatch(output);
  }

  graph_def_ << "  }\n";
}

void IrGraphGenerator::generateScheduleGraph() {
  graph_def_ << "  subgraph cluster_schedule {\n"
             << "    label=\"schedule\";\n"
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

void IrGraphGenerator::dispatch(const Statement* s) {
  OptInConstDispatch::dispatch(s);
}

void IrGraphGenerator::dispatch(const Val* v) {
  if (!visited(v)) {
    visited_.insert(v);
    if (const auto* def = v->definition()) {
      dispatch(def);
    }
    OptInConstDispatch::dispatch(v);
  }
}

void IrGraphGenerator::dispatch(const Expr* e) {
  if (!visited(e)) {
    visited_.insert(e);

    // node
    printExpr(e, e->getGraphvizLabel());

    // inputs & outputs
    for (auto v : e->inputs()) {
      addArc(v, e);
    }
    for (auto v : e->outputs()) {
      addArc(e, v);
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
  graph_def_ << "    " << getid(id) << " [label=\"" << IrNodeLabel::gen(id)
             << "\", shape=cds, color=gray, fontsize=10];\n";

  if (!id->start()->isZeroInt()) {
    addArc(id->start(), id, "[color=gray]");
  }

  addArc(id->extent(), id, "[color=gray]");
}

void IrGraphGenerator::handle(const Val* s) {
  printValue(s, IrNodeLabel::gen(s, detail_level_));
}

void IrGraphGenerator::handle(const NamedScalar* i) {
  printValue(i, IrNodeLabel::gen(i, detail_level_));
}

void IrGraphGenerator::handle(const TensorView* tv) {
  std::stringstream label;
  label << "{T" << tv->name() << "|";
  label << "{";
  bool first_axis = true;
  for (auto iter_domain : tv->getLeafDomain()) {
    if (first_axis) {
      first_axis = false;
    } else {
      label << "|";
    }
    label << IrNodeLabel::gen(iter_domain);
  }
  label << "}}";

  const bool is_input = inputs_.find(tv) != inputs_.end();
  const bool is_output = outputs_.find(tv) != outputs_.end();

  const char* style = is_input ? "style=filled, fillcolor=palegreen"
      : is_output              ? "style=filled, fillcolor=lightblue"
                               : "style=filled, fillcolor=beige";

  graph_def_ << "    " << getid(tv) << " [label=\"" << label.str()
             << "\", shape=Mrecord, color=brown, " << style << "];\n";

  tensor_views_.push_back(tv);
}

} // namespace nvfuser
