// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/id_model.h>
#include <id_model/utils.h>
#include <ir/utils.h>

namespace nvfuser {

std::string TransformToDot::get(Fusion* fusion) {
  TransformToDot dot;
  dot.handle(fusion);
  return dot.buf_.str();
}

void TransformToDot::handle(Fusion* fusion) {
  indent() << "digraph {\n";
  ++indent_;
  indent() << "node [shape=plaintext fontsize=\"20\"];\n";
  indent() << "graph [ordering=\"out\"];\n";
  // indent() << "compound = true;\n";

  for (const auto tv : ir_utils::allTvs(fusion)) {
    handle(tv);
  }

#if 0
  // Draw edges between tensors
  for (const auto expr: fusion->exprs()) {
    for (const auto inp: expr->inputs()) {
      if (!inp->isA<TensorView>()) {
        continue;
      }
      auto inp_tv = inp->as<TensorView>();
      for (const auto out: expr->outputs()) {
        if (!out->isA<TensorView>()) {
          continue;
        }
        auto out_tv = out->as<TensorView>();
        NVF_ERROR(inp_tv->nDims() > 0);
        NVF_ERROR(out_tv->nDims() > 0);
        indent() << "t" << inp_tv->name() << "r"
                 << " -> "
                 << "t" << out_tv->name() << "l"
                 << " [ "
                 << "ltail=cluster_t" << inp->name()
                 << " lhead=cluster_t" << out->name()
                 << " ];\n";
      }
    }
  }
#endif
  --indent_;
  indent() << "}\n";
}

void TransformToDot::handle(TensorView* tv) {
  indent() << "subgraph cluster_t" << tv->name() << " {\n";
  ++indent_;
  indent() << "label = \"t" << tv->name() << " ca_pos("
           << tv->getComputeAtPosition() << ")\"\n";
  indent() << "fontsize = \"20\";\n";
  indent() << "graph [style=dotted];\n";

  // Root domain
  // handle(tv->getMaybeRFactorDomain(), "root");

  // Invisible connecting points
  // indent() << "t" << tv->name() << "l;\n";
  // indent() << "t" << tv->name() << "r;\n";

  // Rfactor domain
  // if (tv->hasRFactor()) {
  //    handle(tv->getMaybeRFactorDomain(), "rfactor");
  //}

  // Leaf
  // handle(tv->getLeafDomain(), "leaf");

  markRfactor(tv);

  const auto all_exp = DependencyCheck::getAllExprsBetween(
      {tv->getRootDomain().begin(), tv->getRootDomain().end()},
      {tv->getLeafDomain().begin(), tv->getLeafDomain().end()});

  for (auto exp : all_exp) {
    handle(exp);
  }

  enforceRootOrder(tv);

  --indent_;
  indent() << "}\n";
}

void TransformToDot::markRfactor(TensorView* tv) {
  for (auto id : tv->getMaybeRFactorDomain()) {
    indent() << id->name() << " [shape=circle];\n";
  }
}

void TransformToDot::enforceRootOrder(TensorView* tv) {
  indent() << "{\n";
  ++indent_;
  indent() << "rank=same;\n";
  indent() << "edge [style=invis];\n";
  bool first = true;
  std::stringstream ss;
  // ss << "t" << tv->name() << "l";
  for (auto id : tv->getRootDomain()) {
    if (!first) {
      ss << " -> ";
    }
    ss << id->name();
    first = false;
  }
  // ss << " -> t" << tv->name() << "r";
  indent() << ss.str() << ";\n";
  indent() << "rankdir = LR;\n";
  --indent_;
  indent() << "}\n";

#if 0
  indent() << "{\n";
  ++indent_;
  indent() << "rank=same;\n";
  indent() << "edge [style=invis];\n";
  bool first = true;
  std::stringstream ss_leaf;
  for (auto id: tv->getLeafDomain()) {
    if (!first) {
      ss_leaf << " -> ";
    }
    ss_leaf << id->name();
    first = false;
  }
  indent() << ss_leaf.str() << ";\n";
  indent() << "rankdir = LR;\n";
  --indent_;
  indent() << "}\n";
#endif
}

void TransformToDot::handle(
    const std::vector<IterDomain*>& domain,
    std::string label) {
  indent() << "subgraph cluster_" << label << " {\n";
  ++indent_;
  if (label == "rfactor") {
    indent() << "label=" << label << ";\n";
    indent() << "graph[style=dotted];\n";
  } else {
    indent() << "graph[style=invis];\n";
  }

  indent() << "rank = same;\n";

  for (const auto id : domain) {
    indent() << id->name() << ";\n";
  }

  --indent_;
  indent() << "}\n";
}

void TransformToDot::handle(Expr* expr) {
  for (auto inp : expr->inputs()) {
    handle(inp->as<IterDomain>());
  }
  for (auto out : expr->outputs()) {
    handle(out->as<IterDomain>());
  }

  for (auto inp : expr->inputs()) {
    for (auto out : expr->outputs()) {
      indent() << inp->name() << " -> " << out->name() << "\n";
    }
  }

  if (getenv("OUTPUT_ORDER")) {
    if (expr->outputs().size() > 1) {
      // Enfore ordering
      indent() << "{\n";
      ++indent_;
      indent() << "rank=same;\n";
      indent() << "edge [style=invis];\n";
      bool first = true;
      std::stringstream ss;
      for (auto id : expr->outputs()) {
        if (!first) {
          ss << " -> ";
        }
        ss << id->name();
        first = false;
      }
      indent() << ss.str() << ";\n";
      indent() << "rankdir = LR;\n";
      --indent_;
      indent() << "}\n";
    }
  }
}

void TransformToDot::handle(IterDomain* id) {
  if (printed_vals_.find(id) != printed_vals_.end()) {
    return;
  }

  indent() << id->name()
           << " [fontcolor=" << (id->isBroadcast() ? "blue" : "black")
           << "];\n";

  printed_vals_.insert(id);
}

std::stringstream& TransformToDot::indent() {
  for (int i = 0; i < indent_; ++i) {
    buf_ << "  ";
  }
  return buf_;
}

} // namespace nvfuser
