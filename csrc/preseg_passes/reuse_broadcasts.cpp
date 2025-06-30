// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/reuse_broadcasts.h>

#include <expr_evaluator.h>
#include <id_model/id_model.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <polymorphic_value.h>
#include <utils.h>
#include <val_graph.h>

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

namespace nvfuser::preseg_passes {

namespace {

struct BroadcastInfo {
  // The unbroadcasted input
  TensorView* input;
  // The output of BroadcastOp or BroadcastOp+ExpandOp
  TensorView* output;
  std::vector<ValGroup> output_groups;

  bool operator==(const BroadcastInfo& other) const {
    if (input != other.input) {
      return false;
    }
    if (output_groups.size() != other.output_groups.size()) {
      return false;
    }
    for (const auto& [g, other_g] : zip(output_groups, other.output_groups)) {
      for (Val* v : *g) {
        auto* extent = v->as<IterDomain>()->getMaybeExpandedExtent();
        if (extent->isOneInt()) {
          continue;
        }
        for (Val* other_v : *other_g) {
          auto* other_extent = other_v->as<IterDomain>()->getMaybeExpandedExtent();
          if (other_extent->isOneInt()) {
            continue;
          }
          if (!extent->sameAs(other_extent)) {
            return false;
          }
        }
      }
    }
    return true;
  }
};

class BroadcastReuser : OptOutDispatch {
 public:
  BroadcastReuser(Fusion* fusion) : fusion_(fusion), id_model_(fusion, /*build_graphs=*/false) {
    id_model_.buildPermissiveGraph();
  }

  void run() {
    for (Expr* expr : fusion_->exprs()) {
      dispatch(expr);
    }

    std::unordered_map<Val*, Val*> replacement_map;
    std::vector<const BroadcastInfo*> kept_bcasts;
    // NOTE: bcasts_ is built in topological order
    for (const BroadcastInfo& bcast : bcasts_) {
      for (const BroadcastInfo* kept_bcast : kept_bcasts) {
        if (bcast == *kept_bcast) {
          // Reuse kept_bcast
          replacement_map[bcast.output] = kept_bcast->output;
        } else {
          // We'll keep bcast instead of reusing
          kept_bcasts.push_back(&bcast);
        }
      }
    }

    ir_utils::replaceValue(fusion_, replacement_map);
  }
 
 private:
  using OptOutDispatch::handle;

  void handle(BroadcastOp* bop) {
    auto* tv = bop->out()->as<TensorView>();
    if (!std::all_of(tv->uses().begin(), tv->uses().end(), [](Expr* use) {
          return use->isA<ExpandOp>();
          })) {
      registerBroadcast(tv);
    }
  }

  void handle(ExpandOp* eop) {
    auto* tv = eop->out()->as<TensorView>();
    Expr* def = eop->in()->definition();
    if (def && def->isA<BroadcastOp>()) {
      registerBroadcast(tv);
    }
  }

  void registerBroadcast(TensorView* output) {
    TensorView* unbroadcasted = output;
    Expr* def = unbroadcasted->definition();
    while (def && def->isOneOf<BroadcastOp, ExpandOp>()) {
      unbroadcasted = def->input(0)->as<TensorView>();
      def = unbroadcasted->definition();
    }

    const ValGraph& graph = id_model_.idGraph(IdMappingMode::PERMISSIVE);
    std::vector<ValGroup> abs_domain;
    abs_domain.reserve(output->getLogicalDomain().size());
    for (IterDomain* id : output->getLogicalDomain()) {
      abs_domain.push_back(graph.toGroup(id));
    }

    bcasts_.emplace_back(unbroadcasted, output, std::move(abs_domain));
  }

 private:
  Fusion* fusion_;
  IdModel id_model_;
  // Results of BroadcastOps, excluding those whose only uses are ExpandOp, and also
  // results of ExpandOps where the definition of the input is a BroadcastOp.
  // These are candidates for reuse.
  std::vector<BroadcastInfo> bcasts_;
};

} // namespace

void ReuseBroadcasts::runPass(Fusion* fusion) {
  BroadcastReuser(fusion).run();
}

} // namespace nvfuser::preseg_passes
