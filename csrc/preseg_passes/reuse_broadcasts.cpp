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
};

class BroadcastReuser : OptOutDispatch {
 public:
  BroadcastReuser(Fusion* fusion)
      : fusion_(fusion), id_model_(fusion, /*build_graphs=*/false) {
    id_model_.buildPermissiveGraph();
  }

  void run() {
    for (Expr* expr : fusion_->exprs()) {
      dispatch(expr);
    }

    // At this point we have found all of the new broadcasts and expands
    // introduced in the Fusion. These introduce new IDs and we need to know
    // which pairs of those IDs are safe to swap with one another.
    //
    // For example...
    //  TODO: more examples
    const ValGraph& graph = id_model_.idGraph(IdMappingMode::PERMISSIVE);
    std::unordered_map<ValGroup, std::unordered_set<ValGroup>>
        forbidden_replacements;
    for (TensorView* tv : fusion_->allTvs()) {
      for (IterDomain* id1 : tv->getLogicalDomain()) {
        ValGroup g1 = graph.toGroup(id1);
        for (IterDomain* id2 : tv->getLogicalDomain()) {
          ValGroup g2 = graph.toGroup(id2);
          if (g1 == g2) {
            continue;
          }
          forbidden_replacements[g1].insert(g2);
          forbidden_replacements[g2].insert(g1);
        }
      }
    }

    std::unordered_map<Val*, Val*> replacement_map;
    std::vector<const BroadcastInfo*> kept_bcasts;
    // NOTE: bcasts_ is built in topological order
    for (const BroadcastInfo& bcast : bcasts_) {
      bool reusing = false;
      for (const BroadcastInfo* kept_bcast : kept_bcasts) {
        bool reuse_is_safe = bcast.input == kept_bcast->input &&
            bcast.output->getLogicalDomain().size() ==
                kept_bcast->output->getLogicalDomain().size();
        if (reuse_is_safe) {
          // Verify reuse is safe by checking that any replaced ValGroups are
          // not forbidden
          for (const auto& [id, kept_id] :
               zip(bcast.output->getLogicalDomain(),
                   kept_bcast->output->getLogicalDomain())) {
            ValGroup g = graph.toGroup(id);
            ValGroup kept_g = graph.toGroup(kept_id);
            if (forbidden_replacements[g].count(kept_g)) {
              reuse_is_safe = false;
              break;
            }
          }
        }

        if (reuse_is_safe) {
          // Reuse kept_bcast
          replacement_map[bcast.output] = kept_bcast->output;
          reusing = true;
          break;
        }
      }
      if (!reusing) {
        // We'll keep bcast instead of reusing
        kept_bcasts.push_back(&bcast);
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
    bcasts_.emplace_back(unbroadcasted, output);
  }

 private:
  Fusion* fusion_;
  IdModel id_model_;
  // Results of BroadcastOps, excluding those whose only uses are ExpandOp, and
  // also results of ExpandOps where the definition of the input is a
  // BroadcastOp. These are candidates for reuse.
  std::vector<BroadcastInfo> bcasts_;
};

} // namespace

void ReuseBroadcasts::runPass(Fusion* fusion) {
  BroadcastReuser(fusion).run();
}

} // namespace nvfuser::preseg_passes
