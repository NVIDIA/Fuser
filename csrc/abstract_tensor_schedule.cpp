// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <abstract_tensor.h>
#include <abstract_tensor_schedule.h>
#include <ir/internal_base_nodes.h>
#include <iter_visitor.h>
#include <val_graph.h>

namespace nvfuser {

namespace {

class AbstractTensorSchedule : public IterVisitor {
 public:
  static void apply(
      const AbstractTensor& abstract,
      TensorView* concrete,
      ValGraph* graph) {
    AbstractTensorSchedule ats(abstract, concrete, graph);
    ats.run();
  }

 private:
  AbstractTensorSchedule(
      const AbstractTensor& abstract,
      TensorView* concrete,
      ValGraph* graph)
      : abstract_(abstract), concrete_(concrete), graph_(graph) {}

  void run() {
    findNearestProducers();

    // Now, for each ValGroup in abstract, find the closest producer ValGroups
    // with entries in concrete_ids_ and replay the path from them. If none
    // exists, then do not include this dimension in the output
    std::vector<IterDomain*> loop_domain;
    for (const AbstractId& abs_id : abstract_.domain) {
      IterDomain* new_id = replayAbstractId(abs_id);
      if (new_id == nullptr) {
        continue;
      }
      loop_domain.push_back(new_id);
    }
    concrete_->setLoopDomain(loop_domain);
  }

  ValGroup abstractIdToValGroup(const AbstractId& abs_id) {
    if (abs_id.is<IterDomain*>()) {
      return graph_->toGroup(abs_id.as<IterDomain*>());
    } else if (abs_id.is<ValGroupAndItsGraph>()) {
      if (graph_ == nullptr) {
        graph_ = abs_id.as<ValGroupAndItsGraph>().graph;
      } else {
        NVF_ERROR(graph_ == abs_id.as<ValGroupAndItsGraph>().graph);
      }
      return abs_id.as<ValGroupAndItsGraph>().group;
    }
    NVF_ERROR(false, "AbstractId must be IterDomain* or ValGroupAndItsGraph");
  }

  //! Work backward from each loop IterDomain in concrete. When we find an
  //! IterDomain* with a ValGroup in scheduled_val_groups, we map to it
  //! (including siblings) in concrete_ids_. Assert that we do find a
  //! scheduled ValGroup, otherwise we would not know where to place this
  //! IterDomain.
  void findNearestProducers() {
    // Record all ValGroups that are producers of those in abstract_. This lets
    // us find starting IterDomains for scheduling concrete's loop domain.
    // TODO: this should probably be a more general utility like DependencyCheck
    // in val_graph_visitor.h
    std::unordered_set<ValGroup> scheduled_val_groups;
    std::stack<ValGroup> vg_stack;
    for (const AbstractId& abs_id : abstract_.domain) {
      vg_stack.push(abstractIdToValGroup(abs_id));
    }
    while (!vg_stack.empty()) {
      ValGroup vg = vg_stack.top();
      vg_stack.pop();
      scheduled_val_groups.insert(vg);
      // TODO: Do we need to check for cycles?
      for (const ExprGroup& eg : graph_->getDefinitions(vg)) {
        for (Expr* e : *eg) {
          for (Val* inp : e->inputs()) {
            ValGroup vg_inp = graph_->toGroup(inp);
            vg_stack.push(vg_inp);
          }
        }
      }
    }

    // Now traverse c2p from concrete loop domain, stopping when we find a
    // scheduled ValGroup
    std::stack<IterDomain*> id_stack;
    for (IterDomain* id : concrete_->getLoopDomain()) {
      id_stack.push(id);
    }
    while (!id_stack.empty()) {
      IterDomain* id = id_stack.top();
      id_stack.pop();
      ValGroup g = graph_->toGroup(id);
      if (scheduled_val_groups.find(g) != scheduled_val_groups.end()) {
        concrete_ids_.emplace(g, id);
        continue;
      }
      NVF_ERROR(
          id->definition() != nullptr,
          "Root IterDomain ",
          id->toString(),
          " does not appear in the history of any ValGroups in abstract tensor");
      for (Val* inp : id->definition()->inputs()) {
        if (auto inp_id = dynamic_cast<IterDomain*>(inp)) {
          id_stack.push(inp_id);
        }
      }
    }
  }

  IterDomain* replayAbstractId(AbstractId abs_id) {
    ValGroup g = abstractIdToValGroup(abs_id);

    // This holds ValGroups that we cannot compute from concrete_'s root. When
    // we detect that any inputs of an ExprGroup are in these groups, we know
    // that we cannot compute that ExprGroup so we should skip it.
    std::unordered_set<ValGroup> uncomputable_groups;

    std::stack<ValGroup> vg_stack({g});
    // Strategy: fill out concrete_ids_ in the direction of g
    auto propagate = [&](ValGroup vg) {
      // We need to try and produce an ID for this group, so we look at
      // definitions for the Vals in this group and check whether we have their
      // inputs computed yet (concrete_ids_) and whether they can be computed
      // (uncomputable_groups).
      std::vector<ValGroup> uncomputed_producer_groups;
      for (const ExprGroup& eg : graph_->getDefinitions(vg)) {
        for (Expr* e : *eg) {
          if (!uncomputed_producer_groups.empty()) {
            // We already have something to compute in the next round, so skip.
            continue;
          }
          std::vector<ValGroup> vg_inps;
          std::vector<IterDomain*> id_inps;
          bool all_inputs_computed = true;
          for (Val* inp : e->inputs()) {
            ValGroup vg_inp = graph_->toGroup(inp);
            vg_inps.push_back(vg_inp);
            auto inp_it = concrete_ids_.find(vg_inp);
            if (inp_it != concrete_ids_.end()) {
              id_inps.push_back(inp_it->second);
            } else {
              // this input is not yet computed
              all_inputs_computed = false;
              if (uncomputable_groups.count(vg_inp) == 0) {
                // Input is not yet proven to be incomputable, so try and
                // compute it in the next iteration
                uncomputed_producer_groups.push_back(vg_inp);
              }
            }
          }
          // some input is not yet computed
          if (!all_inputs_computed) {
            if (uncomputed_producer_groups.empty() && !id_inps.empty()) {
              // There might be some uncomputable producer groups, but some are
              // computed. Just pass those on:
              // TODO: this behavior should be specialized to the type of Expr I
              // believe
              NVF_ERROR(id_inps.size() == 1);
              concrete_ids_.emplace(vg, id_inps.front());
            }
            continue;
          }
          // Compute new ID expression
          Expr* id_expr = nullptr;
          if (auto* m = dynamic_cast<Merge*>(e)) {
            NVF_ERROR(id_inps.size() == 2);
            auto* new_id = IterDomain::merge(id_inps[0], id_inps[1]);
            id_expr = new_id->definition();
          } else if (auto* s = dynamic_cast<Split*>(e)) {
            NVF_ERROR(id_inps.size() == 1);
            auto new_ids =
                IterDomain::split(id_inps[0], s->factor(), s->innerSplit());
            id_expr = new_ids.first->definition();
          } else if (auto* s = dynamic_cast<Swizzle*>(e)) {
            NVF_ERROR(id_inps.size() == 2);
            auto new_ids =
                IterDomain::swizzle(s->swizzleType(), id_inps[0], id_inps[1]);
            id_expr = new_ids.first->definition();
          } else if (auto* s = dynamic_cast<Swizzle2D*>(e)) {
            NVF_ERROR(id_inps.size() == 2);
            auto new_ids = IterDomain::swizzle(
                s->swizzleType(), id_inps[0], id_inps[1], s->swizzleMode());
            id_expr = new_ids.first->definition();
          } else {
            NVF_ERROR(false, "Unhandled IterDomain expression ", e->toString());
          }
          // Update the mapping to point to all of the newly created Expr's
          // outputs
          NVF_ERROR(id_expr != nullptr);
          NVF_ERROR(id_expr->outputs().size() == e->outputs().size());
          for (size_t i : c10::irange(e->outputs().size())) {
            ValGroup vg_outp = graph_->toGroup(e->output((int64_t)i));
            auto* id_outp = id_expr->output((int64_t)i)->as<IterDomain>();
            concrete_ids_.emplace(vg_outp, id_outp);
          }
        }
      }
      if (uncomputed_producer_groups.empty()) {
        // All of the defining expressions are proven uncomputable, so mark vg
        // uncomputable
        uncomputable_groups.insert(vg);
        return;
      } else {
        // There are some uncomputed producer groups that might be computable,
        // so try again after processing those producer groups
        vg_stack.push(vg);
        for (ValGroup next_vg : uncomputed_producer_groups) {
          vg_stack.push(next_vg);
        }
      }
    };

    while (!vg_stack.empty()) {
      ValGroup vg = vg_stack.top();
      vg_stack.pop();
      if (concrete_ids_.count(vg) != 0) {
        continue;
      }
      propagate(vg);
    }

    // Now check whether there is a concrete entry for abs_id (i.e. g). If not,
    // that means this dimension is missing and should not be included in the
    // loop domain, so return nullptr.
    auto it = concrete_ids_.find(g);
    return it == concrete_ids_.end() ? nullptr : it->second;
  }

 private:
  const AbstractTensor& abstract_;
  TensorView* concrete_;
  ValGraph* graph_;

  std::unordered_map<ValGroup, IterDomain*> concrete_ids_;
};

} // namespace

void applyAbstractSchedule(
    const AbstractTensor& abstract,
    TensorView* concrete,
    ValGraph* graph) {
  AbstractTensorSchedule::apply(abstract, concrete, graph);
}

} // namespace nvfuser
