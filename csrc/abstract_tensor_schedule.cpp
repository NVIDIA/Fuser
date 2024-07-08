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

class AbstractTensorSchedule {
 public:
  static void apply(
      const AbstractTensor& abstract_tensor,
      TensorView* concrete,
      ValGraph* graph) {
    AbstractTensorSchedule ats(abstract_tensor, concrete, graph);
    ats.run();
  }

 private:
  AbstractTensorSchedule(
      const AbstractTensor& abstract_tensor,
      TensorView* tv,
      ValGraph* graph)
      : abstract_tensor_(abstract_tensor), tv_(tv), graph_(graph) {}

  void run() {
    findNearestProducers();

    // Now, for each ValGroup in abstract, find the closest producer ValGroups
    // with entries in tv_ids_ and replay the path from them. If none
    // exists, then do not include this dimension in the output
    std::vector<IterDomain*> loop_domain;
    for (const AbstractId& abs_id : abstract_tensor_.domain) {
      IterDomain* new_id = replayAbstractId(abs_id);
      if (new_id == nullptr) {
        continue;
      }
      loop_domain.push_back(new_id);
    }
    tv_->setLoopDomain(loop_domain);
  }

  ValGroup abstractIdToValGroup(const AbstractId& abs_id) {
    NVF_ERROR(
        abs_id.is<ValGroupAndItsGraph>(),
        "AbstractId must be a ValGroupAndItsGraph");
    const ValGroupAndItsGraph& vgg = abs_id.as<ValGroupAndItsGraph>();
    if (graph_ == nullptr) {
      graph_ = vgg.graph;
    } else {
      NVF_ERROR(graph_ == vgg.graph);
    }
    return vgg.group;
  }

  //! Work backward from each loop IterDomain in concrete. When we find an
  //! IterDomain* with a ValGroup in scheduled_val_groups, we map to it
  //! (including siblings) in tv_ids_. Assert that we do find a
  //! scheduled ValGroup, otherwise we would not know where to place this
  //! IterDomain.
  //!
  //! For example, suppose we have the following:
  //!
  //!   tv_:
  //!     root: iS0{i0} iS1{i1}
  //!     loop: iS3{ceilDiv(i0 * i1, 16)} iS4{16}
  //!     transforms:
  //!       iS2 = merge(iS0, iS1)
  //!       iS3, iS4 = split(iS2, 16)
  //!   graph:
  //!     ValGroup 0: 0 8
  //!     ValGroup 1: 1 9
  //!     ValGroup 2: 2 10
  //!     ValGroup 3: 3
  //!     ValGroup 4: 4
  //!     ValGroup 5: 11
  //!     ValGroup 6: 12
  //!     ExprGroup 0: iS11{ceilDiv(i0 * i1, 32)}, iS12{32} = split(iS10, 32)
  //!   abstract_tensor_:
  //!     {{ValGroup 5, graph}, {ValGroup 6, graph}}
  //!
  //! Then findNearestProducers() works backward from the loop domain of
  //! tv_ (ValGroups 3 and 4) and will identify ValGroup 2 as a producer
  //! to abstract_tensor_ and map it to the concrete domain iS2{i0}.
  //!
  //! Note that any dangling transforms will be discarded; in this example since
  //! we work from iS2, we will discard the split that produced iS3 and iS4.
  void findNearestProducers() {
    // Record all ValGroups that are producers of those in abstract_tensor_.
    // This lets us find starting IterDomains for scheduling concrete's loop
    // domain.
    // TODO: this should probably use a new more general utility that resembles
    // StmtSort::getExprsTo in val_graph_visitor.h
    std::unordered_set<ValGroup> scheduled_val_groups;
    std::stack<ValGroup> vg_stack;
    for (const AbstractId& abs_id : abstract_tensor_.domain) {
      vg_stack.push(abstractIdToValGroup(abs_id));
    }
    while (!vg_stack.empty()) {
      ValGroup vg = vg_stack.top();
      vg_stack.pop();
      bool inserted = scheduled_val_groups.insert(vg).second;
      if (!inserted) {
        // Avoid cycles
        continue;
      }
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
    for (IterDomain* id : tv_->getLoopDomain()) {
      id_stack.push(id);
    }
    while (!id_stack.empty()) {
      IterDomain* id = id_stack.top();
      id_stack.pop();
      ValGroup g = graph_->toGroup(id);
      if (scheduled_val_groups.find(g) != scheduled_val_groups.end()) {
        tv_ids_.emplace(g, id);
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

    // This holds ValGroups that we cannot compute from tv_'s root. When
    // we detect that any inputs of an ExprGroup are in these groups, we know
    // that we cannot compute that ExprGroup so we should skip it.
    //
    // For example, suppose we had
    //   tv_:
    //     root: iS0{i0} iS1{i1} iS2{i2}
    //
    //   graph:
    //     ValGroup 0: iS0
    //     ValGroup 1: iS1
    //     ValGroup 2: iS2 iS6
    //     ValGroup 3: iS3
    //     ValGroup 4: iS4
    //     ValGroup 5: iS5
    //     ValGroup 6: iS7
    //     ValGroup 7: iS8
    //     ExprGroup 0:
    //       iS3 = merge(iS0, iS1)
    //     ExprGroup 1:
    //       iS4, iS5 = split(iS3, 16)
    //     ExprGroup 2:
    //       iS8 = merge(iS6, iS7)
    //
    //   abstract_tensor_.domain:
    //     ValGroup 4, ValGroup 5, ValGroup 7
    //
    // In this case, ValGroups 4 and 5 are computable since those ValGroups are
    // produced by ExprGroup 1 which itself produced by ExprGroup 0, and
    // tv_ includes iS0 and iS1.
    //
    // However, ValGroup 7 is not computable. It is produced by ExprGroup 2
    // whose producer ValGroups are 2 and 6. ValGroup 2 is computable since iS0
    // is in tv_, however there is no IterDomain in tv_ that can be
    // used to represent ValGroup 6 which also has no producer ValGroups.
    std::unordered_set<ValGroup> uncomputable_groups;

    std::stack<ValGroup> vg_stack({g});
    // Strategy: fill out tv_ids_ in the direction of g
    auto propagate = [&](ValGroup vg) {
      // We need to try and produce an ID for this group, so we look at
      // definitions for the Vals in this group and check whether we have their
      // inputs computed yet (tv_ids_) and whether they can be computed
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
            auto inp_it = tv_ids_.find(vg_inp);
            if (inp_it != tv_ids_.end()) {
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
              // computed. Just pass those computed inputs through.
              NVF_ERROR(id_inps.size() == 1);
              tv_ids_.emplace(vg, id_inps.front());
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
            graph_->initializeVal(id_outp, vg_outp);
            tv_ids_.emplace(vg_outp, id_outp);
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
        for (const ValGroup& next_vg : uncomputed_producer_groups) {
          vg_stack.push(next_vg);
        }
      }
    };

    while (!vg_stack.empty()) {
      ValGroup vg = vg_stack.top();
      vg_stack.pop();
      if (tv_ids_.count(vg) != 0) {
        continue;
      }
      propagate(vg);
    }

    // Now check whether there is a concrete entry for abs_id (i.e. g). If not,
    // that means this dimension is missing and should not be included in the
    // loop domain, so return nullptr.
    auto it = tv_ids_.find(g);
    return it == tv_ids_.end() ? nullptr : it->second;
  }

 private:
  const AbstractTensor& abstract_tensor_;
  TensorView* tv_;
  ValGraph* graph_;

  std::unordered_map<ValGroup, IterDomain*> tv_ids_;
};

} // namespace

void applyAbstractTransforms(
    const AbstractTensor& abstract_tensor,
    TensorView* tv,
    ValGraph* graph) {
  AbstractTensorSchedule::apply(abstract_tensor, concrete, graph);
}

} // namespace nvfuser
