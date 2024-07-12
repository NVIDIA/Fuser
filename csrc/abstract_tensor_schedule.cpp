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
#include <transform_iter.h>
#include <val_graph.h>

namespace nvfuser {

namespace {

// Given an IterDomain expression, replay it using the provided inputs and
// return the new Expr
Expr* replayIdExpr(Expr* expr, const std::vector<IterDomain*>& inputs) {
  std::vector<IterDomain*> target_domain;
  target_domain.reserve(expr->outputs().size());
  for (Val* outp : expr->outputs()) {
    target_domain.push_back(outp->as<IterDomain>());
  }
  std::unordered_map<IterDomain*, IterDomain*> id_map;
  NVF_ERROR(inputs.size() == expr->inputs().size());
  for (size_t i : c10::irange(inputs.size())) {
    id_map.emplace(expr->input(i)->as<IterDomain>(), inputs[i]);
  }
  ReplayTransformations replay(target_domain, id_map);
  const std::unordered_map<IterDomain*, IterDomain*>& replay_map =
      replay.getReplay();
  auto it = replay_map.find(target_domain.front());
  NVF_ERROR(it != replay_map.end());
  return it->second->definition();
}

class AbstractTensorSchedule {
 public:
  static void apply(
      const AbstractTensor& abstract_tensor,
      const std::vector<TensorView*>& tvs,
      ValGraph* graph) {
    AbstractTensorSchedule ats(abstract_tensor, graph);
    for (TensorView* tv : tvs) {
      ats.run(tv);
    }
  }

 private:
  AbstractTensorSchedule(const AbstractTensor& abstract_tensor, ValGraph* graph)
      : abstract_tensor_(abstract_tensor), graph_(graph) {
    findScheduledValGroups();
  }

  using GroupIdMap = std::unordered_map<ValGroup, IterDomain*>;

  void run(TensorView* tv) {
    // This holds a mapping from scheduled val groups to IterDomains in tv.
    // Note this is non-const since we insert new IDs into it while replaying.
    GroupIdMap computed_ids = mapScheduledGroupsToLoopIterDomains(tv);

    // Now, for each ValGroup in abstract, find the closest producer ValGroups
    // with entries in computed_ids and replay the path from them. If none
    // exists, then do not include this dimension in the output
    std::vector<IterDomain*> loop_domain;
    for (const AbstractId& abs_id : abstract_tensor_.domain) {
      IterDomain* new_id = replayAbstractId(abs_id, computed_ids);
      if (new_id == nullptr) {
        continue;
      }
      loop_domain.push_back(new_id);
    }
    tv->setLoopDomain(loop_domain);
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

  //! Record all ValGroups that are producers of those in abstract_tensor_.
  //! This lets us find starting IterDomains for scheduling concrete's loop
  //! domain.
  //! TODO: this should probably use a new more general utility that resembles
  //! StmtSort::getExprsTo in val_graph_visitor.h
  void findScheduledValGroups() {
    std::stack<ValGroup> vg_stack;
    for (const AbstractId& abs_id : abstract_tensor_.domain) {
      vg_stack.push(abstractIdToValGroup(abs_id));
    }
    while (!vg_stack.empty()) {
      ValGroup vg = vg_stack.top();
      vg_stack.pop();
      bool inserted = scheduled_val_groups_.insert(vg).second;
      if (!inserted) {
        // Avoid cycles
        continue;
      }
      for (const ExprGroup& eg : graph_->getDefinitions(vg)) {
        for (const ValGroup& vg_inp : graph_->inputGroups(eg)) {
          vg_stack.push(vg_inp);
        }
      }
    }
  }

  //! Look at all ValGroups that are producers of abstract_tensor_.domain and
  //! map any of those containing loop IterDomains in tv to those loop
  //! IterDomains. These provide the starting points for scheduling.
  GroupIdMap mapScheduledGroupsToLoopIterDomains(TensorView* tv) const {
    GroupIdMap computed_ids;

    // Look up each loop IterDomain in tv and assert that it is in
    // scheduled_val_groups_, then map it.
    for (IterDomain* loop_id : tv->getLoopDomain()) {
      ValGroup vg = graph_->toGroup(loop_id);
      // NOTE: this check guarantees that we will only ever _append_ transforms
      // to the loop domain of tv. If instead we searched for producer IDs in
      // the root->loop path of tv, then we could wind up silently erasing
      // some pre-existing transforms.
      NVF_CHECK(
          scheduled_val_groups_.count(vg) > 0,
          "Found loop IterDomain ",
          loop_id->toString(),
          " that is not described by abstract tensor.");
      computed_ids.emplace(vg, loop_id);
    }

    return computed_ids;
  }

  IterDomain* replayAbstractId(AbstractId abs_id, GroupIdMap& computed_ids) {
    const ValGroup& g = abstractIdToValGroup(abs_id);

    // Uncomputable ValGroups are those that we cannot compute from tv's root.
    // When we detect that any inputs of an ExprGroup are in these groups, we
    // know that we cannot compute that ExprGroup so we should skip it when
    // scheduling tv.
    //
    // For example, suppose we had
    //   tv:
    //     logical/loop: iS0{i0} iS1{i1} iS2{i2}
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
    //     ValGroup 8: iS9
    //     ExprGroup 0:
    //       iS3 = merge(iS0, iS1)
    //     ExprGroup 1:
    //       iS4, iS5 = split(iS3, 16)
    //     ExprGroup 2:
    //       iS8 = merge(iS6, iS7)
    //
    //   Note that IterDomains iS6, iS7, iS8, and iS9 might be associated to
    //   other tensors in the fusion besides tv. Consider the following
    //   abstract tensor:
    //
    //     abstract_tensor_.domain: ValGroup 4, ValGroup 5, ValGroup 2
    //
    //   The following diagrams show the transformations that are embedded in
    //   the abstract tensor in this case:
    //
    //      VG0   VG1
    //        \   /
    //         EG0
    //          |
    //         VG3
    //          |
    //         EG1
    //        /   \
    //      VG4   VG5       VG2
    //
    //   In this case the schedule is computable since VG4, VG5, and VG2 are
    //   all computable (since producer groups VG0, VG1, and VG2 are occupied
    //   by tv).
    //
    //   Now consider a case where we replace VG2 with {VG5, VG7} in the
    //   abstract domain and we add VG8:
    //
    //     abstract_tensor_.domain: ValGroup 4, ValGroup 5, ValGroup 7, ValGroup 8
    //
    //   The following diagrams show the transformations that are embedded in
    //   the abstract tensor in this case:
    //
    //      VG0   VG1
    //        \   /
    //         EG0
    //          |
    //         VG3       VG2   VG6
    //          |          \   /
    //         EG1          EG2
    //        /   \          |
    //      VG4   VG5       VG7      VG8
    //
    //   Again ValGroups 4 and 5 are computable since those ValGroups are
    //   produced by ExprGroup 1 which itself produced by ExprGroup 0, and tv
    //   includes iS0 and iS1. VG8 is not computable, but it has no computable
    //   producers either, so we simply ignore it when scheduling tv.
    //
    //   However, ValGroup 7 is not computable. It is produced by ExprGroup 2
    //   whose input ValGroups are 2 and 6. ValGroup 2 is computable since iS2
    //   is in tv, however there is no IterDomain in tv that can be used to
    //   represent ValGroup 6 which also has no producer ValGroups. In this
    //   case we will throw an error instead of ignoring VG7; the reason we
    //   cannot ignore it in this case like we did with VG8 is that if we did
    //   then we would leave VG2 out of the loop domain, which would validate
    //   the consistency of the logical->loop mapping.

    // Any computable ValGroup should have an entry in computed_ids. If we prove
    // that the ValGroup is not computable, we insert it here.
    // In the case of the above example this would include VG6 since it cannot
    // be computed. VG7 is not included since it can be computed despite VG6
    // being uncomputable:
    std::unordered_set<ValGroup> uncomputable_groups;

    // We are trying to evaluate the ValGroup g which acts like a symbolic
    // IterDomain using the concrete IterDomains found in the TensorView. To do
    // so, we recursively evaluate the input ValGroups then create a new Expr
    // for the corresponding ExprGroup. This recursion is implemented using a
    // stack of ValGroups that are left to be evaluated.
    std::stack<ValGroup> eval_stack({g});
    while (!eval_stack.empty()) {
      ValGroup vg = eval_stack.top();
      eval_stack.pop();
      if (computed_ids.count(vg) != 0) {
        // vg is already computed
        continue;
      }

      // We need to try and produce an ID for the group vg, so we look at
      // definitions for the Vals in vg and check whether we have their inputs
      // computed yet (computed_ids) and whether they can be computed
      // (uncomputable_groups). If an ExprGroup does not have any computable
      // inputs, we move on to the next ExprGroup. There might be inputs that
      // have not yet been computed or proven uncomputable; in those cases we
      // will place them into the vector below so that we can push them onto
      // the stack so that we can try again.
      std::vector<ValGroup> uncomputed_producer_groups;
      for (const ExprGroup& eg : graph_->getDefinitions(vg)) {
        // NOTE: it suffices to use any Expr* in eg, since they are all
        // guaranteed to have the same type and identical attributes.
        Expr* expr = eg->front();

        std::vector<IterDomain*> id_inps;
        bool all_inputs_computed = true;
        for (Val* inp : expr->inputs()) {
          ValGroup vg_inp = graph_->toGroup(inp);
          auto inp_it = computed_ids.find(vg_inp);
          if (inp_it != computed_ids.end()) {
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
        if (all_inputs_computed) {
          // Compute new ID expression
          Expr* id_expr = replayIdExpr(expr, id_inps);
          // Update the mapping to point to all of the newly created Expr's
          // outputs
          NVF_ERROR(id_expr->outputs().size() == expr->outputs().size());
          for (size_t i : c10::irange(expr->outputs().size())) {
            ValGroup vg_outp = graph_->toGroup(expr->output((int64_t)i));
            auto* id_outp = id_expr->output((int64_t)i)->as<IterDomain>();
            graph_->initializeVal(id_outp, vg_outp);
            computed_ids.emplace(vg_outp, id_outp);
          }
          // No need to look at next ExprGroup
          break;
        } else {
          // Some input is not yet computed
          if (uncomputed_producer_groups.empty() && !id_inps.empty()) {
            // The only uncomputed producer groups are uncomputable, but some
            // are computed. Just pass those computed inputs through. Note that
            // in this case, id_inps.front() will be the representative of vg
            // even though it is _not_ actually mapped into that ValGroup.
            NVF_ERROR(id_inps.size() == 1);
            computed_ids.emplace(vg, id_inps.front());
            break;
          }
        }
        if (!uncomputed_producer_groups.empty()) {
          // Do not look at other ExprGroups before we try computing these
          // uncomputed producer groups
          break;
        }
      }

      if (uncomputed_producer_groups.empty()) {
        // All of the defining expressions are proven uncomputable, so mark vg
        // uncomputable
        uncomputable_groups.insert(vg);
      } else {
        // There are some uncomputed producer groups that might be computable,
        // so try again after processing those producer groups
        eval_stack.push(vg);
        for (const ValGroup& next_vg : uncomputed_producer_groups) {
          eval_stack.push(next_vg);
        }
      }
    }

    // Now check whether there is a concrete entry for abs_id (i.e. g). If not,
    // that means this dimension is missing and should not be included in the
    // loop domain, so return nullptr.
    auto it = computed_ids.find(g);
    return it == computed_ids.end() ? nullptr : it->second;
  }

 private:
  const AbstractTensor& abstract_tensor_;
  ValGraph* graph_;

  std::unordered_set<ValGroup> scheduled_val_groups_;
};

} // namespace

void applyAbstractTransforms(
    const AbstractTensor& abstract_tensor,
    const std::vector<TensorView*>& tvs,
    ValGraph* graph) {
  AbstractTensorSchedule::apply(abstract_tensor, tvs, graph);
}

} // namespace nvfuser
