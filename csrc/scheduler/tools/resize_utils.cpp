// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <id_model/id_model.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {
namespace scheduler_tools {

bool isResizeBasedOp(Expr* expr) {
  return expr->isOneOf<SliceOp, PadOp>();
}

bool hasResizeBasedOps(Fusion* fusion) {
  return ir_utils::hasOpsOfType<SliceOp, PadOp>(fusion);
}

std::vector<Expr*> getResizeBasedOps(Fusion* fusion) {
  return ir_utils::getOpsOfType<SliceOp, PadOp>(fusion);
}

void propagateResizeToInputs(Expr* resize_tensor_op) {
  NVF_ERROR(
      resize_tensor_op->isA<SliceOp>() || resize_tensor_op->isA<PadOp>(),
      "Unexpected resize tensor op: ",
      resize_tensor_op->toString());

  auto producer_tv = resize_tensor_op->input(0)->as<TensorView>();
  auto consumer_tv = resize_tensor_op->output(0)->as<TensorView>();

  // Note: DependencyCheck::getAllValsBetween with fusion inputs fails
  // to grab factory-created tensors (#4202)
  auto all_dep_stmts = StmtSort::getStmtsTo({producer_tv});

  std::vector<TensorView*> tvs_to_schedule;
  for (auto tv : ir_utils::filterByType<TensorView>(all_dep_stmts)) {
    if (!tv->isFusionInput()) {
      tvs_to_schedule.push_back(tv);
    }
  }

  // Ideally, this should be just calling
  // scheduler_tools::scheduleLoopDomainsLike once with the consumer
  // tensor as a reference. However, due to the indexing issue with
  // resize, propagating the Resize iter-domain op may fail. To avoid
  // the problem, the propagation of the resize op is explicitly done
  // by using scheduler_tools::scheduleLoopDomainsBy.
  //
  // Before doing so, all the dependent tensors need to have the exact-mapped
  // loop domain.
  scheduler_tools::scheduleLoopDomainsLike(
      tvs_to_schedule,
      producer_tv->getLoopDomain(),
      /*update_loop_domain_only=*/true);

  // Now that all the dependent tensors have the uniform, exact-mapped
  // loop domains, we just need to propagte the specific Resize ops of
  // this tensor.
  for (const auto i : arange(consumer_tv->getLogicalDomain().size())) {
    auto out_logical_id = consumer_tv->getLogicalDomain().at(i);
    auto resize = dynamic_cast<Resize*>(out_logical_id->definition());
    if (resize == nullptr) {
      continue;
    }

    scheduler_tools::scheduleLoopDomainsBy(
        tvs_to_schedule, resize, Direction::Forward);
  }
}

std::unordered_map<TensorView*, ResizeExclusivityInfo> getNonExclusiveResizeInfo(
    const std::vector<Expr*>& ordered_resize_tensor_ops,
    const ValGraph& exact_graph) {
  NVF_ERROR(!ordered_resize_tensor_ops.empty());
  Fusion* fusion = ordered_resize_tensor_ops[0]->fusion();

  std::unordered_map<TensorView*, ResizeExclusivityInfo> non_exclusive_resizes;

  // Start with both fusion inputs and factory-created tensors. Fusion
  // inputs are not enough (#4202).
  const auto inputs_vec = InputsOf::outputs(fusion->outputs());
  std::unordered_set<Val*> inputs{inputs_vec.begin(), inputs_vec.end()};

  auto get_root_to_logical_resizes =
      [&exact_graph](TensorView* tv) -> ValGroups {
    // This should be only used for outputs of resize-based ops,
    // so it should always have a root domain.
    NVF_ERROR(tv->hasRoot());
    auto out_tv_root_to_logical_exprs = DependencyCheck::getAllExprsBetween(
        {tv->getRootDomain().begin(), tv->getRootDomain().end()},
        {tv->getLogicalDomain().begin(), tv->getLogicalDomain().end()});
    ValGroups resize_inp_ids;
    for (auto resize :
         ir_utils::filterByType<Resize>(out_tv_root_to_logical_exprs)) {
      resize_inp_ids.pushBack(exact_graph.toGroup(resize->in()));
    }
    return resize_inp_ids;
  };

  // Traverse the ops in a topological order
  for (Expr* resize_tensor_op : ordered_resize_tensor_ops) {
    auto inp_tv = dynamic_cast<TensorView*>(resize_tensor_op->inputs().at(0));
    auto out_tv = dynamic_cast<TensorView*>(resize_tensor_op->outputs().at(0));

    ResizeExclusivityInfo info;

    ValGroups resize_inp_ids = get_root_to_logical_resizes(out_tv);
    NVF_ERROR(!resize_inp_ids.empty());

    auto dep_vals =
        DependencyCheck::getAllValsBetween(inputs, std::vector<Val*>{inp_tv});

    // For each tensor that inp_tv depends on, check if the resize op
    // is considered non-exclusive with respect to the tensor. That
    // is, if propagation of the resize may result in externally
    // visible changes through the tensor, the resize is considered
    // non-exclusive.
    for (auto dep_tv : ir_utils::filterByType<TensorView>(dep_vals)) {
      bool maybe_non_exclusive = dep_tv->isFusionOutput();

      if (!maybe_non_exclusive) {
        // If a dependent tv has a consumer that inp_tv does not
        // depend on, propagation of resize would escape to outputs,
        // which needs to be avoided.
        for (auto consumer_tv : ir_utils::consumerTvsOf(dep_tv)) {
          // We are interested in if resized IDs are used by other tensors
          // than out_tv
          if (consumer_tv != out_tv &&
              std::find(dep_vals.begin(), dep_vals.end(), consumer_tv) ==
                  dep_vals.end()) {
            maybe_non_exclusive = true;
            break;
          }
        }
      }

      if (!maybe_non_exclusive) {
        continue;
      }

      // dep_tv potentially is either a fusion output or it has a
      // consumer outside of the dependency set to the resized
      // tensor. Propagating the resize to dep_tv should be
      // avoided. However, if the dep_tv iter domain that corresponds
      // to the resized ID is a broadcast or there's no such ID, it
      // should still be safe to consider the resize op exclusive as
      // there's no iter domain to resize. For a concrete example, see
      // ResizeSchedulerTest.PropagateMultipleSlicesToInputs4.
      const auto inp_tv_logical_groups =
          exact_graph.toGroups(inp_tv->getLogicalDomain());
      const auto dep_tv_logical_groups =
          exact_graph.toGroups(dep_tv->getLogicalDomain());
      auto vals_between = getValsBetween<ValGraphBFS>(
          {inp_tv_logical_groups.begin(), inp_tv_logical_groups.end()},
          {dep_tv_logical_groups.begin(), dep_tv_logical_groups.end()},
          exact_graph);

      for (const ValGroup& resize_inp_id : resize_inp_ids) {
        if (std::find(
                vals_between.begin(), vals_between.end(), resize_inp_id) ==
            vals_between.end()) {
          // This resize can be ignored as there's no corresponding ID
          // in the dep tv
          continue;
        }

        // This resize input ID is not exclusively used
        info.non_exclusive_dep_tvs.push_back(dep_tv);
        info.resized_ids.pushBack(resize_inp_id);
      }
    }

    if (!info.non_exclusive_dep_tvs.empty()) {
      NVF_ERROR(non_exclusive_resizes.emplace(out_tv, info).second);
    }

    // Analysis of exclusiveness until in_tv is done. Following
    // resize-based tensor ops do not need to check the same section
    // of the fusion and can start from out_tv.
    inputs.insert(out_tv);
  }

  return non_exclusive_resizes;
}

} // namespace scheduler_tools
} // namespace nvfuser
