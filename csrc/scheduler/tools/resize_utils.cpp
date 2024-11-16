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
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <val_graph_visitor.h>

namespace nvfuser {
namespace scheduler_tools {

bool propagateSqueezedSliceToOutputs(Fusion* fusion) {
  std::cerr << "propagateSqueezedSliceToOutputs\n";

  fusion->printMath();

  IdModel id_model(fusion, /*build_models=*/false);
  const auto& graph = id_model.buildExactGraph();

  auto squeezed_slices = ir_utils::getSqueezedSlices(fusion);
  std::unordered_set<IterDomain*> squeezed_slice_set{
      squeezed_slices.begin(), squeezed_slices.end()};

  std::cerr << "All squeezed slices: " << toDelimitedString(squeezed_slices)
            << "\n";

  const auto exprs = fusion->exprs();
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto slice = dynamic_cast<SliceOp*>(*it);
    if (slice == nullptr) {
      continue;
    }

    std::cerr << "Slice: " << slice->toString();

    auto slice_out = slice->output(0)->as<TensorView>();

    auto dep_outputs = DependencyCheck::getAllValsBetween(
        {slice_out}, {fusion->outputs().begin(), fusion->outputs().end()});

    for (const auto logical_id : slice_out->getLogicalDomain()) {
      if (squeezed_slice_set.count(logical_id) == 0) {
        continue;
      }

      std::cerr << "propagateSqueezedSliceToOutputs: slice candidate: "
                << slice->toString() << ", " << logical_id->toString() << "\n";

      std::cerr << "All dep outputs: " << toDelimitedString(dep_outputs)
                << "\n";

      // squeezed slice found
      // Assume this ID remains in the loop domain
      NVF_ERROR(
          std::find(
              slice_out->getLoopDomain().begin(),
              slice_out->getLoopDomain().end(),
              logical_id) != slice_out->getLoopDomain().end());

      std::vector<TensorView*> tvs_to_schedule;
      tvs_to_schedule.reserve(dep_outputs.size());
      for (Val* dep_output : dep_outputs) {
        auto tv = dep_output->as<TensorView>();
        if (std::find_if(
                tv->getLogicalDomain().begin(),
                tv->getLogicalDomain().end(),
                [&](IterDomain* id) {
                  return graph.disjointValSets().strictAreMapped(
                      id, logical_id);
                }) != tv->getLogicalDomain().end()) {
          // Not yet squeezed
          std::cerr << "Not yet squeezed: " << tv->toString() << "\n";
          continue;
        }
        tvs_to_schedule.push_back(tv);
      }

      if (tvs_to_schedule.empty()) {
        continue;
      }

      std::cerr << "propagate squeezed slice: " << logical_id->toString()
                << " of " << slice->toString();
      std::cerr << "To tensors: " << toDelimitedString(tvs_to_schedule) << "\n";
      for (auto tv : tvs_to_schedule) {
        tv->broadcast(-1);
        auto inserted_broadcast = tv->getLoopDomain().back();
        std::cerr << "New inserted broadcast: "
                  << inserted_broadcast->toString() << ", "
                  << logical_id->toString() << "\n";
        tv->fusion()->registerExactMapping(logical_id, inserted_broadcast);
      }
    }
  }

  std::cerr << "propagateSqueezedSliceToOutputs done\n";
  fusion->printMath();

  return true;
}

void propagateResizeTensorOpToInputs(Expr* resize_op) {
  DebugStreamGuard dsg(std::cerr);

  NVF_ERROR(
      resize_op->isA<SliceOp>() || resize_op->isA<PadOp>(),
      "Unexpected resize tensor op: ",
      resize_op->toString());

  Fusion* fusion = resize_op->fusion();

  std::cerr << "propagateResizeTensorOpToInputs: " << resize_op->toString();

  // NOTE: privatization assumed

  auto producer_tv = resize_op->input(0)->as<TensorView>();
  auto consumer_tv = resize_op->output(0)->as<TensorView>();

  auto all_dep_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, {producer_tv});

  std::vector<TensorView*> tvs_to_schedule;
  tvs_to_schedule.reserve(all_dep_vals.size());
  for (auto val : all_dep_vals) {
    if (val->isA<TensorView>() && !val->isFusionInput()) {
      tvs_to_schedule.push_back(val->as<TensorView>());
    }
  }

  std::cerr << "Propagating pre-resize producer loop domain "
            << toDelimitedString(tvs_to_schedule) << " with "
            << producer_tv->toString() << "\n";

  scheduler_tools::scheduleLoopDomainsLike(
      tvs_to_schedule, producer_tv->getLoopDomain(), false);

  for (const auto i : c10::irange(consumer_tv->getLogicalDomain().size())) {
    auto out_logical_id = consumer_tv->getLogicalDomain().at(i);
    auto resize = dynamic_cast<Resize*>(out_logical_id->definition());
    if (resize == nullptr) {
      continue;
    }

    std::cerr << "Scheduling " << toDelimitedString(tvs_to_schedule) << " with "
              << out_logical_id->toString() << "\n";

    scheduler_tools::scheduleLoopDomainsBy(tvs_to_schedule, resize);
  }
}

} // namespace scheduler_tools
} // namespace nvfuser
