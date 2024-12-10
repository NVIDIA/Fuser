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

std::vector<IterDomain*> getSqueezedSlices(Fusion* fusion) {
  // ValGroups slice_groups;
  std::vector<IterDomain*> slice_ids;
  std::unordered_map<ValGroup, IterDomain*> slice_id_map;

  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      slice_dep_map;

  VectorOfUniqueEntries<IterDomain*> squeezed_slices;

  for (auto expr : fusion->exprs()) {
    // Propagate the slice ID dependencies. Assuming no reshape, no
    // further resize
    for (auto p_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto c_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto p2c = PairwiseLogicalDomainMap(p_tv, c_tv)
                       .mapBroadcast(false)
                       .mapProducerToConsumer();
        for (auto p_id : p_tv->getLogicalDomain()) {
          if (p2c.count(p_id) == 0) {
            continue;
          }
          for (auto& [id, id_set] : slice_dep_map) {
            if (id_set.count(p_id)) {
              id_set.insert(p2c.at(p_id));
            }
          }
        }
      }
    }

    if (auto slice = dynamic_cast<SliceOp*>(expr)) {
      auto output_tv = expr->output(0)->as<TensorView>();
      for (const auto logical_id : output_tv->getLogicalDomain()) {
        auto resize = dynamic_cast<Resize*>(logical_id->definition());
        if (resize == nullptr) {
          continue;
        }

        if (!resize->out()->isBroadcast()) {
          continue;
        }

        // Can the input be a broadcast?
        NVF_ERROR(
            !resize->in()->isBroadcast(),
            "Unexpected broadcast input: ",
            resize->in()->toString());

        auto slice_id = resize->out();
        std::cerr << "Registering slice: " << slice_id->toString() << ", "
                  << slice->toString();
        slice_ids.push_back(slice_id);
        slice_dep_map.emplace(
            slice_id, std::unordered_set<IterDomain*>{slice_id});
      }
    } else if (auto squeeze = dynamic_cast<SqueezeOp*>(expr)) {
      auto input_tv = expr->input(0)->as<TensorView>();
      for (const auto i : c10::irange(input_tv->getLogicalDomain().size())) {
        if (!squeeze->isSqueezeDim(i)) {
          continue;
        }

        auto squeezed_id = input_tv->getLogicalDomain().at(i);
        NVF_CHECK(squeezed_id->isBroadcast());

        for (const auto& [slice_id, dep_set] : slice_dep_map) {
          if (dep_set.count(squeezed_id)) {
            std::cerr << "Squeeze of slice detected: " << squeeze->toString();
            squeezed_slices.pushBack(slice_id);
          }
        }
      }
    }
  }

  return squeezed_slices.vector();
}

void propagateSqueezedSliceToOutputs(Fusion* fusion) {
  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto& graph = id_model.buildExactGraph();

  auto squeezed_slices = getSqueezedSlices(fusion);
  std::unordered_set<IterDomain*> squeezed_slice_set{
      squeezed_slices.begin(), squeezed_slices.end()};

  const auto exprs = fusion->exprs();

  std::unordered_map<TensorView*, std::unordered_set<IterDomain*>>
      propagated_squeezed_slices;

  // TODO: Does the traversal order matter?
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto slice = dynamic_cast<SliceOp*>(*it);
    if (slice == nullptr) {
      continue;
    }

    auto slice_out = slice->output(0)->as<TensorView>();

    auto dep_outputs = DependencyCheck::getAllValsBetween(
        {slice_out}, {fusion->outputs().begin(), fusion->outputs().end()});

    for (const auto squeezed_id : slice_out->getLogicalDomain()) {
      if (squeezed_slice_set.count(squeezed_id) == 0) {
        continue;
      }

      // squeezed slice found
      // Assume this ID remains in the loop domain
      NVF_ERROR(
          std::find(
              slice_out->getLoopDomain().begin(),
              slice_out->getLoopDomain().end(),
              squeezed_id) != slice_out->getLoopDomain().end());

      std::vector<TensorView*> tvs_to_schedule;
      tvs_to_schedule.reserve(dep_outputs.size());
      for (Val* dep_output : dep_outputs) {
        auto tv = dep_output->as<TensorView>();
        if (std::find_if(
                tv->getLogicalDomain().begin(),
                tv->getLogicalDomain().end(),
                [&](IterDomain* id) {
                  return graph.disjointValSets().strictAreMapped(
                      id, squeezed_id);
                }) != tv->getLogicalDomain().end()) {
          // The ID still exists in tv
          continue;
        }
        tvs_to_schedule.push_back(tv);
      }

      if (tvs_to_schedule.empty()) {
        continue;
      }

      // Insert a new broadcast ID for the squeezed ID
      for (auto tv : tvs_to_schedule) {
        // Skip if already inserted
        bool already_done = false;
        for (const auto& id : propagated_squeezed_slices[tv]) {
          if (graph.disjointValSets().strictAreMapped(id, squeezed_id)) {
            already_done = true;
            break;
          }
        }
        if (already_done) {
          continue;
        }

        // Insert a broadcast ID at the innermost position
        tv->broadcast(-1);
        auto inserted_broadcast = tv->getLoopDomain().back();

        // Mark the new ID as mapped as the squeezed ID
        tv->fusion()->registerExactMapping(squeezed_id, inserted_broadcast);

        // Keep track of which squeezed IDs have been processed
        propagated_squeezed_slices[tv].emplace(squeezed_id);
      }
    }
  }
}

void propagateResizeToInputs(Expr* resize_op) {
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
      tvs_to_schedule, producer_tv->getLoopDomain());

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
