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

void propagateResizeToInputs(Expr* resize_tensor_op) {
  NVF_ERROR(
      resize_tensor_op->isA<SliceOp>() || resize_tensor_op->isA<PadOp>(),
      "Unexpected resize tensor op: ",
      resize_tensor_op->toString());

  Fusion* fusion = resize_tensor_op->fusion();

  auto producer_tv = resize_tensor_op->input(0)->as<TensorView>();
  auto consumer_tv = resize_tensor_op->output(0)->as<TensorView>();

  auto all_dep_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, {producer_tv});

  std::vector<TensorView*> tvs_to_schedule;
  tvs_to_schedule.reserve(all_dep_vals.size());
  for (auto val : all_dep_vals) {
    if (val->isA<TensorView>() && !val->isFusionInput()) {
      tvs_to_schedule.push_back(val->as<TensorView>());
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
      tvs_to_schedule, producer_tv->getLoopDomain());

  // Now that all the dependent tensors have the uniform, exact-mapped
  // loop domains, we just need to propagte the specific Resize ops of
  // this tensor.
  for (const auto i : c10::irange(consumer_tv->getLogicalDomain().size())) {
    auto out_logical_id = consumer_tv->getLogicalDomain().at(i);
    auto resize = dynamic_cast<Resize*>(out_logical_id->definition());
    if (resize == nullptr) {
      continue;
    }

    scheduler_tools::scheduleLoopDomainsBy(tvs_to_schedule, resize);
  }
}

} // namespace scheduler_tools
} // namespace nvfuser
