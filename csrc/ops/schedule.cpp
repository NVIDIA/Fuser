// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <ops/schedule.h>
#include <algorithm>

namespace nvfuser {

namespace {

// Given a reference TensorView, create a new TensorView with the same sized
// logical domain but only with broadcast IterDomains.
TensorView* createBroadcastTv(TensorView* reference) {
  auto logical_domain =
      reference->getLogicalDomain() | TensorDomain::kNoReductions;
  std::vector<IterDomain*> out_domain;
  out_domain.reserve(std::ranges::distance(logical_domain));
  std::transform(
      logical_domain.begin(),
      logical_domain.end(),
      std::back_inserter(out_domain),
      [](IterDomain* id) {
        return IterDomainBuilder(
                   FusionGuard::getCurFusion()->zeroVal(),
                   FusionGuard::getCurFusion()->oneVal())
            .iter_type(IterType::Broadcast)
            .build();
      });
  TensorView* out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(out_domain),
      reference->getDataType().value());
  return out;
}

} // namespace

TensorView* launch_dependent_grid(std::vector<Val*> inputs) {
  auto tensorview_input_iter =
      std::find_if(inputs.begin(), inputs.end(), [](const Val* val) {
        return val->isA<TensorView>();
      });
  NVF_ERROR(
      tensorview_input_iter != inputs.end(),
      "Expected at least one TensorView input.");
  TensorView* out =
      createBroadcastTv((*tensorview_input_iter)->as<TensorView>());
  IrBuilder::create<LaunchDependentGridOp>(out, inputs);
  return out;
}

TensorView* wait_for_prior_grid(std::vector<Val*> inputs) {
  auto tensorview_input_iter =
      std::find_if(inputs.begin(), inputs.end(), [](const Val* val) {
        return val->isA<TensorView>();
      });
  NVF_ERROR(
      tensorview_input_iter != inputs.end(),
      "Expected at least one TensorView input.");
  TensorView* out =
      createBroadcastTv((*tensorview_input_iter)->as<TensorView>());
  IrBuilder::create<WaitForPriorGridOp>(out, inputs);
  return out;
}

} // namespace nvfuser
