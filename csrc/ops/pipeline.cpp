// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <ops/pipeline.h>
#include <algorithm>

namespace nvfuser {

namespace {

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
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      reference->getDataType().value());
  return out;
}

} // namespace

TensorView* launch_dependent_grid(std::vector<Val*> inputs) {
  NVF_ERROR(inputs.size() > 0, "Expected at least one input tensor view.");
  NVF_ERROR(inputs.front()->isA<TensorView>(), "Expected a tensor view.");
  TensorView* out = createBroadcastTv(inputs.front()->as<TensorView>());
  IrBuilder::create<LaunchDependentGridOp>(out, inputs);
  return out;
}

TensorView* wait_for_prior_grid(std::vector<Val*> inputs) {
  NVF_ERROR(inputs.size() > 0, "Expected at least one input tensor view.");
  NVF_ERROR(inputs.front()->isA<TensorView>(), "Expected a tensor view.");
  TensorView* out = createBroadcastTv(inputs.front()->as<TensorView>());
  IrBuilder::create<WaitForPriorGridOp>(out, inputs);
  return out;
}

} // namespace nvfuser
