// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <transform_replay.h>

namespace nvfuser::python {

namespace {

void bindTensorviewScheduleOps(py::module_& schedule) {
  schedule.def(
      "transform_like",
      [](TensorView* reference_tv,
         const std::vector<TensorView*>& selected_tensors) {
        TransformPropagator propagator(reference_tv);
        if (selected_tensors.empty()) {
          // Propagate scheduler transformations on reference TensorView to the
          // rest of the fusion.
          MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);
        } else {
          // Propagate scheduler transformations on reference TensorView to the
          // subset of the fusion.
          std::unordered_set<TensorView*> selected_tv_set(
              selected_tensors.begin(), selected_tensors.end());
          SetSelector selector(
              {selected_tv_set.begin(), selected_tv_set.end()});
          MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
              .traverse(&propagator);
        }
      },
      R"(
      Propagate scheduler transformations from a reference TensorView to other TensorViews.

      Parameters
      ----------
      reference_tv : TensorView
          The reference TensorView whose transformations will be propagated.
      selected_tensors : List[TensorView], optional
          List of TensorViews to propagate transformations to. If empty, propagates to all TensorViews.

      Returns
      -------
      None
    )",
      py::arg("reference_tv"),
      py::arg("selected_tensors") = std::vector<TensorView*>());
}

} // namespace

void bindScheduleOperators(py::module& nvfuser) {
  py::module_ nvf_schedule = nvfuser.def_submodule(
      "schedule",
      "This submodule contains all schedule operators for NvFuser.");
  bindTensorviewScheduleOps(nvf_schedule);
}

} // namespace nvfuser::python
