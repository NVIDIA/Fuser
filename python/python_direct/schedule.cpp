// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::python {

namespace {

void bindTensorviewScheduleOps(py::module_& schedule) {
  schedule.def(
      "bounded_transform_backward",
      [](TensorView* from,
         int64_t pos,
         std::vector<TensorView*> to,
         bool propagate_parallel_type) {
        using TransformPropagator =
            scheduler_utils::BoundedDirectionalTransformPropagator;
        TransformPropagator::Options options;
        if (propagate_parallel_type) {
          options.propagateParallelType();
        }
        TransformPropagator::backward(from, pos, to, options);
      },
      R"(
      Propagate scheduler transformations from a reference TensorView to other TensorViews.

      Parameters
      ----------
      from : TensorView
          The reference TensorView whose transformations will be propagated.
      pos : int
          The position up to which dimensions should be selected. -1 means all dimensions.
      to : List[TensorView]
          List of TensorViews to propagate transformations to.
      propagate_parallel_type : bool
          Whether to propagate parallel type.

      Returns
      -------
      None
      )",
      py::arg("from"),
      py::arg("pos"),
      py::arg("to"),
      py::arg("propagate_parallel_type") = false);

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

  schedule.def(
      "parallelize_like",
      [](TensorView* reference_tv,
         int64_t pos,
         const std::vector<TensorView*>& selected_tensors,
         const std::unordered_set<ParallelType>& selected_parallel_types,
         bool propagate_padding) {
        scheduler_utils::parallelizeAllLike(
            reference_tv,
            pos,
            selected_tensors,
            selected_parallel_types,
            propagate_padding);
      },
      R"(
          Propagate the parallelization from the selected dimensions of the
          reference tensor to their corresponding dimensions in all selected
          tensors in the DAG.

          Parameters
          ----------
          reference_tv : TensorView
              The reference TensorView whose parallelization will be propagated.
          pos : int, optional
              The position up to which dimensions should be selected. -1 means all dimensions.
          selected_tensors : List[TensorView], optional
              List of TensorViews to propagate parallelization to. If empty, propagates to all TensorViews.
          selected_parallel_types : Set[ParallelType], optional
              Set of parallel types to propagate. If empty, propagates all parallel types.
          propagate_padding : bool, optional
              Whether to propagate padding (default: True).

          Returns
          -------
          None
        )",
      py::arg("reference_tv"),
      py::arg("pos") = -1,
      py::arg("selected_tensors") = std::vector<TensorView*>(),
      py::arg("selected_parallel_types") = std::unordered_set<ParallelType>(),
      py::arg("propagate_padding") = true);

  schedule.def(
      "inline_most",
      [](const std::vector<TensorView*>& selected_tensors) {
        if (selected_tensors.empty()) {
          inlineMost();
        } else {
          inlineMost(selected_tensors);
        }
      },
      R"(
          Inline operations to the right most allowed position for the selected tensors.

          Parameters
          ----------
          selected_tensors : List[TensorView], optional
              List of TensorViews to inline. If empty, inlines all operations.

          Returns
          -------
          None
        )",
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
