// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include <bindings.h>
#include <direct_utils.h>
#include <options.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::python {

namespace {

void bindTensorviewScheduleOps(nb::module_& schedule) {
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
      nb::arg("from"),
      nb::arg("pos"),
      nb::arg("to"),
      nb::arg("propagate_parallel_type") = false);

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
      nb::arg("reference_tv"),
      nb::arg("selected_tensors") = std::vector<TensorView*>());

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
      nb::arg("reference_tv"),
      nb::arg("pos") = -1,
      nb::arg("selected_tensors") = std::vector<TensorView*>(),
      nb::arg("selected_parallel_types") = std::unordered_set<ParallelType>(),
      nb::arg("propagate_padding") = true);

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
      nb::arg("selected_tensors") = std::vector<TensorView*>());

  schedule.def(
      "can_schedule",
      [](Fusion* fusion,
         SchedulerType scheduler_type,
         const nb::iterable& inputs) {
        // Enable collection of messages from canScheduleRejectReason
        DebugDumpOptionsGuard debug_dump_options_guard;
        DebugDumpOptionsGuard::getCurOptions().set(
            DebugDumpOption::FusionSegmenterLog);

        // Send debug messages to stringstream
        std::stringstream ss;
        DebugStreamGuard dsg(ss);

        // Create runtime info from inputs
        auto args = from_pyiterable(inputs);
        SchedulerRuntimeInfo runtime_info(fusion, args);

        bool can_schedule =
            Schedule::canSchedule(scheduler_type, fusion, runtime_info);
        return std::make_tuple(can_schedule, ss.str());
      },
      nb::arg("fusion"),
      nb::arg("scheduler_type"),
      nb::arg("inputs"),
      R"(
          Check if a scheduler can schedule the given fusion with the provided inputs.

          Parameters
          ----------
          fusion : Fusion
              The fusion to check.
          scheduler_type : SchedulerType
              The type of scheduler to check.
          inputs : iterable
              The input tensors/values for the fusion.

          Returns
          -------
          tuple of (bool, str)
              A tuple containing:
              - bool: True if the scheduler can schedule the fusion, False otherwise.
              - str: Debug message explaining why the scheduler was accepted or rejected.
        )");

  schedule.def(
      "find_compatible_schedulers",
      [](Fusion* fusion, const nb::iterable& inputs) {
        // Create runtime info from inputs
        auto args = from_pyiterable(inputs);
        SchedulerRuntimeInfo runtime_info(fusion, args);

        std::vector<SchedulerType> compatible_schedulers;

        // Check all scheduler types except None
        for (const auto& scheduler_type : all_heuristics_in_priority_order) {
          if (scheduler_type != SchedulerType::None &&
              Schedule::canSchedule(scheduler_type, fusion, runtime_info)) {
            compatible_schedulers.push_back(scheduler_type);
          }
        }

        return compatible_schedulers;
      },
      nb::arg("fusion"),
      nb::arg("inputs"),
      R"(
          Find all schedulers compatible with the given fusion and inputs.

          Parameters
          ----------
          fusion : Fusion
              The fusion to check.
          inputs : iterable
              The input tensors/values for the fusion.

          Returns
          -------
          list of SchedulerType
              A list of scheduler types that can schedule the fusion.
        )");

  schedule.def(
      "compute_heuristics",
      [](Fusion* fusion,
         SchedulerType scheduler_type,
         const nb::iterable& inputs) {
        auto args = from_pyiterable(inputs);
        SchedulerRuntimeInfo runtime_info(fusion, args);
        NVF_ERROR(
            Schedule::canSchedule(scheduler_type, fusion, runtime_info),
            "Could not schedule fusion with the SchedulerType: ",
            scheduler_type);
        auto scheduler_instance =
            SchedulerEntry::makeSchedulerInstance(scheduler_type);
        return scheduler_instance->computeHeuristics(fusion, runtime_info);
      },
      nb::arg("fusion"),
      nb::arg("scheduler_type"),
      nb::arg("inputs"),
      R"(
          Compute the heuristics for the specified scheduler type.

          Parameters
          ----------
          fusion : Fusion
              The fusion to compute heuristics for.
          scheduler_type : SchedulerType
              The type of scheduler to compute heuristics for.
          inputs : iterable
              The input tensors/values for the fusion.

          Returns
          -------
          HeuristicParams
              The heuristics for the given fusion.

          Notes
          -----
          This function will raise an error if the scheduler cannot schedule the fusion.
        )");

  schedule.def(
      "schedule",
      [](Fusion* fusion,
         SchedulerType scheduler_type,
         const nb::iterable& inputs) {
        auto args = from_pyiterable(inputs);
        return SchedulerEntry::scheduleWith(
            fusion, scheduler_type, args, /*validate_scheduler=*/true);
      },
      nb::arg("fusion"),
      nb::arg("scheduler_type"),
      nb::arg("inputs"),
      R"(
          Schedule the fusion with the specified scheduler type.

          Parameters
          ----------
          fusion : Fusion
              The fusion to schedule.
          scheduler_type : SchedulerType
              The type of scheduler to use.
          inputs : iterable
              The input tensors/values for the fusion.

          Returns
          -------
          HeuristicParams
              The heuristics for the scheduled fusion.

          Notes
          -----
          This function will raise an error if the scheduler cannot schedule the fusion.
        )");

  schedule.def(
      "schedule",
      [](Fusion* fusion,
         SchedulerType scheduler_type,
         const HeuristicParams* heuristic_params) {
        auto scheduler_instance =
            SchedulerEntry::makeSchedulerInstance(scheduler_type);
        scheduler_instance->schedule(fusion, heuristic_params);
      },
      nb::arg("fusion"),
      nb::arg("scheduler_type"),
      nb::arg("heuristic_params"),
      R"(
          Schedule the fusion with the specified scheduler type.

          Parameters
          ----------
          fusion : Fusion
              The fusion to schedule.
          scheduler_type : SchedulerType
              The type of scheduler to use.
          heuristic_params : HeuristicParams
              The heuristics for the scheduled fusion.

          Returns
          -------
          None
        )");
}

} // namespace

void bindScheduleOperators(nb::module_& nvfuser) {
  nb::module_ nvf_schedule = nvfuser.def_submodule(
      "schedule",
      "This submodule contains all schedule operators for NvFuser.");
  bindTensorviewScheduleOps(nvf_schedule);
}

} // namespace nvfuser::python
