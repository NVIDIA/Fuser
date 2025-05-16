// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <ir/interface_nodes.h>
#include <multidevice/utils.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/python_bindings.h>
#include <scheduler/tools/inlining.h>
#include <transform_replay.h>

namespace nvfuser::python_frontend {

void bindSchedule(py::class_<FusionDefinition>& fusion_def) {
  //! The SchedOperators class is a nested class of FusionDefinition to allow
  //! the user to query the class for the list of schedule operators.
  //!
  //! Example:
  //!   help(FusionDefinition.SchedOperators)
  //!
  //! Additional operators are expected to be defined below as needed.
  py::class_<FusionDefinition::SchedOperators> nvf_sched(
      fusion_def, "SchedOperators");
  nvf_sched.def(py::init<FusionDefinition*>());
  nvf_sched.def(
      "to_string",
      [](FusionDefinition::SchedOperators& self, Tensor tensor) {
        // NOTE: For debugging purposes, print the state of TensorView
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        // Determine if tensor is a result from a reduction operation.
        FusionDefinition* fd = self.fusion_definition;
        TensorView* tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();
        return tv->toString();
      },
      py::arg("tensor"));
  nvf_sched.def(
      "user_schedule_ir",
      [](FusionDefinition::SchedOperators& self) {
        return self.fusion_definition->userScheduleIr();
      },
      py::return_value_policy::reference);
  //! experimental API for multidevice support
  nvf_sched.def(
      "_set_device_mesh",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         const DeviceMesh& mesh) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto tv = fd->getFusionState(tensor.index)->template as<TensorView>();
        tv->setDeviceMesh(mesh);
      },
      py::arg("tensor"),
      py::arg("mesh"));
  nvf_sched.def(
      "parallelize",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         int axis,
         const ParallelType& parallel_type) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto tv = fd->getFusionState(tensor.index)->template as<TensorView>();
        tv->axis(axis)->parallelize(parallel_type);
      },
      py::arg("tensor"),
      py::arg("axis"),
      py::arg("parallel_type"));
  nvf_sched.def(
      "merge",
      [](FusionDefinition::SchedOperators& self, Tensor arg, int dim) {
        FUSER_PERF_SCOPE("SchedOperators.merge");
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->merge(dim);
      },
      py::arg("arg"),
      py::arg("dim"));
  auto reduction_factor_func = [](FusionDefinition::SchedOperators& self,
                                  Tensor arg,
                                  const std::vector<int64_t>& dims) -> Tensor {
    FUSER_PERF_SCOPE("SchedOperators.reduction_factor");
    NVF_CHECK(
        self.validUse(),
        "Attempting to use a SchedOperators Op prior to definition!");
    FusionDefinition* fd = self.fusion_definition;
    TensorView* input_tv =
        fd->getFusionState(arg.index)->template as<TensorView>();
    TensorView* output_tv = input_tv->rFactor(dims);
    return fd->addTensor(output_tv);
  };
  nvf_sched.def(
      "reduction_factor",
      reduction_factor_func,
      py::arg("arg"),
      py::arg("dims"));
  nvf_sched.def(
      "rfactor", reduction_factor_func, py::arg("arg"), py::arg("dims"));
  nvf_sched.def(
      "reorder",
      [](FusionDefinition::SchedOperators& self,
         Tensor arg,
         const std::unordered_map<int64_t, int64_t>& old2new) {
        FUSER_PERF_SCOPE("SchedOperators.reorder");
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->reorder(old2new);
      },
      py::arg("arg"),
      py::arg("old2new"));
  nvf_sched.def(
      "split",
      [](FusionDefinition::SchedOperators& self,
         Tensor arg,
         int64_t dim,
         int64_t factor,
         bool inner_split) {
        FUSER_PERF_SCOPE("SchedOperators.split");
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto input_tv =
            fd->getFusionState(arg.index)->template as<TensorView>();
        input_tv->split(dim, factor, inner_split);
      },
      py::arg("arg"),
      py::arg("dim"),
      py::arg("factor"),
      py::arg("inner_split") = true);
  nvf_sched.def(
      "set_allocation_as_loop",
      [](FusionDefinition::SchedOperators& self, Tensor arg) {
        FUSER_PERF_SCOPE("SchedOperators.set_allocation_as_loop");
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        auto* tv = fd->getFusionState(arg.index)->template as<TensorView>();
        tv->setAllocationDomain(tv->getLoopDomain(), true);
      },
      py::arg("arg"));
  nvf_sched.def(
      "cache_after",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         const LoadStoreOpType& op_type,
         const CacheOp& cache_op) -> Tensor {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        TensorView* input_tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();
        TensorView* output_tv = input_tv->cacheAfter(op_type, cache_op);
        return fd->addTensor(output_tv);
      },
      py::arg("tensor"),
      py::arg("op_type") = LoadStoreOpType::Set,
      py::arg("cache_op") = CacheOp::Unspecified);
  nvf_sched.def(
      "cache_before",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         const LoadStoreOpType& op_type) -> Tensor {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        TensorView* input_tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();
        TensorView* output_tv = input_tv->cacheBefore(op_type);
        return fd->addTensor(output_tv);
      },
      py::arg("tensor"),
      py::arg("op_type") = LoadStoreOpType::Set);
  nvf_sched.def(
      "cache_fork",
      [](FusionDefinition::SchedOperators& self, Tensor tensor) -> Tensor {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        TensorView* input_tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();
        TensorView* output_tv = input_tv->cacheFork();
        return fd->addTensor(output_tv);
      },
      py::arg("tensor"));
  nvf_sched.def(
      "set_memory_type",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         const MemoryType& memory_type) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        FusionDefinition* fd = self.fusion_definition;
        TensorView* tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();
        tv->setMemoryType(memory_type);
      },
      py::arg("tensor"),
      py::arg("memory_type"));
  nvf_sched.def(
      "transform_like",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         const std::vector<Tensor>& selected_tensors) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");

        FusionDefinition* fd = self.fusion_definition;
        TensorView* reference_tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();

        TransformPropagator propagator(reference_tv);
        if (selected_tensors.empty()) {
          // Propagate scheduler transformations on reference TensorView to the
          // rest of the fusion.
          MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);
        } else {
          // Propagate scheduler transformations on reference TensorView to the
          // subset of the fusion.
          std::unordered_set<TensorView*> selected_tv_set;
          selected_tv_set.reserve(selected_tensors.size());
          std::transform(
              selected_tensors.begin(),
              selected_tensors.end(),
              std::inserter(selected_tv_set, selected_tv_set.end()),
              [&fd](const Tensor& t) {
                return fd->getFusionState(t.index)->template as<TensorView>();
              });
          SetSelector selector(
              {selected_tv_set.begin(), selected_tv_set.end()});
          MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
              .traverse(&propagator);
        }
      },
      py::arg("tensor"),
      py::arg("selected_tensors") = std::vector<Tensor>());
  nvf_sched.def(
      "parallelize_like",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         int64_t pos,
         const std::vector<Tensor>& selected_tensors,
         const std::unordered_set<ParallelType>& selected_parallel_types,
         bool propagate_padding) {
        // Propagate the parallelization from the selected dimensions of the
        // reference tensor to their corresponding dimensions in all selected
        // tensors in the DAG.
        //
        // 1. Position `pos` means selecting all the dimensions
        // [0, 1, ..., pos - 1]. pos = -1 means selecting all dimensions.
        // 2. `selected_tvs` are selected tensors in the DAG. Empty
        // `selected_tvs` means selecting all tensors in the fusion of
        // `reference_tv`.
        // 3. `selected_parallel_types` are the selected parallel types. Empty
        // `selected_parallel_types` means selecting all parallel types.

        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");

        FusionDefinition* fd = self.fusion_definition;
        TensorView* reference_tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();

        std::vector<TensorView*> selected_tvs;
        selected_tvs.reserve(selected_tensors.size());
        std::transform(
            selected_tensors.begin(),
            selected_tensors.end(),
            std::back_inserter(selected_tvs),
            [&fd](const Tensor& t) {
              return fd->getFusionState(t.index)->template as<TensorView>();
            });

        nvfuser::scheduler_utils::parallelizeAllLike(
            reference_tv,
            pos,
            selected_tvs,
            selected_parallel_types,
            propagate_padding);
      },
      py::arg("tensor"),
      py::arg("pos") = -1,
      py::arg("selected_tensors") = std::vector<Tensor>(),
      py::arg("selected_parallel_types") = std::unordered_set<ParallelType>(),
      py::arg("propagate_padding") = true);
  nvf_sched.def(
      "inline_most",
      [](FusionDefinition::SchedOperators& self,
         const std::vector<Tensor>& selected_tensors) {
        // Inline to the right most allowed position for the selected tensors in
        // the current fusion.

        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");

        FusionDefinition* fd = self.fusion_definition;

        if (selected_tensors.empty()) {
          nvfuser::inlineMost();
        } else {
          std::vector<TensorView*> selected_tvs;
          selected_tvs.reserve(selected_tensors.size());
          std::transform(
              selected_tensors.begin(),
              selected_tensors.end(),
              std::back_inserter(selected_tvs),
              [&fd](const Tensor& t) {
                return fd->getFusionState(t.index)->template as<TensorView>();
              });
          nvfuser::inlineMost(selected_tvs);
        }
      },
      py::arg("selected_tensors") = std::vector<Tensor>());
  nvf_sched.def(
      "inline_at",
      [](FusionDefinition::SchedOperators& self,
         Tensor tensor,
         int64_t pos,
         bool best_effort,
         const std::vector<Tensor>& selected_tensors) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");

        FusionDefinition* fd = self.fusion_definition;
        TensorView* reference_tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();

        if (selected_tensors.empty()) {
          // Inline to the position corresponding to the reference position in
          // the reference tensor for all tensors in the current fusion.
          nvfuser::inlineAllAt(reference_tv, pos, best_effort);
        } else {
          // Inline to the position corresponding to the reference position in
          // the reference tensor for selected tensors in the current fusion.
          std::unordered_set<TensorView*> selected_tvs;
          selected_tvs.reserve(selected_tensors.size());
          std::transform(
              selected_tensors.begin(),
              selected_tensors.end(),
              std::inserter(selected_tvs, selected_tvs.end()),
              [&fd](const Tensor& t) {
                return fd->getFusionState(t.index)->template as<TensorView>();
              });

          nvfuser::inlineSelectedAt(
              selected_tvs, reference_tv, pos, best_effort);
        }
      },
      py::arg("tensor"),
      py::arg("pos") = -1,
      py::arg("best_effort") = false,
      py::arg("selected_tensors") = std::vector<Tensor>());
  nvf_sched.def("tensors", [](FusionDefinition::SchedOperators& self) {
    NVF_CHECK(
        self.validUse(),
        "Attempting to use a SchedOperators Op prior to definition!");
    // Return all Tensors in FusionDefinition
    return self.fusion_definition->tensors();
  });
  nvf_sched.def(
      "is_reduction",
      [](FusionDefinition::SchedOperators& self, Tensor tensor) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        // Determine if tensor is a result from a reduction operation.
        FusionDefinition* fd = self.fusion_definition;
        TensorView* tv =
            fd->getFusionState(tensor.index)->template as<TensorView>();
        return (
            !tv->isFusionInput() &&
            std::any_of(
                tv->getMaybeRootDomain().begin(),
                tv->getMaybeRootDomain().end(),
                [](IterDomain* id) { return id->isReduction(); }) &&
            !isResharding(tv->definition()));
      },
      py::arg("tensor"));
  nvf_sched.def(
      "can_schedule",
      [](FusionDefinition::SchedOperators& self,
         const SchedulerType& scheduler_type) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        return self.fusion_definition->userSchedule()->canScheduleDebug(
            scheduler_type);
      },
      py::arg("scheduler_type"));
  nvf_sched.def(
      "find_compatible_schedulers", [](FusionDefinition::SchedOperators& self) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");

        std::vector<SchedulerType> valid_scheduler_types;
        valid_scheduler_types.reserve(all_heuristics_in_priority_order.size());
        std::copy_if(
            all_heuristics_in_priority_order.begin(),
            all_heuristics_in_priority_order.end(),
            std::back_inserter(valid_scheduler_types),
            [sched = self.fusion_definition->userSchedule()](
                SchedulerType scheduler_type) {
              return sched->canSchedule(scheduler_type);
            });
        return valid_scheduler_types;
      });
  nvf_sched.def(
      "schedule",
      [](FusionDefinition::SchedOperators& self,
         const SchedulerType& scheduler_type) {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        UserSchedule* sched = self.fusion_definition->userSchedule();
        auto&& [can_schedule, error_msg] =
            sched->canScheduleDebug(scheduler_type);
        NVF_CHECK(can_schedule, error_msg);
        sched->scheduleWithType(scheduler_type);
      },
      py::arg("heuristic"));
  nvf_sched.def("schedule", [](FusionDefinition::SchedOperators& self) {
    NVF_CHECK(
        self.validUse(),
        "Attempting to use a SchedOperators Op prior to definition!");
    UserSchedule* sched = self.fusion_definition->userSchedule();
    sched->schedule();
  });
  nvf_sched.def(
      "compute_pointwise_heuristics",
      [](FusionDefinition::SchedOperators& self) -> PointwiseParams& {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        UserSchedule* sched = self.fusion_definition->userSchedule();
        HeuristicParams* parameters =
            sched->computeHeuristics(SchedulerType::PointWise);
        return *parameters->as<PointwiseParams>();
      },
      py::return_value_policy::reference);
  nvf_sched.def(
      "compute_reduction_heuristics",
      [](FusionDefinition::SchedOperators& self) -> ReductionParams& {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        UserSchedule* sched = self.fusion_definition->userSchedule();
        HeuristicParams* parameters =
            sched->computeHeuristics(SchedulerType::Reduction);
        return *parameters->as<ReductionParams>();
      },
      py::return_value_policy::reference);
  nvf_sched.def(
      "compute_matmul_heuristics",
      [](FusionDefinition::SchedOperators& self) -> MatmulParams& {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        UserSchedule* sched = self.fusion_definition->userSchedule();
        HeuristicParams* parameters =
            sched->computeHeuristics(SchedulerType::Matmul);
        return *parameters->as<MatmulParams>();
      },
      py::return_value_policy::reference);
  nvf_sched.def(
      "schedule_hyperparameters",
      [](FusionDefinition::SchedOperators& self)
          -> scheduler_utils::SchedulerHyperParameters& {
        NVF_CHECK(
            self.validUse(),
            "Attempting to use a SchedOperators Op prior to definition!");
        UserSchedule* sched = self.fusion_definition->userSchedule();
        auto scheduler_hyperparameters_entry = HeuristicDataCacheEntry<
            HeuristicCompileTime::SchedulerHyperParameters>(
            sched->data_cache.get(), []() {
              return std::make_unique<
                  scheduler_utils::SchedulerHyperParameters>(
                  /*vectorize_factor=*/1,
                  /*unroll_factor=*/1,
                  /*threads_per_block_min=*/1,
                  /*threads_per_block_max=*/1,
                  /*is_warp_specialized=*/false);
            });
        return scheduler_hyperparameters_entry.get();
      },
      py::return_value_policy::reference);
}

} // namespace nvfuser::python_frontend
