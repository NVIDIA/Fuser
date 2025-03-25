// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <multidevice/utils.h>
#include <options.h>
#include <python_frontend/direct_bindings/fusion_definition.h>
#include <runtime/executor_params.h>
#include <scheduler/matmul.h>
#include <scheduler/pointwise.h>
#include <scheduler/reduction.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::python_frontend {

namespace {

void bindTensorViewScheduleOps(
    py::class_<DirectFusionDefinition::ScheduleOperators>& sched) {
  sched.def(
      "to_string",
      [](DirectFusionDefinition::ScheduleOperators& self, TensorView* tv) {
        return tv->toString();
      },
      R"(
        Get a string representation of the TensorView's schedule state.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to get the string representation of.

        Returns
        -------
        str
            A string containing the schedule state of the TensorView.
      )",
      py::arg("tensor"));

  sched.def(
      "parallelize",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         int axis,
         const ParallelType& parallel_type) {
        tv->axis(axis)->parallelize(parallel_type);
      },
      R"(
        Parallelize a specific axis of a TensorView.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to parallelize.
        axis : int
            The axis to parallelize.
        parallel_type : ParallelType
            The type of parallelization to apply (e.g., TIDx, TIDy, BIDx, etc.).

        Returns
        -------
        None
      )",
      py::arg("tensor"),
      py::arg("axis"),
      py::arg("parallel_type"));

  sched.def(
      "merge",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         int dim) { tv->merge(dim); },
      R"(
        Merge a dimension with the next dimension in the TensorView.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to merge dimensions in.
        dim : int
            The dimension to merge with the next dimension.

        Returns
        -------
        None
      )",
      py::arg("tensor"),
      py::arg("dim"));

  auto reduction_factor_func =
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         const std::vector<int64_t>& dims) -> TensorView* {
    return tv->rFactor(dims);
  };

  sched.def(
      "reduction_factor",
      reduction_factor_func,
      R"(
        Create a reduction factor TensorView by splitting the reduction axes.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to create a reduction factor from.
        dims : List[int]
            The dimensions to split for the reduction factor.

        Returns
        -------
        TensorView
            The new reduction factor TensorView.
      )",
      py::arg("tensor"),
      py::arg("dims"));

  sched.def(
      "rfactor",
      reduction_factor_func,
      R"(
        Alias for reduction_factor. Creates a reduction factor TensorView by splitting the reduction axes.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to create a reduction factor from.
        dims : List[int]
            The dimensions to split for the reduction factor.

        Returns
        -------
        TensorView
            The new reduction factor TensorView.
      )",
      py::arg("tensor"),
      py::arg("dims"));

  sched.def(
      "reorder",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         const std::unordered_map<int64_t, int64_t>& old2new) {
        tv->reorder(old2new);
      },
      R"(
        Reorder the dimensions of a TensorView according to the specified mapping.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to reorder.
        old2new : Dict[int, int]
            A dictionary mapping old dimension indices to new dimension indices.

        Returns
        -------
        None
      )",
      py::arg("tensor"),
      py::arg("old2new"));

  sched.def(
      "split",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         int64_t dim,
         int64_t factor,
         bool inner_split) { tv->split(dim, factor, inner_split); },
      R"(
        Split a dimension of a TensorView into two dimensions.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to split a dimension in.
        dim : int
            The dimension to split.
        factor : int
            The factor to split the dimension by.
        inner_split : bool, optional
            Whether to perform an inner split (default: True).

        Returns
        -------
        None
      )",
      py::arg("tensor"),
      py::arg("dim"),
      py::arg("factor"),
      py::arg("inner_split") = true);

  sched.def(
      "cache_after",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         const LoadStoreOpType& op_type,
         const CacheOp& cache_op) -> TensorView* {
        return tv->cacheAfter(op_type, cache_op);
      },
      R"(
        Cache the TensorView after the specified operation.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to cache.
        op_type : LoadStoreOpType, optional
            The type of load/store operation (default: Set).
        cache_op : CacheOp, optional
            The type of cache operation (default: Unspecified).

        Returns
        -------
        TensorView
            The new cached TensorView.
      )",
      py::arg("tensor"),
      py::arg("op_type") = LoadStoreOpType::Set,
      py::arg("cache_op") = CacheOp::Unspecified);

  sched.def(
      "cache_before",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         const LoadStoreOpType& op_type) -> TensorView* {
        return tv->cacheBefore(op_type);
      },
      R"(
        Cache the TensorView before the specified operation.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to cache.
        op_type : LoadStoreOpType, optional
            The type of load/store operation (default: Set).

        Returns
        -------
        TensorView
            The new cached TensorView.
      )",
      py::arg("tensor"),
      py::arg("op_type") = LoadStoreOpType::Set);

  sched.def(
      "cache_fork",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv) -> TensorView* { return tv->cacheFork(); },
      R"(
        Create a forked cache of the TensorView.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to create a cache fork from.

        Returns
        -------
        TensorView
            The new forked TensorView.
      )",
      py::arg("tensor"));

  sched.def(
      "set_memory_type",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         const MemoryType& memory_type) { tv->setMemoryType(memory_type); },
      R"(
        Set the memory type for the TensorView.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to set the memory type for.
        memory_type : MemoryType
            The memory type to set.

        Returns
        -------
        None
      )",
      py::arg("tensor"),
      py::arg("memory_type"));

  sched.def(
      "transform_like",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* reference_tv,
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

  sched.def(
      "parallelize_like",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* reference_tv,
         int64_t pos,
         const std::vector<TensorView*>& selected_tensors,
         const std::unordered_set<ParallelType>& selected_parallel_types,
         bool propagate_padding) {
        nvfuser::scheduler_utils::parallelizeAllLike(
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

  sched.def(
      "inline_most",
      [](DirectFusionDefinition::ScheduleOperators& self,
         const std::vector<TensorView*>& selected_tensors) {
        if (selected_tensors.empty()) {
          nvfuser::inlineMost();
        } else {
          nvfuser::inlineMost(selected_tensors);
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

  sched.def(
      "inline_at",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* reference_tv,
         int64_t pos,
         bool best_effort,
         const std::vector<TensorView*>& selected_tensors) {
        if (selected_tensors.empty()) {
          // Inline to the position corresponding to the reference position in
          // the reference tensor for all tensors in the current fusion.
          nvfuser::inlineAllAt(reference_tv, pos, best_effort);
        } else {
          // Inline to the position corresponding to the reference position in
          // the reference tensor for selected tensors in the current fusion.
          std::unordered_set<TensorView*> selected_tv_set(
              selected_tensors.begin(), selected_tensors.end());
          nvfuser::inlineSelectedAt(
              selected_tv_set, reference_tv, pos, best_effort);
        }
      },
      R"(
        Inline operations at a specific position for the selected tensors.
        If selected_tensors is empty, inlines all operations.

        Parameters
        ----------
        reference_tv : TensorView
            The reference TensorView whose position will be used for inlining.
        pos : int, optional
            The position to inline at. -1 means the last position.
        best_effort : bool, optional
            Whether to try to inline even if the exact position is not possible (default: False).
        selected_tensors : List[TensorView], optional
            List of TensorViews to inline. If empty, inlines all operations.

        Returns
        -------
        None
      )",
      py::arg("reference_tv"),
      py::arg("pos") = -1,
      py::arg("best_effort") = false,
      py::arg("selected_tensors") = std::vector<TensorView*>());

  sched.def(
      "is_reduction",
      [](DirectFusionDefinition::ScheduleOperators& self, TensorView* tv) {
        // Determine if tensor is a result from a reduction operation.
        return (
            !tv->isFusionInput() &&
            std::any_of(
                tv->getMaybeRootDomain().begin(),
                tv->getMaybeRootDomain().end(),
                [](IterDomain* id) { return id->isReduction(); }) &&
            !isResharding(tv->definition()));
      },
      R"(
        Check if a TensorView is a result from a reduction operation.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to check.

        Returns
        -------
        bool
            True if the TensorView is a reduction result, False otherwise.
            A TensorView is considered a reduction if:
            1. It is not a fusion input
            2. It has at least one reduction domain
            3. It is not a resharding operation
      )",
      py::arg("tensor"));

  sched.def(
      "set_allocation_as_loop",
      [](DirectFusionDefinition::ScheduleOperators& self, TensorView* tv) {
        tv->setAllocationDomain(tv->getLoopDomain(), true);
      },
      R"(
        Set the allocation domain of a TensorView to match its loop domain.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to set the allocation domain for.

        Returns
        -------
        None
      )",
      py::arg("tensor"));

  sched.def(
      "_set_device_mesh",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         const DeviceMesh& mesh) { tv->setDeviceMesh(mesh); },
      R"(
        Set the device mesh for a TensorView.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to set the device mesh for.
        mesh : DeviceMesh
            The device mesh to set.

        Returns
        -------
        None
      )",
      py::arg("tensor"),
      py::arg("mesh"));
}

//! Debug function to check if a fusion can be scheduled with a given scheduler
//! type. It enables collection of messages from canScheduleRejectReason and
//! returns a tuple of (can_schedule, debug_messages).
std::tuple<bool, std::string> canScheduleDebug(
    Fusion* fusion,
    const py::iterable& iter,
    const SchedulerType& scheduler_type) {
  // Enable collection of messages from canScheduleRejectReason
  DebugDumpOptionsGuard debug_dump_options_guard;
  DebugDumpOptionsGuard::getCurOptions().set(
      DebugDumpOption::FusionSegmenterLog);

  // Send debug messages to stringstream
  std::stringstream ss;
  DebugStreamGuard dsg(ss);

  SchedulerRuntimeInfo runtime_info(
      fusion,
      from_pyiterable(iter),
      /*precomputed_values=*/nullptr,
      fusion->allTvs());
  bool can_schedule =
      Schedule::canSchedule(scheduler_type, fusion, runtime_info);
  return std::make_tuple(can_schedule, ss.str());
}

void bindFusionScheduleOps(
    py::class_<DirectFusionDefinition::ScheduleOperators>& sched) {
  sched.def(
      "can_schedule",
      [](DirectFusionDefinition::ScheduleOperators& self,
         Fusion* fusion,
         const py::iterable& iter,
         const SchedulerType& scheduler_type) {
        return canScheduleDebug(fusion, iter, scheduler_type);
      },
      R"(
        Check if the fusion can be scheduled with the given scheduler type.

        Parameters
        ----------
        scheduler_type : SchedulerType
            The type of scheduler to check compatibility with.

        Returns
        -------
        bool
            True if the fusion can be scheduled with the given scheduler type,
            False otherwise.

        Notes
        -----
        This is a debug function that checks if the fusion's structure is compatible
        with the specified scheduler type. It does not actually schedule the fusion.
      )",
      py::arg("fusion"),
      py::arg("inputs"),
      py::arg("scheduler_type"));

  sched.def(
      "find_compatible_schedulers",
      [](DirectFusionDefinition::ScheduleOperators& self,
         Fusion* fusion,
         const py::iterable& iter) {
        SchedulerRuntimeInfo runtime_info(
            fusion,
            from_pyiterable(iter),
            /*precomputed_values=*/nullptr,
            fusion->allTvs());
        std::vector<SchedulerType> valid_scheduler_types;
        valid_scheduler_types.reserve(all_heuristics_in_priority_order.size());
        std::copy_if(
            all_heuristics_in_priority_order.begin(),
            all_heuristics_in_priority_order.end(),
            std::back_inserter(valid_scheduler_types),
            [fusion, &runtime_info](SchedulerType scheduler_type) {
              return Schedule::canSchedule(
                  scheduler_type, fusion, runtime_info);
            });
        return valid_scheduler_types;
      },
      R"(
        Find all compatible scheduler types for the given fusion and inputs.

        Parameters
        ----------
        fusion : Fusion
            The fusion to find compatible scheduler types for.
        inputs : iterable
            The inputs to the fusion.

        Returns
        -------
        list
            A list of compatible scheduler types.
      )",
      py::arg("fusion"),
      py::arg("inputs"));

  sched.def(
      "auto_schedule",
      [](DirectFusionDefinition::ScheduleOperators& self,
         Fusion* fusion,
         const py::iterable& iter,
         const SchedulerType& heuristic,
         HeuristicParams* heuristic_params) {
        SchedulerRuntimeInfo runtime_info(
            fusion,
            from_pyiterable(iter),
            /*precomputed_values=*/nullptr,
            fusion->allTvs());
        auto&& [can_schedule, error_msg] =
            canScheduleDebug(fusion, iter, heuristic);
        NVF_CHECK(can_schedule, error_msg);

        std::unique_ptr<SchedulerEntry> scheduler =
            SchedulerEntry::makeSchedulerInstance(heuristic);
        if (heuristic_params == nullptr) {
          std::unique_ptr<HeuristicParams> heuristic_params =
              scheduler->computeHeuristics(fusion, runtime_info, nullptr);
          scheduler->schedule(fusion, heuristic_params.get());
        } else {
          scheduler->schedule(fusion, heuristic_params);
        }
      },
      R"(
        Schedule the fusion with the given scheduler type.

        Parameters
        ----------
        heuristic : SchedulerType
            The scheduler type to use for scheduling.

        Returns
        -------
        None
      )",
      py::arg("fusion"),
      py::arg("inputs"),
      py::arg("heuristic"),
      py::arg("heuristic_params") = py::none());

  sched.def(
      "compute_heuristics",
      [](DirectFusionDefinition::ScheduleOperators& self,
         Fusion* fusion,
         const py::iterable& iter,
         const SchedulerType& scheduler_type) {
        std::unique_ptr<SchedulerEntry> scheduler =
            SchedulerEntry::makeSchedulerInstance(scheduler_type);
        SchedulerRuntimeInfo runtime_info(
            fusion,
            from_pyiterable(iter),
            /*precomputed_values=*/nullptr,
            fusion->allTvs());
        NVF_CHECK(
            scheduler->canScheduleCompileTime(fusion) &&
                scheduler->canScheduleRunTime(fusion, runtime_info),
            "Could not schedule fusion with ",
            scheduler_type,
            " scheduler.");
        return scheduler->computeHeuristics(fusion, runtime_info, nullptr);
      },
      R"(
        Compute the heuristics for the given fusion and inputs.

        Parameters
        ----------
        fusion : Fusion
            The fusion to compute heuristics for.
        inputs : iterable
            The inputs to the fusion.

        Returns
        -------
        PointwiseParams
            The pointwise heuristics for the given fusion and inputs.
      )",
      py::arg("fusion"),
      py::arg("inputs"),
      py::arg("scheduler_type"));
}

void bindCircularBuffering(
    py::class_<DirectFusionDefinition::ScheduleOperators>& sched) {
  sched.def(
      "warp_specialize",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         int64_t number_of_stages,
         int64_t prefetch_distance,
         ParallelType parallel_type,
         std::optional<std::pair<int64_t, int64_t>> num_registers) {
        CircularBufferType circular_buffer_type = (num_registers.has_value())
            ? WarpSpecialized(parallel_type, num_registers.value())
            : WarpSpecialized(parallel_type);
        tv->circularBuffer(
            number_of_stages, prefetch_distance, circular_buffer_type);
      },
      R"(
        Apply warp specialization circular buffering to the given TensorView.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to apply warp specialization circular buffering to. 
        number_of_stages : int
            The number of stages in the circular buffer.
        prefetch_distance : int
            The prefetch distance for the circular buffer.
        parallel_type : ParallelType
            The parallel type to apply warp specialization to.
        num_registers : tuple of int, optional
            The number of registers to use for the warp specialization.
      )",
      py::arg("tensor"),
      py::arg("number_of_stages"),
      py::arg("prefetch_distance"),
      py::arg("parallel_type"),
      py::arg("num_registers") = py::none());
  sched.def(
      "pipeline",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         int64_t number_of_stages,
         int64_t prefetch_distance,
         bool uses_mbarrier_for_war) {
        CircularBufferType circular_buffer_type =
            Pipelined(uses_mbarrier_for_war);
        tv->circularBuffer(
            number_of_stages, prefetch_distance, circular_buffer_type);
      },
      R"(
        Apply circular buffering pipelining to the given TensorView.

        Parameters
        ----------
        tensor : TensorView
            The TensorView to apply circular buffering pipelining to. 
        number_of_stages : int
            The number of stages in the circular buffer.
        prefetch_distance : int
            The prefetch distance for the circular buffer.
        uses_mbarrier_for_war : bool
            Whether to use mbarrier synchronization for the circular buffer pipelining.
      )",
      py::arg("tensor"),
      py::arg("number_of_stages"),
      py::arg("prefetch_distance"),
      py::arg("uses_mbarrier_for_war") = false);
}

void bindAbstractTensor(
    py::class_<DirectFusionDefinition::ScheduleOperators>& sched) {
  // Bind AbstractTensor specialized with EmptyInfo
  py::class_<AbstractTensor>(sched, "AbstractTensor", R"(
Abstract representation of a tensor with multiple dimensions.

The AbstractTensor is similar to TensorView, having multiple dimensions where each dimension
is represented by an Abstract IterDomain. It provides operations like merge, split, etc.,
but only has a single "domain" instead of multiple domains like "logical domain", "loop domain", etc.)")
      .def(py::init<>(), R"(
Default constructor creating an empty AbstractTensor.)")
      .def(py::init<const std::vector<IterDomain*>&>(), R"(
Create an AbstractTensor from a vector of IterDomains.

Parameters
----------
domains : List[IterDomain]
    List of IterDomains representing the tensor dimensions.)")

      // Access methods
      .def(
          "__getitem__",
          [](AbstractTensor& self, int64_t i) { return self[i]; },
          R"(
Access dimension at the given index.

Parameters
----------
index : int
    Index of the dimension to access.

Returns
-------
AbstractId
    The abstract domain at the specified index.)",
          py::arg("index"))
      .def("size", &AbstractTensor::size, R"(
Get the number of dimensions in the tensor.

Returns
-------
int
    Number of dimensions.)")
      .def("empty", &AbstractTensor::empty, R"(
Check if the tensor has no dimensions.

Returns
-------
bool
    True if the tensor has no dimensions, False otherwise.)")

      // Transformation methods
      .def(
          "parallelize",
          &AbstractTensor::parallelize,
          R"(
Parallelize a specific dimension.

Parameters
----------
axis : int
    The dimension to parallelize
parallel_type : ParallelType
    The type of parallelization to apply

Returns
-------
self : AbstractTensor
    Returns self for method chaining.)",
          py::arg("axis"),
          py::arg("parallel_type"))
      .def(
          "split",
          py::overload_cast<int64_t, int64_t, bool>(&AbstractTensor::split),
          R"(
Split a dimension into two dimensions using an integer factor.

Parameters
----------
axis : int
    The dimension to split
factor : int
    The splitting factor
inner_split : bool, optional
    If True, the factor determines the size of the inner dimension (default: True)

Returns
-------
self : AbstractTensor
    Returns self for method chaining.)",
          py::arg("axis"),
          py::arg("factor"),
          py::arg("inner_split") = true)
      .def(
          "merge",
          py::overload_cast<int64_t>(&AbstractTensor::merge),
          R"(
Merge a dimension with the next dimension.

Parameters
----------
axis : int
    The dimension to merge with the next dimension

Returns
-------
self : AbstractTensor
    Returns self for method chaining.)",
          py::arg("axis"))
      .def(
          "reorder",
          py::overload_cast<const std::vector<int64_t>&>(
              &AbstractTensor::reorder),
          R"(
Reorder the dimensions according to the given permutation.

Parameters
----------
permutation : List[int]
    The new order of dimensions

Returns
-------
self : AbstractTensor
    Returns self for method chaining.)",
          py::arg("permutation"))
      .def(
          "swizzle",
          py::overload_cast<SwizzleType, int64_t, int64_t>(
              &AbstractTensor::swizzle),
          R"(
Apply a swizzle operation between two dimensions.

Parameters
----------
swizzle_type : SwizzleType
    The type of swizzle operation to apply
dim_x : int
    First dimension for swizzling
dim_y : int
    Second dimension for swizzling

Returns
-------
self : AbstractTensor
    Returns self for method chaining.)",
          py::arg("swizzle_type"),
          py::arg("dim_x"),
          py::arg("dim_y"))

      // Comparison operators
      .def(
          "__eq__",
          py::overload_cast<const AbstractTensor&>(
              &AbstractTensor::operator==, py::const_),
          R"(
Compare this AbstractTensor with another for equality.

Parameters
----------
other : AbstractTensor
    The other AbstractTensor to compare with

Returns
-------
bool
    True if the tensors are equal, False otherwise.)",
          py::arg("other"));
}

void bindLdStMatrix(
    py::class_<DirectFusionDefinition::ScheduleOperators>& sched) {
  sched.def(
      "ldst_matrix",
      [](DirectFusionDefinition::ScheduleOperators& self,
         TensorView* tv,
         const LoadStoreOpType& op_type,
         int64_t m_tile,
         int64_t n_tile,
         int64_t m_smem,
         int64_t n_smem) {
        tv->fusion()->manage("ldst_matrix_m_tile", m_tile);
        tv->fusion()->manage("ldst_matrix_n_tile", n_tile);
        tv->fusion()->manage("ldst_matrix_m_smem", m_smem);
        tv->fusion()->manage("ldst_matrix_n_smem", n_smem);
        TensorView* result = tv->cacheAfter();
        result->definition()->as<LoadStoreOp>()->setOpType(op_type);
        return result;
      },
      R"(
        Apply LDMatrix to the given TensorView.

        Parameters
        ----------
        tv : TensorView
            The TensorView to apply LDMatrix to.
        op_type : LoadStoreOpType
            Select LoadStoreOpType.load_matrix or LoadStoreOpType.store_matrix.
        m_tile : int
            The size of the tile in the M dimension.
        n_tile : int
            The size of the tile in the N dimension.
        m_smem : int
            The size of the SMEM in the M dimension.
        n_smem : int
            The size of the SMEM in the N dimension.

        Returns
        -------
        TensorView
            The TensorView with LDMatrix applied.
      )",
      py::arg("tv"),
      py::arg("op_type"),
      py::arg("m_tile"),
      py::arg("n_tile"),
      py::arg("m_smem"),
      py::arg("n_smem"));
}

} // namespace

void bindDirectScheduleOperators(
    py::class_<DirectFusionDefinition>& fusion_def) {
  //! The ScheduleOperators class is a nested class of DirectFusionDefinition to
  //! allow the user to query the class for the list of schedule operators.
  //!
  //! Example:
  //!   help(DirectFusionDefinition.ScheduleOperators)
  py::class_<DirectFusionDefinition::ScheduleOperators> nvf_sched(
      fusion_def, "ScheduleOperators");
  nvf_sched.def(py::init<DirectFusionDefinition*>());
  bindTensorViewScheduleOps(nvf_sched);
  bindFusionScheduleOps(nvf_sched);
  bindCircularBuffering(nvf_sched);
  bindAbstractTensor(nvf_sched);
  bindLdStMatrix(nvf_sched);
}

} // namespace nvfuser::python_frontend
