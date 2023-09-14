// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <executor_utils.h>
#include <instrumentation.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/pointwise.h>
#include <scheduler/registry.h>
#include <scheduler/registry_utils.h>
#include <scheduler/transpose.h>
#include <scheduler/utils.h>
#include <tensor_metadata.h>

namespace nvfuser {

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    KernelArgumentHolder args,
    PrecomputedValues* precomputed_values,
    const std::vector<TensorView*>& all_tvs,
    std::optional<PrimDataType> forced_index_type)
    : complete_fusion_(complete_fusion) {
  NVF_ERROR(
      complete_fusion_->inputs().size() == args.size(),
      "Invalid number of arguments passed in for provided fusion group.");

  expression_evaluator_ = getExpressionEvaluator(args, precomputed_values);

  if (forced_index_type.has_value()) {
    index_type_ = forced_index_type.value();
  } else {
    index_type_ = registry_utils::getIndexTypeOfKernel(
        complete_fusion_,
        all_tvs.empty() ? ir_utils::allTvs(complete_fusion_) : all_tvs,
        args,
        *expression_evaluator_);
  }

  for (auto inp_i : c10::irange(static_cast<int64_t>(args.size()))) {
    auto fusion_inp = complete_fusion_->inputs().at(inp_i);
    auto input_tv = dynamic_cast<TensorView*>(fusion_inp);
    // Note: we are skipping CpuScalar tensor here
    if (input_tv != nullptr && !input_tv->isCpuScalar()) {
      const auto& metadata =
          expression_evaluator_->evaluate(IrBuilder::metadataExpr(input_tv));
      const auto& alloc_sizes = metadata->*&TensorMetaData::alloc_size;
      const auto& alloc_strides = metadata->*&TensorMetaData::alloc_stride;
      NVF_ERROR(alloc_sizes.size() == alloc_strides.size());

      input_ptrs_[fusion_inp] = (size_t)(metadata->*&TensorMetaData::data);

      // find and push discontiguous stride
      int64_t dtype_size = dataTypeSize(input_tv->dtype());
      input_discontig_strides_[fusion_inp] = {};
      int64_t dims = (int64_t)alloc_strides.size();
      int64_t expected_stride = 1;
      for (int64_t dim = dims - 1; dim >= 0; dim--) {
        auto size = alloc_sizes.at(dim);
        if (size <= 1) {
          continue;
        }
        auto stride = alloc_strides.at(dim);
        if (stride != expected_stride) {
          input_discontig_strides_[fusion_inp].push_back(stride * dtype_size);
          expected_stride = stride;
        }
        expected_stride *= size;
      }
    }
  }
}

SchedulerRuntimeInfo::SchedulerRuntimeInfo(
    Fusion* complete_fusion,
    const at::ArrayRef<c10::IValue>& aten_inputs)
    : SchedulerRuntimeInfo(
          complete_fusion,
          KernelArgumentHolder::createKernelArgumentHolder(aten_inputs)) {}

// TODO: Output tensors could have an alignment that is not 16 Bytes passed in
// from user.
size_t SchedulerRuntimeInfo::ptrOf(TensorView* tv) const {
  if (input_ptrs_.find(tv) != input_ptrs_.end()) {
    return input_ptrs_.at(tv);
  }
  return max_alignment_size_in_byte;
}

std::unique_ptr<ExpressionEvaluator> SchedulerRuntimeInfo::
    getExpressionEvaluator(
        const KernelArgumentHolder& args,
        PrecomputedValues* precomputed_values) {
  std::unique_ptr<ExpressionEvaluator> ee =
      std::make_unique<ExpressionEvaluator>(
          executor_utils::bindInputs(args, complete_fusion_));
  if (precomputed_values) {
    ee->bindPrecomputedValues(precomputed_values);
  }
  return ee;
}

size_t SchedulerRuntimeInfo::computeAlignmentSize(size_t ptr_address) {
  size_t alignment_size = 1;
  size_t next_alignment_size = 2;

  while (next_alignment_size <= max_alignment_size_in_byte &&
         ptr_address % next_alignment_size == 0) {
    alignment_size = next_alignment_size;
    next_alignment_size *= 2;
  }
  return alignment_size;
}

size_t SchedulerRuntimeInfo::getAlignmentSize(TensorView* tv) {
  auto alignment_entry = alignment_map_.find(tv);
  if (alignment_entry != alignment_map_.end()) {
    return alignment_entry->second;
  }

  auto alignment_size = SchedulerRuntimeInfo::computeAlignmentSize(ptrOf(tv));
  auto strides_it = input_discontig_strides_.find(tv);
  if (strides_it != input_discontig_strides_.end()) {
    for (auto stride : strides_it->second) {
      alignment_size = std::min(
          alignment_size, SchedulerRuntimeInfo::computeAlignmentSize(stride));
    }
  }
  alignment_map_[tv] = alignment_size;
  return alignment_size;
}

bool SchedulerEntry::sameAs(const SchedulerEntry* other) {
  return heuristic_ == other->heuristic_ && params_->sameAs(other->params_);
}

namespace {

//! Scheduler interface:
//!    Each of the scheduler needs to provide 3 interface functions:
//!
//!      1. canScheduleCompileTime(Fusion* fusion) :
//!
//!        This function contains compiled-time checks on the graph itself
//!        without runtime input information. Only `fusion` is given in the
//!        argument to make sure only compile-time available info is needed in
//!        the check.
//!
//!        This function is to be called exactly once on each segmented group
//!        created in a segmented fusion so this part will not contribute to
//!        dynamic shape latency.
//!
//!     2. canScheduleRunTime(
//!            Fusion* fusion,
//!            SchedulerRuntimeInfo& runtime_info,
//!           HeuristicSummary* data_cache = nullptr):
//!        This function contains all canSchedule checks that will have to
//!        involve runtime input information, and will be run both by the
//!        segmenter and the kernel cache. The latency of this function will
//!        contribute to dynamic shape latency so `data_cache` should be used as
//!        much as possible to save re-computation.
//!
//!     3. schedule(fusion):
//!
//!        This function will be called when compiling a kernel. It should apply
//!        scheduling to the given fusion

//! NoOp scheduler represents the case where scheduler will
//!  not do any scheduling operations and forward the un-scheduled
//!  fusion directly to code generation and kernel compilation.
//!
//! Typical use case of this scheduler is to handle edge cases
//!  such as where all tensors are size-1 or size-0.
class NoOpScheduler : public SchedulerEntry {
  //! Provides a dummy heuristic type to ensure
  //!  unified interface on NoOp scheduler.
  class NoOpHeuristic : public HeuristicParams {
   public:
    using HeuristicParams::HeuristicParams;

    size_t hash() const override {
      return 0;
    }
    std::shared_ptr<HeuristicParams> clone() const override {
      return std::make_shared<NoOpHeuristic>();
    }
    bool sameAs(const std::shared_ptr<HeuristicParams>& other) const override {
      auto other_casted = std::dynamic_pointer_cast<ReductionParams>(other);
      return other_casted != nullptr && other_casted->cparams == cparams;
    };
  };

 public:
  explicit NoOpScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::NoOp) {
    params_ = std::make_shared<NoOpHeuristic>("", runtime_info.getIndexType());
  }

  //! Check if the no-op heuristics apply in given fusion
  static bool canScheduleCompileTime(Fusion* fusion) {
    if (fusion->isNoOp()) {
      return true;
    }
    // Check there're no non-trivial reduction ops.
    for (auto reduction : ir_utils::getReductionOps(fusion)) {
      for (auto output :
           ir_utils::filterByType<TensorView>(reduction->outputs())) {
        auto concrete_dimension =
            TensorDomain::noReductions(output->getRootDomain());
        auto all_nonzero = std::none_of(
            concrete_dimension.begin(),
            concrete_dimension.end(),
            [](IterDomain* id) { return id->extent()->isZeroInt(); });
        if (all_nonzero) {
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::NoOp,
              "reduction of non-zero elements is not supported");
          return false;
        }
      }
    }

    // Check that all outputs are either broadcast or ignored reduction.
    for (auto out_tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
      auto concrete_dimension = TensorDomain::noReductions(
          TensorDomain::noBroadcasts(out_tv->getLeafDomain()));
      if (!concrete_dimension.empty()) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::NoOp, "output has a concrete dimension");
        return false;
      }
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (registry_utils::rejectScheduleForMemoryPromotion(
            fusion, ScheduleHeuristic::NoOp)) {
      return false;
    }

    // We have verified that all iterdomains on all output tv's are trivial
    // reductions,
    //  broadcasts or zero-sized. Therefore accepting this fusion for NoOp
    //  scheduling.
    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    // TODO:
    //  Pipe through dynamic zero checks.
    return true;
  }

  void schedule(Fusion* fusion) override {
    // Schedule is no-op.
    return;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    // Heuristics is no-op.
    return;
  }
};

class ReductionScheduler : public SchedulerEntry {
 public:
  explicit ReductionScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Reduction) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  //! Check if the reduction heuristics apply in given fusion
  static bool canScheduleCompileTime(Fusion* fusion) {
    // Needs at least one reduction to consider.
    if (ir_utils::getReductionOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction, "No reduction op to schedule");
      return false;
    }

    if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "Scheduling not supported with no input");
      return false;
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (registry_utils::rejectScheduleForMemoryPromotion(
            fusion, ScheduleHeuristic::Reduction)) {
      return false;
    }

    // Fusions handled by reduction scheduler cannot have MmaOp.
    if (!ir_utils::getMmaOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction, "no support for mma ops.");
      return false;
    }

    auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

    if (reduction_tvs.empty()) {
      // Use pointwise logic
      return false;
    }

    if (registry_utils::hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    if (!ir_utils::getViewOps(fusion).empty()) {
      ComputeAtMap ca_map(fusion);
      if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Reduction,
            "Fusion requires view being reversible.");
        return false;
      }

      // Reduction scheduler simply uses reduction_tvs[0] as the reference, if
      // that changes, this needs to be changed.
      if (registry_utils::reductionInterferingView(
              fusion, ca_map, reduction_tvs[0])) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Reduction,
            "View may interfere with reduction scheduling.");
        return false;
      }
    }

    // Make sure reduction axes are consistent through the fusion
    auto reduction_ops = ir_utils::getReductionOps(fusion);
    if (reduction_ops.size() > 1) {
      // Before examining the reduction axes want to quickly
      //   check the reductions have the same axis width
      //   to avoid building root domain map in easier cases
      bool valid_axis_count = false;
      size_t axis_count = 0;
      auto reduction_root_size = [](TensorView* red_tv) {
        size_t count = 0;
        for (auto id : red_tv->getRootDomain()) {
          if (!id->isBroadcast()) {
            count++;
          }
        }
        return count;
      };

      for (auto red : reduction_tvs) {
        if (!valid_axis_count) {
          valid_axis_count = true;
          axis_count = reduction_root_size(red);
        } else {
          if (reduction_root_size(red) != axis_count) {
            scheduler_debug_utils::canScheduleRejectReason(
                ScheduleHeuristic::Reduction,
                "Inconsistent reduction axes ",
                red,
                "is not ",
                axis_count);
            return false;
          }
        }
      }

      // Use root domain map to check the reduction ops have the same axes
      FusionGuard fg(fusion);
      ComputeAtRootDomainMap root_map;
      root_map.build(true);

      // red_ops.size()>1 checked before
      for (size_t it = 1; it < reduction_tvs.size(); it++) {
        if (!registry_utils::checkPatternEquivalence(
                reduction_tvs[it - 1], reduction_tvs[it], root_map)) {
          scheduler_debug_utils::canScheduleRejectReason(
              ScheduleHeuristic::Reduction,
              "Un-mapped multi-reduction: ",
              reduction_tvs[it - 1],
              " ",
              reduction_tvs[it]);
          return false;
        }
      }
    }

    // Doesn't allow persistent kernels in this scheduler
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
    if (!persistent_buffer_info.persistent_buffers.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "need persistent buffers that reduction scheduler doesn't handle");
      return false;
    }

    if (!registry_utils::SchedulerTopologyChecker::supportedPostReductionFusion(
            fusion, reduction_tvs) ||
        registry_utils::SchedulerTopologyChecker::hasPostReductionBCast(
            fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "has unsupported post reduction fusion");
      return false;
    }

    if (registry_utils::SchedulerTopologyChecker::
            hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Reduction,
          "has unsupported gather-like ops before reduction");
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Single Reduction");
    scheduleReduction(fusion, reductionParams());
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getReductionHeuristics(fusion, runtime_info, data_cache);
    NVF_ERROR(params_ != nullptr);
  }
};

class TransposeScheduler : public SchedulerEntry {
 public:
  explicit TransposeScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Transpose) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    // Check that inputs of all select/gather-like ops are fusion inputs
    if (registry_utils::rejectScheduleForMemoryPromotion(
            fusion, ScheduleHeuristic::Transpose)) {
      return false;
    }

    // Fusions handled by transpose scheduler cannot have MmaOp.
    if (!ir_utils::getMmaOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "no support for mma ops.");
      return false;
    }

    for (auto select : ir_utils::getSelectOps(fusion)) {
      auto root = TensorDomain::noReductions(
          select->input(0)->as<TensorView>()->getMaybeRFactorDomain());
      if (select->getIndexedID() == root[root.size() - 1]) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Transpose,
            "SelectOp on inner dim is not supported by transpose scheduler yet."
            "In transpose scheduler, we want to leave the select dim alone, instead of creating a tile for it.");
        return false;
      }
    }
    for (auto idx_sel : ir_utils::getIndexSelectOps(fusion)) {
      auto root = TensorDomain::noReductions(
          idx_sel->input(0)->as<TensorView>()->getMaybeRFactorDomain());
      if (idx_sel->getIndexedID() == root[root.size() - 1]) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Transpose,
            "IndexSelectOp on inner dim is not supported by transpose scheduler yet."
            "In transpose scheduler, we want to leave the select dim alone, instead of creating a tile for it.");
        return false;
      }
    }
    for (auto torch_gather : ir_utils::getTorchGatherOps(fusion)) {
      auto root = TensorDomain::noReductions(
          torch_gather->input(0)->as<TensorView>()->getMaybeRFactorDomain());
      if (torch_gather->dim() == (int)root.size() - 1) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::Transpose,
            "TorchGatherOp on inner dim is not supported by transpose scheduler yet."
            "In transpose scheduler, we want to leave the select dim alone, instead of creating a tile for it.");
        return false;
      }
    }

    if (!hasAtLeastTwoValidGroups(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose,
          "cannot find two mismatching inner most dimensions");
      return false;
    }

    auto reduction_ops = ir_utils::getReductionOps(fusion);

    if (!reduction_ops.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, "no support for reduction ops");
      return false;
    }

    if (registry_utils::hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    FUSER_PERF_SCOPE("TransposeScheduler::canScheduleRunTime");

    auto reason =
        getTransposeRuntimeRejectReason(fusion, data_cache, runtime_info);
    if (!reason.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Transpose, reason);
      return false;
    }
    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Transpose Fusion");
    scheduleTranspose(fusion, transposeParams());
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getTransposeHeuristics(fusion, runtime_info, data_cache);
    NVF_ERROR(params_ != nullptr);
  }
};

class PointWiseScheduler : public SchedulerEntry {
 public:
  explicit PointWiseScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::PointWise) {
    computeHeuristics(fusion, runtime_info, data_cache);
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    //   Currently using the same path as the scheduler
    // to eliminate mismatch between canSchedule and
    // schedule pointwise.
    if (!hasReferenceTensorView(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "cannot find reference tensor");
      return false;
    }

    // Check that inputs of all select/gather-like ops are fusion inputs
    if (registry_utils::rejectScheduleForMemoryPromotion(
            fusion, ScheduleHeuristic::PointWise)) {
      return false;
    }

    // Fusions handled by pointwise scheduler cannot have MmaOp.
    if (!ir_utils::getMmaOps(fusion).empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "no support for mma ops.");
      return false;
    }

    if (!ir_utils::getViewOps(fusion).empty()) {
      ComputeAtMap ca_map(fusion);
      if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            ScheduleHeuristic::PointWise,
            "Fusion requires view being reversible.");
        return false;
      }
    }

    auto reduction_ops = ir_utils::getReductionOps(fusion);

    if (!reduction_ops.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise, "no support for reduction ops");
      return false;
    }

    if (registry_utils::hasNonUniqueBcast(fusion)) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::PointWise,
          "Broadcasting dimension might be broadcasting to multiple sizes.");
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    auto can_schedule_transpose_entry =
        HeuristicSummaryEntry<HeuristicCompileTime::CanScheduleTranspose>(
            data_cache, [fusion]() {
              return std::make_unique<bool>(
                  TransposeScheduler::canScheduleCompileTime(fusion));
            });
    if (can_schedule_transpose_entry.get()) {
      auto reason =
          getTransposeRuntimeRejectReason(fusion, data_cache, runtime_info);
      return !reason.empty();
    }

    return true;
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule PointWise Fusion");
    schedulePointwise(fusion, pointwiseParams());
  }

  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getPointwiseHeuristics(fusion, runtime_info, data_cache);
    NVF_ERROR(params_ != nullptr);
  }
};

class MatmulScheduler : public SchedulerEntry {
 public:
  explicit MatmulScheduler(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr)
      : SchedulerEntry(ScheduleHeuristic::Matmul) {
    computeHeuristics(fusion, runtime_info);
  }

  void schedule(Fusion* fusion) override {
    FUSER_PERF_SCOPE("Schedule Matmul Fusion");
    scheduleMatmul(fusion, matmulParams());
  }

  static bool canScheduleCompileTime(Fusion* fusion) {
    const auto msg = getMatmulCompileTimeRejectReason(fusion);
    if (!msg.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Matmul, msg);
      return false;
    }

    return true;
  }

  static bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    FUSER_PERF_SCOPE("MatmulScheduler::canSchedule");
    auto reason =
        getMatmulRunTimeRejectReason(fusion, data_cache, runtime_info);
    if (!reason.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          ScheduleHeuristic::Matmul, reason);
      return false;
    }
    return true;
  }

 private:
  void computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr) {
    params_ = getMatmulHeuristics(fusion, runtime_info, data_cache);
    NVF_ERROR(params_ != nullptr);
  }
};

// Schedule Table
const std::vector<ScheduleHeuristic>& all_heuristics() {
  static const std::vector<ScheduleHeuristic> hlist = {
      ScheduleHeuristic::NoOp,
      ScheduleHeuristic::Reduction,
      ScheduleHeuristic::Transpose,
      ScheduleHeuristic::PointWise,
      ScheduleHeuristic::InnerPersistent,
      ScheduleHeuristic::OuterPersistent,
      ScheduleHeuristic::InnerOuterPersistent,
      ScheduleHeuristic::Matmul};
  return hlist;
}

//! A Utility for checking both dynamic and static part of
//!  can schedule
template <typename SchedulerType>
bool checkCanSchedule(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr) {
  FusionGuard fg(fusion);
  // If a data cache is given, the compile time part doesn't need to be checked,
  // since for all current use cases
  //  it has to pass all the compile time checks to create a data cache for this
  //  fusion.
  if (!data_cache) {
    if (!registry_utils::isConnectedFusionGraph(fusion)) {
      return false;
    }
    if (IterDomainGraph(fusion, /*allow_self_mapping=*/true).hasSelfMapping()) {
      return false;
    }
    if (!SchedulerType::canScheduleCompileTime(fusion)) {
      return false;
    }
  }

  return SchedulerType::canScheduleRunTime(fusion, runtime_info, data_cache);
}

} // namespace

// Simple dispatcher interface
bool SchedulerEntry::canSchedule(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  switch (sh) {
    case ScheduleHeuristic::NoOp:
      return checkCanSchedule<NoOpScheduler>(fusion, runtime_info, data_cache);
    case ScheduleHeuristic::PointWise:
      return checkCanSchedule<PointWiseScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Reduction:
      return checkCanSchedule<ReductionScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::InnerPersistent:
      return checkCanSchedule<InnerPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::OuterPersistent:
      return checkCanSchedule<OuterPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::InnerOuterPersistent:
      return checkCanSchedule<InnerOuterPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Transpose:
      return checkCanSchedule<TransposeScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Matmul:
      return checkCanSchedule<MatmulScheduler>(
          fusion, runtime_info, data_cache);
    default:
      NVF_ERROR(false, "unreachable");
      return false;
  }
  return false;
}

std::unique_ptr<SchedulerEntry> SchedulerEntry::makeEntry(
    ScheduleHeuristic sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  std::unique_ptr<SchedulerEntry> scheduler_entry = nullptr;
  switch (sh) {
    case ScheduleHeuristic::NoOp:
      scheduler_entry =
          std::make_unique<NoOpScheduler>(fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::PointWise:
      scheduler_entry = std::make_unique<PointWiseScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Reduction:
      scheduler_entry = std::make_unique<ReductionScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::InnerPersistent:
      scheduler_entry = std::make_unique<InnerPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::OuterPersistent:
      scheduler_entry = std::make_unique<OuterPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::InnerOuterPersistent:
      scheduler_entry = std::make_unique<InnerOuterPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Transpose:
      scheduler_entry = std::make_unique<TransposeScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Matmul:
      scheduler_entry =
          std::make_unique<MatmulScheduler>(fusion, runtime_info, data_cache);
      break;
    default:
      NVF_ERROR(false, "unreachable");
  }

  return scheduler_entry;
}

// Simply loop through the list as baseline strategy
std::optional<ScheduleHeuristic> SchedulerEntry::proposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  for (auto sh : all_heuristics()) {
    if (canSchedule(sh, fusion, runtime_info)) {
      scheduler_debug_utils::canScheduleMessage("***Accepted*** as: ", sh);
      return sh;
    }
  }
  return std::nullopt;
}

size_t SchedulerEntryHash::operator()(const SchedulerEntry& se) const {
  return se.params()->hash();
}

namespace {

//! CompileTimeInfo is the actual subclass of CompileTimeInfoBase that will
//!  be stored in the data cache. It owns a data_ state internally of the
//!  dataType defined within the entry class, which are listed in compile
//!  time info header.
template <typename EntryClass>
class CompileTimeInfo : public HeuristicCompileTime::CompileTimeInfoBase {
 public:
  CompileTimeInfo(std::unique_ptr<typename EntryClass::DataType> data)
      : CompileTimeInfoBase(EntryClass::EntryType), data_(std::move(data)) {}

  typename EntryClass::DataType* get() {
    return data_.get();
  }

 private:
  std::unique_ptr<typename EntryClass::DataType> data_;
};

} // namespace

HeuristicSummary::HeuristicSummary(
    Fusion* fusion,
    ScheduleHeuristic heuristic,
    SchedulerRuntimeInfo& runtime_info)
    : heuristic_(heuristic), recording_(true) {
  switch (heuristic) {
    case ScheduleHeuristic::NoOp:
      NoOpScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::PointWise:
      getPointwiseHeuristics(fusion, runtime_info, this);
      PointWiseScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Reduction:
      getReductionHeuristics(fusion, runtime_info, this);
      ReductionScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::InnerPersistent:
      InnerPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, this);
      InnerPersistentKernelScheduler::canScheduleRunTime(
          fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::OuterPersistent:
      OuterPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, this);
      OuterPersistentKernelScheduler::canScheduleRunTime(
          fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::InnerOuterPersistent:
      InnerOuterPersistentKernelScheduler::getPersistentHeuristic(
          fusion, runtime_info, this);
      InnerOuterPersistentKernelScheduler::canScheduleRunTime(
          fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Transpose:
      getTransposeHeuristics(fusion, runtime_info, this);
      TransposeScheduler::canScheduleRunTime(fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Matmul: {
      const auto heuristics = getMatmulHeuristics(fusion, runtime_info, this);
      NVF_ERROR(heuristics, "Failed to get matmul heuristics");
      const auto canSchedule =
          MatmulScheduler::canScheduleRunTime(fusion, runtime_info, this);
      NVF_ERROR(canSchedule, "Could not schedule matmul (run time)");
      break;
    }
    default:
      NVF_ERROR(false, "unknown heuristic");
  }
  validate();
  recording_ = false;
}

void HeuristicSummary::validate() const {
  switch (heuristic_) {
    case ScheduleHeuristic::NoOp: {
      // TODO: need to cache the dynamically zero inputs?
      break;
    }
    case ScheduleHeuristic::Transpose:
    case ScheduleHeuristic::PointWise: {
      if (heuristic_ == ScheduleHeuristic::PointWise) {
        NVF_ERROR(entry_type_map_.count(EntryType::DOMAIN_MAP));
        NVF_ERROR(entry_type_map_.count(EntryType::REFERENCE_TENSORS));
        NVF_ERROR(
            entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
        NVF_ERROR(
            entry_type_map_.count(EntryType::TV_TO_CONTIG_INNER_SIZE_MAPS));
        NVF_ERROR(entry_type_map_.count(EntryType::BROADCAST_BYTE_MULTIPLES));
        NVF_ERROR(entry_type_map_.count(EntryType::CAN_SCHEDULE_TRANSPOSE));
        auto can_schedule_transpose =
            entry_type_map_.at(EntryType::CAN_SCHEDULE_TRANSPOSE)
                ->as<CompileTimeInfo<
                    HeuristicCompileTime::CanScheduleTranspose>>()
                ->get();
        if (!*can_schedule_transpose) {
          break;
        }
      }
      NVF_ERROR(entry_type_map_.count(EntryType::TRANSPOSE_DOMAIN_MAP));
      NVF_ERROR(entry_type_map_.count(
          EntryType::INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS));
      NVF_ERROR(entry_type_map_.count(EntryType::REFERENCE_TENSORS_FOR_GROUPS));
      NVF_ERROR(entry_type_map_.count(EntryType::INNER_MOST_DIMS_INFO));
      break;
    }
    case ScheduleHeuristic::Reduction: {
      NVF_ERROR(entry_type_map_.count(EntryType::REDUCTION_TVS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      NVF_ERROR(entry_type_map_.count(EntryType::TV_TO_CONTIG_INNER_SIZE_MAPS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
      break;
    }
    case ScheduleHeuristic::InnerPersistent:
    case ScheduleHeuristic::OuterPersistent:
      NVF_ERROR(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
    // No break, fall through additional checks
    case ScheduleHeuristic::InnerOuterPersistent: {
      NVF_ERROR(entry_type_map_.count(EntryType::REDUCTION_TVS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      NVF_ERROR(entry_type_map_.count(EntryType::TV_TO_CONTIG_INNER_SIZE_MAPS));

      NVF_ERROR(entry_type_map_.count(EntryType::PERSISTENT_BUFFER_INFO));
      // If check persistent factor only when persistent buffers needed.
      auto persistent_buffer_info =
          entry_type_map_.at(EntryType::PERSISTENT_BUFFER_INFO)
              ->as<
                  CompileTimeInfo<HeuristicCompileTime::PersistentBufferInfo>>()
              ->get();
      NVF_ERROR(
          !persistent_buffer_info->persistent_buffers.empty() &&
          entry_type_map_.count(EntryType::SCOPE_PERSISTENT_FACTOR_INFO));
      break;
    }
    case ScheduleHeuristic::Matmul: {
      // TODO: add a proper set of checks
      break;
    }
    default:
      NVF_ERROR(false, "unknown heuristic");
  }
}

void HeuristicSummary::insert(HeuristicSummary::EntryOwningPtr new_entry) {
  NVF_ERROR(recording_, "should only insert entries at recording phase");
  // Just override when insertion duplicates, equality not checked.
  entry_type_map_[new_entry->type()] = new_entry.get();
  entries_.emplace_back(std::move(new_entry));
}

template <typename EntryClass>
HeuristicSummaryEntry<EntryClass>::HeuristicSummaryEntry(
    HeuristicSummary* data_cache,
    MakerFnType fn) {
  using InfoType = CompileTimeInfo<EntryClass>;

  if (!data_cache || data_cache->isRecording()) {
    owned_data_ = fn();
    data_ptr_ = owned_data_.get();

    if (data_cache) {
      std::unique_ptr<HeuristicCompileTime::CompileTimeInfoBase> new_entry =
          std::make_unique<InfoType>(std::move(owned_data_));
      data_cache->insert(std::move(new_entry));
    }
  } else {
    data_ptr_ =
        data_cache->at(EntryClass::EntryType)->template as<InfoType>()->get();
  }
}

// Template instantiation for pre-defined cache entries
template class HeuristicSummaryEntry<HeuristicCompileTime::DomainMap>;
template class HeuristicSummaryEntry<HeuristicCompileTime::TransposeDomainMap>;
template class HeuristicSummaryEntry<HeuristicCompileTime::ReferenceTensors>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::ReferenceTensorsForGroups>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::VectorizableInputsAndOutputs>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::TvToContigInnerSizeMaps>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::InputsOutputsInnerDimGroups>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::UnrollableInputsAndOutputs>;
template class HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::PersistentBufferInfo>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::ScopePersistentFactorInfo>;
template class HeuristicSummaryEntry<HeuristicCompileTime::BroadcastMultiples>;
template class HeuristicSummaryEntry<HeuristicCompileTime::InnerMostDimInfo>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::CanScheduleTranspose>;

} // namespace nvfuser
