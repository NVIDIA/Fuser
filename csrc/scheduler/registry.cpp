// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <instrumentation.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/debug_utils.h>
#include <scheduler/heuristic.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/registry.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

namespace nvfuser {

namespace {
//! A Utility for checking both dynamic and static part of
//!  can schedule
bool checkCanSchedule(Fusion* fusion, SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE("registry.cpp::checkCanSchedule<T>");
  // ExprEval scheduler only requires `canScheduleCompileTime` check and should
  // not use this fn. The following checks build the computeAt map that do not
  // work with SDPAOp.
  if (scheduler_type == SchedulerType::ExprEval) {
    return true;
  }

  FusionGuard fg(fusion);

  // Fusions with `SdpaFwdOp/SdpaBwdOp` are only accepted in `ExprEval`
  // scheduler, all other schedulers should reject them.
  if (ir_utils::hasOpsOfType<SdpaFwdOp, SdpaBwdOp>(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "SdpaOps are not supported.");
    return false;
  }

  // Fusions with `MatmulOp, LinearOp, MmaOp` can only be accepted by Matmul
  // scheduler.
  if (scheduler_type != SchedulerType::Matmul &&
      ir_utils::hasOpsOfType<MatmulOp, LinearOp, MmaOp>(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "Matmul ops are not supported.");
    return false;
  }

  if (!registry_utils::isConnectedFusionGraph(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "Connected fusion graph check failed!");
    return false;
  }
  if (IterDomainGraph(fusion, /*allow_self_mapping=*/true).hasSelfMapping()) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "Iter domain graph check failed!");
    return false;
  }

  return true;
}

} // namespace

// Dispatch heuristic type to the right derived class of scheduler entry.
// Scheduler entries are stateless so it's a lightweight class to dispatch to
// the virtual functions in this abstract class.
std::unique_ptr<SchedulerEntry> SchedulerEntry::makeSchedulerInstance(
    SchedulerType scheduler_type) {
  std::unique_ptr<SchedulerEntry> scheduler = nullptr;
  switch (scheduler_type) {
    case SchedulerType::NoOp:
      return std::make_unique<NoOpScheduler>();
    case SchedulerType::PointWise:
      return std::make_unique<PointWiseScheduler>();
    case SchedulerType::Reduction:
      return std::make_unique<ReductionScheduler>();
    case SchedulerType::InnerPersistent:
      return std::make_unique<InnerPersistentKernelScheduler>();
    case SchedulerType::OuterPersistent:
      return std::make_unique<OuterPersistentKernelScheduler>();
    case SchedulerType::InnerOuterPersistent:
      return std::make_unique<InnerOuterPersistentKernelScheduler>();
    case SchedulerType::Transpose:
      return std::make_unique<TransposeScheduler>();
    case SchedulerType::Matmul:
      return std::make_unique<MatmulScheduler>();
    case SchedulerType::ExprEval:
      return std::make_unique<ExprEvalScheduler>();
    default:
      NVF_THROW("unreachable");
  }
}

namespace Schedule {
// Simple dispatcher interface
bool canSchedule(
    SchedulerType scheduler_type,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  // If a data cache is given, the compile time part doesn't need to be checked,
  // since during segmentation the segmenter will call
  // SchedulerEntry::proposeHeuristics which doesn't pass a data_cache.
  if (data_cache == nullptr && !checkCanSchedule(fusion, scheduler_type)) {
    return false;
  }

  std::unique_ptr<SchedulerEntry> scheduler =
      SchedulerEntry::makeSchedulerInstance(scheduler_type);
  return scheduler->canScheduleCompileTime(fusion) &&
      scheduler->canScheduleRunTime(fusion, runtime_info, data_cache);
}

// Simply loop through the list as baseline strategy
std::optional<SchedulerType> proposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  for (const auto& sh : all_heuristics_in_priority_order) {
    if (canSchedule(sh, fusion, runtime_info)) {
      scheduler_debug_utils::canScheduleMessage("***Accepted*** as: ", sh);
      return sh;
    }
  }
  return std::nullopt;
}
} // namespace Schedule

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
    SchedulerType scheduler_type,
    SchedulerRuntimeInfo& runtime_info)
    : scheduler_type_(scheduler_type), recording_(true) {
  switch (scheduler_type) {
    case SchedulerType::NoOp:
      NoOpScheduler().canScheduleRunTime(fusion, runtime_info, this);
      break;
    case SchedulerType::PointWise:
      getPointwiseHeuristics(fusion, runtime_info, this);
      PointWiseScheduler().canScheduleRunTime(fusion, runtime_info, this);
      break;
    case SchedulerType::Reduction:
      getReductionHeuristics(fusion, runtime_info, this);
      ReductionScheduler().canScheduleRunTime(fusion, runtime_info, this);
      break;
    case SchedulerType::InnerPersistent:
      getInnerPersistentHeuristics(fusion, runtime_info, this);
      InnerPersistentKernelScheduler().canScheduleRunTime(
          fusion, runtime_info, this);
      break;
    case SchedulerType::OuterPersistent:
      getOuterPersistentHeuristics(fusion, runtime_info, this);
      OuterPersistentKernelScheduler().canScheduleRunTime(
          fusion, runtime_info, this);
      break;
    case SchedulerType::InnerOuterPersistent:
      getInnerOuterPersistentHeuristics(fusion, runtime_info, this);
      InnerOuterPersistentKernelScheduler().canScheduleRunTime(
          fusion, runtime_info, this);
      break;
    case SchedulerType::Transpose:
      getTransposeHeuristics(fusion, runtime_info, this);
      TransposeScheduler().canScheduleRunTime(fusion, runtime_info, this);
      break;
    case SchedulerType::Matmul: {
      const auto heuristics = getMatmulHeuristics(fusion, runtime_info, this);
      NVF_ERROR(heuristics, "Failed to get matmul heuristics");
      const auto canSchedule =
          MatmulScheduler().canScheduleRunTime(fusion, runtime_info, this);
      NVF_ERROR(canSchedule, "Could not schedule matmul (run time)");
      break;
    }
    case SchedulerType::ExprEval:
      ExprEvalScheduler().canScheduleRunTime(fusion, runtime_info, this);
      break;
    default:
      NVF_THROW("unknown heuristic");
  }
  validate();
  recording_ = false;
}

void HeuristicSummary::validate() const {
  switch (scheduler_type_) {
    case SchedulerType::NoOp: {
      // TODO: need to cache the dynamically zero inputs?
      break;
    }
    case SchedulerType::Transpose:
    case SchedulerType::PointWise: {
      if (scheduler_type_ == SchedulerType::PointWise) {
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
        NVF_ERROR(entry_type_map_.count(EntryType::LOGICAL_REORDER_MAP));
      }
      NVF_ERROR(entry_type_map_.count(EntryType::TRANSPOSE_DOMAIN_MAP));
      NVF_ERROR(entry_type_map_.count(
          EntryType::INPUTS_AND_OUTPUTS_INNER_DIM_GROUPS));
      NVF_ERROR(entry_type_map_.count(EntryType::REFERENCE_TENSORS_FOR_GROUPS));
      NVF_ERROR(entry_type_map_.count(EntryType::INNER_MOST_DIMS_INFO));
      break;
    }
    case SchedulerType::Reduction: {
      NVF_ERROR(entry_type_map_.count(EntryType::REDUCTION_TVS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      NVF_ERROR(entry_type_map_.count(EntryType::TV_TO_CONTIG_INNER_SIZE_MAPS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
      break;
    }
    case SchedulerType::InnerPersistent:
    case SchedulerType::OuterPersistent:
      NVF_ERROR(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
    // No break, fall through additional checks
    case SchedulerType::InnerOuterPersistent: {
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
    case SchedulerType::ExprEval:
    case SchedulerType::Matmul: {
      // TODO: add a proper set of checks for matmul
      break;
    }
    default:
      NVF_THROW("unknown heuristic");
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
template class HeuristicSummaryEntry<HeuristicCompileTime::LogicalReorderMap>;
template class HeuristicSummaryEntry<
    HeuristicCompileTime::VectorizationBreakPointOfReductionProducer>;

} // namespace nvfuser
