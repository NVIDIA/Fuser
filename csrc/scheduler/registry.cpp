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
#include <scheduler/resize.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>
#include <visibility.h>

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

  // These ops are  are only accepted in `ExprEval`
  // scheduler, all other schedulers should reject them.
  // TODO: remove IndexPutAccumulateOp
  if (ir_utils::hasOpsOfType<
          IndexShuffleOp,
          ScatterOp,
          SdpaFwdOp,
          SdpaBwdOp,
          EmbeddingFwdOp,
          IndexPutAccumulateOp,
          ArgsortOp,
          GroupedMmaOp,
          ScaledMmaOp,
          CutlassNvfp4GroupedMmaOp,
          TopKOp,
          ScanOp>(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "Has unsupported ops");
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

  if (registry_utils::SchedulerTopologyChecker::hasResizeAndIndexOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "has resize-based ops and index ops");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::hasCyclicReshape(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        scheduler_type, "Fusion has cyclic reshapes.");
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
    case SchedulerType::Resize:
      return std::make_unique<ResizeScheduler>();
    case SchedulerType::Communication:
      return std::make_unique<CommunicationScheduler>();
    default:
      NVF_THROW("unreachable");
  }
}

std::unique_ptr<HeuristicParams> SchedulerEntry::scheduleWith(
    Fusion* fusion,
    SchedulerType scheduler_type,
    const KernelArgumentHolder& runtime_inputs,
    bool validate_scheduler) {
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs);
  NVF_ERROR(
      !validate_scheduler ||
          Schedule::canSchedule(scheduler_type, fusion, runtime_info),
      "Could not schedule fusion with the SchedulerType: ",
      scheduler_type);
  auto scheduler_instance =
      SchedulerEntry::makeSchedulerInstance(scheduler_type);
  auto heuristic_params =
      scheduler_instance->computeHeuristics(fusion, runtime_info);
  scheduler_instance->schedule(fusion, heuristic_params.get());
  return heuristic_params;
}

namespace Schedule {
// Simple dispatcher interface
bool canSchedule(
    SchedulerType scheduler_type,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    bool skip_compile_time_checks) {
  // If a data cache is given, the compile time part doesn't need to be checked,
  // since during segmentation the segmenter will call
  // SchedulerEntry::proposeHeuristics which doesn't pass a data_cache.
  if (data_cache == nullptr && !checkCanSchedule(fusion, scheduler_type)) {
    return false;
  }

  std::unique_ptr<SchedulerEntry> scheduler =
      SchedulerEntry::makeSchedulerInstance(scheduler_type);

  if (!skip_compile_time_checks && !scheduler->canScheduleCompileTime(fusion)) {
    return false;
  }

  return scheduler->canScheduleRunTime(fusion, runtime_info, data_cache);
}

// Simply loop through the list as baseline strategy
SchedulerType proposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info) {
  for (const auto& sh : all_heuristics_in_priority_order) {
    if (canSchedule(sh, fusion, runtime_info)) {
      scheduler_debug_utils::canScheduleMessage("***Accepted*** as: ", sh);
      return sh;
    }
  }
  return SchedulerType::None;
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

void HeuristicDataCache::insert(HeuristicDataCache::EntryOwningPtr new_entry) {
  // Just override when insertion duplicates, equality not checked.
  entry_type_map_[new_entry->type()] = new_entry.get();
  entries_.emplace_back(std::move(new_entry));
}

template <typename EntryClass>
HeuristicDataCacheEntry<EntryClass>::HeuristicDataCacheEntry(
    HeuristicDataCache* data_cache,
    MakerFnType fn) {
  if (data_cache && data_cache->hasEntry(EntryClass::EntryType)) {
    data_ptr_ = data_cache->at(EntryClass::EntryType)
                    ->template as<CompileTimeInfo<EntryClass>>()
                    ->get();
  } else {
    owned_data_ = fn();
    data_ptr_ = owned_data_.get();

    if (data_cache) {
      std::unique_ptr<HeuristicCompileTime::CompileTimeInfoBase> new_entry =
          std::make_unique<CompileTimeInfo<EntryClass>>(std::move(owned_data_));
      data_cache->insert(std::move(new_entry));
    }
  }
}

// Template instantiation for pre-defined cache entries
template class HeuristicDataCacheEntry<HeuristicCompileTime::DomainMap>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::TransposeDomainMap>;
template class HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensors>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::ReferenceTensorsForGroups>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::VectorizableInputsAndOutputs>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::TvToContigInnerSizeMaps>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::ResizeVectorizationFactors>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::InputsOutputsInnerDimGroups>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::UnrollableInputsAndOutputs>;
template class HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::PersistentBufferInfo>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::ScopePersistentFactorInfo>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::BroadcastMultiples>;
template class HeuristicDataCacheEntry<HeuristicCompileTime::InnerMostDimInfo>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::CanScheduleTranspose>;
template class HeuristicDataCacheEntry<HeuristicCompileTime::LogicalReorderMap>;
template class HeuristicDataCacheEntry<
    HeuristicCompileTime::VectorizationBreakPointOfReductionProducer>;
template class NVF_API
    HeuristicDataCacheEntry<HeuristicCompileTime::SchedulerHyperParameters>;
} // namespace nvfuser
