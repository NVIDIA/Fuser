// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <fusion_executor/executor_utils.h>
#include <instrumentation.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/registry.h>
#include <scheduler/registry_utils.h>
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
  FUSER_PERF_SCOPE("SchedulerRuntimeInfo::SchedulerRuntimeInfo");
  NVF_ERROR(
      complete_fusion_->inputs().size() == args.size(),
      "The provided fusion group expects ",
      complete_fusion_->inputs().size(),
      " arguments, but ",
      args.size(),
      " arguments were passed in.");

  expression_evaluator_ = getExpressionEvaluator(args, precomputed_values);

  if (forced_index_type.has_value()) {
    index_type_ = forced_index_type.value();
  } else {
    index_type_ = registry_utils::getIndexTypeOfKernel(
        complete_fusion_,
        all_tvs.empty() ? complete_fusion_->allTvs() : all_tvs,
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

      std::optional<std::vector<int64_t>> alloc_perm_opt =
          ir_utils::computePermutation(
              TensorDomain::noReductions(input_tv->getLogicalDomain()),
              TensorDomain::noReductions(input_tv->getMaybeAllocationDomain()));
      if (alloc_perm_opt.has_value()) {
        // Save the strides in order of allocation domain in case the
        // allocation domain is a permutation of RFactor domain
        std::vector<int64_t> orig_sizes = alloc_sizes.vec();
        std::vector<int64_t> orig_strides = alloc_strides.vec();
        std::vector<int64_t> ordered_sizes, ordered_strides;
        ordered_sizes.reserve(orig_sizes.size());
        ordered_strides.reserve(orig_strides.size());
        NVF_ERROR(orig_strides.size() == alloc_perm_opt->size());
        for (int64_t i : alloc_perm_opt.value()) {
          ordered_sizes.push_back(orig_sizes[i]);
          ordered_strides.push_back(orig_strides[i]);
        }
        input_sizes_[fusion_inp] = std::move(ordered_sizes);
        input_strides_elements_[fusion_inp] = std::move(ordered_strides);
      }

      // find and push discontiguous stride
      int64_t dtype_size = dataTypeSize(input_tv->dtype());
      input_discontig_strides_[fusion_inp] = {};
      auto dims = static_cast<int64_t>(alloc_strides.size());
      int64_t expected_stride = 1;
      for (int64_t dim = dims - 1; dim >= 0; dim--) {
        auto size = alloc_sizes.at(dim);
        auto stride = alloc_strides.at(dim);
        // Skip broadcast dimensions because they don't affect contiguity.
        // Consider to change this to check IterDomain::isBroadcast instead:
        // https://github.com/NVIDIA/Fuser/pull/2854#discussion_r1733205035
        if (size <= 1 || stride == 0) {
          continue;
        }

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

namespace {
//! A Utility for checking both dynamic and static part of
//!  can schedule
bool checkCanSchedule(Fusion* fusion, SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE("SchedulerRuntimeInfo::checkCanSchedule<T>");
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
    HeuristicDataCache* data_cache) {
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

HeuristicDataCache::HeuristicDataCache(
    Fusion* fusion,
    SchedulerType scheduler_type,
    SchedulerRuntimeInfo& runtime_info)
    : scheduler_type_(scheduler_type) {}

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

//! TODO: Move to another file, or make it so it can be in Heuristics.h by
//! moving SchedulerRuntimeInfo outisde registry.h
HeuristicParamsList::HeuristicParamsList(
    SchedulerType schedule_heuristic,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache)
    : is_segmented_(false) {
  heuristics_.emplace_back(
      SchedulerEntry::makeSchedulerInstance(schedule_heuristic)
          ->computeHeuristics(runtime_info.fusion(), runtime_info, data_cache));
}

} // namespace nvfuser
