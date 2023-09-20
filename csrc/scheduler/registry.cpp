// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <executor_utils.h>
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
    case ScheduleHeuristic::OuterPersistent:
      return checkCanSchedule<InnerPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
    case ScheduleHeuristic::Persistent:
      return checkCanSchedule<PersistentKernelScheduler>(
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
    case ScheduleHeuristic::OuterPersistent:
      scheduler_entry = std::make_unique<InnerPersistentKernelScheduler>(
          fusion, runtime_info, data_cache);
      break;
    case ScheduleHeuristic::Persistent:
      scheduler_entry = std::make_unique<PersistentKernelScheduler>(
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
  for (const auto& sh : all_heuristics_in_priority_order) {
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
    case ScheduleHeuristic::OuterPersistent:
      getInnerPersistentHeuristics(fusion, runtime_info, this);
      InnerPersistentKernelScheduler::canScheduleRunTime(
          fusion, runtime_info, this);
      break;
    case ScheduleHeuristic::Persistent:
      getPersistentHeuristics(fusion, runtime_info, this);
      PersistentKernelScheduler::canScheduleRunTime(fusion, runtime_info, this);
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
    case ScheduleHeuristic::OuterPersistent: {
      NVF_ERROR(entry_type_map_.count(EntryType::REDUCTION_TVS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      NVF_ERROR(entry_type_map_.count(EntryType::TV_TO_CONTIG_INNER_SIZE_MAPS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
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
    case ScheduleHeuristic::Persistent: {
      NVF_ERROR(entry_type_map_.count(EntryType::REDUCTION_TVS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::VECTORIZABLE_INPUTS_AND_OUTPUTS));
      NVF_ERROR(entry_type_map_.count(EntryType::TV_TO_CONTIG_INNER_SIZE_MAPS));
      NVF_ERROR(
          entry_type_map_.count(EntryType::UNROLLABLE_INPUTS_AND_OUTPUTS));
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
