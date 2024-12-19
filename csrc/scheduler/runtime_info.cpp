// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// #include <ATen/cuda/CUDAContext.h>
#include <instrumentation.h>
#include <runtime/executor_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/vectorize_helper.h>
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
        // NOTE: alloc_sizes and alloc_strides are already in order of
        // allocation domain
        input_sizes_.emplace(fusion_inp, alloc_sizes.vec());
        input_strides_elements_.emplace(fusion_inp, alloc_strides.vec());
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

size_t SchedulerRuntimeInfo::getAlignmentSize(
    TensorView* tv,
    int64_t datatype_size,
    const std::unordered_map<
        TensorView*,
        vectorize_helper::TensorResizeAlignmentInfo>& resize_alignment_map) {
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

  // resize operation has special checks on alignment sizes.
  auto resize_id_it = resize_alignment_map.find(tv);
  if (resize_id_it != resize_alignment_map.end()) {
    // get tensor stride
    auto strides = getInputAllocationStrides(tv);
    // resize, specifically negative resize (slicing), makes its outer dimension
    // non-contiguous, `non_contig_idx_alloc` is the index for allocation domain
    // that needs alignment check on its stride.
    for (int64_t alloc_idx : resize_id_it->second.non_contig_idx_alloc) {
      alignment_size = std::min(
          alignment_size,
          SchedulerRuntimeInfo::computeAlignmentSize(
              strides[alloc_idx] * datatype_size));
    }
  }

  alignment_map_[tv] = alignment_size;
  return alignment_size;
}

} // namespace nvfuser
