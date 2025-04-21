// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <scheduler/normalization_inner_outer_utils.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {
namespace inner_outer_utils {

int64_t roundUpSharedMemory(
    int64_t tv_buffer_size,
    int64_t data_type_size,
    int64_t vectorize_factor,
    int64_t threads_per_block_min,
    int64_t threads_per_block_max,
    int64_t threads_per_block_step) {
  int64_t dim_size = tv_buffer_size / data_type_size;
  int64_t after_vect = dim_size / vectorize_factor;
  int64_t max_smem = 0;
  for (int64_t threads_per_block = threads_per_block_min;
       threads_per_block <= threads_per_block_max;
       threads_per_block += threads_per_block_step) {
    int64_t n_batch = ceilDiv(after_vect, threads_per_block);
    max_smem = std::max(
        max_smem,
        n_batch * vectorize_factor * threads_per_block * data_type_size);
  }
  return max_smem;
}

std::vector<TensorView*> getOuterBroadcastTvs(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs) {
  // set reference broadcast mask using the first inner reduction tv
  std::vector<bool> ref_broadcast_mask;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      const auto& logical = tv->getLogicalDomain();
      ref_broadcast_mask.reserve(logical.size());
      for (const auto i : arange(logical.size())) {
        ref_broadcast_mask.push_back(!logical.at(i)->isReduction());
      }
      break;
    }
  }
  NVF_ERROR(!ref_broadcast_mask.empty(), "ref_broadcast_mask is empty!");

  // find the broadcast tensor whose broadcast mask is same to the reference
  std::vector<TensorView*> outer_broadcast_tvs;
  for (auto tv : fusion->allTvs()) {
    if (std::any_of(
            tv->getLoopDomain().begin(),
            tv->getLoopDomain().end(),
            [](IterDomain* id) { return id->isBroadcast(); })) {
      if (auto bcast = dynamic_cast<BroadcastOp*>(tv->definition())) {
        if (bcast->getBroadcastDimFlags() == ref_broadcast_mask) {
          outer_broadcast_tvs.emplace_back(tv);
        }
      }
    }
  }
  return outer_broadcast_tvs;
}

int64_t partialOuterReductionBufferSize(
    const std::vector<TensorView*>& reduction_tvs,
    SchedulerRuntimeInfo& runtime_info) {
  int64_t partial_reduction_buffer_size = 0;
  for (auto buffer : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(buffer)) {
      continue;
    }
    int64_t buffer_size = -1;
    for (auto id : buffer->getLogicalDomain()) {
      if (id->isReduction() || id->isBroadcast()) {
        continue;
      }
      auto id_size = runtime_info.expressionEvaluator().evaluate(id->extent());
      NVF_ERROR(id_size.hasValue(), "Could not infer persistent buffer size.");
      if (buffer_size == -1) {
        buffer_size = id_size.as<int64_t>();
      } else {
        buffer_size *= id_size.as<int64_t>();
      }
    }
    buffer_size = (buffer_size == -1) ? 0
                                      : buffer_size *
            (int64_t)dataTypeSize(buffer->getDataType().value(),
                                  runtime_info.getIndexType());
    partial_reduction_buffer_size += buffer_size;
  }
  return partial_reduction_buffer_size;
}

std::vector<TensorView*> sortProjectableBufferInputs(
    const std::vector<TensorView*>& projectable_buffer_inputs,
    const std::vector<TensorView*>& outer_broadcast_tvs) {
  // mark whether the buffer is used by outer broadcast tensors
  std::unordered_map<TensorView*, bool> is_used_by_outer_bcast;
  for (auto buffer : projectable_buffer_inputs) {
    is_used_by_outer_bcast[buffer] = std::any_of(
        outer_broadcast_tvs.begin(),
        outer_broadcast_tvs.end(),
        [&buffer](TensorView* tv) {
          return DependencyCheck::isDependencyOf(buffer, tv);
        });
  }

  // sort based on [is_used_by_outer_bcast]
  std::vector<TensorView*> sorted_buffer = projectable_buffer_inputs;
  std::sort(
      sorted_buffer.begin(),
      sorted_buffer.end(),
      [&](TensorView* a, TensorView* b) {
        return !is_used_by_outer_bcast[a] && is_used_by_outer_bcast[b];
      });
  return sorted_buffer;
}

PersistentBufferStorageParams getPersistentBufferStorageParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t vectorize_factor,
    const int64_t threads_per_block_min,
    const int64_t threads_per_block_max) {
  FUSER_PERF_SCOPE(
      "normalization_inner_outer::getPersistentBufferStorageParams");

  PersistentBufferStorageParams buffer_params;

  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSize(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  // Project to inputs when there is at least one outer broadcast tensor or
  // projected persistent buffer size is smaller. When projecting to inputs, the
  // outer broadcast tensor is reused in the loop over the iteration dimension,
  // test shows it is faster than the non-projected version which requires
  // reload from gmem for each iteration.
  // Note: in current use cases (layer norm bwd and RMS norm bwd), there are
  // outer broadcast tvs and always project to inputs.
  // Warp specialized persistent kernel always cache inputs in shared memory,
  // should project to inputs.
  const auto& outer_broadcast_tvs = getOuterBroadcastTvs(fusion, reduction_tvs);
  bool skip_check_buffer_size = !outer_broadcast_tvs.empty() ||
      isOptionEnabled(EnableOption::WarpSpecializedNormalization);
  normalization_scheduler_utils::BufferProjectionStrategy project_strategy =
      normalization_scheduler_utils::isProjectBufferToInputs(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          persistent_buffer_size_info,
          InnerOuterPersistentKernelScheduler::schedulerType(),
          /*can_use_smem_persistent=*/true,
          !skip_check_buffer_size);

  buffer_params.project_to_input =
      (project_strategy ==
       normalization_scheduler_utils::BufferProjectionStrategy::
           ProjectToInputs);

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t smem_overhead = scheduler_utils::getSharedMemoryOverheadPerBlock(
      fusion, reduction_tvs, threads_per_block_max);
  int64_t available_smem =
      (int64_t)dev_prop->sharedMemPerMultiprocessor - smem_overhead;
  int64_t available_regs = scheduler_utils::register_file_size_56k;
  buffer_params.smem_overhead = smem_overhead;

  // (1) Use both register and shared memory.
  // Start with all the cached input buffers in shared memory, they are loaded
  // from global memory uses async copy which bypasses L1 cache. Outer reduction
  // buffers are used to accumulate partial results of the outer reduction. They
  // are not loaded from global memory and requires frequent read/write. So,
  // they are always stored in registers.
  // TODO: We may also move outer reduction buffers to shared
  // memory to avoid segmentation when there are many outer reductions and
  // hardware has larger shared memory, but these applications are rare, so this
  // is not considered here.
  auto buffers = buffer_params.project_to_input
      ? persistent_buffer_info.projectable_buffer_inputs
      : persistent_buffer_info.persistent_buffers;

  // Add buffers that are inputs to the fusion. They are not included in
  // projectable_buffer_inputs since they are not projectable.
  if (buffer_params.project_to_input) {
    for (auto tv : persistent_buffer_info.persistent_buffers) {
      if (tv->isFusionInput()) {
        buffers.push_back(tv);
      }
    }
  }

  // Needs to use rounded shared memory size to avoid over usage.
  // key : buffer tv.
  // val : register size and rounded shared memory size
  std::unordered_map<TensorView*, std::pair<int64_t, int64_t>>
      required_size_regs_smem_map;
  int64_t total_smem_buffer_size = 0;
  for (auto buffer : buffers) {
    int64_t buffer_size_regs = scheduler_utils::getPersistentBufferSizeOfTensor(
        buffer, runtime_info, persistent_buffer_info);
    int64_t buffer_size_smem = roundUpSharedMemory(
        buffer_size_regs,
        dataTypeSize(buffer->getDataType().value()),
        vectorize_factor,
        threads_per_block_min,
        threads_per_block_max,
        dev_prop->warpSize);
    required_size_regs_smem_map[buffer] =
        std::make_pair(buffer_size_regs, buffer_size_smem);
    total_smem_buffer_size += buffer_size_smem;
  }
  buffer_params.smem_buffer_size = total_smem_buffer_size;
  buffer_params.regs_buffer_size =
      partialOuterReductionBufferSize(reduction_tvs, runtime_info);
  if (buffer_params.regs_buffer_size <= available_regs &&
      buffer_params.smem_buffer_size <= available_smem) {
    buffer_params.smem_persistent_buffers = buffers;
    buffer_params.has_enough_regs_and_smem = true;
    return buffer_params;
  }

  // Moving outer reduction buffer to shared memory is not considered yet,
  // set to false if the outer reduction buffer size exceeds the register size.
  if (buffer_params.regs_buffer_size > available_regs) {
    buffer_params.has_enough_regs_and_smem = false;
    return buffer_params;
  }

  // (2) Now, shared memory is overused, move some buffers to registers.
  // (2.1) Sort the candidate persistent buffers. No need to sort since the
  // sorting is based on whether the buffer is used by outer broadcast tensors.
  if (!outer_broadcast_tvs.empty()) {
    buffers = sortProjectableBufferInputs(buffers, outer_broadcast_tvs);
  }
  // (2.2) Before this loop, all cached input buffers are in shared memory. Move
  // buffer from shared memory to register.
  int64_t n_regs_buffer = -1;
  const int n_buffers = (int)buffers.size();
  for (int i = 0; i < n_buffers; i++) {
    auto current_tv = buffers[i];
    auto [buffer_size_regs, buffer_size_smem] =
        required_size_regs_smem_map.at(current_tv);
    buffer_params.regs_buffer_size += buffer_size_regs;
    buffer_params.smem_buffer_size -= buffer_size_smem;

    // The first-i buffers to are moved from shared memory to register
    // If both the register buffer size and shared memory buffer size are within
    // the allowable limit, we found a good configuration.
    if (buffer_params.regs_buffer_size <= available_regs &&
        buffer_params.smem_buffer_size <= available_smem) {
      n_regs_buffer = i + 1;
      break;
    }
    // Register buffer size exceeds the limit, can't move more to registers.
    // Break the loop.
    if (buffer_params.regs_buffer_size > available_regs) {
      break;
    }
  }

  // n_regs_buffer > 0 indicats a good configuration is found.
  // The first n_regs_buffer buffers are stored in registers and last [n_buffers
  // - n_regs_buffer] are stored in shared memory.
  if (n_regs_buffer > 0) {
    buffer_params.has_enough_regs_and_smem = true;
    auto n_smem_buffer = n_buffers - n_regs_buffer;
    buffer_params.smem_persistent_buffers.reserve(n_smem_buffer);
    for (int i = 0; i < n_smem_buffer; i++) {
      buffer_params.smem_persistent_buffers.emplace_back(
          buffers[n_buffers - 1 - i]);
    }
  } else {
    buffer_params.has_enough_regs_and_smem = false;
  }
  return buffer_params;
}

} // namespace inner_outer_utils
} // namespace nvfuser
