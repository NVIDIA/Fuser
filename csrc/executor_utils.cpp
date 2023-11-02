// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/cuda/jit_utils.h>

#include <c10/util/irange.h>

#include <contiguity.h>
#include <debug.h>
#include <driver_api.h>
#include <executor_utils.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel_db/kernel_db.h>
#include <options.h>
#include <tensor_metadata.h>
#include <torch/csrc/jit/resource_guard.h>
#include <utils.h>

#include <cuda_occupancy.h>
#include <nvfuser_resources/array.h>
#include <nvfuser_resources/basic_type_traits.h>
#include <nvfuser_resources/bf16_support.h>
#include <nvfuser_resources/bit.h>
#include <nvfuser_resources/block_reduction.h>
#include <nvfuser_resources/block_sync_atomic.h>
#include <nvfuser_resources/block_sync_default.h>
#include <nvfuser_resources/block_welford_outer.h>
#include <nvfuser_resources/broadcast.h>
#include <nvfuser_resources/complex_number.h>
#include <nvfuser_resources/fp16_support.h>
#include <nvfuser_resources/fused_reduction.h>
#include <nvfuser_resources/fused_welford_helper.h>
#include <nvfuser_resources/fused_welford_impl.h>
#include <nvfuser_resources/fused_welford_impl_outer.h>
#include <nvfuser_resources/grid_broadcast.h>
#include <nvfuser_resources/grid_reduction.h>
#include <nvfuser_resources/grid_sync.h>
#include <nvfuser_resources/helpers.h>
#include <nvfuser_resources/index_utils.h>
#include <nvfuser_resources/mbarrier.h>
#include <nvfuser_resources/memory.h>
#include <nvfuser_resources/random_numbers.h>
#include <nvfuser_resources/tensor.h>
#include <nvfuser_resources/tensorcore.h>
#include <nvfuser_resources/tuple.h>
#include <nvfuser_resources/type_traits.h>
#include <nvfuser_resources/warp.h>
#include <nvfuser_resources/welford.h>

#include <cstdlib>
#include <fstream>
#include <variant>

#include <nvrtc.h>

namespace nvfuser {
namespace executor_utils {

std::string kernelPreamble() {
  std::stringstream ss;
  ss << nvfuser_resources::basic_type_traits_cu;
  ss << nvfuser_resources::bit_cu;
  ss << nvfuser_resources::complex_number_cu;

  ss << nvfuser_resources::fp16_support_cu;
  ss << nvfuser_resources::bf16_support_cu;

  // Base classes and helpers
  ss << nvfuser_resources::type_traits_cu;
  ss << nvfuser_resources::array_cu;
  ss << nvfuser_resources::tensor_cu;
  ss << nvfuser_resources::random_numbers_cu;
  ss << nvfuser_resources::helpers_cu;
  ss << nvfuser_resources::index_utils_cu;
  ss << nvfuser_resources::tuple_cu;

  // Synchronization classes
  if (getNvFuserEnv("USE_BLOCK_SYNC_ATOMIC")) {
    ss << nvfuser_resources::block_sync_atomic_cu;
  } else {
    ss << nvfuser_resources::block_sync_default_cu;
  }
  ss << nvfuser_resources::grid_sync_cu;
  ss << nvfuser_resources::mbarrier_cu;

  // Communication classes
  ss << nvfuser_resources::block_reduction_cu;
  ss << nvfuser_resources::grid_reduction_cu;
  ss << nvfuser_resources::grid_broadcast_cu;
  ss << nvfuser_resources::broadcast_cu;
  ss << nvfuser_resources::welford_cu;
  ss << nvfuser_resources::warp_cu;
  ss << nvfuser_resources::tensorcore_cu;
  ss << nvfuser_resources::memory_cu;
  ss << nvfuser_resources::fused_welford_helper_cu;
  ss << nvfuser_resources::fused_reduction_cu;
  ss << nvfuser_resources::fused_welford_impl_cu;
  ss << nvfuser_resources::block_welford_outer_cu;
  ss << nvfuser_resources::fused_welford_impl_outer_cu;

  return ss.str();
}

namespace {

// Query the target GPU version number NVRTC compiles CUDA kernels for
void queryTargetGPUVersion(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor,
    bool& compile_to_sass) {
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;
  NVFUSER_NVRTC_SAFE_CALL(
      nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  NVF_CHECK(
      nvrtc_version.first >= 6,
      "NVRTC versions less than 6 are not supported. Is: ",
      nvrtc_version.first);

  // Version supported by device
  // Usually any lower version works too but is less efficient
  const CudaVersion dev_version = CudaVersion(prop->major, prop->minor);
  // Maximum version supported by the driver, cap dev_version to this
  CudaVersion max_dev_version;
  if (nvrtc_version.first <= 7) { // 7 supports 2-5.x
    max_dev_version = CudaVersion(5, 0);
  } else if (nvrtc_version.first <= 8) { // 8 supports 2-6.x
    max_dev_version = CudaVersion(6, 0);
  } else if (nvrtc_version.first <= 9) { // 9 supports 3-7.2
    max_dev_version = CudaVersion(7, 2);
  } else if (nvrtc_version.first <= 10) { // 10 supports 3-7.5
    max_dev_version = CudaVersion(7, 5);
  } else if (nvrtc_version == CudaVersion(11, 0)) { // 11.0 supports 3-8.0
    max_dev_version = CudaVersion(8, 0);
  } else if (nvrtc_version.first == 11 && nvrtc_version.second < 8) {
    max_dev_version = CudaVersion(8, 6);
  } else {
    // If the driver version is unknown (i.e. newer than this code)
    // assume the driver supports this device
    max_dev_version = dev_version;
  }
  if (dev_version > max_dev_version) {
    major = max_dev_version.first;
    minor = max_dev_version.second;
    // if we are clamping major/minor, sass is not compatible
    compile_to_sass = false;
  } else {
    major = dev_version.first;
    minor = dev_version.second;
    compile_to_sass = true;
  }
}

// Return true if all the tensors have the same stride, assumes all tensors are
// contiguous
bool checkSameStride(const std::vector<c10::IValue>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }
  for (const auto idx : c10::irange(tensors.size() - 1)) {
    auto current = tensors[idx];
    auto next = tensors[idx + 1];
    if (!current.isTensor() || !next.isTensor()) {
      return false;
    }

    const auto& current_tensor = current.toTensor();
    const auto& next_tensor = next.toTensor();
    if (current_tensor.ndimension() != next_tensor.ndimension()) {
      return false;
    }

    for (const auto i : c10::irange(current_tensor.ndimension())) {
      if (current_tensor.stride(i) != next_tensor.stride(i)) {
        return false;
      }
    }
  }
  return true;
}

// Return true if all the tensors are contiguous and have the same striding
bool checkSameContiguity(const std::vector<c10::IValue>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }

  auto reference = tensors.front();
  if (!reference.isTensor()) {
    return false;
  }

  // Determine if the reference tensor is contiguous
  const auto& reference_tensor = reference.toTensor();
  int64_t expected_stride = 1;
  for (const auto i : c10::irange(1, reference_tensor.ndimension() + 1)) {
    int64_t ind = reference_tensor.ndimension() - i;
    if (reference_tensor.size(ind) == 1) {
      continue;
    }
    if (reference_tensor.stride(ind) != expected_stride) {
      return false;
    }
    expected_stride *= reference_tensor.size(ind);
  }

  // Check if all the tensors have the same contiguity
  return checkSameStride(tensors);
}

bool checkValidMisalignedTensors(
    const std::unordered_set<TensorView*>& inp_tv,
    const std::unordered_set<TensorView*>& out_tv,
    const std::vector<c10::IValue>& inp_tensors,
    const std::vector<c10::IValue>& out_tensors) {
  if (out_tv.empty()) {
    // Only check input tensors
    return checkSameStride(inp_tensors);
  } else if (!out_tv.empty() && out_tensors.empty()) {
    // out_tensors is empty unless outputs are given to runFusion.
    // Assume out tensors are contiguous
    return checkSameContiguity(inp_tensors);
  } else {
    // Only check input and output tensors
    std::vector<c10::IValue> tensors;
    tensors.insert(tensors.end(), inp_tensors.begin(), inp_tensors.end());
    tensors.insert(tensors.end(), out_tensors.begin(), out_tensors.end());
    return checkSameStride(tensors);
  }
}

} // namespace

namespace {

// Finds a fusion input or output tensor, this function is used to grab tensors
// to validate the strides of the tensors for vectorization.
//
// Returns a pair consisting of a flag indicating if it's a fusion input (else
// is output) and an integer position within in the input or output tensor list.
std::vector<std::pair<bool, int>> getVectorizedFusionInputOutput(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    Fusion* fusion) {
  std::vector<std::pair<bool, int>> input_output;

  // When the producer is a fusion input, only return the producer
  // (vectorization validation assumes consumer of input is vectorizable).
  // Similarly, when the consumer is a fusion output, only return the consumer
  // (vectorization validation assumes producer of output is vectorizable). If
  // producer is input and consumer is output, return both.

  if (producer_tv->isFusionInput()) {
    auto producer_it = std::find(
        fusion->inputs().begin(), fusion->inputs().end(), producer_tv);
    NVF_ERROR(
        producer_it != fusion->inputs().end(),
        "Could not find ",
        producer_tv,
        " in fusion inputs.");
    auto pos = std::distance(fusion->inputs().begin(), producer_it);
    input_output.push_back(
        std::make_pair<bool, int>(true, static_cast<int>(pos)));
  }

  if (consumer_tv->isFusionOutput()) {
    auto consumer_it = std::find(
        fusion->outputs().begin(), fusion->outputs().end(), consumer_tv);
    NVF_ERROR(
        consumer_it != fusion->outputs().end(),
        "Could not find ",
        consumer_tv,
        " in fusion outputs.");
    auto pos = std::distance(fusion->outputs().begin(), consumer_it);
    input_output.push_back(
        std::make_pair<bool, int>(false, static_cast<int>(pos)));
  }

  return input_output;
}

//! Returns the information of vectorized input/output tensors
//! in the given fusion.
std::unique_ptr<caching::VectorizedTensorInfo> getVectorizedTensorValidationInfo(
    kir::Kernel* kernel) {
  auto vectorized_tensor_info_ptr =
      std::make_unique<caching::VectorizedTensorInfo>();

  for (const auto& vector_info : kernel->summary().vectorized_set_info) {
    auto consumer_tv = vector_info.consumer_tv;
    auto producer_tv = vector_info.producer_tv;

    auto vector_dim = vector_info.vectorized_leaf_id;
    const auto is_aligned =
        vector_dim->getParallelType() == ParallelType::Vectorize;

    // Find fusion inputs and outputs that are used with misaligned
    // vectorization.
    if (!is_aligned) {
      NVF_ERROR(
          producer_tv->isFusionInput() || consumer_tv->isFusionOutput(),
          "MisalignedVectorize is assumed to be used with either input or output tensor");
      if (consumer_tv->getMemoryType() == MemoryType::Global &&
          producer_tv->getMemoryType() == MemoryType::Local) {
        vectorized_tensor_info_ptr->global_out_misaligned_tv.insert(
            consumer_tv);
      } else if (
          producer_tv->getMemoryType() == MemoryType::Global &&
          consumer_tv->getMemoryType() == MemoryType::Local) {
        vectorized_tensor_info_ptr->global_inp_misaligned_tv.insert(
            producer_tv);
      } else {
        NVF_ERROR(
            false,
            "Unsupported memory configuration for misaligned vectorization.");
      }
    }

    // Collect information on corresponding fusion input and output
    // tensors to verify strides.
    auto inp_or_out_info =
        getVectorizedFusionInputOutput(producer_tv, consumer_tv, kernel);

    // If both producer and consumer are contig and intermediate,
    // nothing to validate with respect to strides.
    if (inp_or_out_info.empty()) {
      continue;
    }

    // Misaligned vectorize only allows from input to local or local
    // to output
    if (!is_aligned) {
      NVF_ERROR(inp_or_out_info.size() == 1);
    }

    for (const auto& inp_or_out : inp_or_out_info) {
      const bool is_input = inp_or_out.first;
      const int pos = inp_or_out.second;

      if (is_aligned) {
        auto& pos_list = is_input
            ? vectorized_tensor_info_ptr->aligned_vectorized_inp_tensor_pos
            : vectorized_tensor_info_ptr->aligned_vectorized_out_tensor_pos;
        pos_list.push_back(pos);
      } else {
        auto& map = is_input
            ? vectorized_tensor_info_ptr->inp_misaligned_tensors_pos
            : vectorized_tensor_info_ptr->out_misaligned_tensors_pos;
        map.emplace_back(pos);
      }
    }
  }

  return vectorized_tensor_info_ptr;
}

// Make sure the root domain(s) comprising the vectorized leaf domain
// have the (merged) extent that is divisible by the vectorization
// word size.
void validateAlignedVectorizeExtents(
    const VectorizedSetInfo& info,
    ExpressionEvaluator& expr_eval) {
  NVF_ERROR(
      !info.contig_alloc_ids.empty(),
      "No root ID found for vectorization with ",
      info.consumer_tv->toString(),
      " and ",
      info.producer_tv->toString());

  // TODO: Rewrite validation of the vectorized dimension
  // int64_t vectorized_merged_domain_extent = 1;
  for (auto id : info.contig_alloc_ids) {
    auto extent_val = expr_eval.evaluate(id->extent());
    NVF_ERROR(
        extent_val.hasValue(),
        "Error vectorizing, ",
        info.consumer_tv->toString(),
        " as the extent of a vectorized root domain, ",
        id->toString(),
        ", is unknown.");
    // TODO: Rewrite validation of the vectorized dimension
    // vectorized_merged_domain_extent *= extent_val->as<int64_t>();
  }

  // TODO: Rewrite validation of the vectorized dimension, we can't just used a
  // single merged extent because we could be splitting a dimension then merging
  // it in order to the right of it. Contig merged index simply isn't exactly
  // what we need to validate for vectorization, and we're relying on better
  // vectorization support than that would offer. This validation needs to be
  // rewritten based on updated indexing logic that traverses loop->rfactor
  // domains and tracks partial mappings like scheduler/vectorize_helper.cpp
  //
  // NVF_ERROR(
  //     vectorized_merged_domain_extent % info.word_size == 0,
  //     "Error vectorizing, ",
  //     info.consumer_tv->toString(),
  //     " as the extent of the indexed domain, ",
  //     vectorized_merged_domain_extent,
  //     ", is not divisible by vector word size ",
  //     info.word_size);
}

namespace {

// Return offsets of the first points accessed as well as sliced root
// domains. Currently only non-zero when tensor is sliced.
std::pair<std::unordered_set<size_t>, std::unordered_set<IterDomain*>>
getTensorOffsets(
    TensorView* tv,
    c10::IntArrayRef logical_strides,
    ExpressionEvaluator& eval) {
  if (!tv->isFusionInput()) {
    return {{0}, {}};
  }

  std::unordered_set<size_t> offsets;
  std::unordered_set<IterDomain*> sliced_domains;

  const auto root_ids = TensorDomain::noReductions(tv->getMaybeRFactorDomain());

  for (auto use : tv->uses()) {
    auto slice = dynamic_cast<SliceOp*>(use);

    if (slice == nullptr) {
      offsets.insert(0);
      continue;
    }

    NVF_ERROR(logical_strides.size() == root_ids.size());
    const auto slice_info = slice->getRanges();

    size_t offset = 0;
    for (const auto i : c10::irange(root_ids.size())) {
      auto slice_start_eval = eval.evaluate(slice_info.at(i).start);
      NVF_ERROR(slice_start_eval.hasValue());
      auto slice_stop_eval = eval.evaluate(slice_info.at(i).stop);
      NVF_ERROR(slice_stop_eval.hasValue());
      auto extent_eval =
          eval.evaluate(root_ids.at(i)->getMaybeExpandedExtent());
      NVF_ERROR(extent_eval.hasValue());

      offset += static_cast<size_t>(
          slice_start_eval.as<int64_t>() * logical_strides.at(i));

      // Keep track of the root domain unless this slice is
      // effectively no-op
      if (slice_start_eval.as<int64_t>() != 0 ||
          slice_stop_eval.as<int64_t>() != extent_eval.as<int64_t>()) {
        sliced_domains.insert(root_ids.at(i));
      }
    }

    offsets.insert(offset);
  }

  return std::make_pair(offsets, sliced_domains);
}

} // namespace

void validateAlignedVectorizedFusionInputOutput(
    const at::Tensor& aten_tensor,
    int word_size,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  eval.bind(tv, aten_tensor);
  const auto& metadata = eval.evaluate(IrBuilder::metadataExpr(tv));

  const auto [offsets, sliced_domains] =
      getTensorOffsets(tv, metadata->*&TensorMetaData::logical_stride, eval);
  const bool is_sliced = !sliced_domains.empty();

  const auto& domain_to_validate =
      is_sliced ? tv->getMaybeRFactorDomain() : tv->getMaybeAllocationDomain();

  std::vector<int64_t> no_reduction_to_full;
  for (int64_t i : c10::irange((int64_t)domain_to_validate.size())) {
    auto alloc_id = domain_to_validate.at(i);
    if (!alloc_id->isReduction()) {
      no_reduction_to_full.emplace_back(i);
    }
  }

  const auto& sizes = is_sliced ? metadata->*&TensorMetaData::logical_size
                                : metadata->*&TensorMetaData::alloc_size;
  const auto& strides = is_sliced ? metadata->*&TensorMetaData::logical_stride
                                  : metadata->*&TensorMetaData::alloc_stride;
  NVF_ERROR(sizes.size() == no_reduction_to_full.size());
  NVF_ERROR(strides.size() == no_reduction_to_full.size());

  for (auto offset : offsets) {
    NVF_ERROR(
        (reinterpret_cast<size_t>(aten_tensor.data_ptr()) +
         offset * aten_tensor.dtype().itemsize()) %
                (word_size * aten_tensor.dtype().itemsize()) ==
            0,
        "Vectorization of ",
        tv->toString(),
        " not possible as the memory address is not aligned. ",
        "Address: ",
        aten_tensor.data_ptr(),
        ", offset: ",
        offset,
        ", vector word size: ",
        word_size,
        ", data type: ",
        aten_tensor.dtype());
  }

  // Traverse strides from the right-most domains. The rightmost
  // domain must have stride 1.
  int64_t cur_contig_stride = 1;
  bool still_rightmost = true;
  bool non_contig_due_to_slice = false;
  for (int64_t i = (int64_t)sizes.size() - 1; i >= 0; --i) {
    const auto size = sizes.at(i);
    const auto stride = strides.at(i);
    auto id = domain_to_validate.at(no_reduction_to_full.at(i));
    const auto is_expanded_broadcasting =
        id->isBroadcast() && id->hasExpandedExtent();

    if (is_expanded_broadcasting) {
      NVF_ERROR(
          stride == 0,
          "Dimension ",
          i,
          " should be an expanded broadcasting, but it does not have stride zero.");
    }

    // If this domain is contiguous or size == 1, then not necessary to check
    // the stride. Otherwise, stride must be 1 if it's rightmost or
    // divisible by word_size

    bool is_contiguous =
        stride == cur_contig_stride && !non_contig_due_to_slice;

    NVF_ERROR(
        is_contiguous || size == 1 || is_expanded_broadcasting ||
            (still_rightmost && stride == 1) ||
            (!still_rightmost && stride % word_size == 0),
        "Vectorization of ",
        tv->toString(),
        " with word size ",
        word_size,
        " not possible due to invalid stride.",
        " Domain: ",
        tv->axis(i)->toString(),
        ", stride: ",
        stride,
        ", cur_contig_stride: ",
        cur_contig_stride,
        ", non contig due to slice: ",
        non_contig_due_to_slice);
    // If the domain is size-1, the next domain is still considered
    // rightmost.
    still_rightmost =
        still_rightmost && (size == 1 || is_expanded_broadcasting);
    // We do not update cur_contig_stride for size==1 dimensions,
    // since we have specialized vectorization stride check for
    // them. Same for non_contig_due_to_slice.
    if (size != 1 && !is_expanded_broadcasting) {
      cur_contig_stride = stride * size;
      // Note that when a domain is sliced, the next outer domain is
      // no longer contiguous.
      non_contig_due_to_slice = sliced_domains.count(
          domain_to_validate.at(no_reduction_to_full.at(i)));
    }
  }
}

void validateAlignedVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval) {
  auto tensor_vectorization_validation_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::VectorizedTensorValidation>(
          data_cache, [kernel]() {
            return executor_utils::getVectorizedTensorValidationInfo(kernel);
          });

  // Verify extents of aligned vectorized tensors
  for (const auto& vec_info : kernel->summary().vectorized_set_info) {
    if (vec_info.vectorized_leaf_id->getParallelType() ==
        ParallelType::Vectorize) {
      validateAlignedVectorizeExtents(vec_info, expr_eval);
    }
  }

  // Validate input and output tensors with aligend
  // vectorization.
  for (auto pos : tensor_vectorization_validation_entry.get()
                      .aligned_vectorized_inp_tensor_pos) {
    auto tv = kernel->inputs().at(pos)->as<TensorView>();
    auto word_size = kernel->summary().vectorized_accesses.at(tv);
    NVF_ERROR(args[pos]->is<at::Tensor>(), "alias io only supports tensor");
    validateAlignedVectorizedFusionInputOutput(
        args[pos]->as<at::Tensor>(), word_size, tv, expr_eval);
  }
  if (!outputs.empty()) {
    for (auto pos : tensor_vectorization_validation_entry.get()
                        .aligned_vectorized_out_tensor_pos) {
      auto tv = kernel->outputs().at(pos)->as<TensorView>();
      auto word_size = kernel->summary().vectorized_accesses.at(tv);
      validateAlignedVectorizedFusionInputOutput(
          outputs[pos], word_size, tv, expr_eval);
    }
  }
}

// Misaligned vectorization check. Currently misaligned vectorization is limited
// to global-register and register-global load/store patterns. However, this
// could be improved to include shared memory.
void validateMisalignedVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval) {
  auto tensor_vectorization_validation_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::VectorizedTensorValidation>(
          data_cache, [kernel]() {
            return executor_utils::getVectorizedTensorValidationInfo(kernel);
          });

  std::vector<c10::IValue> inp_misaligned_tensors;
  std::vector<c10::IValue> out_misaligned_tensors;

  const auto& inp_misaligned_tensors_pos =
      tensor_vectorization_validation_entry.get().inp_misaligned_tensors_pos;
  inp_misaligned_tensors.reserve(inp_misaligned_tensors_pos.size());
  std::transform(
      inp_misaligned_tensors_pos.begin(),
      inp_misaligned_tensors_pos.end(),
      std::back_inserter(inp_misaligned_tensors),
      [&args](int idx) {
        NVF_ERROR(args[idx]->is<at::Tensor>(), "alias io only supports tensor");
        return args[idx]->as<at::Tensor>();
      });

  const auto& out_misaligned_tensors_pos =
      tensor_vectorization_validation_entry.get().out_misaligned_tensors_pos;
  if (!outputs.empty()) {
    out_misaligned_tensors.reserve(out_misaligned_tensors_pos.size());
    std::transform(
        out_misaligned_tensors_pos.begin(),
        out_misaligned_tensors_pos.end(),
        std::back_inserter(out_misaligned_tensors),
        [&outputs](int idx) { return outputs[idx]; });
  }
  // If input stride is non-contiguous + no outputs, return false
  NVF_ERROR(
      checkValidMisalignedTensors(
          tensor_vectorization_validation_entry.get().global_inp_misaligned_tv,
          tensor_vectorization_validation_entry.get().global_out_misaligned_tv,
          inp_misaligned_tensors,
          out_misaligned_tensors),
      "All global tensors must have the same stride for misaligned vectorization.");
}

// Check if there's any split that is non-divisible and vectorized. If
// found, Vectorize is illegal.
void validateVectorizedSplits(
    kir::Kernel* kernel,
    ExpressionEvaluator& expr_eval) {
  for (const auto& extent_factor : kernel->summary().splits_to_validate) {
    auto input_extent = expr_eval.evaluate(extent_factor.first);
    auto split_factor = expr_eval.evaluate(extent_factor.second);
    auto divisible = (input_extent % split_factor == 0);
    NVF_ERROR(
        divisible,
        "Non-divisible split with vectorization is detected. ",
        "Extent: ",
        input_extent,
        ". Factor: ",
        split_factor);
  }
}

} // namespace

void validateVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("FusionExecutor::validateVectorizedTensors");

  validateAlignedVectorizedTensors(
      kernel, args, outputs, data_cache, expr_eval);

  validateMisalignedVectorizedTensors(
      kernel, args, outputs, data_cache, expr_eval);

  validateVectorizedSplits(kernel, expr_eval);
}

ExpressionEvaluator bindInputs(
    const KernelArgumentHolder& args,
    Fusion* kernel) {
  FUSER_PERF_SCOPE("executor_utils::bindInputs");

  // args may contains more than just inputs, but inputs are always at the
  // beginning.
  NVF_ERROR(
      kernel->inputs().size() <= args.size(),
      "KernelArgumentHolder contains less argument than kernel's input.");

  ExpressionEvaluator expr_eval;
  const auto& inputs = kernel->inputs();
  for (const auto i : c10::irange(inputs.size())) {
    // NOTE: we bind all inputs here, including at::Tensors. This means that
    // expr_eval will create a PolymorphicValue containing *args[i], which means
    // that at::Tensor's lifetime will be at least as long as that of expr_eval.
    expr_eval.bind(inputs[i], *args[i], true);
  }

  return expr_eval;
}

namespace {

std::vector<char> compileNvrtcProgramToPtx(const nvrtcProgram& program) {
  size_t size = 0;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &size));
  std::vector<char> code(size);
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTX(program, code.data()));
  return code;
}

std::vector<char> compileNvrtcProgramToCubin(const nvrtcProgram& program) {
#if CUDA_VERSION < 11010
  NVF_ERROR(false, "SASS not supported in CUDA versions older than 11.1");
#endif

  size_t size = 0;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetCUBINSize(program, &size));
  std::vector<char> code(size);
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetCUBIN(program, code.data()));
  return code;
}

// Returns the name of the dumped file.
std::string dumpCompiledCodeToFile(
    const std::vector<char>& code,
    const std::string& id,
    const std::string& suffix) {
  std::stringstream file_name;
  file_name << "__tmp_kernel_" << id << suffix;
  debug() << "PRINTING: " << file_name.str() << std::endl;
  std::ofstream out(file_name.str());
  NVF_ERROR(out.is_open());
  out.write(code.data(), (std::streamsize)code.size());
  out.close();
  return file_name.str();
}

// Get the max register count passed as -maxrregcount ptxas
// option. The count is determined based on block sizes, an optional
// heuristic and an environment variable.
std::optional<int64_t> getMaxRegCount(
    std::optional<int64_t> opt_block_size,
    const int64_t max_register_heuristic) {
  // The maximum possible count allowed by ptxas is 255
  constexpr int64_t max_register_limit = 255;

  // Temporary set the max register count to be larger than the
  // limit.
  int64_t max_register = max_register_limit + 1;

  // If the block size is known, set the maximum that at least allows
  // one block to be resident on an SM
  if (opt_block_size.has_value() && opt_block_size.value() > 0) {
    constexpr int64_t block_per_sm = 1;
    max_register = std::min(
        max_register_limit,
        getRegPerThreadGivenThreadsPerSM(
            opt_block_size.value() * block_per_sm));
  }

  // If a heuristic value is given, i.e., max_register_heuristic is
  // less than the limit, use that value if it's smaller than the
  // block-size based count
  if (max_register_heuristic < max_register_limit) {
    max_register = std::min(max_register, max_register_heuristic);
  }

  // Overwrite the count by the environment variable
  if (auto env_count = getNvFuserEnv("MAX_REG_COUNT")) {
    auto env_max_reg_count = std::atoi(env_count);
    NVF_CHECK(
        env_max_reg_count > 0 && env_max_reg_count <= max_register_limit,
        "Invalid max register count specified by NVFUSER_MAX_REG_COUNT: ",
        env_max_reg_count);
    max_register = env_max_reg_count;
  }

  // At this point, max_register should be <= max_register_limit if set
  if (max_register <= max_register_limit) {
    return max_register;
  } else {
    return std::optional<int64_t>();
  }
}

//! Utility class to invoke nvrtcCompileProgram. Mainly for setting up
//! the c-str options.
class NvrtcCompileDriver {
 public:
  void setOption(const std::string& opt) {
    options_.push_back(opt);
  }

  const std::vector<std::string>& options() const {
    return options_;
  }

  std::string invoke(nvrtcProgram program, const std::string& src) const {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::CompileProgram");
    auto opts = getOptions();
    auto result = nvrtcCompileProgram(
        program, static_cast<int>(opts.size()), opts.data());
    size_t logsize = 0;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &logsize));
    // The log size, as returned by 'nvrtcGetProgramLogSize', appears larger
    // than its actual size by 2. This discrepancy was noticed in NVRTC
    // version 12.1. The log returned from 'nvrtcGetProgramLog' terminates with
    // a NULL character, ensuring it's safe to use 'std::vector<char>' for
    // storage before converting it to 'std::string'.
    std::vector<char> log_backing_buf(logsize);
    char* log_buf = log_backing_buf.data();
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log_buf));
    if (result != NVRTC_SUCCESS) {
      NVF_ERROR(false, src, "\nCUDA NVRTC compile error: ", log_buf);
    }
    if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
      debug() << log_buf << std::endl;
    }
    return std::string(log_buf);
  }

 private:
  // Get options that can be passed to nvrtcCompileProgram
  std::vector<const char*> getOptions() const {
    std::vector<const char*> opts(options_.size());
    for (const auto i : c10::irange(options_.size())) {
      opts.at(i) = options_.at(i).c_str();
    }
    return opts;
  }

 private:
  std::vector<std::string> options_;
};

//! Utility class to invoke cuModuleLoadDataEx. Similar to
//! NvrtcCompileDriver, the main task is to set up the option lists
//! of type void**
class CuModuleLoadDataDriver {
 public:
  //! Valid option type is either int or char*
  using OptionType = std::variant<int, char*>;

  template <typename OptionValType>
  void setOption(CUjit_option key, OptionValType val) {
    options_.push_back(key);
    option_vals_.push_back(val);
  }

  //! Enable logging of cuModuleLoadData
  void enableLogging() {
    logging_enabled_ = true;
    log_.resize(kLogSize);
  }

  const std::string& log() const {
    NVF_ERROR(logging_enabled_, "Logging not enabled");
    return log_;
  }

  //! Invoke cuModuleLoadDataEx with ptx or cubin. Dump logging output
  //! if enabled
  std::string invoke(CUmodule& module, const void* image) {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::LoadPTX");

    auto [opts, opt_vals] = getOptions();

    NVFUSER_CUDA_SAFE_CALL(cuModuleLoadDataEx(
        &module, image, opts.size(), opts.data(), opt_vals.data()));

    if (logging_enabled_) {
      debug() << log_ << std::endl;
    }

    return log_;
  }

 private:
  // Get options that can be passed to cuModuleLoadDataEx
  std::pair<std::vector<CUjit_option>, std::vector<void*>> getOptions() {
    auto opts = options_;
    auto opt_vals = option_vals_;

    // Append options for saving log message to log_
    if (logging_enabled_) {
      opts.push_back(CU_JIT_LOG_VERBOSE);
      opt_vals.emplace_back(1);

      opts.push_back(CU_JIT_INFO_LOG_BUFFER);
      opt_vals.emplace_back(log_.data());

      opts.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
      opt_vals.emplace_back(kLogSize);
    }

    // Convert the options to void**. This is ugly, but that's how
    // cuModuleLoadDataEx works. See initCUDA in the
    // matrixMulDynlinkJIT sample
    // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp#L169-L204.
    std::vector<void*> opt_val_voidp(opt_vals.size());
    for (const auto i : c10::irange(opt_vals.size())) {
      auto opt_val = opt_vals.at(i);
      if (std::holds_alternative<int>(opt_val)) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        opt_val_voidp.at(i) = (void*)(int64_t)std::get<int>(opt_val);
      } else if (std::holds_alternative<char*>(opt_val)) {
        opt_val_voidp.at(i) = std::get<char*>(opt_val);
      } else {
        NVF_ERROR(false, "Invalid option");
      }
    }

    return std::make_pair(opts, opt_val_voidp);
  }

 private:
  static constexpr int kLogSize = 8196;
  //! cuModuleLoadDataEx options
  std::vector<CUjit_option> options_;
  //! Option parameters
  std::vector<OptionType> option_vals_;
  //! Save log to log_ if true
  bool logging_enabled_ = false;
  std::string log_;
};

// Fill options for nvrtcCompileProgram and cuModuleLoadDataEx
void fillCompileOptions(
    NvrtcCompileDriver& nvrtc_compile_driver,
    CuModuleLoadDataDriver& module_load_driver,
    bool compile_to_sass,
    int major,
    int minor,
    const CompileParams& compile_params,
    std::optional<int64_t> opt_block_size) {
  nvrtc_compile_driver.setOption("--std=c++17");

  // CUDA 11.1 allows going directly to SASS (sm_) instead of PTX (compute_)
  // which gives better backwards compatibility to work on older driver,
  // (since older driver doesn't necessarily recognize PTX emitted by new
  // toolkit);
  // Meanwhile, for forward compatibility (future device with
  // `unsupported_arch==True`), since SASS are not necessarily compatible,
  // we fallback to PTX instead.
  const std::string compute = std::string("--gpu-architecture=") +
      (compile_to_sass ? "sm_" : "compute_") + std::to_string(major) +
      std::to_string(minor);
  nvrtc_compile_driver.setOption(compute);

  nvrtc_compile_driver.setOption("-default-device");

  if (isOptionDisabled(DisableOption::Fma)) {
    nvrtc_compile_driver.setOption("--fmad=false");
  } else {
    nvrtc_compile_driver.setOption("--fmad=true");
  }

  // Add line info to generated kernels
  if (isDebugDumpEnabled(DebugDumpOption::DebugInfo)) {
    nvrtc_compile_driver.setOption("-lineinfo");
  }

#ifdef NDEBUG
  // Avoid excessive register usage from assertion
  nvrtc_compile_driver.setOption("-DNDEBUG");
#endif

  if (isOptionEnabled(EnableOption::KernelProfile)) {
    nvrtc_compile_driver.setOption("-DNVFUSER_PROFILE_KERNEL");
  }
  if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose) ||
      isOptionEnabled(EnableOption::WarnRegisterSpill) ||
      compile_params.enable_ptxas_verbose) {
    // show register usage in compilation log
    if (compile_to_sass) {
      nvrtc_compile_driver.setOption("--ptxas-options");
      nvrtc_compile_driver.setOption("--verbose");
    } else {
      module_load_driver.enableLogging();
    }
  }

  const char* ptxas_opt_level = getNvFuserEnv("JIT_OPT_LEVEL");

  if (ptxas_opt_level) {
    int val = atoi(ptxas_opt_level);
    if (val <= 4 && val >= 0) {
      if (val < 4) {
        TORCH_WARN(
            "ptxas optimization level manually set as ",
            val,
            ", which could negatively affect performance. Try removing env variable NVFUSER_JIT_OPT_LEVEL for optimal performance.");
      }
      if (compile_to_sass) {
        nvrtc_compile_driver.setOption("--ptxas-options");
        nvrtc_compile_driver.setOption("-O" + std::to_string(val));
      } else {
        module_load_driver.setOption(CU_JIT_OPTIMIZATION_LEVEL, val);
      }
    } else {
      TORCH_WARN_ONCE(
          "acceptable range for NVFUSER_JIT_OPT_LEVEL is between 0 and 4, but received ",
          val,
          ", ignoring the option");
    }
  }

  const auto max_register =
      getMaxRegCount(opt_block_size, compile_params.maxrregcount);

  // If the max register count is set
  if (max_register.has_value()) {
    if (compile_to_sass) {
      nvrtc_compile_driver.setOption(
          "--maxrregcount=" + std::to_string(*max_register));
    } else {
      module_load_driver.setOption(CU_JIT_MAX_REGISTERS, (int)*max_register);
    }
  }
}

// Dump ptxas output if register spill is detected
void warnRegisterSpill(const std::string& compile_log) {
  auto getRegisterSpillInfo = [](const std::string& log, const char* subStr) {
    auto it_end =
        std::search(log.begin(), log.end(), subStr, subStr + strlen(subStr)) -
        1;
    auto it_beg = it_end - 1;
    while (!std::isspace(*(it_beg - 1))) {
      it_beg--;
    }
    std::string str(it_beg, it_end);
    return std::stoi(str);
  };

  const char* str_stack = "bytes stack frame";
  const char* str_store = "bytes spill stores";
  const char* str_load = "bytes spill loads";
  int stack_count = getRegisterSpillInfo(compile_log, str_stack);
  int store_count = getRegisterSpillInfo(compile_log, str_store);
  int load_count = getRegisterSpillInfo(compile_log, str_load);
  int allowed_spill = 0;
  if (isOptionEnabled(EnableOption::WarnRegisterSpill)) {
    auto optionArgs = getEnableOptionArguments(EnableOption::WarnRegisterSpill);
    if (!optionArgs.empty()) {
      try {
        allowed_spill = std::stoi(optionArgs[0]);
      } catch (const std::exception& e) {
        debug() << "skip invalid argument for WarnRegisterSpill, arg = "
                << optionArgs[0] << std::endl;
      }
    }
  }
  if (stack_count > allowed_spill || store_count > allowed_spill ||
      load_count > allowed_spill) {
    debug() << "WARNING: Register spill detected\n" << compile_log << std::endl;
  }
}

void createNvrtcProgram(
    nvrtcProgram& program,
    const std::string& id,
    const std::string& full_src_code) {
  std::stringstream ss;
  ss << "__tmp_kernel_" << id << ".cu";
  std::string name = ss.str();
  FUSER_PERF_SCOPE("executor_utils::NvrtcCreateProgram");
  NVFUSER_NVRTC_SAFE_CALL(nvrtcCreateProgram(
      &program, full_src_code.c_str(), name.c_str(), 0, nullptr, nullptr));
}

// Compile the given source code with the NVRTC compiler driver.
std::unique_ptr<CompiledKernel> compileSource(
    const std::string& full_src_code,
    const std::string& func_name,
    const std::string& id,
    const bool compile_to_sass,
    NvrtcCompileDriver& nvrtc_compile) {
  std::stringstream log;

  nvrtcProgram program; // NOLINT(cppcoreguidelines-init-variables)
  torch::jit::ResourceGuard holdProgram([&] {
    FUSER_PERF_SCOPE("executor_utils::NvrtcDestroyProgram");
    NVFUSER_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
  });

  createNvrtcProgram(program, id, full_src_code);

  NVFUSER_NVRTC_SAFE_CALL(nvrtcAddNameExpression(program, func_name.c_str()));
  log << nvrtc_compile.invoke(program, full_src_code) << std::endl;

  auto compiled_kernel = std::make_unique<CompiledKernel>();
  const char* lowered_kernel_name = nullptr;
  NVFUSER_NVRTC_SAFE_CALL(
      nvrtcGetLoweredName(program, func_name.c_str(), &lowered_kernel_name));
  compiled_kernel->kernel_name = lowered_kernel_name;
  compiled_kernel->compile_log = log.str();

  if (compile_to_sass) {
    compiled_kernel->cubin = compileNvrtcProgramToCubin(program);
    if (isDebugDumpEnabled(DebugDumpOption::Cubin)) {
      compiled_kernel->cubin_filename =
          dumpCompiledCodeToFile(compiled_kernel->cubin, id, ".cubin");
    }
  }

  if (!compile_to_sass || isDebugDumpEnabled(DebugDumpOption::Ptx)) {
    compiled_kernel->ptx = compileNvrtcProgramToPtx(program);
    if (isDebugDumpEnabled(DebugDumpOption::Ptx)) {
      compiled_kernel->ptx_filename =
          dumpCompiledCodeToFile(compiled_kernel->ptx, id, ".ptx");
    }
  }

  return compiled_kernel;
}

} // namespace

CompiledKernel::~CompiledKernel() {
  if (module != nullptr) {
    NVFUSER_CUDA_SAFE_CALL(cuModuleUnload(module));
  }
}

// Compile the source if no existing compiled binary is found in KernelDB
std::unique_ptr<CompiledKernel> getCompiledKernel(
    std::optional<std::reference_wrapper<const std::string>> kernel_code,
    const std::string& full_src_code,
    const std::string& func_name,
    const std::string& id,
    const CompileParams& compile_params,
    std::optional<int64_t> opt_block_size) {
  FUSER_PERF_SCOPE("executor_utils::NVRTC");

  at::cuda::jit::initializeCudaContext();

  // The above initialization works in some cases. However, it seems to
  // occasionally fail to initialize a primary context. Here we check for that
  // and if we detect that no context exists, we create one manually.
  int device = 0;
  cudaGetDevice(&device);
  if (!at::detail::getCUDAHooks().hasPrimaryContext((c10::DeviceIndex)device)) {
    // CUDA>=12 creates a context when cudaSetDevice is called. However, before
    // cu12, that context is not necessarily created. In that case, we create
    // one here implicitly. See https://github.com/NVIDIA/Fuser/issues/429
    cudaFree(nullptr);
  }

  const auto prop = at::cuda::getCurrentDeviceProperties();

  int major = 0, minor = 0;
  bool compile_to_sass = false;
  queryTargetGPUVersion(prop, major, minor, compile_to_sass);

#if CUDA_VERSION < 11010
  // compile to sass is not allowed prior to CUDA 11.1
  compile_to_sass = false;
#endif

  if (isOptionDisabled(DisableOption::CompileToSass)) {
    compile_to_sass = false;
  }

  NvrtcCompileDriver nvrtc_compile_driver;
  CuModuleLoadDataDriver module_load_driver;

  fillCompileOptions(
      nvrtc_compile_driver,
      module_load_driver,
      compile_to_sass,
      major,
      minor,
      compile_params,
      opt_block_size);

  std::stringstream log;

  if (compile_to_sass) {
    log << "\nCompile options: ";
    for (const auto& opt : nvrtc_compile_driver.options()) {
      log << opt << " ";
    }
    if (opt_block_size.has_value()) {
      log << " ; block size=" << opt_block_size.value() << "\n";
    }
  }

  auto compiled_kernel = std::make_unique<CompiledKernel>();
  const auto compile_args =
      toDelimitedString(nvrtc_compile_driver.options(), " ");

  auto& kernel_db = KernelDb::get();
  const auto use_kernel_db = kernel_db.enabled() && kernel_code.has_value();

  // If the Kernel Query fails, the Kernel is recompiled
  if (!(use_kernel_db &&
        kernel_db.query(
            kernel_code.value(),
            compile_args,
            compiled_kernel->kernel_name,
            (compile_to_sass ? compiled_kernel->cubin
                             : compiled_kernel->ptx)))) {
    compiled_kernel = compileSource(
        full_src_code, func_name, id, compile_to_sass, nvrtc_compile_driver);
    log << compiled_kernel->compile_log << std::endl;
    if (use_kernel_db) {
      auto result = kernel_db.write(
          kernel_code.value(),
          compile_args,
          compiled_kernel->kernel_name,
          (compile_to_sass ? compiled_kernel->cubin : compiled_kernel->ptx));
      if (!result) {
        TORCH_WARN(
            "kernel_db was unable to write kernel: ",
            compiled_kernel->kernel_name);
      }
    }
  }

  log << module_load_driver.invoke(
             compiled_kernel->module,
             (compile_to_sass ? compiled_kernel->cubin.data()
                              : compiled_kernel->ptx.data()))
      << std::endl;
  compiled_kernel->compile_log = log.str();
  compiled_kernel->compile_args = compile_args;

  if (isOptionEnabled(EnableOption::WarnRegisterSpill) ||
      compile_params.enable_ptxas_verbose) {
    warnRegisterSpill(compiled_kernel->compile_log);
  }

  NVFUSER_CUDA_SAFE_CALL(cuModuleGetFunction(
      &(compiled_kernel->function),
      compiled_kernel->module,
      compiled_kernel->kernel_name.c_str()));

  // Store block size used to generate compile arguments
  if (opt_block_size.has_value()) {
    compiled_kernel->block_size = opt_block_size.value();
  }

  return compiled_kernel;
}

std::unique_ptr<CompiledKernel> getCompiledKernel(
    const serde::CudaKernel* buffer,
    const CompileParams& compile_params) {
  FUSER_PERF_SCOPE("executor_utils::serde_NVRTC");

  NVF_ERROR(buffer != nullptr, "serde::CudaKernel is nullptr.");

  // Deserialize flatbuffer into CompiledKernel
  auto compiled_kernel = std::make_unique<CompiledKernel>();
  compiled_kernel->kernel_name = buffer->kernel_name()->str();
  compiled_kernel->compile_args = buffer->compile_args()->str();
  compiled_kernel->block_size = buffer->block_size();

  if (buffer->cubin() != nullptr) {
    compiled_kernel->cubin.reserve(buffer->cubin()->size());
    std::copy(
        buffer->cubin()->begin(),
        buffer->cubin()->end(),
        std::back_inserter(compiled_kernel->cubin));
    compiled_kernel->cubin_filename = buffer->cubin_filename()->str();
  }

  if (buffer->ptx() != nullptr) {
    compiled_kernel->ptx.reserve(buffer->ptx()->size());
    std::copy(
        buffer->ptx()->begin(),
        buffer->ptx()->end(),
        std::back_inserter(compiled_kernel->ptx));
    compiled_kernel->ptx_filename = buffer->ptx_filename()->str();
  }

  at::cuda::jit::initializeCudaContext();

  // The above initialization works in some cases. However, it seems to
  // occasionally fail to initialize a primary context. Here we check for that
  // and if we detect that no context exists, we create one manually.
  int device = 0;
  cudaGetDevice(&device);
  if (!at::detail::getCUDAHooks().hasPrimaryContext((c10::DeviceIndex)device)) {
    // CUDA>=12 creates a context when cudaSetDevice is called. However, before
    // cu12, that context is not necessarily created. In that case, we create
    // one here implicitly. See https://github.com/NVIDIA/Fuser/issues/429
    cudaFree(nullptr);
  }

  const auto prop = at::cuda::getCurrentDeviceProperties();

  // Generate compile args and compare against saved args in compiled_kernel
  NvrtcCompileDriver nvrtc_compile_driver;
  CuModuleLoadDataDriver module_load_driver;

  int major = 0, minor = 0;
  bool compile_to_sass = false;
  queryTargetGPUVersion(prop, major, minor, compile_to_sass);

  std::optional<int64_t> opt_block_size;
  if (compiled_kernel->block_size >= -1) {
    opt_block_size = compiled_kernel->block_size;
  }

  fillCompileOptions(
      nvrtc_compile_driver,
      module_load_driver,
      compile_to_sass,
      major,
      minor,
      compile_params,
      opt_block_size);

  const auto latest_compile_args =
      toDelimitedString(nvrtc_compile_driver.options(), " ");
  NVF_ERROR(
      latest_compile_args == compiled_kernel->compile_args,
      "The compile arguments for the serialized cuda kernel does not ",
      "match the latest generated compile args.\t",
      latest_compile_args,
      "\t",
      compiled_kernel->compile_args);

  NVF_ERROR(
      !compile_to_sass || !compiled_kernel->cubin.empty(),
      "Expected compiled cubin after deserializing CompiledKernel.");

  NVF_ERROR(
      compile_to_sass || !compiled_kernel->ptx.empty(),
      "Expected compiled ptx after deserializing CompiledKernel.");

  std::stringstream log;
  log << module_load_driver.invoke(
             compiled_kernel->module,
             (compile_to_sass ? compiled_kernel->cubin.data()
                              : compiled_kernel->ptx.data()))
      << std::endl;
  compiled_kernel->compile_log = log.str();

  NVFUSER_CUDA_SAFE_CALL(cuModuleGetFunction(
      &(compiled_kernel->function),
      compiled_kernel->module,
      compiled_kernel->kernel_name.c_str()));

  return compiled_kernel;
}

namespace caching {

//! CompileTimeInfo is the actual subclass of CompileTimeInfoBase that will
//!  be stored in the data cache. It owns a data_ state internally of the
//!  dataType defined within the entry class, which are listed in header file.
template <typename EntryClass>
class CompileTimeInfo : public CompileTimeInfoBase {
 public:
  CompileTimeInfo(std::unique_ptr<typename EntryClass::DataType> data)
      : CompileTimeInfoBase(EntryClass::EntryType), data_(std::move(data)) {}

  typename EntryClass::DataType* get() {
    return data_.get();
  }

 private:
  std::unique_ptr<typename EntryClass::DataType> data_;
};

void ExecutorCompileTimeInfoCache::insert(EntryOwningPtr new_entry) {
  // Just overwrite when insertion duplicates, equality not checked.
  entry_type_map_[new_entry->type()] = new_entry.get();
  entries_.emplace_back(std::move(new_entry));
}

template <typename EntryClass>
ExecutorCompileTimeEntry<EntryClass>::ExecutorCompileTimeEntry(
    ExecutorCompileTimeInfoCache* data_cache,
    MakerFnType fn) {
  using InfoType = CompileTimeInfo<EntryClass>;

  if (!data_cache || !data_cache->has(EntryClass::EntryType)) {
    owned_data_ = fn();
    data_ptr_ = owned_data_.get();

    if (data_cache) {
      std::unique_ptr<CompileTimeInfoBase> new_entry =
          std::make_unique<InfoType>(std::move(owned_data_));
      data_cache->insert(std::move(new_entry));
    }
  } else {
    data_ptr_ =
        data_cache->at(EntryClass::EntryType)->template as<InfoType>()->get();
  }
}

// Template instantiation
template class ExecutorCompileTimeEntry<ParallelBindingIterDomains>;
template class ExecutorCompileTimeEntry<ParallelIterExtentMap>;
template class ExecutorCompileTimeEntry<VectorizedTensorValidation>;
template class ExecutorCompileTimeEntry<InputOutputAliases>;

} // namespace caching

std::vector<IterDomain*> getParallelBindingsIterDomains(
    GpuLower* lower,
    const std::vector<TensorView*>& used_tvs) {
  std::vector<IterDomain*> parallel_ids;
  for (auto tv : used_tvs) {
    for (auto id : tv->getLeafDomain()) {
      if (id->isThread()) {
        if (id->isBroadcast()) {
          // Want to keep the broadcast dimensions if they are not resolved
          // TODO: piping down the parallel dimension map here would
          //  be helpful
          if (lower->caMap()->getConcreteMappedID(id, IdMappingMode::LOOP) ==
              id) {
            parallel_ids.push_back(id);
          }
        } else {
          // Non broadcast ids are directly added to the binding
          //  ids.
          parallel_ids.push_back(id);
        }
      }
    }
  }
  return parallel_ids;
}

namespace {

void insertParallelExtent(
    IterDomain* binding_id,
    const std::unique_ptr<ParallelExtentMap>& parallel_iter_extents_ptr) {
  auto extent = binding_id->extent();
  const auto it =
      parallel_iter_extents_ptr->find(binding_id->getParallelType());
  if (it != parallel_iter_extents_ptr->end()) {
    it->second.push_back(extent);
  } else {
    parallel_iter_extents_ptr->operator[](binding_id->getParallelType()) = {
        extent};
  }
}

} // namespace

std::unique_ptr<ParallelExtentMap> getParallelIterExtents(
    std::vector<IterDomain*>& parallel_binding_ids) {
  auto parallel_iter_extents_ptr = std::make_unique<ParallelExtentMap>();
  for (auto id : parallel_binding_ids) {
    insertParallelExtent(id, parallel_iter_extents_ptr);
  }

  return parallel_iter_extents_ptr;
}

} // namespace executor_utils
} // namespace nvfuser
