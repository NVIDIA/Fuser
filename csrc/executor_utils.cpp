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
#include <executor_utils.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel_db/kernel_db.h>
#include <torch/csrc/jit/resource_guard.h>

#include <cuda_occupancy.h>
#include <nvfuser_resources/PhiloxCudaStateRaw.h>
#include <nvfuser_resources/array.h>
#include <nvfuser_resources/basic_type_traits.h>
#include <nvfuser_resources/bf16_support.h>
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
  ss << nvfuser_resources::complex_number_cu;

  ss << nvfuser_resources::fp16_support_cu;
  ss << nvfuser_resources::bf16_support_cu;

  // Base classes and helpers
  ss << nvfuser_resources::tensor_cu;
  ss << nvfuser_resources::type_traits_cu;
  ss << nvfuser_resources::array_cu;
  ss << nvfuser_resources::random_numbers_cu;
  ss << nvfuser_resources::helpers_cu;
  ss << nvfuser_resources::index_utils_cu;
  ss << nvfuser_resources::tuple_cu;

  // Synchronization classes
  if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
    ss << nvfuser_resources::block_sync_atomic_cu;
  } else {
    ss << nvfuser_resources::block_sync_default_cu;
  }
  ss << nvfuser_resources::grid_sync_cu;

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

  // Random utilities
  ss << nvfuser_resources::PhiloxCudaStateRaw_cu;

  return ss.str();
}

namespace {

// Query the target GPU version number NVRTC compiles CUDA kernels for
TORCH_CUDA_CU_API void queryTargetGPUVersion(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor,
    bool& compile_to_sass) {
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;
  NVRTC_SAFE_CALL(nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  TORCH_CHECK(
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

// return false if arg's type, number of dimensions, and device, doesn't match
// param and provided c10:device
bool validateKernelArgTensor(
    const at::Tensor& arg,
    const Val* param,
    const c10::Device& device,
    std::stringstream& msg) {
  // Arg is a tensor. Param must be a tensor too.
  if (*param->getValType() != ValType::TensorView) {
    msg << "Argument is a tensor, but the parameter is not.\n";
    return false;
  }

  if (is_cpu_scalar(arg) && !param->as<TensorView>()->isCpuScalar()) {
    msg << "Argument is CPU Scalar Tensor, but parameter is not.\n";
    return false;
  }

  if (!is_cpu_scalar(arg) && !arg.is_cuda()) {
    msg << "Argument is a CPU tensor which is not supported in fusions.\n";
    return false;
  }

  // Check the rank of the tensors.
  size_t arg_dim = arg.dim();
  // Note: This requires current Fusion to be active.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t param_dim = TensorDomain::noReductions(
                         param->as<TensorView>()->getMaybeRFactorDomain())
                         .size();
  // see [Note - broadcast support in integration]
  // Because of broadcasting support handled in integration, we relax the rank
  // check as necessary.
  if (arg_dim > param_dim) {
    msg << "Argument tensor's rank is " << arg_dim << ", but the parameter is "
        << param_dim << "\n";
    return false;
  }

  if (!is_cpu_scalar(arg) && arg.device() != device) {
    msg << "Argument is on device that is not compiled for."
        << "\n";
    return false;
  }
  // Check element type
  at::ScalarType arg_data_type = arg.scalar_type();
  DataType param_data_type = *param->getDataType();
  bool match = false;
  // TODO: remove this switch with `aten_to_data_type`
  switch (arg_data_type) {
    case at::ScalarType::Double:
      match = param_data_type == DataType::Double;
      break;
    case at::ScalarType::Half:
      match = param_data_type == DataType::Half;
      break;
    case at::ScalarType::BFloat16:
      match = param_data_type == DataType::BFloat16;
      break;
    case at::ScalarType::Float:
      match = param_data_type == DataType::Float;
      break;
    case at::ScalarType::Long:
      match = param_data_type == DataType::Int;
      break;
    case at::ScalarType::Int:
      match = param_data_type == DataType::Int32;
      break;
    case at::ScalarType::Bool:
      match = param_data_type == DataType::Bool;
      break;
    case at::ScalarType::ComplexFloat:
      match = param_data_type == DataType::ComplexFloat;
      break;
    case at::ScalarType::ComplexDouble:
      match = param_data_type == DataType::ComplexDouble;
      break;
    default:
      msg << "Argument element type, " << arg_data_type << ", is not supported."
          << "\n";
      return false;
  }
  if (!match)
    msg << "Argument element type is " << arg_data_type
        << ", but the parameter is " << param_data_type << "\n";
  return match;
}

// Return false if  arg_type doesn't match the type in param
bool validateKernelArgScalar(
    const ArgAbstract* arg,
    const Val* param,
    std::stringstream& msg) {
  TORCH_INTERNAL_ASSERT(
      param->getDataType().has_value(), "kernel param should have data type");
  DataType param_type = *param->getDataType();
  bool match = false;
  switch (arg->type()) {
    case ArgType::Long:
      match = param_type == DataType::Int || param_type == DataType::Int32;
      break;
    case ArgType::Double:
      match = param_type == DataType::Double || param_type == DataType::Float ||
          param_type == DataType::Half || param_type == DataType::BFloat16;
      break;
    case ArgType::Bool:
      match = param_type == DataType::Bool;
      break;
    case ArgType::ComplexDouble:
      match = param_type == DataType::ComplexDouble ||
          param_type == DataType::ComplexFloat;
      break;
    default:
      // TODO: We need to verify that param is actually a scalar
      msg << "Argument is not a scalar, but the parameter is."
          << "\n";
      return false;
  }
  if (!match) {
    msg << "Argument type is " << argTypeToString(arg->type())
        << ", but the parameter is " << param_type << "\n";
  }
  return match;
}

// Return false if arg and param don't match up and if arg's device (if a
// tensor) doesn't match provided device
bool validateKernelArg(
    const ArgAbstract* arg,
    const Val* param,
    const c10::Device& device,
    std::stringstream& msg) {
  // clang-tidy complains that arg may be null without this assertion
  TORCH_INTERNAL_ASSERT(arg != nullptr);
  if (auto tensor_arg_abstract = dynamic_cast<const TensorArgAbstract*>(arg)) {
    // TODO: don't use get tensor here. We would want to remove tensor reference
    // for async compilation
    return validateKernelArgTensor(
        tensor_arg_abstract->getTensor(), param, device, msg);
  } else if (arg->isType(ArgType::CpuScalarTensor)) {
    // TODO: merge this one with above
    // TODO: we need to check cpu scalar dtyp matches param
    bool match = param->as<TensorView>()->isCpuScalar();
    if (!match) {
      msg << "Argument is scalar type, but kernel parameter is not\n";
    }
    return match;
  } else {
    return validateKernelArgScalar(arg, param, msg);
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

void validateKernelInputs(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const c10::Device& device) {
  FUSER_PERF_SCOPE("executor_utils::ValidateKernelInputs");

  // This is necessary as we were traversing the fusion graph later in the check
  FusionGuard fg(fusion);
  // Check inputs
  TORCH_INTERNAL_ASSERT(
      args.size() == fusion->inputs().size(), "Wrong number of kernel inputs.");

  std::stringstream msg;
  bool mismatch = false;
  for (const auto i : c10::irange(args.size())) {
    const ArgAbstract* arg = args[i];
    const Val* param = fusion->inputs()[i];
    mismatch = !validateKernelArg(arg, param, device, msg) || mismatch;
  }
  TORCH_INTERNAL_ASSERT(
      !mismatch, "Found one or more invalid arguments: ", msg.str());
}

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    const c10::Device& device) {
  FUSER_PERF_SCOPE("executor_utils::ValidateKernelOutputs");

  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(),
      "Kernel should have at least one output tensor.");

  TORCH_INTERNAL_ASSERT(
      outputs.size() == fusion->outputs().size(),
      "Wrong number of kernel outputs.");

  std::stringstream msg;
  bool mismatch = false;
  for (const auto i : c10::irange(outputs.size())) {
    const at::Tensor& arg = outputs[i];
    const Val* param = fusion->outputs()[i];
    mismatch = !validateKernelArgTensor(arg, param, device, msg) || mismatch;
  }
  TORCH_INTERNAL_ASSERT(
      !mismatch, "Found one or more invalid arguments: ", msg.str());
}

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
    TORCH_INTERNAL_ASSERT(
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
    TORCH_INTERNAL_ASSERT(
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
      TORCH_INTERNAL_ASSERT(
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
        TORCH_INTERNAL_ASSERT(
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
      TORCH_INTERNAL_ASSERT(inp_or_out_info.size() == 1);
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
  TORCH_INTERNAL_ASSERT(
      !info.contig_alloc_ids.empty(),
      "No root ID found for vectorization with ",
      info.consumer_tv->toString(),
      " and ",
      info.producer_tv->toString());

  // TODO: Rewrite validation of the vectorized dimension
  // int64_t vectorized_merged_domain_extent = 1;
  for (auto id : info.contig_alloc_ids) {
    auto extent_val = expr_eval.evaluate(id->extent());
    TORCH_INTERNAL_ASSERT(
        extent_val.has_value(),
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
  // TORCH_INTERNAL_ASSERT(
  //     vectorized_merged_domain_extent % info.word_size == 0,
  //     "Error vectorizing, ",
  //     info.consumer_tv->toString(),
  //     " as the extent of the indexed domain, ",
  //     vectorized_merged_domain_extent,
  //     ", is not divisible by vector word size ",
  //     info.word_size);
}

void validateAlignedVectorizedFusionInputOutput(
    const at::Tensor& aten_tensor,
    int word_size,
    TensorView* tv) {
  ExpressionEvaluator eval;
  auto sizes_strides =
      inferAndValidateAllocationSizesAndStrides(aten_tensor, tv, eval);

  std::vector<int64_t> no_reduction_to_full;
  for (int64_t i :
       c10::irange((int64_t)tv->getMaybeAllocationDomain().size())) {
    auto alloc_id = tv->getMaybeAllocationDomain().at(i);
    if (!alloc_id->isReduction()) {
      no_reduction_to_full.emplace_back(i);
    }
  }
  TORCH_INTERNAL_ASSERT(sizes_strides.size() == no_reduction_to_full.size());

  TORCH_INTERNAL_ASSERT(
      reinterpret_cast<size_t>(aten_tensor.data_ptr()) %
              (word_size * aten_tensor.dtype().itemsize()) ==
          0,
      "Vectorization of ",
      tv->toString(),
      " not possible as the memory address is not aligned. ",
      "Address: ",
      aten_tensor.data_ptr(),
      ", vector word size: ",
      word_size,
      ", data type: ",
      aten_tensor.dtype());

  // Traverse strides from the right-most domains. The rightmost
  // domain must have stride 1.
  int64_t cur_contig_stride = 1;
  bool still_rightmost = true;
  for (int64_t i = (int64_t)sizes_strides.size() - 1; i >= 0; --i) {
    const auto [size, stride] = sizes_strides.at(i);
    auto alloc_id =
        tv->getMaybeAllocationDomain().at(no_reduction_to_full.at(i));
    const auto is_expanded_broadcasting =
        alloc_id->isBroadcast() && alloc_id->hasExpandedExtent();

    if (is_expanded_broadcasting) {
      TORCH_INTERNAL_ASSERT(
          stride == 0,
          "Dimension ",
          i,
          " should be an expanded broadcasting, but it does not have stride zero.");
    }

    // If this domain is contiguous or size == 1, then not necessary to check
    // the stride. Otherwise, stride must be 1 if it's rightmost or
    // divisible by word_size
    TORCH_INTERNAL_ASSERT(
        stride == cur_contig_stride || size == 1 || is_expanded_broadcasting ||
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
        stride)
    // If the domain is size-1, the next domain is still considered
    // rightmost.
    still_rightmost =
        still_rightmost && (size == 1 || is_expanded_broadcasting);
    // We do not update cur_contig_stride for size==1 dimensions,
    // since we have specialized vectorization stride check for them
    if (size != 1) {
      cur_contig_stride = stride * size;
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
    auto tensor_arg_abstract =
        dynamic_cast<const TensorArgAbstract*>(args[pos]);
    TORCH_INTERNAL_ASSERT(tensor_arg_abstract, "alias io only supports tensor");
    validateAlignedVectorizedFusionInputOutput(
        tensor_arg_abstract->getTensor(), word_size, tv);
  }
  if (!outputs.empty()) {
    for (auto pos : tensor_vectorization_validation_entry.get()
                        .aligned_vectorized_out_tensor_pos) {
      auto tv = kernel->outputs().at(pos)->as<TensorView>();
      auto word_size = kernel->summary().vectorized_accesses.at(tv);
      validateAlignedVectorizedFusionInputOutput(outputs[pos], word_size, tv);
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
        auto tensor_arg_abstract =
            dynamic_cast<const TensorArgAbstract*>(args[idx]);
        TORCH_INTERNAL_ASSERT(
            tensor_arg_abstract, "alias io only supports tensor");
        return tensor_arg_abstract->getTensor();
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
  TORCH_INTERNAL_ASSERT(
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
    TORCH_INTERNAL_ASSERT(
        input_extent.has_value(),
        "Could not check if a split with vectorization is divisible because the extent, ",
        extent_factor.first->toString(),
        ", is not possible to evaluate.");
    TORCH_INTERNAL_ASSERT(
        input_extent.has_value(),
        "Could not check if a split with vectorization is divisible because the split factor, ",
        extent_factor.second->toString(),
        ", is not possible to evaluate.");
    TORCH_INTERNAL_ASSERT(
        input_extent.value() % split_factor.value() == 0,
        "Non-divisible split with vectorization is detected. ",
        "Extent: ",
        input_extent.value(),
        ". Factor: ",
        split_factor.value());
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

namespace {

void bindInputForExprEvaluation(
    Val* val,
    const ArgAbstract* arg,
    bool check_consistency,
    ExpressionEvaluator& expr_eval) {
  if (val->getValType() == ValType::TensorView) {
    TensorView* cg_tensor = val->as<TensorView>();
    auto root_domain =
        TensorDomain::noReductions(cg_tensor->getMaybeRFactorDomain());

    if (root_domain.empty()) {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::CpuScalarTensor) ||
              (arg->isType(ArgType::Tensor) &&
               dynamic_cast<const TensorArgAbstract*>(arg)->getRank() == 0),
          "Something went wrong configuring launch. Inputs is not rank 0 tensor");
    } else {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::Tensor),
          "Something went wrong configuring launch. Inputs do not match.");

      auto tensor_arg_abstract = dynamic_cast<const TensorArgAbstract*>(arg);

      TORCH_INTERNAL_ASSERT(
          tensor_arg_abstract &&
              tensor_arg_abstract->getRank() == (int64_t)root_domain.size(),
          "Something went wrong configuring launch. Inputs rank does not match.");

      for (const auto dim : c10::irange(root_domain.size())) {
        const auto tensor_arg_size = tensor_arg_abstract->getSize((int)dim);
        const auto extent = root_domain[dim]->extent();
        if (root_domain[dim]->hasExpandedExtent()) {
          // Could support dynamic size on expanded dimension, so may not have
          // an inferable expanded extent here. This check might be better to do
          // once all values are bound.
          auto maybe_expanded_size =
              expr_eval.evaluate(root_domain[dim]->expandedExtent());
          if (maybe_expanded_size.has_value()) {
            TORCH_CHECK(
                *maybe_expanded_size == tensor_arg_size,
                "Expecting expanded extent of ",
                *maybe_expanded_size,
                " but received value of ",
                tensor_arg_size);
          } else {
            expr_eval.bind(root_domain[dim]->expandedExtent(), tensor_arg_size);
          }
        }

        const auto value =
            root_domain[dim]->hasExpandedExtent() ? 1 : tensor_arg_size;
        bool should_bind = true;
        if (check_consistency) {
          const auto prev_value = expr_eval.evaluate(extent);
          if (prev_value.has_value()) {
            TORCH_CHECK(
                *prev_value == value,
                "Attempting to bind ",
                extent->toString(),
                " to ",
                value,
                " but it's already set to ",
                *prev_value);
            should_bind = false;
          }
        }
        if (should_bind && !extent->isConstScalar()) {
          expr_eval.bind(extent, value);
        }
      }
    }
  } else if (val->getValType().value() == ValType::Scalar) {
    if (val->getDataType().value() == DataType::Int) {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::Long),
          "fusion expected Scalar Int inputs, but found ",
          argTypeToString(arg->type()));
      expr_eval.bind(val, *static_cast<const int64_t*>(arg->arg()));
    } else if (val->getDataType().value() == DataType::Double) {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::Double),
          "fusion expected Scalar Double inputs, but found ",
          argTypeToString(arg->type()));
      expr_eval.bind(val, *static_cast<const double*>(arg->arg()));
    }
  }
}

} // namespace

ExpressionEvaluator bindInputs(
    const KernelArgumentHolder& args,
    Fusion* kernel,
    bool check_consistency) {
  FUSER_PERF_SCOPE("executor_utils::bindInputs");

  // args may contains more than just inputs, but inputs are always at the
  // beginning.
  TORCH_INTERNAL_ASSERT(
      kernel->inputs().size() <= args.size(),
      "KernelArgumentHolder contains less argument than kernel's input.");

  ExpressionEvaluator expr_eval;
  const auto& inputs = kernel->inputs();

  for (const auto i : c10::irange(inputs.size())) {
    bindInputForExprEvaluation(
        inputs[i], args[i], check_consistency, expr_eval);
  }
  return expr_eval;
}

namespace {

// Get the size of the program code in nvrtcProgram, which is either PTX or SASS
size_t nvrtcGetSize(const nvrtcProgram& program, bool compile_to_sass) {
#if CUDA_VERSION >= 11010
  const auto getSize = compile_to_sass ? nvrtcGetCUBINSize : nvrtcGetPTXSize;
#else
  TORCH_INTERNAL_ASSERT(
      !compile_to_sass, "SASS not supported in CUDA versions older than 11.1");
  const auto getSize = nvrtcGetPTXSize;
#endif
  size_t size = 0;
  NVRTC_SAFE_CALL(getSize(program, &size));
  return size;
}

// Get the program code from nvrtcProgram
std::vector<char> nvrtcGetCode(
    const nvrtcProgram& program,
    bool compile_to_sass) {
  const auto size = nvrtcGetSize(program, compile_to_sass);

#if CUDA_VERSION >= 11010
  const auto getCode = compile_to_sass ? nvrtcGetCUBIN : nvrtcGetPTX;
#else
  TORCH_INTERNAL_ASSERT(
      !compile_to_sass, "SASS not supported in CUDA versions older than 11.1");
  const auto getCode = nvrtcGetPTX;
#endif

  std::vector<char> code(size);
  NVRTC_SAFE_CALL(getCode(program, code.data()));
  return code;
}

void dumpCompiledCodeToFile(
    const std::vector<char>& code,
    int64_t fusion_id,
    bool dump_cubin) {
  std::stringstream file_name;
  file_name << "__tmp_kernel" << fusion_id << "."
            << (dump_cubin ? "cubin" : "ptx");
  std::cout << "PRINTING: " << file_name.str() << std::endl;
  std::ofstream out(file_name.str());
  TORCH_INTERNAL_ASSERT(out.is_open());
  out.write(code.data(), (std::streamsize)code.size());
  out.close();
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
  if (auto env_count = getenv("PYTORCH_NVFUSER_MAX_REG_COUNT")) {
    auto env_max_reg_count = std::atoi(env_count);
    TORCH_CHECK(
        env_max_reg_count > 0 && env_max_reg_count <= max_register_limit,
        "Invalid max register count specified by PYTORCH_NVFUSER_MAX_REG_COUNT: ",
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

  //! Call nvrtcCompileProgram with set options
  std::string invoke(nvrtcProgram program, const std::string& src) const {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::CompileProgram");
    auto opts = getOptions();
    auto result = nvrtcCompileProgram(
        program, static_cast<int>(opts.size()), opts.data());

    size_t logsize = 0;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &logsize));
    std::string log;
    log.reserve(logsize);
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log.data()));
    if (result != NVRTC_SUCCESS) {
      TORCH_INTERNAL_ASSERT(
          false, src, "\nCUDA NVRTC compile error: ", log.data());
    }

    if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
      std::cout << log.data() << std::endl;
    }

    return log;
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
    log_.reserve(kLogSize);
  }

  const std::string& log() const {
    TORCH_INTERNAL_ASSERT(logging_enabled_, "Logging not enabled");
    return log_;
  }

  //! Invoke cuModuleLoadDataEx with ptx or cubin. Dump logging output
  //! if enabled
  std::string invoke(CUmodule& module, const void* image) {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::LoadPTX");

    auto [opts, opt_vals] = getOptions();

    CUDA_SAFE_CALL(cuModuleLoadDataEx(
        &module, image, opts.size(), opts.data(), opt_vals.data()));

    if (logging_enabled_) {
      std::cout << log_ << std::endl;
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
        TORCH_INTERNAL_ASSERT(false, "Invalid option");
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
    std::optional<int64_t> opt_block_size,
    const int64_t max_register_heuristic) {
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
    nvrtc_compile_driver.setOption("-DPYTORCH_NVFUSER_PROFILE_KERNEL");
  }

  if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose) ||
      isOptionEnabled(EnableOption::WarnRegisterSpill)) {
    // show register usage in compilation log
    if (compile_to_sass) {
      nvrtc_compile_driver.setOption("--ptxas-options");
      nvrtc_compile_driver.setOption("--verbose");
    } else {
      module_load_driver.enableLogging();
    }
  }

  const char* ptxas_opt_level = getenv("PYTORCH_NVFUSER_JIT_OPT_LEVEL");

  if (ptxas_opt_level) {
    int val = atoi(ptxas_opt_level);
    if (val <= 4 && val >= 0) {
      if (val < 4) {
        TORCH_WARN(
            "ptxas optimization level manually set as ",
            val,
            ", which could negatively affect performance. Try removing env variable PYTORCH_NVFUSER_JIT_OPT_LEVEL for optimal performance.");
      }
      if (compile_to_sass) {
        nvrtc_compile_driver.setOption("--ptxas-options");
        nvrtc_compile_driver.setOption("-O" + std::to_string(val));
      } else {
        module_load_driver.setOption(CU_JIT_OPTIMIZATION_LEVEL, val);
      }
    } else {
      TORCH_WARN_ONCE(
          "acceptable range for PYTORCH_NVFUSER_JIT_OPT_LEVEL is between 0 and 4, but received ",
          val,
          ", ignoring the option");
    }
  }

  const auto max_register =
      getMaxRegCount(opt_block_size, max_register_heuristic);

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
  auto optionArgs = getEnableOptionArguments(EnableOption::WarnRegisterSpill);
  int allowed_spill = 0;
  if (!optionArgs.empty()) {
    try {
      allowed_spill = std::stoi(optionArgs[0]);
    } catch (const std::exception& e) {
      std::cout << "skip invalid argument for WarnRegisterSpill, arg = "
                << optionArgs[0] << std::endl;
    }
  }
  if (stack_count > allowed_spill || store_count > allowed_spill ||
      load_count > allowed_spill) {
    std::cout << "WARNING: Register spill detected\n"
              << compile_log << std::endl;
  }
}

void createNvrtcProgram(
    nvrtcProgram& program,
    int64_t id,
    const std::string& full_src_code) {
  std::stringstream ss;
  ss << "__tmp_kernel" << id << ".cu";
  std::string name = ss.str();
  FUSER_PERF_SCOPE("executor_utils::NvrtcCreateProgram");
  NVRTC_SAFE_CALL(nvrtcCreateProgram(
      &program, full_src_code.c_str(), name.c_str(), 0, nullptr, nullptr));
}

// Compile the given source code with the NVRTC compiler
// driver. Return the binary of the kernel and its lowered name
std::tuple<std::vector<char>, std::string> compileSource(
    const std::string& full_src_code,
    const std::string& func_name,
    int64_t id,
    bool compile_to_sass,
    NvrtcCompileDriver& nvrtc_compile) {
  std::stringstream log;

  nvrtcProgram program; // NOLINT(cppcoreguidelines-init-variables)
  torch::jit::ResourceGuard holdProgram([&] {
    FUSER_PERF_SCOPE("executor_utils::NvrtcDestroyProgram");
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
  });

  createNvrtcProgram(program, id, full_src_code);

  NVRTC_SAFE_CALL(nvrtcAddNameExpression(program, func_name.c_str()));
  log << nvrtc_compile.invoke(program, full_src_code) << std::endl;

  const char* lowered_kernel_name = nullptr;
  NVRTC_SAFE_CALL(
      nvrtcGetLoweredName(program, func_name.c_str(), &lowered_kernel_name));
  auto lowered_kernel_name_str = std::string(lowered_kernel_name);

  auto object_code = nvrtcGetCode(program, compile_to_sass);

  if (isDebugDumpEnabled(DebugDumpOption::Ptx) ||
      isDebugDumpEnabled(DebugDumpOption::Cubin)) {
    dumpCompiledCodeToFile(object_code, id, compile_to_sass);
  }

  return {object_code, lowered_kernel_name_str};
}

} // namespace

// Compile the source if no existing compiled binary is found in KernelDB
std::tuple<NvrtcFunction, std::string, std::vector<char>> getCompiledKernel(
    c10::optional<std::reference_wrapper<const std::string>> kernel_code,
    const std::string& full_src_code,
    const std::string& func_name,
    int64_t id,
    std::optional<int64_t> opt_block_size,
    const int64_t max_register_heuristic,
    bool return_compiled_binary) {
  FUSER_PERF_SCOPE("executor_utils::NVRTC");

  at::cuda::jit::initializeCudaContext();

  const auto prop = at::cuda::getCurrentDeviceProperties();

  int major = 0, minor = 0;
  bool compile_to_sass = false;
  queryTargetGPUVersion(prop, major, minor, compile_to_sass);

#if CUDA_VERSION < 11010
  // compile to sass is not allowed prior to CUDA 11.1
  compile_to_sass = false;
#endif

  if (isOptionDisabled(DisableOption::CompileToSass) ||
      isDebugDumpEnabled(DebugDumpOption::Ptx)) {
    // Allows manually disabling compilation to sass
    //  so the intermediate ptx could be checked.
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
      opt_block_size,
      max_register_heuristic);

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

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<char> object_code;
  std::string lowered_kernel_name_str;
  const auto compile_args =
      toDelimitedString(nvrtc_compile_driver.options(), " ");

  auto& kernel_db = KernelDb::get();
  const auto use_kernel_db = kernel_db.enabled() && kernel_code.has_value();

  // If the Kernel Query failes, the Kernel is recompiled
  if (!(use_kernel_db &&
        kernel_db.query(
            kernel_code.value(),
            compile_args,
            lowered_kernel_name_str,
            object_code))) {
    std::tie(object_code, lowered_kernel_name_str) = compileSource(
        full_src_code, func_name, id, compile_to_sass, nvrtc_compile_driver);

    if (use_kernel_db) {
      auto result = kernel_db.write(
          kernel_code.value(),
          compile_args,
          lowered_kernel_name_str,
          object_code);
      if (!result) {
        TORCH_WARN(
            "kernel_db was unable to write kernel: ", lowered_kernel_name_str);
      }
    }
  }

  NvrtcFunction compiled_kernel;

  log << module_load_driver.invoke(compiled_kernel.module, object_code.data())
      << std::endl;

  if (isOptionEnabled(EnableOption::WarnRegisterSpill)) {
    warnRegisterSpill(log.str());
  }

  CUDA_SAFE_CALL(cuModuleGetFunction(
      &(compiled_kernel.function),
      compiled_kernel.module,
      lowered_kernel_name_str.c_str()));

  if (!return_compiled_binary) {
    object_code.clear();
  }

  return {compiled_kernel, log.str(), object_code};
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
template class ExecutorCompileTimeEntry<InputAliasIndices>;
template class ExecutorCompileTimeEntry<OutputAliasIndices>;

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
