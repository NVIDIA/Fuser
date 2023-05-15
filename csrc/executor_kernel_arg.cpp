// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/util/irange.h>

// Extract size and strides
#include <kernel_cache.h>

#include <executor_kernel_arg.h>

namespace nvfuser {

PrimDataType TensorArgAbstract::getSmallestIndexType() const {
  KernelIndexTypeCompute index_type_helper;
  for (const auto dim_i : c10::irange(tensor_.ndimension())) {
    auto size = tensor_.size(dim_i);
    auto stride = tensor_.stride(dim_i);
    if (index_type_helper.addDim(size, stride) == PrimDataType::Int) {
      return PrimDataType::Int;
    }
  }
  return PrimDataType::Int32;
}

namespace {

template <typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    at::Tensor tensor,
    TensorView* tv) {
  switch (tensor.ndimension()) {
    case (0):
      return std::make_unique<TensorArg<TensorArgCodegen<0, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (1):
      return std::make_unique<TensorArg<TensorArgCodegen<1, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (2):
      return std::make_unique<TensorArg<TensorArgCodegen<2, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (3):
      return std::make_unique<TensorArg<TensorArgCodegen<3, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (4):
      return std::make_unique<TensorArg<TensorArgCodegen<4, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (5):
      return std::make_unique<TensorArg<TensorArgCodegen<5, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (6):
      return std::make_unique<TensorArg<TensorArgCodegen<6, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (7):
      return std::make_unique<TensorArg<TensorArgCodegen<7, nvfuser_index_t>>>(
          std::move(tensor), tv);
    case (8):
      return std::make_unique<TensorArg<TensorArgCodegen<8, nvfuser_index_t>>>(
          std::move(tensor), tv);
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          tensor.ndimension(),
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

std::unique_ptr<TensorArgAbstract> getAbstractTensorArg(at::Tensor tensor) {
  return std::make_unique<TensorArgAbstract>(std::move(tensor));
}

std::unique_ptr<TensorArgAbstract> getTensorArg(
    at::Tensor tensor,
    TensorView* tv,
    std::optional<PrimDataType> index_type) {
  if (index_type.has_value()) {
    switch (index_type.value()) {
      case PrimDataType::Int32:
        return getTensorArg<int>(std::move(tensor), tv);
      case PrimDataType::Int:
        return getTensorArg<int64_t>(std::move(tensor), tv);
      default:
        TORCH_INTERNAL_ASSERT(false, "unknown index mode");
        break;
    }
  } else {
    return getAbstractTensorArg(std::move(tensor));
  }
}

} // namespace

KernelArgumentHolder KernelArgumentHolder::createKernelArgumentHolder(
    const c10::ArrayRef<c10::IValue>& inputs) {
  if (inputs.empty()) {
    // default to device 0
    KernelArgumentHolder args;
    args.setDeviceIndex(0);
    return args;
  }
  auto device_index = getCommonDeviceCUDA(inputs);

  KernelArgumentHolder args;
  args.setDeviceIndex(device_index);
  args.push(inputs);

  return args;
}

namespace {

template <size_t size>
std::unique_ptr<ArgAbstract> makeCpuScalarTensorArg(const at::Tensor& tensor) {
  auto ptr = std::make_unique<CpuScalarTensorArg<size>>();
  static_assert(sizeof(ptr->instance_) == size);
  std::memcpy(&(ptr->instance_), tensor.data_ptr(), size);
  return ptr;
}

} // namespace

// Push a tensor to the arguments
void KernelArgumentHolder::push(const at::Tensor& tensor) {
  if (is_cpu_scalar(tensor)) {
    switch (tensor.element_size()) {
      case 1:
        arguments_.push_back(makeCpuScalarTensorArg<1>(tensor));
        break;
      case 2:
        arguments_.push_back(makeCpuScalarTensorArg<2>(tensor));
        break;
      case 4:
        arguments_.push_back(makeCpuScalarTensorArg<4>(tensor));
        break;
      case 8:
        arguments_.push_back(makeCpuScalarTensorArg<8>(tensor));
        break;
      case 16:
        arguments_.push_back(makeCpuScalarTensorArg<16>(tensor));
        break;
    }
  } else {
    arguments_.push_back(getTensorArg(tensor, nullptr, std::nullopt));
  }
}

// Push a scalar or integer to the arguments
void KernelArgumentHolder::push(const c10::IValue& val) {
  TORCH_INTERNAL_ASSERT(
      val.isScalar(),
      "Tried to push an arg to run in a fused kernel, expected a scalar but got, ",
      val);
  auto scalar_val = val.toScalar();
  switch (scalar_val.type()) {
    case c10::ScalarType::ComplexDouble:
      arguments_.push_back(
          std::make_unique<ComplexDoubleArg>(scalar_val.toComplexDouble()));
      return;
    case c10::ScalarType::Double:
      arguments_.push_back(std::make_unique<DoubleArg>(scalar_val.toDouble()));
      return;
    case c10::ScalarType::Long:
      arguments_.push_back(std::make_unique<LongArg>(scalar_val.toLong()));
      return;
    case c10::ScalarType::Bool:
      arguments_.push_back(std::make_unique<BoolArg>(scalar_val.toBool()));
      return;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          " Tried to create argument to send to a fused kernel, but got an unexpected type.");
  }
  TORCH_INTERNAL_ASSERT(
      false,
      " Tried to create argument to send to a fused kernel, but got a non-scalar type.");
}

void KernelArgumentHolder::push(int64_t val) {
  arguments_.push_back(std::make_unique<LongArg>(val));
}

void KernelArgumentHolder::push(const at::PhiloxCudaState& val) {
  arguments_.push_back(std::make_unique<PhiloxCudaStateArg>(val));
}

// Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
// in the buffer
void** KernelArgumentHolder::getBuffer(
    PrimDataType index_type,
    std::vector<TensorView*> tvs) {
  TORCH_INTERNAL_ASSERT(
      arguments_.size() == tvs.size(),
      "The size of arguments and the size of tvs does not match.");
  if (void_ptrs_.size() < arguments_.size()) {
    void_ptrs_.resize(arguments_.size());
  }
  for (const auto i : c10::irange(arguments_.size())) {
    if (auto tensor_arg =
            dynamic_cast<TensorArgAbstract*>(arguments_.at(i).get())) {
      if (tensor_arg->isAbstract() ||
          tensor_arg->getIndexType() != index_type) {
        auto resolved_arg =
            getTensorArg(tensor_arg->getTensor(), tvs.at(i), index_type);
        arguments_.at(i) = std::move(resolved_arg);
      }
    }
    void_ptrs_.at(i) = static_cast<void*>(arguments_.at(i)->arg());
  }
  return void_ptrs_.data();
}

void KernelArgumentHolder::push(const c10::ArrayRef<c10::IValue>& args) {
  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (const auto& arg : args) {
    if (arg.isTensor()) {
      push(arg.toTensor());
    } else {
      push(arg);
    }
  }
}

void KernelArgumentHolder::push(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    push(tensor);
  }
}

void KernelArgumentHolder::push(const ArgAbstract* arg) {
  arguments_.emplace_back(arg->clone());
}

void KernelArgumentHolder::erase(const ArgAbstract* arg_to_delete) {
  auto iter = std::remove_if(
      arguments_.begin(),
      arguments_.end(),
      [&](const std::unique_ptr<ArgAbstract>& ref) {
        return arg_to_delete == ref.get();
      });
  arguments_.erase(iter, arguments_.end());
}

void KernelArgumentHolder::swap(int i, const ArgAbstract* arg) {
  auto holder = arg->clone();
  arguments_[i].swap(holder);
}

void KernelArgumentHolder::appendPhiloxRNGSeed(uint64_t rand_offset) {
  at::PhiloxCudaState philox_engine_inputs;
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    philox_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(
            rand_offset);
  }
  push(philox_engine_inputs);
}

std::string KernelArgumentHolder::toString() const {
  std::stringstream ss;
  for (const auto& arg : arguments_) {
    ss << arg->toString() << "\n";
  }
  return ss.str();
}

PrimDataType KernelArgumentHolder::getSmallestIndexTypeOfArguments() const {
  for (const auto& arg : arguments_) {
    if (auto tensor_arg = dynamic_cast<const TensorArgAbstract*>(arg.get())) {
      if (tensor_arg->getSmallestIndexType() == PrimDataType::Int) {
        return PrimDataType::Int;
      }
    }
  }
  return PrimDataType::Int32;
}

} // namespace nvfuser
