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
#include <serde/arg_abstract_serde.h>

namespace nvfuser {

std::string TensorArgAbstract::toString() const {
  std::stringstream ss;
  auto rank = getRank();
  ss << "tensor dtype: " << getDataType() << " sizes: (";
  for (auto i = 0; i < rank; i++) {
    ss << getSize(i) << ", ";
  }
  ss << ") stride: (";
  for (auto i = 0; i < rank; i++) {
    ss << getStride(i) << ", ";
  }
  ss << ") pointer: " << getPointer();
  return ss.str();
}

namespace {

template <typename T, typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const at::Tensor& tensor,
    bool index_type_resolved) {
  switch (tensor.ndimension()) {
    case (0):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 0, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (1):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 1, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (2):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 2, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (3):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 3, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (4):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 4, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (5):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 5, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (6):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 6, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (7):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 7, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    case (8):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 8, nvfuser_index_t>>>(
          tensor, index_type_resolved);
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          tensor.ndimension(),
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

template <typename nvfuser_index_t>
struct GetTensorArgWithNativeType {
  template <typename T>
  std::unique_ptr<TensorArgAbstract> operator()(
      const at::Tensor& tensor,
      bool index_type_resolved) {
    return getTensorArg<T, nvfuser_index_t>(tensor, index_type_resolved);
  };
};

template <typename INDEX_TYPE>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const at::Tensor& tensor,
    bool index_type_resolved) {
  return atenTypeDispatchWithC10Complex(
      tensor.scalar_type(),
      GetTensorArgWithNativeType<INDEX_TYPE>(),
      tensor,
      index_type_resolved);
}

std::unique_ptr<TensorArgAbstract> getTensorArg(
    const at::Tensor& tensor,
    std::optional<PrimDataType> index_type) {
  if (index_type.has_value()) {
    switch (index_type.value()) {
      case PrimDataType::Int32:
        return getTensorArg<int>(tensor, true);
      case PrimDataType::Int:
        return getTensorArg<int64_t>(tensor, true);
      default:
        TORCH_INTERNAL_ASSERT(false, "unknown index mode");
        break;
    }
  } else {
    // Tentatively create TensorArgAbstract with int64_t
    return getTensorArg<int64_t>(tensor, false);
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

struct MakeCpuScalarTensor {
  template <typename T>
  std::unique_ptr<ArgAbstract> operator()(const at::Tensor& tensor) const {
    return std::make_unique<CpuScalarTensorArg<CpuScalarTensorCodegen<T>>>(
        tensor.data_ptr<T>()[0]);
  }
};

PrimDataType getIndexTypeOfAtenTensor(const at::Tensor& tensor) {
  KernelIndexTypeCompute index_type_helper;
  for (const auto i : c10::irange(tensor.ndimension())) {
    index_type_helper.addDim(tensor.sizes()[i], tensor.strides()[i]);
  }
  return index_type_helper.getType();
}

} // namespace

// Push a tensor to the arguments
void KernelArgumentHolder::push(const at::Tensor& tensor) {
  if (is_cpu_scalar(tensor)) {
    arguments_.push_back(atenTypeDispatchWithC10Complex(
        tensor.scalar_type(), MakeCpuScalarTensor(), tensor));
  } else {
    arguments_.push_back(getTensorArg(tensor, std::nullopt));
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
void** KernelArgumentHolder::getBuffer(PrimDataType index_type) {
  if (void_ptrs_.size() < arguments_.size()) {
    void_ptrs_.resize(arguments_.size());
  }
  for (const auto i : c10::irange(arguments_.size())) {
    if (auto tensor_arg =
            dynamic_cast<TensorArgAbstract*>(arguments_.at(i).get())) {
      if (!tensor_arg->isIndexTypeResolved() ||
          tensor_arg->getIndexType() != index_type) {
        auto resolved_arg = getTensorArg(tensor_arg->getTensor(), index_type);
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
  arguments_.emplace_back(arg->copy_unique_ptr());
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
  auto holder = arg->copy_unique_ptr();
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
    auto tensor_arg = dynamic_cast<const TensorArgAbstract*>(arg.get());
    if (tensor_arg == nullptr) {
      continue;
    }
    KernelIndexTypeCompute index_type_helper;
    for (const auto dim_i : c10::irange(tensor_arg->getRank())) {
      auto size = tensor_arg->getSize(dim_i);
      auto stride = tensor_arg->getStride(dim_i);
      if (index_type_helper.addDim(size, stride) == PrimDataType::Int) {
        return PrimDataType::Int;
      }
    }
  }
  return PrimDataType::Int32;
}

flatbuffers::Offset<serde::KernelArgumentHolder> KernelArgumentHolder::
    serialize(flatbuffers::FlatBufferBuilder& builder) const {
  // table KernelArgumentHolder {
  //   arguments : [ArgAbstract];
  //   device_index : byte;
  //   cache_id : ulong;
  //   is_int_index_mode : bool;
  // }

  using fb_arg_abstract = flatbuffers::Offset<nvfuser::serde::ArgAbstract>;
  std::vector<fb_arg_abstract> arguments_fb;
  arguments_fb.reserve(arguments_.size());
  for (auto& arg : arguments_) {
    arguments_fb.push_back(arg->serialize(builder));
  }

  return serde::CreateKernelArgumentHolderDirect(
      builder, &arguments_fb, device_index_, cache_id_.value_or(SIZE_MAX));
}

void KernelArgumentHolder::deserialize(
    const serde::KernelArgumentHolder* buffer) {
  device_index_ = buffer->device_index();
  cache_id_ = (buffer->cache_id() != SIZE_MAX)
      ? std::optional<size_t>(buffer->cache_id())
      : std::nullopt;

  serde::ArgAbstractFactory arg_abstract_factory;
  for (auto fb_arg_abstract : *buffer->arguments()) {
    arguments_.push_back(arg_abstract_factory.parse(
        fb_arg_abstract->data_type(), fb_arg_abstract));
  }
}

} // namespace nvfuser
