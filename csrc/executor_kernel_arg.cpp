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
std::unique_ptr<TensorArgAbstract> getTensorArg(const at::Tensor& tensor) {
  switch (tensor.ndimension()) {
    case (0):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 0, nvfuser_index_t>>>(tensor);
    case (1):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 1, nvfuser_index_t>>>(tensor);
    case (2):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 2, nvfuser_index_t>>>(tensor);
    case (3):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 3, nvfuser_index_t>>>(tensor);
    case (4):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 4, nvfuser_index_t>>>(tensor);
    case (5):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 5, nvfuser_index_t>>>(tensor);
    case (6):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 6, nvfuser_index_t>>>(tensor);
    case (7):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 7, nvfuser_index_t>>>(tensor);
    case (8):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 8, nvfuser_index_t>>>(tensor);
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
  std::unique_ptr<TensorArgAbstract> operator()(const at::Tensor& tensor) {
    return getTensorArg<T, nvfuser_index_t>(tensor);
  };
};

template <typename INDEX_MODE>
std::unique_ptr<TensorArgAbstract> getTensorArg(const at::Tensor& tensor) {
  return atenTypeDispatchWithC10Complex(
      tensor.scalar_type(), GetTensorArgWithNativeType<INDEX_MODE>(), tensor);
}

std::unique_ptr<TensorArgAbstract> getTensorArg(
    KernelIndexMode index_mode,
    const at::Tensor& tensor) {
  switch (index_mode) {
    case KernelIndexMode::INT32:
      return getTensorArg<int>(tensor);
    case KernelIndexMode::INT64:
      return getTensorArg<int64_t>(tensor);
    default:
      break;
  }

  TORCH_INTERNAL_ASSERT(false, "unknown index mode");
  return nullptr;
}

} // namespace

KernelArgumentHolder KernelArgumentHolder::createKernelArgumentHolder(
    const c10::ArrayRef<c10::IValue>& inputs,
    const std::optional<KernelIndexMode>& opt_index_mode) {
  if (inputs.empty()) {
    // default to int32 on device 0
    KernelArgumentHolder args(
        opt_index_mode.has_value() ? opt_index_mode.value()
                                   : KernelIndexMode::INT32);
    args.setDeviceIndex(0);
    return args;
  }
  auto device_index = getCommonDeviceCUDA(inputs);
  auto input_index_mode = collectIndexMode(inputs);

  auto index_mode = input_index_mode;

  // Use index_mode if given. Make sure it is as large as the index
  // mode required for the inputs
  if (opt_index_mode.has_value()) {
    TORCH_INTERNAL_ASSERT(
        (opt_index_mode == input_index_mode) ||
            opt_index_mode == KernelIndexMode::INT64,
        "Given index mode and argument index mode don't match.",
        "Index mode: ",
        opt_index_mode.value(),
        ", argument index mode: ",
        input_index_mode);
    index_mode = opt_index_mode.value();
  }

  KernelArgumentHolder args(index_mode);
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

} // namespace

// Push a tensor to the arguments
void KernelArgumentHolder::push(const at::Tensor& tensor) {
  changed_ = true;
  if (is_cpu_scalar(tensor)) {
    arguments_.push_back(atenTypeDispatchWithC10Complex(
        tensor.scalar_type(), MakeCpuScalarTensor(), tensor));
  } else {
    arguments_.push_back(getTensorArg(index_mode_, tensor));
  }
}

// Push a scalar or integer to the arguments
void KernelArgumentHolder::push(const c10::IValue& val) {
  changed_ = true;
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
void** KernelArgumentHolder::getBuffer() {
  if (changed_) {
    void_ptrs_ = std::vector<void*>(arguments_.size(), nullptr);
    for (const auto i : c10::irange(arguments_.size())) {
      void_ptrs_[i] = static_cast<void*>(arguments_[i]->arg());
    }
    changed_ = false;
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
  changed_ = true;
  arguments_.emplace_back(arg->copy_unique_ptr());
}

void KernelArgumentHolder::swap(int i, const ArgAbstract* arg) {
  changed_ = true;
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

} // namespace nvfuser
