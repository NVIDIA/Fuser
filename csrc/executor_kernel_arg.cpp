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

PrimDataType getSmallestIndexType(const at::Tensor& tensor) {
  KernelIndexTypeCompute index_type_helper;
  for (const auto dim_i : c10::irange(tensor.ndimension())) {
    auto size = tensor.size(dim_i);
    auto stride = tensor.stride(dim_i);
    if (index_type_helper.addDim(size, stride) == PrimDataType::Int) {
      return PrimDataType::Int;
    }
  }
  return PrimDataType::Int32;
}

PrimDataType TensorArgAbstract::getSmallestIndexType() const {
  return nvfuser::getSmallestIndexType(tensor_);
}

namespace {

template <int nalloc, typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    at::Tensor tensor,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  switch (tensor.ndimension()) {
    case (0):
      return std::make_unique<
          TensorArg<TensorArgCodegen<0, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (1):
      return std::make_unique<
          TensorArg<TensorArgCodegen<1, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (2):
      return std::make_unique<
          TensorArg<TensorArgCodegen<2, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (3):
      return std::make_unique<
          TensorArg<TensorArgCodegen<3, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (4):
      return std::make_unique<
          TensorArg<TensorArgCodegen<4, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (5):
      return std::make_unique<
          TensorArg<TensorArgCodegen<5, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (6):
      return std::make_unique<
          TensorArg<TensorArgCodegen<6, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (7):
      return std::make_unique<
          TensorArg<TensorArgCodegen<7, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (8):
      return std::make_unique<
          TensorArg<TensorArgCodegen<8, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
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
std::unique_ptr<TensorArgAbstract> getTensorArg(
    at::Tensor tensor,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  // When tv is nullptr, the given sizes and strides should already be in the
  // target format.
  int64_t alloc_size =
      (tv != nullptr
           ? (int64_t)TensorDomain::noReductions(tv->getMaybeAllocationDomain())
                 .size()
           : tensor.dim());
  switch (alloc_size) {
    case (0):
      return getTensorArg<0, nvfuser_index_t>(tensor, tv, eval);
    case (1):
      return getTensorArg<1, nvfuser_index_t>(tensor, tv, eval);
    case (2):
      return getTensorArg<2, nvfuser_index_t>(tensor, tv, eval);
    case (3):
      return getTensorArg<3, nvfuser_index_t>(tensor, tv, eval);
    case (4):
      return getTensorArg<4, nvfuser_index_t>(tensor, tv, eval);
    case (5):
      return getTensorArg<5, nvfuser_index_t>(tensor, tv, eval);
    case (6):
      return getTensorArg<6, nvfuser_index_t>(tensor, tv, eval);
    case (7):
      return getTensorArg<7, nvfuser_index_t>(tensor, tv, eval);
    case (8):
      return getTensorArg<8, nvfuser_index_t>(tensor, tv, eval);
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
    ExpressionEvaluator& eval,
    PrimDataType index_type) {
  switch (index_type) {
    case PrimDataType::Int32:
      return getTensorArg<int>(std::move(tensor), tv, eval);
    case PrimDataType::Int:
      return getTensorArg<int64_t>(std::move(tensor), tv, eval);
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown index mode");
      break;
  }
}

} // namespace

KernelArgumentHolder KernelArgumentHolder::createKernelArgumentHolder(
    const c10::ArrayRef<c10::IValue>& inputs,
    std::optional<int8_t> selected_device) {
  if (inputs.empty()) {
    // default to device 0
    KernelArgumentHolder args;
    args.setDeviceIndex(
        selected_device.has_value() ? selected_device.value() : (int8_t)0);
    return args;
  }
  auto device_index = getCommonDeviceCUDA(inputs, selected_device);

  KernelArgumentHolder args;
  args.setDeviceIndex(device_index);
  args.push(inputs);

  return args;
}

namespace {

PolymorphicValue IValueToPolymorphicValue(const c10::IValue& val) {
  if (val.isTensor()) {
    return val.toTensor();
  }

  auto scalar_val = val.toScalar();
  switch (scalar_val.type()) {
    case c10::ScalarType::ComplexDouble:
      return (std::complex<double>)scalar_val.toComplexDouble();
    case c10::ScalarType::Double:
      return scalar_val.toDouble();
    case c10::ScalarType::Long:
      return scalar_val.toLong();
    case c10::ScalarType::Bool:
      return scalar_val.toBool();
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Can not convert IValue to PolymorphicValue");
  }
}

} // namespace

void KernelArgumentHolder::push(const c10::ArrayRef<c10::IValue>& args) {
  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (const auto& arg : args) {
    push(IValueToPolymorphicValue(arg));
  }
}

void KernelArgumentHolder::push(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    push(tensor);
  }
}

void KernelArgumentHolder::erase(const PolymorphicValue* arg_to_delete) {
  auto iter = std::remove_if(
      arguments_.begin(), arguments_.end(), [&](const auto& ref) {
        return arg_to_delete == &ref;
      });
  arguments_.erase(iter, arguments_.end());
}

std::string KernelArgumentHolder::toString() const {
  std::stringstream ss;
  for (const auto& arg : arguments_) {
    ss << arg << "\n";
  }
  return ss.str();
}

PrimDataType KernelArgumentHolder::getSmallestIndexTypeOfArguments() const {
  for (const auto& arg : arguments_) {
    if (arg.is<at::Tensor>()) {
      if (getSmallestIndexType(arg.as<at::Tensor>()) == PrimDataType::Int) {
        return PrimDataType::Int;
      }
    }
  }
  return PrimDataType::Int32;
}

void KernelArgumentHolder::pushTensorProxy(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    at::ScalarType dtype) {
  TORCH_INTERNAL_ASSERT(strides.size() == sizes.size());
  auto meta_tensor = at::detail::empty_strided_meta(
      sizes,
      strides,
      dtype,
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt);
  arguments_.push_back(at::Tensor(meta_tensor));
}

std::vector<std::byte> getKernelArgumentData(
    const PolymorphicValue& argument,
    const DataType& dtype,
    PrimDataType index_type) {
  if (argument.is<Struct>()) {
    auto dtype_ = std::get<StructOf>(dtype.type);
    auto struct_ = argument.as<Struct>();
    std::vector<std::byte> buffer;
    for (const auto& field : dtype_.field_names) {
      auto field_data = getKernelArgumentData(
          struct_[field],
          NVFUSER_MAYBE_STAR dtype_.types.at(field),
          index_type);
      buffer.insert(buffer.end(), field_data.begin(), field_data.end());
    }
    return buffer;
  } else if (argument.is<at::Tensor>()) {
    auto tensor = argument.as<at::Tensor>();
    TORCH_INTERNAL_ASSERT(
        tensor.is_cpu() && tensor.numel() == 1,
        "Only CPU scalar tensors are supported here. ",
        "For GPU tensors, please use their metadata.");
    auto scalar_type = tensor.scalar_type();
    TORCH_INTERNAL_ASSERT(dtype == aten_to_data_type(scalar_type));
    std::vector<std::byte> buffer;
    buffer.reserve(tensor.element_size());
    buffer.insert(
        buffer.end(),
        (std::byte*)tensor.data_ptr(),
        (std::byte*)tensor.data_ptr() + tensor.element_size());
    return buffer;
  } else if (argument.is<Pointer>()) {
    TORCH_INTERNAL_ASSERT(std::holds_alternative<PointerOf>(dtype.type));
    void* ptr = (void*)argument;
    std::vector<std::byte> buffer;
    buffer.reserve(sizeof(void*));
    buffer.insert(
        buffer.end(), (std::byte*)&ptr, (std::byte*)&ptr + sizeof(void*));
    return buffer;
  } else if (argument.is<std::vector>()) {
    auto dtype_ = std::get<ArrayOf>(dtype.type);
    std::vector<std::byte> buffer;
    for (const auto& elem : argument.as<std::vector>()) {
      auto elem_data = getKernelArgumentData(elem, *dtype_.type, index_type);
      buffer.insert(buffer.end(), elem_data.begin(), elem_data.end());
    }
    return buffer;
  } else if (argument.is<int64_t>()) {
    int64_t v = argument.as<int64_t>();
    if (dtype == DataType::Int ||
        (index_type == PrimDataType::Int && dtype == DataType::Index)) {
      return std::vector<std::byte>((std::byte*)&v, (std::byte*)&v + 8);
    } else if (
        dtype == DataType::Int32 ||
        (index_type == PrimDataType::Int32 && dtype == DataType::Index)) {
      int32_t v32 = (int32_t)v;
      return std::vector<std::byte>((std::byte*)&v32, (std::byte*)&v32 + 4);
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Cannot convert int64_t to ",
          dtype,
          " type: only int32 and int64 are supported.");
    }
  } else if (argument.is<bool>()) {
    bool v = argument.as<bool>();
    TORCH_INTERNAL_ASSERT(dtype == DataType::Bool);
    return std::vector<std::byte>((std::byte*)&v, (std::byte*)&v + 1);
  } else if (argument.is<double>()) {
    double v = argument.as<double>();
    if (dtype == DataType::Double) {
      return std::vector<std::byte>(
          (std::byte*)&v, (std::byte*)&v + sizeof(double));
    } else if (dtype == DataType::Float) {
      float v32 = (float)v;
      return std::vector<std::byte>(
          (std::byte*)&v32, (std::byte*)&v32 + sizeof(float));
    } else if (dtype == DataType::Half) {
      at::Half v16 = (at::Half)(float)v;
      return std::vector<std::byte>(
          (std::byte*)&v16, (std::byte*)&v16 + sizeof(at::Half));
    } else if (dtype == DataType::BFloat16) {
      at::BFloat16 v16 = (at::BFloat16)(float)v;
      return std::vector<std::byte>(
          (std::byte*)&v16, (std::byte*)&v16 + sizeof(at::BFloat16));
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Cannot convert double to ",
          dtype,
          " type: only half, bfloat16, float and double are supported.");
    }
  } else if (argument.is<std::complex<double>>()) {
    std::complex<double> v = argument.as<std::complex<double>>();
    if (dtype == DataType::ComplexDouble) {
      return std::vector<std::byte>(
          (std::byte*)&v, (std::byte*)&v + sizeof(std::complex<double>));
    } else if (dtype == DataType::ComplexFloat) {
      std::complex<float> v32 = (std::complex<float>)v;
      return std::vector<std::byte>(
          (std::byte*)&v32, (std::byte*)&v32 + sizeof(std::complex<float>));
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Cannot convert complex double to ",
          dtype,
          " type: only complex float and complex double are supported.");
    }
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Cannot convert ",
        argument.type().name(),
        " to kernel argument data.");
  }
}

std::vector<std::byte> getKernelArgument(
    ExpressionEvaluator& ee,
    Val* parameter,
    PrimDataType index_type) {
  TORCH_INTERNAL_ASSERT(parameter != nullptr);
  PolymorphicValue pv = ee.evaluate(parameter);
  if (auto tv = dynamic_cast<TensorView*>(parameter)) {
    if (tv->isCpuScalar()) {
      return getKernelArgumentData(pv, tv->dtype(), index_type);
    } else {
      auto metadata_val = IrBuilder::metadataExpr(tv);
      auto metadata = ee.evaluate(metadata_val);
      return getKernelArgumentData(metadata, metadata_val->dtype(), index_type);
    }
  }
  return getKernelArgumentData(pv, parameter->dtype(), index_type);
}

} // namespace nvfuser
