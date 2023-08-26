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
#include <instrumentation.h>
#include <serde/polymorphic_value_serde.h>

namespace nvfuser {

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
        return arg_to_delete == ref.get();
      });
  arguments_.erase(iter, arguments_.end());
}

std::string KernelArgumentHolder::toString() const {
  std::stringstream ss;
  for (const auto& arg : arguments_) {
    ss << *arg << "\n";
  }
  return ss.str();
}

PrimDataType KernelArgumentHolder::getSmallestIndexTypeOfArguments() const {
  for (const auto& arg : arguments_) {
    if (arg->is<at::Tensor>()) {
      if (getSmallestIndexType(arg->as<at::Tensor>()) == PrimDataType::Int) {
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
  at::Tensor meta_tensor = at::detail::empty_strided_meta(
      sizes,
      strides,
      dtype,
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt);
  push(std::move(meta_tensor));
}

flatbuffers::Offset<serde::KernelArgumentHolder> KernelArgumentHolder::
    serialize(flatbuffers::FlatBufferBuilder& builder) const {
  // See table definitions for KernelArgumentHolder and PolymorphicValue
  // in serde/fusion_cache.fbs

  using fb_poly_value = flatbuffers::Offset<serde::PolymorphicValue>;

  std::vector<fb_poly_value> arguments_fb;
  arguments_fb.reserve(arguments_.size());
  for (auto& arg : arguments_) {
    arguments_fb.push_back(serde::serializePolymorphicValue(builder, arg));
  }

  return serde::CreateKernelArgumentHolderDirect(
      builder, &arguments_fb, device_index_, cache_id_.value_or(SIZE_MAX));
}

void KernelArgumentHolder::deserialize(
    const serde::KernelArgumentHolder* buffer) {
  // See table definitions for KernelArgumentHolder and PolymorphicValue
  // in serde/fusion_cache.fbs

  TORCH_INTERNAL_ASSERT(
      buffer != nullptr, "serde::KernelArgumentHolder is nullptr.");

  device_index_ = buffer->device_index();
  cache_id_ = (buffer->cache_id() != SIZE_MAX)
      ? std::optional<size_t>(buffer->cache_id())
      : std::nullopt;

  serde::PolymorphicValueFactory poly_value_factory;
  for (auto fb_poly_value : *buffer->arguments()) {
    TORCH_INTERNAL_ASSERT(
        fb_poly_value != nullptr, "serde::PolymorphicValue is nullptr.");
    push(poly_value_factory.parse(fb_poly_value->data_type(), fb_poly_value));
  }
}

std::vector<std::byte> polymorphicValueToBytes(
    const PolymorphicValue& argument,
    const DataType& dtype,
    PrimDataType index_type) {
  if (argument.is<LegacyStruct>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(LegacyStruct)");
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<StructType>(dtype.type),
        "Expected StructType type.");
    auto dtype_ = std::get<StructType>(dtype.type);
    auto struct_ = argument.as<LegacyStruct>();
    std::vector<std::byte> buffer;
    for (const auto& field : dtype_.fields) {
      if (!field.used_in_kernel) {
        continue;
      }
      auto field_data =
          polymorphicValueToBytes(struct_[field.name], *field.type, index_type);
      buffer.insert(buffer.end(), field_data.begin(), field_data.end());
    }
    return buffer;
  } else if (argument.is<at::Tensor>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(at::Tensor)");
    const auto& tensor = argument.as<at::Tensor>();
    TORCH_INTERNAL_ASSERT(
        tensor.is_cpu() && tensor.numel() == 1,
        "Only CPU scalar tensors are supported here. ",
        "For GPU tensors, please use their metadata.");
    auto scalar_type = tensor.scalar_type();
    TORCH_INTERNAL_ASSERT(
        dtype == aten_to_data_type(scalar_type),
        "Expected ",
        dtype,
        " but got ",
        aten_to_data_type(scalar_type),
        ".");
    std::vector<std::byte> buffer;
    buffer.reserve(tensor.element_size());
    buffer.insert(
        buffer.end(),
        (std::byte*)tensor.data_ptr(),
        (std::byte*)tensor.data_ptr() + tensor.element_size());
    return buffer;
  } else if (argument.is<Pointer>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(Pointer)");
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<PointerType>(dtype.type),
        "Expected PointerType type.");
    void* ptr = (void*)argument;
    std::vector<std::byte> buffer;
    buffer.reserve(sizeof(void*));
    buffer.insert(
        buffer.end(), (std::byte*)&ptr, (std::byte*)&ptr + sizeof(void*));
    return buffer;
  } else if (argument.is<std::vector>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(std::vector)");
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<ArrayType>(dtype.type),
        "Expected ArrayType type.");
    auto dtype_ = std::get<ArrayType>(dtype.type);
    std::vector<std::byte> buffer;
    for (const auto& elem : argument.as<std::vector>()) {
      auto elem_data = polymorphicValueToBytes(elem, *dtype_.type, index_type);
      buffer.insert(buffer.end(), elem_data.begin(), elem_data.end());
    }
    return buffer;
  } else if (argument.is<int64_t>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(int64_t)");
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
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(bool)");
    bool v = argument.as<bool>();
    TORCH_INTERNAL_ASSERT(dtype == DataType::Bool, "Expected Bool type.");
    return std::vector<std::byte>((std::byte*)&v, (std::byte*)&v + 1);
  } else if (argument.is<double>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(double)");
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
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(std::complex<double>)");
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
  } else if (argument.is<Opaque>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(Opaque)");
    return argument.as<Opaque>().bytes();
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
  FUSER_PERF_SCOPE("getKernelArgument");
  TORCH_INTERNAL_ASSERT(parameter != nullptr);
  PolymorphicValue pv = ee.evaluate(parameter);
  if (auto tv = dynamic_cast<TensorView*>(parameter)) {
    if (tv->isCpuScalar()) {
      return polymorphicValueToBytes(pv, tv->dtype(), index_type);
    } else {
      auto metadata_val = IrBuilder::metadataExpr(tv);
      auto metadata = ee.evaluate(metadata_val);
      return polymorphicValueToBytes(
          metadata, metadata_val->dtype(), index_type);
    }
  }
  return polymorphicValueToBytes(pv, parameter->dtype(), index_type);
}

} // namespace nvfuser
