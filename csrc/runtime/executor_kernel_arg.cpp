// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// Extract size and strides
#include <runtime/allocations.h>
#include <runtime/fusion_executor_cache.h>

#include <instrumentation.h>
#include <polymorphic_value.h>
#include <runtime/executor_kernel_arg.h>
#include <serde/polymorphic_value.h>
#include <tensor_metadata.h>

namespace nvfuser {

namespace {

PrimDataType getSmallestIndexType(const at::Tensor& tensor) {
  KernelIndexTypeCompute index_type_helper;
  for (const auto dim_i : arange(tensor.ndimension())) {
    auto size = tensor.size(dim_i);
    auto stride = tensor.stride(dim_i);
    if (index_type_helper.addDim(size, stride) == PrimDataType::Int) {
      return PrimDataType::Int;
    }
  }
  return PrimDataType::Int32;
}

} // namespace

void KernelArgumentHolder::push(const std::vector<PolymorphicValue>& args) {
  arguments_.insert(arguments_.end(), args.begin(), args.end());
}

void KernelArgumentHolder::push(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    arguments_.emplace_back(PolymorphicValue(tensor));
  }
}

void KernelArgumentHolder::push(const c10::ArrayRef<c10::IValue>& args) {
  for (const auto& arg : args) {
    arguments_.emplace_back(
        PolymorphicValue_functions::IValueToPolymorphicValue(arg));
  }
}

void KernelArgumentHolder::push(const std::vector<c10::IValue>& args) {
  for (const auto& arg : args) {
    arguments_.emplace_back(
        PolymorphicValue_functions::IValueToPolymorphicValue(arg));
  }
}

void KernelArgumentHolder::push(at::Tensor tensor) {
  arguments_.emplace_back(PolymorphicValue(tensor));
}

void KernelArgumentHolder::push(PolymorphicValue val) {
  arguments_.emplace_back(std::move(val));
}

void KernelArgumentHolder::push(std::optional<at::Tensor> tensor) {
  NVF_ERROR(
      tensor.has_value(),
      "KernelArgumentHolder doesn't support empty optional values, it's "
      "expected that when pushed they exist.");
  arguments_.emplace_back(PolymorphicValue(tensor.value()));
}

void KernelArgumentHolder::erase(const PolymorphicValue& arg_to_delete) {
  auto iter = std::remove_if(
      arguments_.begin(), arguments_.end(), [&](const auto& ref) {
        return &arg_to_delete == &ref;
      });
  arguments_.erase(iter, arguments_.end());
}

std::string KernelArgumentHolder::toString() const {
  std::stringstream ss;
  for (const auto& arg : arguments_) {
    if (arg.is<at::Tensor>()) {
      ss << debug_str(arg.as<at::Tensor>()) << "\n";
    } else {
      ss << PolymorphicValue_functions::toString(arg) << "\n";
    }
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
  NVF_ERROR(strides.size() == sizes.size());
  auto meta_tensor = at::empty_strided(
      sizes,
      strides,
      dtype,
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt);
  push(meta_tensor);
}

void KernelArgumentHolder::setDeviceIndex(std::optional<int8_t> index) {
  if (index.has_value()) {
    device_index_ = index.value();
  } else {
    device_index_ = getCommonDeviceCUDA(*this);
  }
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

  NVF_ERROR(buffer != nullptr, "serde::KernelArgumentHolder is nullptr.");

  device_index_ = buffer->device_index();
  cache_id_ = (buffer->cache_id() != SIZE_MAX)
      ? std::optional<size_t>(buffer->cache_id())
      : std::nullopt;

  serde::PolymorphicValueFactory poly_value_factory;
  for (auto fb_poly_value : *buffer->arguments()) {
    NVF_ERROR(fb_poly_value != nullptr, "serde::PolymorphicValue is nullptr.");
    push(poly_value_factory.parse(fb_poly_value->data_type(), fb_poly_value));
  }
}

std::vector<std::byte> polymorphicValueToBytes(
    const PolymorphicValue& argument,
    const DataType& dtype,
    PrimDataType index_type) {
  if (argument.is<at::Tensor>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(at::Tensor)");
    const auto& tensor = argument.as<at::Tensor>();
    NVF_ERROR(
        tensor.is_cpu() && tensor.numel() == 1,
        "Only CPU scalar tensors are supported here. ",
        "For GPU tensors, please use their metadata.");
    auto scalar_type = tensor.scalar_type();
    NVF_ERROR(
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
    NVF_ERROR(
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
    NVF_ERROR(
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
      NVF_THROW(
          "Cannot convert int64_t to ",
          dtype,
          " type: only int32 and int64 are supported.");
    }
  } else if (argument.is<bool>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(bool)");
    bool v = argument.as<bool>();
    NVF_ERROR(dtype == DataType::Bool, "Expected Bool type.");
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
    } else if (dtype == DataType::Float8_e4m3fn) {
      at::Float8_e4m3fn v8 = (at::Float8_e4m3fn)(float)v;
      return std::vector<std::byte>(
          (std::byte*)&v8, (std::byte*)&v8 + sizeof(at::Float8_e4m3fn));
    } else if (dtype == DataType::Float8_e5m2) {
      at::Float8_e5m2 v8 = (at::Float8_e5m2)(float)v;
      return std::vector<std::byte>(
          (std::byte*)&v8, (std::byte*)&v8 + sizeof(at::Float8_e5m2));
    } else {
      NVF_THROW(
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
      NVF_THROW(
          "Cannot convert complex double to ",
          dtype,
          " type: only complex float and complex double are supported.");
    }
  } else if (argument.is<StructHandle>()) {
    // FUSER_PERF_SCOPE("polymorphicValueToBytes(StructHandle)");
    std::vector<std::byte> buffer;
    if (argument.as<StructHandle>().is<TensorMetaData>()) {
      NVF_THROW(
          "Don't send tensor metadata to this function directly, use "
          "tensorToBytes.");
    } else {
      const auto& dtype_ = std::get<StructType>(dtype.type);
      for (const auto& field : dtype_.fields) {
        if (!field.used_in_kernel) {
          continue;
        }
        auto field_data = polymorphicValueToBytes(
            argument->*field.name, *field.type, index_type);
        buffer.insert(buffer.end(), field_data.begin(), field_data.end());
      }
      return buffer;
    }
  } else if (argument.is<Opaque>()) {
    return argument.as<Opaque>().bytes();
  } else {
    NVF_THROW(
        "Cannot convert ", argument.type().name(), " to kernel argument data.");
  }
}

std::vector<std::byte> tensorToBytes(
    const PolymorphicValue& argument,
    const std::vector<int64_t>& logical_sizes,
    const std::vector<int64_t>& alloc_strides,
    PrimDataType idx_type,
    AdjustLastDim adjust_last_dim,
    const std::vector<int64_t>& unsharded_logical_sizes) {
  std::vector<std::byte> bytes;
  NVF_ERROR(
      argument.is<at::Tensor>() && argument.as<at::Tensor>().is_cuda(),
      "Argument is not a CUDA tensor.");
  const auto& tensor = argument.as<at::Tensor>();
  auto data = tensor.data_ptr();

  const auto& size_to_use =
      logical_sizes.size() == unsharded_logical_sizes.size()
      ? unsharded_logical_sizes
      : logical_sizes;
  // special handle for TensorMetaData so that CPU overhead is minimal.
  if (idx_type == PrimDataType::Int) {
    bytes.reserve(
        sizeof(void*) + sizeof(int64_t) * size_to_use.size() +
        sizeof(int64_t) * alloc_strides.size());
    bytes.insert(bytes.end(), (std::byte*)&data, (std::byte*)(&data + 1));
    bytes.insert(
        bytes.end(),
        (std::byte*)size_to_use.data(),
        (std::byte*)size_to_use.data() + sizeof(int64_t) * size_to_use.size());

    // Adjust the last dimension of the logical domain to support DataType
    // that is not supported by PyTorch. See the comment of getLastDimAdjustment
    // in type.h for more details.
    if (!size_to_use.empty()) {
      int64_t& last_size = *reinterpret_cast<int64_t*>(
          bytes.data() + bytes.size() - sizeof(int64_t));
      last_size = adjust_last_dim.fromATenToNVF(last_size);
    } else {
      NVF_ERROR(
          adjust_last_dim.denominator == 1 && adjust_last_dim.numerator == 1,
          "DataType not supported");
    }

    bytes.insert(
        bytes.end(),
        (std::byte*)alloc_strides.data(),
        (std::byte*)alloc_strides.data() +
            sizeof(int64_t) * alloc_strides.size());
  } else {
    bytes.reserve(
        sizeof(void*) + sizeof(int32_t) * size_to_use.size() +
        sizeof(int32_t) * alloc_strides.size());
    bytes.insert(bytes.end(), (std::byte*)&data, (std::byte*)(&data + 1));
    std::vector<int32_t> logical_size32(size_to_use.begin(), size_to_use.end());

    // Adjust the last dimension of the logical domain to support DataType
    // that is not supported by PyTorch. See the comment of getLastDimAdjustment
    // in type.h for more details.
    if (!logical_size32.empty()) {
      int32_t& last_size = logical_size32.back();
      last_size = (int32_t)adjust_last_dim.fromATenToNVF(last_size);
    } else {
      NVF_ERROR(
          adjust_last_dim.denominator == 1 && adjust_last_dim.numerator == 1,
          "DataType not supported");
    }

    bytes.insert(
        bytes.end(),
        (std::byte*)logical_size32.data(),
        (std::byte*)logical_size32.data() +
            sizeof(int32_t) * logical_size32.size());
    std::vector<int32_t> alloc_stride32(
        alloc_strides.begin(), alloc_strides.end());
    bytes.insert(
        bytes.end(),
        (std::byte*)alloc_stride32.data(),
        (std::byte*)alloc_stride32.data() +
            sizeof(int32_t) * alloc_stride32.size());
  }
  return bytes;
}

int64_t computeBytes(const KernelArgumentHolder& args) {
  int64_t num_bytes = 0;
  // Figure how many bytes are inputs, outputs, and temporary buffers
  for (auto i : arange(args.size())) {
    if (args[i].is<at::Tensor>()) {
      auto t = args[i].as<at::Tensor>();
      num_bytes += static_cast<int64_t>(t.storage().nbytes());
    }
  }
  return num_bytes;
}

} // namespace nvfuser
