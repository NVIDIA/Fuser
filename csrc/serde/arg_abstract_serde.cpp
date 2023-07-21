// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/EmptyTensor.h>
#include <serde/arg_abstract_serde.h>
#include <utils.h>

namespace nvfuser::serde {

namespace {

template <size_t size>
std::unique_ptr<nvfuser::ArgAbstract> makeCpuScalarTensor(
    const serde::ScalarCpu* scalar) {
  TORCH_INTERNAL_ASSERT(scalar != nullptr);
  auto ptr = std::make_unique<CpuScalarTensorArg<size>>();
  static_assert(sizeof(ptr->instance_) == size);
  std::memcpy(&(ptr->instance_), scalar->instance()->data(), size);
  return ptr;
}

std::unique_ptr<TensorArgAbstract> getAbstractTensorArg(
    const serde::TensorArg* tensor) {
  TORCH_INTERNAL_ASSERT(tensor != nullptr);
  if (tensor->strides() != nullptr) {
    auto meta_tensor = at::detail::empty_strided_meta(
        parseVector(tensor->sizes()),
        parseVector(tensor->strides()),
        mapToAtenDtype(tensor->dtype()),
        c10::nullopt,
        c10::Device(c10::DeviceType::Meta, 0),
        c10::nullopt);
    return std::make_unique<TensorArgAbstract>(
        at::Tensor(meta_tensor), tensor->ptr());
  }
  auto meta_tensor = at::empty(
      parseVector(tensor->sizes()),
      mapToAtenDtype(tensor->dtype()),
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt,
      c10::nullopt);
  return std::make_unique<TensorArgAbstract>(meta_tensor);
}

std::unique_ptr<nvfuser::ArgAbstract> getAbstractScalar(
    const serde::Scalar* scalar) {
  TORCH_INTERNAL_ASSERT(scalar != nullptr);
  if (!scalar->has_value()) {
    return nullptr;
  } else if (scalar->value_type() == serde::DataType_Double) {
    return std::make_unique<DoubleArg>(scalar->double_value());
  } else if (scalar->value_type() == serde::DataType_Int) {
    return std::make_unique<LongArg>(scalar->long_value());
  } else if (scalar->value_type() == serde::DataType_Bool) {
    return std::make_unique<BoolArg>(scalar->bool_value());
  } else if (scalar->value_type() == serde::DataType_ComplexDouble) {
    c10::complex<double> number{scalar->real_value(), scalar->imag_value()};
    return std::make_unique<ComplexDoubleArg>(number);
  }
  TORCH_INTERNAL_ASSERT(
      false, "Unable to deserialize serde::Scalar as PolymorphicValue.");
}

} // namespace

void ArgAbstractFactory::registerAllParsers() {
  auto deserializeScalar = [](const serde::ArgAbstract* buffer) {
    return getAbstractScalar(buffer->data_as_Scalar());
  };
  registerParser(serde::ArgAbstractData_Scalar, deserializeScalar);

  auto deserializePhilox = [](const serde::ArgAbstract* buffer) {
    auto data = buffer->data_as_PhiloxCudaState();
    at::PhiloxCudaState state{data->seed(), data->offset()};
    return std::make_unique<PhiloxCudaStateArg>(state);
  };
  registerParser(serde::ArgAbstractData_PhiloxCudaState, deserializePhilox);

  auto deserializeScalarCpu = [](const serde::ArgAbstract* buffer) {
    auto scalar = buffer->data_as_ScalarCpu();
    switch (scalar->size()) {
      case 1:
        return makeCpuScalarTensor<1>(scalar);
      case 2:
        return makeCpuScalarTensor<2>(scalar);
      case 4:
        return makeCpuScalarTensor<4>(scalar);
      case 8:
        return makeCpuScalarTensor<8>(scalar);
      case 16:
        return makeCpuScalarTensor<16>(scalar);
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unexpected data type size");
    }
  };
  registerParser(serde::ArgAbstractData_ScalarCpu, deserializeScalarCpu);

  auto deserializeTensorArg = [](const serde::ArgAbstract* buffer) {
    return getAbstractTensorArg(buffer->data_as_TensorArg());
  };
  registerParser(serde::ArgAbstractData_TensorArg, deserializeTensorArg);
}

} // namespace nvfuser::serde
