// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <serde/arg_abstract_serde.h>

namespace nvfuser::serde {

namespace {

template <typename T>
std::unique_ptr<nvfuser::ArgAbstract> makeCpuScalarTensor(T value) {
  return std::make_unique<CpuScalarTensorArg<CpuScalarTensorCodegen<T>>>(value);
}

} // namespace

void ArgAbstractFactory::registerAllParsers() {
  auto deserializeBool = [](const serde::ArgAbstract* buffer) {
    return std::make_unique<BoolArg>(buffer->data_as_Bool()->bool_val());
  };
  registerParser(serde::ArgAbstractData_Bool, deserializeBool);

  auto deserializeDouble = [](const serde::ArgAbstract* buffer) {
    return std::make_unique<DoubleArg>(buffer->data_as_Double()->double_val());
  };
  registerParser(serde::ArgAbstractData_Double, deserializeDouble);

  auto deserializeInt = [](const serde::ArgAbstract* buffer) {
    return std::make_unique<LongArg>(buffer->data_as_Int()->int_val());
  };
  registerParser(serde::ArgAbstractData_Int, deserializeInt);

  auto deserializePhilox = [](const serde::ArgAbstract* buffer) {
    auto data = buffer->data_as_PhiloxCudaState();
    at::PhiloxCudaState state{data->seed(), data->offset()};
    return std::make_unique<PhiloxCudaStateArg>(state);
  };
  registerParser(serde::ArgAbstractData_PhiloxCudaState, deserializePhilox);

  auto deserializeComplexDouble = [](const serde::ArgAbstract* buffer) {
    auto data = buffer->data_as_ComplexDouble();
    c10::complex<double> number{data->real(), data->imag()};
    return std::make_unique<ComplexDoubleArg>(number);
  };
  registerParser(
      serde::ArgAbstractData_ComplexDouble, deserializeComplexDouble);

  auto deserializeScalarCpu = [](const serde::ArgAbstract* buffer) {
    auto scalar = buffer->data_as_ScalarCpu();

    switch (scalar->dtype()) {
      case serde::DataType_Bool: {
        auto value = scalar->data_as_Bool()->bool_val();
        return makeCpuScalarTensor<bool>(value);
      }
      case serde::DataType_Double: {
        auto value = scalar->data_as_Double()->double_val();
        return makeCpuScalarTensor<double>(value);
      }
      case serde::DataType_Float: {
        auto value = (float)scalar->data_as_Double()->double_val();
        return makeCpuScalarTensor<float>(value);
      }
      case serde::DataType_Half: {
        at::Half value{
            scalar->data_as_Half()->half_val(), at::Half::from_bits()};
        return makeCpuScalarTensor<at::Half>(value);
      }
      case serde::DataType_BFloat16: {
        at::BFloat16 value{
            scalar->data_as_Half()->half_val(), at::BFloat16::from_bits()};
        return makeCpuScalarTensor<at::BFloat16>(value);
      }
      case serde::DataType_Int: {
        auto value = scalar->data_as_Int()->int_val();
        return makeCpuScalarTensor<int64_t>(value);
      }
      case serde::DataType_Int32: {
        auto value = (int)scalar->data_as_Int()->int_val();
        return makeCpuScalarTensor<int>(value);
      }
      case serde::DataType_ComplexFloat: {
        auto data = buffer->data_as_ComplexDouble();
        c10::complex<float> value{(float)data->real(), (float)data->imag()};
        return makeCpuScalarTensor<c10::complex<float>>(value);
      }
      case serde::DataType_ComplexDouble: {
        auto data = buffer->data_as_ComplexDouble();
        c10::complex<double> value{data->real(), data->imag()};
        return makeCpuScalarTensor<c10::complex<double>>(value);
      }
      default:
        TORCH_INTERNAL_ASSERT(false, "Unexpected data type");
    }
  };
  registerParser(serde::ArgAbstractData_ScalarCpu, deserializeScalarCpu);
}

} // namespace nvfuser::serde
