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
    return std::make_unique<BoolArg>(buffer->data_as_Bool()->value());
  };
  registerParser(serde::ArgAbstractData_Bool, deserializeBool);

  auto deserializeDouble = [](const serde::ArgAbstract* buffer) {
    return std::make_unique<DoubleArg>(buffer->data_as_Double()->value());
  };
  registerParser(serde::ArgAbstractData_Double, deserializeDouble);

  auto deserializeLong = [](const serde::ArgAbstract* buffer) {
    return std::make_unique<LongArg>(buffer->data_as_Long()->value());
  };
  registerParser(serde::ArgAbstractData_Long, deserializeLong);

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
    switch (scalar->data_type()) {
      case serde::ScalarCpuData_Bool: {
        auto value = scalar->data_as_Bool()->value();
        return makeCpuScalarTensor<bool>(value);
      }
      case serde::ScalarCpuData_Double: {
        auto value = scalar->data_as_Double()->value();
        return makeCpuScalarTensor<double>(value);
      }
      case serde::ScalarCpuData_Float: {
        auto value = scalar->data_as_Float()->value();
        return makeCpuScalarTensor<float>(value);
      }
      case serde::ScalarCpuData_Half: {
        at::Half value{scalar->data_as_Half()->value(), at::Half::from_bits()};
        return makeCpuScalarTensor<at::Half>(value);
      }
      case serde::ScalarCpuData_BFloat16: {
        at::BFloat16 value{
            scalar->data_as_Half()->value(), at::BFloat16::from_bits()};
        return makeCpuScalarTensor<at::BFloat16>(value);
      }
      case serde::ScalarCpuData_Long: {
        auto value = scalar->data_as_Long()->value();
        return makeCpuScalarTensor<int64_t>(value);
      }
      case serde::ScalarCpuData_Int: {
        auto value = scalar->data_as_Int()->value();
        return makeCpuScalarTensor<int>(value);
      }
      case serde::ScalarCpuData_ComplexFloat: {
        auto data = scalar->data_as_ComplexFloat();
        c10::complex<float> value{data->real(), data->imag()};
        return makeCpuScalarTensor<c10::complex<float>>(value);
      }
      case serde::ScalarCpuData_ComplexDouble: {
        auto data = scalar->data_as_ComplexDouble();
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
