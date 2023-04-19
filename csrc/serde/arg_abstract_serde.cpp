// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <serde/arg_abstract_serde.h>
#include <utils.h>

namespace nvfuser::serde {

namespace {

template <typename T>
std::unique_ptr<nvfuser::ArgAbstract> makeCpuScalarTensor(T value) {
  return std::make_unique<CpuScalarTensorArg<CpuScalarTensorCodegen<T>>>(value);
}

template <typename T, typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const serde::TensorArg* tensor) {
  switch (tensor->ndims()) {
    case (0):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 0, nvfuser_index_t>>>(tensor);
    case (1):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 1, nvfuser_index_t>>>(tensor);
    case (2):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 2, nvfuser_index_t>>>(tensor);
    case (3):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 3, nvfuser_index_t>>>(tensor);
    case (4):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 4, nvfuser_index_t>>>(tensor);
    case (5):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 5, nvfuser_index_t>>>(tensor);
    case (6):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 6, nvfuser_index_t>>>(tensor);
    case (7):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 7, nvfuser_index_t>>>(tensor);
    case (8):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<T, 8, nvfuser_index_t>>>(tensor);
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          tensor->ndims(),
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

template <typename nvfuser_index_t>
struct GetTensorArgWithNativeType {
  template <typename T>
  std::unique_ptr<TensorArgAbstract> operator()(
      const serde::TensorArg* tensor) {
    return getTensorArg<T, nvfuser_index_t>(tensor);
  };
};

template <typename INDEX_MODE>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const serde::TensorArg* tensor) {
  return atenTypeDispatchWithC10Complex(
      mapToAtenDtype(tensor->dtype()),
      GetTensorArgWithNativeType<INDEX_MODE>(),
      tensor);
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

  auto deserializeTensorArg = [](const serde::ArgAbstract* buffer) {
    auto tensor = buffer->data_as_TensorArg();
    if (tensor->is_int_index_mode()) {
      return getTensorArg<int>(tensor);
    }
    return getTensorArg<int64_t>(tensor);
  };
  registerParser(serde::ArgAbstractData_TensorArg, deserializeTensorArg);
}

} // namespace nvfuser::serde
