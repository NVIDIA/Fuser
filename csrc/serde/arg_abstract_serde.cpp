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

template <size_t size>
std::unique_ptr<nvfuser::ArgAbstract> makeCpuScalarTensor(
    const serde::ScalarCpu* scalar) {
  auto ptr = std::make_unique<CpuScalarTensorArg<size>>();
  static_assert(sizeof(ptr->instance_) == size);
  std::memcpy(&(ptr->instance_), scalar->instance()->data(), size);
  return ptr;
}

template <typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const serde::TensorArg* tensor) {
  auto meta_tensor = at::native::empty_strided_cpu(
      parseVector(tensor->sizes()),
      parseVector(tensor->strides()),
      mapToAtenDtype(tensor->dtype()),
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt);
  switch (tensor->sizes()->size()) {
    case (0):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<0, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (1):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<1, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (2):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<2, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (3):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<3, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (4):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<4, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (5):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<5, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (6):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<6, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (7):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<7, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    case (8):
      return std::make_unique<
          nvfuser::TensorArg<TensorArgCodegen<8, nvfuser_index_t>>>(
          meta_tensor, nullptr /* tv */);
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          tensor->sizes()->size(),
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

std::unique_ptr<TensorArgAbstract> getAbstractTensorArg(
    const serde::TensorArg* tensor) {
  auto meta_tensor = at::empty(
      parseVector(tensor->sizes()),
      mapToAtenDtype(tensor->dtype()),
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt,
      c10::nullopt);
  return std::make_unique<TensorArgAbstract>(meta_tensor);
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
    auto tensor = buffer->data_as_TensorArg();
    if (tensor->index_type_resolved()) {
      if (tensor->is_int_index_mode()) {
        return getTensorArg<int>(tensor);
      }
      return getTensorArg<int64_t>(tensor);
    }
    return getAbstractTensorArg(tensor);
  };
  registerParser(serde::ArgAbstractData_TensorArg, deserializeTensorArg);
}

} // namespace nvfuser::serde
