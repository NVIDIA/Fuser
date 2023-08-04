// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/EmptyTensor.h>
#include <polymorphic_value.h>
#include <serde/arg_abstract_serde.h>
#include <utils.h>

namespace nvfuser::serde {

namespace {

PolymorphicValue makeCpuScalarTensor(const serde::ScalarCpu* scalar_cpu) {
  TORCH_INTERNAL_ASSERT(scalar_cpu != nullptr);
  auto scalar = parsePolymorphicValue(scalar_cpu->scalar_value());
  return nvfuser::PolymorphicValue_functions::toTensor(scalar, at::kCPU);
}

PolymorphicValue getAbstractTensorArg(const serde::TensorArg* tensor) {
  TORCH_INTERNAL_ASSERT(tensor != nullptr);
  if (tensor->strides() != nullptr) {
    auto meta_tensor = at::detail::empty_strided_meta(
        parseVector(tensor->sizes()),
        parseVector(tensor->strides()),
        mapToAtenDtype(tensor->dtype()),
        c10::nullopt,
        c10::Device(c10::DeviceType::Meta, 0),
        c10::nullopt);
    return at::Tensor(meta_tensor);
  }
  return at::empty(
      parseVector(tensor->sizes()),
      mapToAtenDtype(tensor->dtype()),
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt,
      c10::nullopt);
}

} // namespace

void ArgAbstractFactory::registerAllParsers() {
  auto deserializeScalar = [](const serde::ArgAbstract* buffer) {
    return parsePolymorphicValue(buffer->data_as_Scalar());
  };
  registerParser(serde::ArgAbstractData_Scalar, deserializeScalar);

  auto deserializeScalarCpu = [](const serde::ArgAbstract* buffer) {
    return makeCpuScalarTensor(buffer->data_as_ScalarCpu());
  };
  registerParser(serde::ArgAbstractData_ScalarCpu, deserializeScalarCpu);

  auto deserializeTensorArg = [](const serde::ArgAbstract* buffer) {
    return getAbstractTensorArg(buffer->data_as_TensorArg());
  };
  registerParser(serde::ArgAbstractData_TensorArg, deserializeTensorArg);
}

} // namespace nvfuser::serde
