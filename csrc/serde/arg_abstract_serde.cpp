// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <serde/arg_abstract_serde.h>

namespace nvfuser::serde {

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
    c10::complex number{data->real(), data->imag()};
    return std::make_unique<ComplexDoubleArg>(number);
  };
  registerParser(
      serde::ArgAbstractData_ComplexDouble, deserializeComplexDouble);
}

} // namespace nvfuser::serde
