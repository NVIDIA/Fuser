// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <exceptions.h>
#include <executor_kernel_arg.h>
#include <ir/container.h>
#include <ir/serde.h>
#include <serde/factory.h>
#include <serde/fusion_cache_generated.h>
#include <functional>
#include <memory>

namespace nvfuser::serde {

//! The PolymorphicValueFactory class is used to deserialize the flatbuffer
//! PolymorphicValue table. This factory creates Bool, ComplexDouble, Double,
//! Long, CPU Scalar, and CUDA Tensor objects. These arguments are stored in
//! KernelArgumentHolder, which is used to schedule the fusion in
//! FusionKernelRuntime and to run a kernel in FusionExecutor.
class PolymorphicValueFactory
    : public Factory<PolymorphicValue, nvfuser::PolymorphicValue> {
 public:
  PolymorphicValueFactory()
      : Factory((nvfuser::toUnderlying(PolymorphicValueData::MAX) + 1)) {
    registerAllParsers();
  }

 private:
  void registerAllParsers();

  template <typename T>
  std::vector<T> makeArray(const serde::Array* data);

  nvfuser::PolymorphicValue makeArray(const serde::Array* data);
};

nvfuser::PolymorphicValue deserializePolymorphicValue(
    const PolymorphicValue* c);

void deserializeManagedData(
    nvfuser::Fusion* fusion,
    const PolymorphicValue* pv);

void deserializeManagedNamedData(
    nvfuser::Fusion* fusion,
    const std::string& name,
    const PolymorphicValue* pv);

nvfuser::PolymorphicValue makeScalar(const Scalar* c);

flatbuffers::Offset<PolymorphicValue> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    const IrSerde& container,
    const nvfuser::PolymorphicValue& v);

flatbuffers::Offset<PolymorphicValue> serializeBasicPolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v);

flatbuffers::Offset<PolymorphicValue> serializeTensor(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor);

flatbuffers::Offset<Scalar> serializeScalarCpu(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor);

flatbuffers::Offset<PolymorphicValue> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v);

flatbuffers::Offset<Scalar> serializeScalarRecord(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v,
    nvfuser::DataType t);

} // namespace nvfuser::serde
