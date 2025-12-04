
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <exceptions.h>
#include <runtime/executor_kernel_arg.h>
#include <serde/factory.h>
#include <serde/fusion_cache_generated.h>
#include <visibility.h>
#include <memory>

namespace nvfuser::serde {

//! The PolymorphicValueFactory class is used to deserialize the flatbuffer
//! PolymorphicValue table. This factory creates Bool, ComplexDouble, Double,
//! Long, CPU Scalar, and CUDA Tensor objects. These arguments are stored in
//! KernelArgumentHolder, which is used to schedule the fusion in
//! FusionKernelRuntime and to run a kernel in KernelExecutor.
class PolymorphicValueFactory
    : public Factory<PolymorphicValue, nvfuser::PolymorphicValue> {
 public:
  PolymorphicValueFactory()
      : Factory((nvfuser::toUnderlying(PolymorphicValueData::MAX) + 1)) {
    registerAllParsers();
  }

 private:
  void registerAllParsers();
};

nvfuser::PolymorphicValue deserializePolymorphicValue(const Scalar* c);

flatbuffers::Offset<PolymorphicValue> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v);

flatbuffers::Offset<Scalar> serializeScalarCpu(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor);

NVF_API flatbuffers::Offset<Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v,
    nvfuser::DataType t);

} // namespace nvfuser::serde
