
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <executor_kernel_arg.h>
#include <serde/factory.h>
#include <serde/fusion_cache_generated.h>
#include <functional>
#include <memory>

namespace nvfuser::serde {

//! The ArgAbstractFactory class is used to deserialize the flatbuffer
//! ArgAbstract table. This factory creates Bool, ComplexDouble, Double, Long,
//! PhiloxCudaState, ScalarCpu, TensorArgAbstract objects. These arguments are
//! stored in KernelArgumentHolder, which is used to schedule the fusion in
//! FusionKernelRuntime and to run a kernel in FusionExecutor.
class ArgAbstractFactory : public Factory<
                               serde::ArgAbstract,
                               std::unique_ptr<nvfuser::ArgAbstract>> {
 public:
  ArgAbstractFactory() : Factory((serde::ArgAbstractData_MAX + 1)) {
    registerAllParsers();
  }

 private:
  void registerAllParsers();
};

} // namespace nvfuser::serde
