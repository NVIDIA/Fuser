
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
