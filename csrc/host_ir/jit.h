// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h>
#include <host_ir/container.h>
#include <multidevice/communicator.h>
#include <utils.h>
#include <memory>
namespace nvfuser {

// Forward declaration
struct LlvmJitImpl;

class HostIrJit {
 public:
  KernelArgumentHolder runWithInputs(
      const KernelArgumentHolder& args);
    
  KernelArgumentHolder runWithInput(
      const std::unordered_map<Val*, PolymorphicValue>& val_to_PValue);

  HostIrJit(
      std::unique_ptr<hir::HostIrContainer> container,
      int num_threads = kHostIrJitCompileThreads);

  ~HostIrJit();

 private:
  std::unique_ptr<LlvmJitImpl> pimpl_;
};
} // namespace nvfuser
