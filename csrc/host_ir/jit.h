// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <memory>

#include "fusion.h"
#include "host_ir/container.h"
#include "multidevice/communicator.h"
namespace nvfuser {

constexpr int64_t kHostIrJitCompileThreads = 4;
struct HostIrJitImpl;

class HostIrJit {
 public:
  HostIrJit(
      std::unique_ptr<hir::HostIrContainer> container,
      int num_threads = kHostIrJitCompileThreads);

  KernelArgumentHolder runWithInputs(const KernelArgumentHolder& args);

  const std::vector<Val*>& inputs() const;
  const std::vector<Val*>& outputs() const;
  const hir::HostIrContainer& container() const;
  ~HostIrJit();

 private:
  std::unique_ptr<HostIrJitImpl> pimpl_;
};
} // namespace nvfuser
