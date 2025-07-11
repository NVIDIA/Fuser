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
#include <memory>
namespace nvfuser {

constexpr int64_t kHostIrJitCompileThreads = 4;
struct HostIrJitImpl;

class HostIrJit {
 public:
  HostIrJit(
      std::unique_ptr<hir::HostIrContainer> container,
      int num_threads = kHostIrJitCompileThreads);

  KernelArgumentHolder runWithInputs(const KernelArgumentHolder& args);

  KernelArgumentHolder runWithInput(
      const std::unordered_map<Val*, PolymorphicValue>& val_to_PValue);

  const std::vector<Val*>& inputs() const;
  const std::vector<Val*>& outputs() const;
  auto* container() const;
  const hir::HostIrContainer& getHostIrContainer() const;
  std::ostream& print(std::ostream& os) const;

  ~HostIrJit();

 private:
  std::unique_ptr<HostIrJitImpl> pimpl_;
};
} // namespace nvfuser
