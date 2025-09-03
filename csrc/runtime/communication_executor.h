// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <host_ir/container.h>
#include <multidevice/communicator.h>
#include <runtime/executor_abstract.h>
#include <runtime/executor_kernel_arg.h>

namespace nvfuser {

class CommunicationExecutor : public ExecutorAbstract {
 public:
  CommunicationExecutor(
      int64_t fusion_id = 0,
      int64_t concrete_id = 0,
      int64_t runtime_id = 0,
      int64_t group_id = 0);

  static bool supported(Fusion* fusion);

  void compile(Fusion* fusion);

  bool isCompiled() const override;

  NVF_API KernelArgumentHolder
  run(const KernelArgumentHolder& args, KernelArgumentHolder outputs = {});

 private:
  std::unique_ptr<hir::HostIrContainer> host_ir_container_;
  Communicator* communicator_;
};

} // namespace nvfuser
