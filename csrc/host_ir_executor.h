// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>
#include <executor.h>
#include <host_ir_container.h>
#include <ir/host_ir.h>

namespace nvfuser {

namespace hir {

class HostIrExecutor final : public OptInDispatch {

 public:
  HostIrExecutor(std::unique_ptr<HostIrContainer> container);
  std::vector<at::Tensor> runWithInput(const std::vector<c10::IValue>& inputs);

 private:
  using OptInDispatch::handle;
  void handle(PostOnStream* post) override;

  std::unique_ptr<HostIrContainer> container_;
};

} // namespace hir

} // namespace nvfuser
