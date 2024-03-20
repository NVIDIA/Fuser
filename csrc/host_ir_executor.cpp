// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/utils.h>
#include <host_ir_executor.h>


namespace nvfuser {

namespace hir {

HostIrExecutor::HostIrExecutor(std::unique_ptr<HostIrContainer> container): container_(std::move(container)){};

std::vector<at::Tensor> HostIrExecutor::runWithInput(const std::vector<c10::IValue>& inputs) {
  for (auto expr: container_->topLevelExprs()) {
    dispatch(expr);
  }
  return {};
}

void HostIrExecutor::handle(PostOnStream* post) {
  std::cout << "blabla" << std::endl;
}

} // namespace hir

} // namespace nvfuser
