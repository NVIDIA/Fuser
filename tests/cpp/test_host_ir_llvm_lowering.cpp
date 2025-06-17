// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <gmock/gmock-more-matchers.h>

#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <host_ir/lower_to_llvm.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using testing::Contains;
using HostIrLLVMTest = NVFuserTest;
// Build with: python setup.py install --build-with-llvm
// NVFUSER_ENABLE=host_ir_lowering ./bin/test_host_ir
// --gtest_filter=HostIrLLVMTest.TestLLVMJIT
TEST_F(HostIrLLVMTest, TestLLVMJIT) {
  HostIrLlvmJit::getInstance(4).compile(nullptr);
}

} // namespace hir

} // namespace nvfuser
