// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

namespace hir {

using MultiDeviceHostIrIntegrationTestParams = std::tuple<bool, bool>;

class MultiDeviceHostIrIntegrationTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<MultiDeviceHostIrIntegrationTestParams> {};

TEST_P(MultiDeviceHostIrIntegrationTest, test_kernel) {
  //auto [use_fusion_executor_cache, with_sharding_annotations] = GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    Manual,
    MultiDeviceHostIrIntegrationTest,
    testing::Combine(testing::Bool(), testing::Bool()),
    [](const testing::TestParamInfo<MultiDeviceHostIrIntegrationTestParams>& info)
        -> std::string {
      std::string s;
      s += std::get<0>(info.param) ? "useFusionExecutorCache"
                                   : "useFusionExecutor";
      s += "_";
      s += std::get<1>(info.param) ? "withShardingAnnotations"
                                   : "withoutShardingAnnotations";
      return s;
    });

} // namespace hir

} // namespace nvfuser
