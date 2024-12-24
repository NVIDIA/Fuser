// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using EmbeddingTest = NVFuserTest;

constexpr int64_t n = 5, s = 2;

TEST_F(EmbeddingTest, Basic) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> inp_shape({s});
  std::vector<int64_t> weight_shape({n, s});

  auto tv_inp = makeConcreteTensor(inp_shape, DataType::Int);
  auto tv_weight = makeConcreteTensor(weight_shape, DataType::Half);

  fusion->addInput(tv_inp);
  fusion->addInput(tv_weight);

  auto tv_output = embedding(tv_inp, tv_weight, nullptr, nullptr, nullptr, nullptr, nullptr);
  fusion->addOutput(tv_output);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor input = at::randint(n, inp_shape, options.dtype(at::kLong));
  at::Tensor weight = at::randn(weight_shape, options.dtype(at::kHalf));

  namespace F = torch::nn::functional;
  auto aten_out = F::embedding(input, weight);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto nvf_out = executor_cache.runFusionWithInputs({input, weight});
  EXPECT_TRUE(at::allclose(nvf_out[0], aten_out));  
}

// INSTANTIATE_TEST_SUITE_P(
//     LinearWithoutBias,
//     LinearNodeParametrizedTest,
//     testing::Combine(
//         testing::Values(
//             Sizes({k}),
//             Sizes({m, k}),
//             Sizes({b, m, k}),
//             Sizes({1, k}),
//             Sizes({b, 1, k})),
//         testing::Values(Sizes({n, k}), Sizes({1, k})),
//         testing::Values(std::nullopt)));
        
} // namespace nvfuser