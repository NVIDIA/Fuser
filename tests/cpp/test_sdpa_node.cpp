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
#include <id_model/id_model.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/mma_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using SDPATest = NVFuserTest;

constexpr int64_t n = 16, h = 32, l = 64, s = 128, e = 32, ev = 32;

TEST (SDPATest, Concrete) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    std::vector<int64_t> q_shape ({n, h, l, e});
    std::vector<int64_t> k_shape ({n, h, s, e});
    std::vector<int64_t> v_shape ({n, h, s, ev});

    auto tvq = makeConcreteTensor(q_shape, DataType::Half);
    auto tvk = makeConcreteTensor(k_shape, DataType::Half);
    auto tvv = makeConcreteTensor(k_shape, DataType::Half);
    auto attn = sdpa(tvq, tvk, tvv, nullptr, /*dropout_p=*/0.0, /*is_causal=*/false, std::nullopt);

    fusion->addInput(tvq);
    fusion->addInput(tvk);
    fusion->addInput(tvv);
    fusion->addOutput(attn);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
    at::Tensor q = at::randn(q_shape, options);
    at::Tensor k = at::randn(k_shape, options);
    at::Tensor v = at::randn(v_shape, options);
    at::Tensor out_ref = at::scaled_dot_product_attention(q, k, v);

    FusionExecutorCache fec(std::move(fusion));
    auto out = fec.runFusionWithInputs({q, k, v});

    EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST (SDPATest, Symbolic) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    std::vector<int64_t> q_shape ({n, h, l, e});
    std::vector<int64_t> k_shape ({n, h, s, e});
    std::vector<int64_t> v_shape ({n, h, s, ev});

    auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
    auto tvk = makeSymbolicTensor(k_shape, DataType::Half);
    auto tvv = makeSymbolicTensor(k_shape, DataType::Half);
    auto attn = sdpa(tvq, tvk, tvv, nullptr, /*dropout_p=*/0.0, /*is_causal=*/false, std::nullopt);

    fusion->addInput(tvq);
    fusion->addInput(tvk);
    fusion->addInput(tvv);
    fusion->addOutput(attn);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
    at::Tensor q = at::randn(q_shape, options);
    at::Tensor k = at::randn(k_shape, options);
    at::Tensor v = at::randn(v_shape, options);
    at::Tensor out_ref = at::scaled_dot_product_attention(q, k, v);

    FusionExecutorCache fec(std::move(fusion));
    auto out = fec.runFusionWithInputs({q, k, v});

    EXPECT_TRUE(at::allclose(out[0], out_ref));
}


}