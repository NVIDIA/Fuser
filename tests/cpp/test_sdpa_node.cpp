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

using Sizes = std::vector<int64_t>;
using SDPAParamType = std::tuple<std::optional<Sizes>, DataType>;
using SDPAParametrizedTest = NVFuserFixtureParamTest<SDPAParamType>;
using SDPATest = NVFuserTest;

constexpr int64_t n = 16, h = 32, l = 64, s = 128, e = 32, ev = 32;

// Check that ID exact mapping works as expected
void checkSdpaOpIdMapping(
    Fusion* fusion,
    TensorView* query,
    TensorView* key,
    TensorView* value,
    TensorView* attn_mask,
    TensorView* output) {
    
    IdModel id_model(fusion);
    const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);
    vg.validateConsistency();

    ASSERT_EQ(output->nDims(), query->nDims());
    // Query: [N,..,L,E], Key: [N,..,S,E], Value: [N,..,S,Ev], Attn_mask = null/[N,..,L,S]/[L,S]/(any broadcastable input)
    // Output: [N,..,L,Ev];

    std::vector<TensorView*> input_tvs ({query, key, value});
    if (attn_mask != nullptr) {
        input_tvs.emplace_back(attn_mask);
    }
    // Check mapping for the first out_size - 2 ids for all roles. attn_mask can have different dimension so start from innermost position.  
    auto out_size = (int64_t)output->nDims();
    for (auto tv: input_tvs){
        auto inp_size = (int64_t)tv->nDims();
        for (auto out_idx = out_size - 3, inp_idx = inp_size - 3; inp_idx >= 0; inp_idx--, out_idx--) {
            if (!tv->axis(inp_idx)->isBroadcast()){
                EXPECT_TRUE(checkMapped(vg, tv->axis(inp_idx), output->axis(out_idx)));
            }
        }
    }
    // Check mapping for L
    if (!query->axis(-2)->isBroadcast()){
        EXPECT_TRUE(checkMapped(vg, query->axis(-2), output->axis(-2)));
    }
    if (attn_mask != nullptr && !attn_mask->axis(-2)->isBroadcast()){
        EXPECT_TRUE(checkMapped(vg, attn_mask->axis(-2), output->axis(-2)));
    }
    // Check mapping for Ev
    if (!value->axis(-1)->isBroadcast()){
        EXPECT_TRUE(checkMapped(vg, value->axis(-1), output->axis(-1)));
    }
}

TEST_P (SDPAParametrizedTest, NonCausalAttnConcrete) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    std::vector<int64_t> q_shape ({n, h, l, e});
    std::vector<int64_t> k_shape ({n, h, s, e});
    std::vector<int64_t> v_shape ({n, h, s, ev});

    const auto& [mask_shape, mask_dtype] = GetParam();

    auto tvq = makeConcreteTensor(q_shape, DataType::Half);
    auto tvk = makeConcreteTensor(k_shape, DataType::Half);
    auto tvv = makeConcreteTensor(k_shape, DataType::Half);

    fusion->addInput(tvq);
    fusion->addInput(tvk);
    fusion->addInput(tvv);

    TensorView* tvmask = nullptr;
    if (mask_shape.has_value()){
        tvmask = makeConcreteTensor(*mask_shape, mask_dtype);
        fusion->addInput(tvmask);
    }
    auto tvattn = sdpa(tvq, tvk, tvv, tvmask, /*dropout_p=*/0.0, /*is_causal=*/false, /*scale=*/std::nullopt);
    fusion->addOutput(tvattn);

    checkSdpaOpIdMapping(fusion.get(), tvq, tvk, tvv, tvmask, tvattn);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
    at::Tensor q = at::randn(q_shape, options);
    at::Tensor k = at::randn(k_shape, options);
    at::Tensor v = at::randn(v_shape, options);
    std::optional<at::Tensor> attn_mask = std::nullopt;
    if (mask_shape.has_value()){
        if (mask_dtype == DataType::Bool){
            attn_mask = at::randint(0, 2, *mask_shape, data_type_to_aten(mask_dtype)).cuda();
        } else {
            attn_mask = at::rand(*mask_shape, data_type_to_aten(mask_dtype)).cuda();
        }
    }
    at::Tensor out_ref = at::scaled_dot_product_attention(q, k, v, attn_mask);

    FusionExecutorCache fec(std::move(fusion));
    std::vector<at::Tensor> out = {};
    if (mask_shape.has_value()){
        out = fec.runFusionWithInputs({q, k, v, attn_mask});
    } else {
        out = fec.runFusionWithInputs({q, k, v});
    }

    EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST_P (SDPAParametrizedTest, NonCausalAttnSymbolic) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    std::vector<int64_t> q_shape ({n, h, l, e});
    std::vector<int64_t> k_shape ({n, h, s, e});
    std::vector<int64_t> v_shape ({n, h, s, ev});

    const auto& [mask_shape, mask_dtype] = GetParam();

    auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
    auto tvk = makeSymbolicTensor(k_shape, DataType::Half);
    auto tvv = makeSymbolicTensor(k_shape, DataType::Half);

    fusion->addInput(tvq);
    fusion->addInput(tvk);
    fusion->addInput(tvv);

    TensorView* tvmask = nullptr;
    if (mask_shape.has_value()){
        tvmask = makeSymbolicTensor(*mask_shape, mask_dtype);
        fusion->addInput(tvmask);
    }
    auto tvattn = sdpa(tvq, tvk, tvv, tvmask, /*dropout_p=*/0.0, /*is_causal=*/false, /*scale=*/std::nullopt);
    fusion->addOutput(tvattn);

    checkSdpaOpIdMapping(fusion.get(), tvq, tvk, tvv, tvmask, tvattn);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
    at::Tensor q = at::randn(q_shape, options);
    at::Tensor k = at::randn(k_shape, options);
    at::Tensor v = at::randn(v_shape, options);
    std::optional<at::Tensor> attn_mask = std::nullopt;
    if (mask_shape.has_value()){
        if (mask_dtype == DataType::Bool){
            attn_mask = at::randint(0, 2, *mask_shape, data_type_to_aten(mask_dtype)).cuda();
        } else {
            attn_mask = at::rand(*mask_shape, data_type_to_aten(mask_dtype)).cuda();
        }
    }
    at::Tensor out_ref = at::scaled_dot_product_attention(q, k, v, attn_mask);

    FusionExecutorCache fec(std::move(fusion));
    std::vector<at::Tensor> out = {};
    if (mask_shape.has_value()){
        out = fec.runFusionWithInputs({q, k, v, attn_mask});
    } else {
        out = fec.runFusionWithInputs({q, k, v});
    }

    EXPECT_TRUE(at::allclose(out[0], out_ref));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SDPAParametrizedTest,
    testing::Combine(
        testing::Values(std::nullopt, Sizes({l, s}), Sizes({h, l, s}), Sizes({n, 1, l, s})),
        testing::Values(DataType::Bool, DataType::Half))
);

TEST (SDPATest, CausalAttn) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    std::vector<int64_t> q_shape ({n, h, l, e});
    std::vector<int64_t> k_shape ({n, h, s, e});
    std::vector<int64_t> v_shape ({n, h, s, ev});

    auto tvq = makeSymbolicTensor(q_shape, DataType::Half);
    auto tvk = makeSymbolicTensor(k_shape, DataType::Half);
    auto tvv = makeSymbolicTensor(k_shape, DataType::Half);

    fusion->addInput(tvq);
    fusion->addInput(tvk);
    fusion->addInput(tvv);

    auto tvattn = sdpa(tvq, tvk, tvv, /*attn_mask=*/nullptr, /*dropout_p=*/0.0, /*is_causal=*/true, /*scale=*/1e-3);
    fusion->addOutput(tvattn);

    checkSdpaOpIdMapping(fusion.get(), tvq, tvk, tvv, /*attn_mask=*/nullptr, tvattn);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
    at::Tensor q = at::randn(q_shape, options);
    at::Tensor k = at::randn(k_shape, options);
    at::Tensor v = at::randn(v_shape, options);
    at::Tensor out_ref = at::scaled_dot_product_attention(q, k, v, /*attn_mask=*/std::nullopt, /*dropout_p=*/0.0, /*is_causal=*/true, /*scale=*/1e-3);

    FusionExecutorCache fec(std::move(fusion));
    auto out = fec.runFusionWithInputs({q, k, v});

    EXPECT_TRUE(at::allclose(out[0], out_ref));
}
}