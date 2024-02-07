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
#include <mma_type.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/mma_utils.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class MatmulDefaultTest : public NVFuserTest {};

TEST (MatmulDefaultTest, ComputeMmaThroughEE){
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    EnableOptionsGuard::getCurOptions().set(EnableOption::MatmulExprEval);

    int64_t m = 2, k = 3, n = 4;
    std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

    auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
    auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
    auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
    auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
    auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addOutput(tv2);

    at::Tensor t0 = at::ones(a_shape, at::kHalf).cuda();
    at::Tensor t1 = at::ones(b_shape, at::kHalf).cuda();
    at::Tensor out_ref = at::full(out_shape, k, at::kFloat).cuda();

    FusionExecutorCache fec(std::move(fusion));
    auto out = fec.runFusionWithInputs({t0, t1});

    EXPECT_TRUE(at::allclose(out[0], out_ref));
}

} // namespace nvfuser