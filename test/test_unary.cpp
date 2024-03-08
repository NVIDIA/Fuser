// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <kernel_cache.h>
#include <ops/arith.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

using UnaryTest = NVFuserTest;

TEST_F(UnaryTest, BFloatNeg) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({3}, DataType::BFloat16);
  TensorView* out = neg(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({3}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

} // namespace nvfuser
