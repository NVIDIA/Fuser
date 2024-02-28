// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>
#include <transform_replay.h>

namespace nvfuser {

using ReplayTest = NVFuserTest;

TEST_F(ReplayTest, HorizontallyMergeTwoReshapes) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 5});
  TensorView* s0 = slice(in, {0, 0}, {4, 2});
  TensorView* s1 = slice(in, {0, 2}, {4, 5});
  TensorView* r0 = reshape(s0, {4, 2}, {2, 2, 2});
  TensorView* r1 = reshape(s1, {4, 3}, {2, 2, 3});
  TensorView* out = cat({r0, r1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  std::vector<IterDomain*> r_root = IterDomain::clone(
      TensorDomain::noReductions(in->getMaybeRFactorDomain()));
  TensorDomain* r_domain = TransformReplay::fullSelfReplay(
      IrBuilder::create<TensorDomain>(r_root), r0->domain());
  TensorView* r = IrBuilder::create<TensorView>(r_domain, *out->getDataType());

  Expr* to_replay = r0->definition();
  auto create_fn = to_replay->newObjectFunc();
  create_fn(to_replay->container(), {in}, {r}, to_replay->attributes());
  // To preserve the output allocation domain, we create a Set between `r` and
  // `out` instead of replacing the fusion output.
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, r);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 5}, options);

  FusionExecutorCache fec(std::move(fusion));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  auto out_tensor = out_tensors[0];

  std::vector<at::Tensor> slices = at::split(in_tensor, {2, 3}, /*dim=*/-1);
  at::Tensor expected_out_tensor = at::cat(
      {slices[0].view({2, 2, 2}), slices[1].view({2, 2, 3})}, /*dim=*/-1);

  EXPECT_TRUE(at::equal(out_tensor, expected_out_tensor));
}

} // namespace nvfuser
