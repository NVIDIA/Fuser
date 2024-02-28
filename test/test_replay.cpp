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

  std::vector<IterDomain*> new_out_root = IterDomain::clone(
      TensorDomain::noReductions(in->getMaybeRFactorDomain()));
  TensorView* new_out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(new_out_root), *out->getDataType());
  new_out->setDomain(
      TransformReplay::fullSelfReplay(new_out->domain(), r0->domain()));
  // FIXME: how about allocation domain and contiguity?

  Expr* to_replay = r0->definition();
  auto create_fn = to_replay->newObjectFunc();
  create_fn(to_replay->container(), {in}, {new_out}, to_replay->attributes());

  fusion->addInput(in);
  fusion->addOutput(out);
  fusion->addOutput(new_out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 5}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  EXPECT_TRUE(at::equal(out_tensors[0], out_tensors[1]));
}

} // namespace nvfuser
