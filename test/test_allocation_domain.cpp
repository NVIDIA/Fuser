// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class AllocationDomainTest : public NVFuserTest {};

// A global->shared->global copy kernel, shared memory allocated transposed to
// avoid bank conflict.
TEST_F(AllocationDomainTest, TransposedIntermediate_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  auto bc = fusion.bankConflictInfo();
  ASSERT_TRUE(bc.size() == 1);
  auto [read, write] = bc.at(tv1);
  ASSERT_EQ(read, std::vector<int>{32});
  ASSERT_EQ(write, std::vector<int>{32});

  std::vector<IterDomain*> tv1_transposed = {tv1->axis(1), tv1->axis(0)};
  tv1->setAllocationDomain(tv1_transposed, true);

  ASSERT_TRUE(fusion.bankConflictInfo().empty());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
