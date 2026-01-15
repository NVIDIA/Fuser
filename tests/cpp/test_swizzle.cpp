// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <torch/torch.h>

#include <gtest/gtest.h>

#include "device_lower/analysis/bank_conflict.h"
#include "exceptions.h"
#include "ops/all_ops.h"
#include "runtime/fusion_executor_cache.h"
#include "scheduler/tools/abstract_tensor.h"
#include "scheduler/tools/inlining.h"
#include "swizzle.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"
#include "transform_iter.h"

namespace nvfuser {

class SwizzleTest : public NVFuserTest {};

TEST_F(SwizzleTest, Transpose1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  std::vector<IterDomain*> dim0{tv1->axis(0), tv2->axis(1)};
  std::vector<IterDomain*> dim1{tv1->axis(1), tv2->axis(0)};
  AbstractTensor loop{dim0, dim1};

  loop.split(1, 32);
  loop.split(0, 32);
  loop.reorder({{1, 2}});
  loop.merge(0);
  loop.parallelize(0, ParallelType::BIDx);
  // BIDx, 32, 32

  auto smem_alloc = loop.unzip()[0];
  smem_alloc.swizzle(SwizzleType::XOR, 1, 2);
  tv1->setAllocationDomain(smem_alloc.as<IterDomain*>(), true);

  std::swap(loop[1][1], loop[2][1]);
  loop.merge(1);
  loop.split(1, 256);
  loop.parallelize(2, ParallelType::TIDx);
  // BIDx, 4, TIDx

  auto uz = loop.unzip();
  tv1->setLoopDomain(uz[0].as<IterDomain*>());
  tv2->setLoopDomain(uz[1].as<IterDomain*>());

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t = at::randn({10240, 10240}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t});
  EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto outputs = ke.run({t});
  EXPECT_TRUE(at::equal(t.t(), outputs[0].as<at::Tensor>()));
}

} // namespace nvfuser
