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

using UnaryTest = NVFuserFixtureParamTest<DataType>;

TEST_P(UnaryTest, Neg) {
  DataType dtype = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({640, 64});

  TensorView* in = makeContigConcreteTensor(shape, dtype);
  TensorView* out = neg(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn(shape, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    UnaryTests,
    UnaryTest,
    testing::Values(
        DataType::Half,
        DataType::BFloat16,
        DataType::Float,
        DataType::Double,
        DataType::Int),
    [](const testing::TestParamInfo<DataType>& info) {
      std::ostringstream os;
      os << info.param;
      return os.str();
    });

} // namespace nvfuser
