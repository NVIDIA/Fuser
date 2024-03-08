// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <limits>

#include <fusion.h>
#include <kernel_cache.h>
#include <ops/arith.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

using UnaryTest = NVFuserFixtureParamTest<PrimDataType>;

TEST_P(UnaryTest, Neg) {
  PrimDataType dtype = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({640, 64});

  TensorView* in = makeContigConcreteTensor(shape, dtype);
  TensorView* out = neg(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor in_tensor;
  switch (dtype) {
    case PrimDataType::Int32: {
      constexpr int kIntMax = std::numeric_limits<int>::max();
      in_tensor = at::randint(-kIntMax, kIntMax, shape, options);
    } break;
    case PrimDataType::Int: {
      constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
      in_tensor = at::randint(-kInt64Max, kInt64Max, shape, options);
    } break;
    default:
      NVF_ERROR(
          isFloatingPointType(dtype),
          "Expect a floating point type, but found: ",
          dtype);
      in_tensor = at::randn(shape, options);
  }

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    UnaryTests,
    UnaryTest,
    testing::Values(
        PrimDataType::Half,
        PrimDataType::BFloat16,
        PrimDataType::Float,
        PrimDataType::Double,
        PrimDataType::Int32,
        PrimDataType::Int),
    [](const testing::TestParamInfo<PrimDataType>& info) {
      std::ostringstream os;
      os << info.param;
      return os.str();
    });

} // namespace nvfuser
