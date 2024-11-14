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
#include <ops/arith.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using UnaryTest = NVFuserFixtureParamTest<PrimDataType>;

TEST_P(UnaryTest, Neg) {
  PrimDataType dtype = GetParam();
  if (dtype == DataType::BFloat16 && !deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "Skip bfloat tests on pre-AMPERE GPUs.";
  }

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
      constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();
      // Set `low` to -INT_MAX instead of INT_MIN, because -INT_MIN is
      // undefined.
      in_tensor = at::randint(-kInt32Max, kInt32Max, shape, options);
    } break;
    case PrimDataType::Int: {
      constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
      in_tensor = at::randint(-kInt64Max, kInt64Max, shape, options);
    } break;
    default:
      NVF_ERROR(
          isFloatingPointType(dtype) || isComplexType(dtype),
          "Expect a floating point type or a complex type, but found: ",
          dtype);
      in_tensor = at::randn(shape, options);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  // Calculate the reference output explicitly. Type promotion happens when
  // building the fusion, e.g., inside `neg`. Relying ExpresionEvaluator to
  // verify the result would hide type promotion errors.
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      {-in_tensor},
      __LINE__,
      __FILE__);
}

namespace {

std::string sanitizeTestName(std::string&& name) {
  std::string test_name(name);
  for (char forbidden_char : {':', '<', '>'}) {
    std::replace(test_name.begin(), test_name.end(), forbidden_char, '_');
  }
  return test_name;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    UnaryTests,
    UnaryTest,
    // Skip unsigned types because they are uncommon and not supported in many
    // places, e.g.,
    // https://github.com/NVIDIA/Fuser/blob/329c3b194e7de5bccd3122468893c1fa83cd2a2e/csrc/executor.cpp#L478.
    testing::Values(
        PrimDataType::Half,
        PrimDataType::BFloat16,
        PrimDataType::Float,
        PrimDataType::Double,
        PrimDataType::Int32,
        PrimDataType::Int,
        PrimDataType::ComplexFloat,
        PrimDataType::ComplexDouble),
    [](const testing::TestParamInfo<PrimDataType>& info) {
      std::ostringstream os;
      os << info.param;
      return sanitizeTestName(os.str());
    });

} // namespace nvfuser
