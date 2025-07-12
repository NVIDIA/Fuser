// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <tests/cpp/scan_test_helper.h>

#include <gtest/gtest.h>

namespace nvfuser {

void validateScanResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    ScanBinaryOpType binary_op_type) {
  ASSERT_EQ(input_tensor.dim(), output_tensor.dim());
  ASSERT_EQ(input_tensor.sizes(), output_tensor.sizes());

  // Convert to CPU for comparison
  auto input_cpu = input_tensor.cpu();
  auto output_cpu = output_tensor.cpu();

  // Create PyTorch reference based on binary operation type
  at::Tensor expected;
  std::string binary_op_string;

  switch (binary_op_type) {
    case ScanBinaryOpType::Add: { // Add (cumsum)
      expected = at::cumsum(input_cpu, -1);
      binary_op_string = "Add";
      break;
    }
    case ScanBinaryOpType::Max: { // Max (cummax)
      expected = std::get<0>(at::cummax(input_cpu, -1));
      binary_op_string = "Max";
      break;
    }
    case ScanBinaryOpType::Min: { // Min (cummin)
      expected = std::get<0>(at::cummin(input_cpu, -1));
      binary_op_string = "Min";
      break;
    }
    case ScanBinaryOpType::Mul: { // Mul (cumprod)
      expected = at::cumprod(input_cpu, -1);
      binary_op_string = "Mul";
      break;
    }
    default:
      FAIL() << "Unsupported operation";
  }

  // Compare with tolerance for floating point types
  bool match = false;
  if (input_tensor.scalar_type() == at::kFloat ||
      input_tensor.scalar_type() == at::kDouble ||
      input_tensor.scalar_type() == at::kHalf ||
      input_tensor.scalar_type() == at::kBFloat16) {
    // Use allclose for floating point comparison
    match = at::allclose(output_cpu, expected, /*rtol=*/1e-5, /*atol=*/1e-6);
  } else {
    // Use exact comparison for integer types
    match = at::equal(output_cpu, expected);
  }

  EXPECT_TRUE(match) << "Scan validation failed for operation "
                     << binary_op_string << "\n"
                     << "Input: " << input_cpu << "\n"
                     << "Expected: " << expected << "\n"
                     << "Got: " << output_cpu;
}

} // namespace nvfuser
