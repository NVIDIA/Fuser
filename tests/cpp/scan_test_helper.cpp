// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <csrc/exceptions.h>
#include <csrc/type.h>
#include <tests/cpp/scan_test_helper.h>

namespace nvfuser {

bool validateScanResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType binary_op_type) {
  NVF_ERROR_EQ(input_tensor.dim(), output_tensor.dim());
  NVF_ERROR_EQ(input_tensor.sizes(), output_tensor.sizes());

  // Convert to CPU for comparison
  auto input_cpu = input_tensor.cpu();
  auto output_cpu = output_tensor.cpu();

  // Create PyTorch reference based on binary operation type
  at::Tensor expected;

  switch (binary_op_type) {
    case BinaryOpType::Add: { // Add (cumsum)
      expected = at::cumsum(input_cpu, -1);
      break;
    }
    case BinaryOpType::Max: { // Max (cummax)
      expected = std::get<0>(at::cummax(input_cpu, -1));
      break;
    }
    case BinaryOpType::Min: { // Min (cummin)
      expected = std::get<0>(at::cummin(input_cpu, -1));
      break;
    }
    case BinaryOpType::Mul: { // Mul (cumprod)
      expected = at::cumprod(input_cpu, -1);
      break;
    }
    default: {
      // Unsupported operation
      return false;
    }
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

  if (!match) {
    std::cout << "Scan validation failed for operation "
              << getBinaryOpName(binary_op_type) << std::endl;
    std::cout << "Input: " << input_cpu << std::endl;
    std::cout << "Expected: " << expected << std::endl;
    std::cout << "Got: " << output_cpu << std::endl;
  }

  return match;
}

const char* getBinaryOpName(BinaryOpType binary_op_type) {
  switch (binary_op_type) {
    case BinaryOpType::Add:
      return "Add";
    case BinaryOpType::Max:
      return "Max";
    case BinaryOpType::Min:
      return "Min";
    case BinaryOpType::Mul:
      return "Mul";
    default:
      return "Unknown";
  }
}

} // namespace nvfuser