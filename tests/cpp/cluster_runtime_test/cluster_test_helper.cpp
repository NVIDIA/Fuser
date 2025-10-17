// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <tests/cpp/cluster_runtime_test/cluster_test_helper.h>

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>

namespace nvfuser {

void validateClusterStoreResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    int cluster_size) {
  const int64_t in_dim = static_cast<int64_t>(input_tensor.dim());
  const int64_t out_dim = static_cast<int64_t>(output_tensor.dim());
  ASSERT_TRUE(in_dim == out_dim);

  const int64_t in_numel = input_tensor.numel();
  const int64_t out_numel = output_tensor.numel();
  ASSERT_TRUE(in_numel == out_numel);

  // Current kernel with CLUSTER_SIZE=2 writes peer CTA values,
  // so the expected output is the input with halves swapped.
  ASSERT_TRUE(cluster_size == 2)
      << "This validation currently assumes cluster_size == 2";

  // Convert to CPU for comparison
  auto input_cpu = input_tensor.cpu();
  auto output_cpu = output_tensor.cpu();

  const auto numel = input_cpu.numel();
  ASSERT_TRUE(numel % 2 == 0)
      << "Number of elements must be even to swap halves";
  const auto half = numel / 2;

  auto expected = at::empty_like(input_cpu);
  expected.narrow(0, 0, half).copy_(input_cpu.narrow(0, half, half));
  expected.narrow(0, half, half).copy_(input_cpu.narrow(0, 0, half));

  ASSERT_TRUE(at::allclose(output_cpu, expected, /*rtol=*/1e-7, /*atol=*/1e-8))
      << "Cluster store validation failed: output is not input with halves "
         "swapped";
}

void validateClusterReduceResult(
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    bool is_all_reduce,
    int threads_per_block) {
  auto input_cpu = input_tensor.cpu();
  auto output_cpu = output_tensor.cpu();

  // Expect the result to be the global sum of the input tensor
  const double expected_scalar = input_cpu.sum().item<double>();

  if (is_all_reduce) {
    // All-reduce: output should be full tensor with every element containing
    // the global sum
    const int64_t in_dim = static_cast<int64_t>(input_tensor.dim());
    const int64_t out_dim = static_cast<int64_t>(output_tensor.dim());
    ASSERT_TRUE(in_dim == out_dim);

    const int64_t in_numel = input_tensor.numel();
    const int64_t out_numel = output_tensor.numel();
    ASSERT_TRUE(in_numel == out_numel);

    auto expected = at::empty_like(input_cpu);
    expected.fill_(expected_scalar);

    ASSERT_TRUE(
        at::allclose(output_cpu, expected, /*rtol=*/1e-6, /*atol=*/1e-7))
        << "Cluster all-reduce validation failed: output is not the global sum";
  } else {
    // Reduce: output should be a single scalar containing the global sum
    ASSERT_TRUE(output_tensor.numel() == 1)
        << "Reduce output should be a scalar (single element), got "
        << output_tensor.numel() << " elements";

    const double actual_scalar = output_cpu.item<double>();
    ASSERT_TRUE(std::abs(actual_scalar - expected_scalar) < 1e-7)
        << "Cluster reduce validation failed: output is not the global sum. "
        << "Expected: " << expected_scalar << ", Got: " << actual_scalar;
  }
}

} // namespace nvfuser
