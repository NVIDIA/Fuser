// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <exceptions.h>
#include <tests/cpp/topk_test_helper.h>

namespace nvfuser {

namespace {

template <typename T>
std::vector<T> get_cpu_vector(at::Tensor tensor) {
  NVF_ERROR_EQ(tensor.dim(), 1);
  if (tensor.dtype() == at::kBFloat16) {
    tensor = tensor.to(at::kFloat);
  }
  auto cpu_tensor = tensor.cpu();
  auto total_elements = tensor.size(0);
  return std::vector<T>(
      cpu_tensor.data_ptr<T>(), cpu_tensor.data_ptr<T>() + total_elements);
};

} // namespace

// Helper function to validate topk correctness
template <typename DataT>
bool validateTopkOrder(
    const at::Tensor& input_tensor,
    const at::Tensor& values_tensor,
    const at::Tensor& indices_tensor,
    int64_t k,
    bool largest) {
  NVF_ERROR_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto input_data = get_cpu_vector<DataT>(input_tensor);
  auto output_values = get_cpu_vector<DataT>(values_tensor);
  auto output_indices = get_cpu_vector<int64_t>(indices_tensor);

  // Check that we have k valid results
  if (static_cast<int64_t>(output_values.size()) < k ||
      static_cast<int64_t>(output_indices.size()) < k) {
    return false;
  }

  // Check valid indices range
  for (int64_t i = 0; i < k; i++) {
    if (output_indices[i] < 0 ||
        output_indices[i] >= static_cast<int64_t>(input_data.size())) {
      return false;
    }
  }

  // Check values match indices
  for (int64_t i = 0; i < k; i++) {
    if (output_values[i] != input_data[output_indices[i]]) {
      return false;
    }
  }

  // Check sorting order of the k elements
  for (int64_t i = 1; i < k; i++) {
    if (largest) {
      // For largest, should be in descending order
      if (output_values[i] > output_values[i - 1]) {
        return false;
      }
    } else {
      // For smallest, should be in ascending order
      if (output_values[i] < output_values[i - 1]) {
        return false;
      }
    }
  }

  // Check that the returned values are actually the true top-k elements
  // Sort the input data to get the expected top-k values
  std::vector<DataT> sorted_input = input_data;
  if (largest) {
    std::sort(sorted_input.begin(), sorted_input.end(), std::greater<DataT>());
  } else {
    std::sort(sorted_input.begin(), sorted_input.end());
  }

  // Extract the expected top-k values
  std::vector<DataT> expected_topk(
      sorted_input.begin(), sorted_input.begin() + k);

  // Extract the actual returned values (first k elements)
  std::vector<DataT> actual_topk(
      output_values.begin(), output_values.begin() + k);

  // Compare the expected and actual top-k values
  if (expected_topk != actual_topk) {
    return false;
  }

  return true;
}

template bool validateTopkOrder<float>(
    const at::Tensor& input_tensor,
    const at::Tensor& values_tensor,
    const at::Tensor& indices_tensor,
    int64_t k,
    bool largest);

template bool validateTopkOrder<double>(
    const at::Tensor& input_tensor,
    const at::Tensor& values_tensor,
    const at::Tensor& indices_tensor,
    int64_t k,
    bool largest);

template bool validateTopkOrder<int>(
    const at::Tensor& input_tensor,
    const at::Tensor& values_tensor,
    const at::Tensor& indices_tensor,
    int64_t k,
    bool largest);

template bool validateTopkOrder<int64_t>(
    const at::Tensor& input_tensor,
    const at::Tensor& values_tensor,
    const at::Tensor& indices_tensor,
    int64_t k,
    bool largest);

} // namespace nvfuser
