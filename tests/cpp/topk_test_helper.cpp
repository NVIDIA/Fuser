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

bool validateTopkOrder(
    at::Tensor input_tensor,
    at::Tensor values_tensor,
    at::Tensor indices_tensor,
    int64_t k,
    bool largest) {
  // at::topk is not stable, so here we check the results in a more
  // manual way

  NVF_ERROR_EQ(input_tensor.dim(), values_tensor.dim());
  NVF_ERROR_EQ(input_tensor.dim(), indices_tensor.dim());

  auto input_size = static_cast<int64_t>(input_tensor.size(-1));

  // Check that we have k valid results
  if (static_cast<int64_t>(values_tensor.size(-1)) < k ||
      static_cast<int64_t>(indices_tensor.size(-1)) < k) {
    return false;
  }

  // We don't care output tensors beyond k
  values_tensor = values_tensor.slice(-1, 0, k);
  indices_tensor = indices_tensor.slice(-1, 0, k);

  // Check valid indices range
  if ((indices_tensor < 0).any().item<bool>()) {
    return false;
  }

  if ((indices_tensor >= input_size).any().item<bool>()) {
    return false;
  }

  // Check values match indices
  if ((values_tensor != torch::gather(input_tensor, -1, indices_tensor))
          .any()
          .item<bool>()) {
    return false;
  }

  if (largest) {
    // For largest, should be in descending order
    if ((values_tensor.slice(-1, 1, k) > values_tensor.slice(-1, 0, k - 1))
            .any()
            .item<bool>()) {
      return false;
    }
  } else {
    // For smallest, should be in ascending order
    if ((values_tensor.slice(-1, 1, k) < values_tensor.slice(-1, 0, k - 1))
            .any()
            .item<bool>()) {
      return false;
    }
  }

  // Check that the returned values are actually the true top-k elements
  // Sort the input data to get the expected top-k values
  auto expected_topk =
      std::get<0>(torch::sort(input_tensor, -1, /*descending=*/largest))
          .slice(-1, 0, k);

  // Compare the expected and actual top-k values
  if (!at::equal(expected_topk, values_tensor)) {
    return false;
  }

  return true;
}

} // namespace nvfuser
