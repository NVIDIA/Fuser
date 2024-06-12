// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <vector>

#include <gtest/gtest.h>

#include <ATen/ATen.h>

namespace nvfuser {

TEST(AllocatorTest, Steady) {
  auto run_and_collect_allocations =
      [&](const int num_stages) -> std::vector<void*> {
    std::vector<void*> allocations;

    const int n = (1 << num_stages);
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    std::vector<at::Tensor> tensors(n);
    for (int j = 0; j < n; j++) {
      tensors[j] = at::randn({1, 1024}, options);
      allocations.push_back(tensors[j].data_ptr());
    }

    at::Tensor ref_output = at::cat(tensors);

    for (int num_alive_tensors = n; num_alive_tensors > 1;
         num_alive_tensors /= 2) {
      for (int j = 0; j * 2 + 1 < num_alive_tensors; j++) {
        tensors[j] = at::cat({tensors[j * 2], tensors[j * 2 + 1]});
        allocations.push_back(tensors[j].data_ptr());
      }
    }
    EXPECT_TRUE(at::equal(tensors[0], ref_output));

    return allocations;
  };

  std::vector<void*> allocations_in_previous_run =
      run_and_collect_allocations(10);
  for (int i = 1; i < 20; i++) {
    std::vector<void*> allocations = run_and_collect_allocations(10);
    EXPECT_EQ(allocations_in_previous_run, allocations)
        << "Run " << i << "'s allocations don't match Run " << i - 1 << ".";
    allocations_in_previous_run = allocations;
  }
}

} // namespace nvfuser
