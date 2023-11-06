#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <torch/cuda.h>
#include <torch/torch.h>

#include <iostream>
#include <string>

std::string add_negative_flag(const std::string& flag) {
  std::string filter = ::testing::GTEST_FLAG(filter);
  if (filter.find('-') == std::string::npos) {
    filter.push_back('-');
  } else {
    filter.push_back(':');
  }
  filter += flag;
  return filter;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  NVF_CHECK(
      torch::cuda::is_available(),
      "nvfuser_tests requires CUDA device being available");

  return RUN_ALL_TESTS();
}
