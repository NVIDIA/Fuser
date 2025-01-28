// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once
#include <multidevice/communicator.h>

namespace nvfuser {

template <typename T>
std::vector<uint8_t> toBytes(T data) {
  return std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(&data),
      reinterpret_cast<uint8_t*>(&data) + sizeof(T));
}

template <typename T>
T fromBytes(std::vector<uint8_t> bytes) {
  return *reinterpret_cast<T*>(bytes.data());
}

void LaunchDummyMultiDeviceKernel();

class AllgatherThroughCudaMemcpyAsync {
 public:
  AllgatherThroughCudaMemcpyAsync(at::Tensor input, std::vector<at::Tensor> outputs, Communicator* communicator);

  void post() const;

 private:
  std::string prefix() const {
    return "AllgatherThroughCudaMemcpyAsync" + std::to_string(unique_id);
  }

  static int64_t running_counter;
  int64_t unique_id;
  Communicator* communicator_;
  std::vector<int64_t> sizes_;
  std::vector<void*> input_ptrs_;
  std::vector<void*> output_ptrs_;
};


} // namespace nvfuser
