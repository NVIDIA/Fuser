// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Warning: this file should not include any header from nvFuser or pytorch
// (except raw headers). Compiling dynamic_type.h with nvcc is not supported.
// Compiling pytorch with nvcc is not supported either.

#include <tests/cpp/multidevice_kernels.h>
#include <cuda.h>

namespace nvfuser {

#define CUDA_CALL(call) NVF_ERROR((call) == cudaSuccess, "CUDA call failed: ", cudaGetErrorString(cudaGetLastError()))

__global__ void DummyMultiDeviceKernel() {}

void LaunchDummyMultiDeviceKernel() {
  DummyMultiDeviceKernel<<<1, 1>>>();
}

int64_t AllgatherThroughCudaMemcpyAsync::running_counter = 0;

AllgatherThroughCudaMemcpyAsync::AllgatherThroughCudaMemcpyAsync(at::Tensor input, std::vector<at::Tensor> outputs, Communicator* communicator) : unique_id(running_counter++), communicator_(communicator) {
  cudaIpcMemHandle_t input_ipc_handle;
  CUDA_CALL(cudaIpcGetMemHandle(&input_ipc_handle, input.data_ptr()));

  auto store = communicator->getTcpStore();
  const int64_t my_rank = communicator->deviceId();
  store->set(prefix() + std::to_string(my_rank), toBytes(input_ipc_handle));

  communicator_->barrier();

  sizes_.resize(communicator_->size(), 0);
  input_ptrs_.resize(communicator_->size(), nullptr);
  output_ptrs_.resize(communicator_->size(), nullptr);
  for (int64_t rank: c10::irange(communicator_->size())) {
    auto output = outputs.at(rank);
    sizes_.at(rank) = output.numel() * output.element_size();

    output_ptrs_.at(rank) = output.data_ptr();
    if (rank == my_rank) {
      input_ptrs_.at(rank) = input.data_ptr();
    } else {
      auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(store->get(prefix() + std::to_string(rank)));
      CUDA_CALL(cudaIpcOpenMemHandle(&input_ptrs_.at(rank), peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
}

void AllgatherThroughCudaMemcpyAsync::post() const {
  for (size_t i = 0; i < sizes_.size(); i++) {
    CUDA_CALL(cudaMemcpyAsync(output_ptrs_.at(i), input_ptrs_.at(i), sizes_.at(i), cudaMemcpyDeviceToDevice));
  }
}


} // namespace nvfuser
