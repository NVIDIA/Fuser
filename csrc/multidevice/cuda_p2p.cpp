// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "multidevice/cuda_p2p.h"

#include "cuda_utils.h"
#include "multidevice/ipc_handle.h"
#include "multidevice/symmetric_tensor.h"

namespace nvfuser {

std::ostream& operator<<(std::ostream& os, P2pProtocol protocol) {
  switch (protocol) {
    case P2pProtocol::Get:
      return os << "Get";
    case P2pProtocol::Put:
      return os << "Put";
  }
  std::unreachable();
}

P2pProtocol getP2pProtocol() {
  return hasEnableOptionArgument(EnableOption::P2pProtocol, "put")
      ? P2pProtocol::Put
      : P2pProtocol::Get;
}

namespace {

// We choose  duplicate the state of the semaphore on both the local and peer
// devices to avoid cuStreamWaitValue32 to poll on a remote buffer and pollutes
// the network. This is a theoretical consideration that we have not proved or
// measured experimentally.
void WriteValue32ToLocalAndPeer(
    CUstream stream,
    const P2pIpcHandle& ipc_handles,
    IpcSemaphore value) {
  CUstreamBatchMemOpParams ops[2] = {};

  ops[0].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
  ops[0].writeValue.address =
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore());
  ops[0].writeValue.value = static_cast<cuuint32_t>(value);
  ops[0].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;

  ops[1].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
  ops[1].writeValue.address =
      reinterpret_cast<CUdeviceptr>(ipc_handles.peer().semaphore());
  ops[1].writeValue.value = static_cast<cuuint32_t>(value);
  ops[1].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;

  NVFUSER_CUDA_SAFE_CALL(cuStreamBatchMemOp(stream, 2, ops, 0));
}

void postBroadcastWithCudaBackend(
    Communication* communication,
    at::Tensor input,
    SymMemForBroadcast* multicast_handle,
    CUstream stream) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();
  const int64_t world_size = communicator.size();
  const int64_t root = communication->root();

  if (my_device_index != root) {
    // Non-root writes kInProgress to its own semaphore
    NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
        stream,
        reinterpret_cast<CUdeviceptr>(
            multicast_handle->semaphoreUnicastPtr(my_device_index)),
        static_cast<cuuint32_t>(IpcSemaphore::kInProgress),
        CU_STREAM_WRITE_VALUE_DEFAULT));
  } else {
    // Root waits on all non-root ranks' semaphores to become kInProgress
    std::vector<CUstreamBatchMemOpParams> ops(world_size - 1);
    int op_idx = 0;
    for (int64_t rank = 0; rank < world_size; ++rank) {
      if (rank == root)
        continue;
      ops[op_idx].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
      ops[op_idx].waitValue.address = reinterpret_cast<CUdeviceptr>(
          multicast_handle->semaphoreUnicastPtr(rank));
      ops[op_idx].waitValue.value =
          static_cast<cuuint32_t>(IpcSemaphore::kInProgress);
      ops[op_idx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
      op_idx++;
    }
    NVFUSER_CUDA_SAFE_CALL(
        cuStreamBatchMemOp(stream, world_size - 1, ops.data(), 0));

    // Root multicast the data
    // Root: compute src_ptr and count
    const void* src_ptr = input.data_ptr();
    const int64_t count = input.numel() * input.element_size();
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
        multicast_handle->bufferMulticastPtr(),
        src_ptr,
        count,
        cudaMemcpyDeviceToDevice,
        stream));

    // Root writes kIdle to all non-root semaphores using batched unicast writes
    std::vector<CUstreamBatchMemOpParams> write_idle_ops(world_size - 1);
    op_idx = 0;
    for (int64_t rank = 0; rank < world_size; ++rank) {
      if (rank == root)
        continue;
      write_idle_ops[op_idx].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      write_idle_ops[op_idx].writeValue.address = reinterpret_cast<CUdeviceptr>(
          multicast_handle->semaphoreUnicastPtr(rank));
      write_idle_ops[op_idx].writeValue.value =
          static_cast<cuuint32_t>(IpcSemaphore::kIdle);
      write_idle_ops[op_idx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      op_idx++;
    }
    NVFUSER_CUDA_SAFE_CALL(
        cuStreamBatchMemOp(stream, world_size - 1, write_idle_ops.data(), 0));
  }
}

void waitBroadcastWithCudaBackend(
    Communication* communication,
    SymMemForBroadcast* multicast_handle,
    CUstream stream) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();
  const int64_t root = communication->root();

  if (my_device_index != root) {
    // Non-root waits for its own semaphore to be kIdle
    NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
        stream,
        reinterpret_cast<CUdeviceptr>(
            multicast_handle->semaphoreUnicastPtr(my_device_index)),
        static_cast<cuuint32_t>(IpcSemaphore::kIdle),
        CU_STREAM_WAIT_VALUE_EQ));
  }
}

void postAllgatherWithCudaBackend(
    Communication* communication,
    at::Tensor input,
    SymMemForAllgather* allgather_handle,
    CUstream stream) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();
  const int64_t world_size = communicator.size();

  // Step 1: Each rank signals it's ready by writing kInProgress to its own
  // semaphore for every root
  std::vector<CUstreamBatchMemOpParams> write_ready_ops(world_size - 1);
  int write_op_idx = 0;
  for (int64_t rank = 0; rank < world_size; ++rank) {
    if (rank == my_device_index)
      continue;
    write_ready_ops[write_op_idx].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    write_ready_ops[write_op_idx].writeValue.address =
        reinterpret_cast<CUdeviceptr>(
            allgather_handle->semaphoreUnicastPtr(rank, my_device_index));
    write_ready_ops[write_op_idx].writeValue.value =
        static_cast<cuuint32_t>(IpcSemaphore::kInProgress);
    write_ready_ops[write_op_idx].writeValue.flags =
        CU_STREAM_WRITE_VALUE_DEFAULT;
    write_op_idx++;
  }
  NVFUSER_CUDA_SAFE_CALL(
      cuStreamBatchMemOp(stream, world_size - 1, write_ready_ops.data(), 0));

  // Step 2: Each rank waits for all other ranks to signal ready using batch
  // operations
  std::vector<CUstreamBatchMemOpParams> wait_ready_ops(world_size - 1);
  int wait_op_idx = 0;
  for (int64_t rank = 0; rank < world_size; ++rank) {
    if (rank == my_device_index)
      continue;
    wait_ready_ops[wait_op_idx].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    wait_ready_ops[wait_op_idx].waitValue.address =
        reinterpret_cast<CUdeviceptr>(
            allgather_handle->semaphoreUnicastPtr(my_device_index, rank));
    wait_ready_ops[wait_op_idx].waitValue.value =
        static_cast<cuuint32_t>(IpcSemaphore::kInProgress);
    wait_ready_ops[wait_op_idx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    wait_op_idx++;
  }
  NVFUSER_CUDA_SAFE_CALL(
      cuStreamBatchMemOp(stream, world_size - 1, wait_ready_ops.data(), 0));

  // Step 3: Each rank copies its data to its multicast buffer
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
      allgather_handle->bufferMulticastPtr(my_device_index),
      input.data_ptr(),
      input.numel() * input.element_size(),
      cudaMemcpyDeviceToDevice,
      stream));

  // Step 4: Each rank signals completion by writing kIdle to all peer
  // semaphores using batched unicast writes
  std::vector<CUstreamBatchMemOpParams> write_complete_ops(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    write_complete_ops[rank].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    write_complete_ops[rank].writeValue.address = reinterpret_cast<CUdeviceptr>(
        allgather_handle->semaphoreUnicastPtr(my_device_index, rank));
    write_complete_ops[rank].writeValue.value =
        static_cast<cuuint32_t>(IpcSemaphore::kIdle);
    write_complete_ops[rank].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
  }
  NVFUSER_CUDA_SAFE_CALL(
      cuStreamBatchMemOp(stream, world_size, write_complete_ops.data(), 0));
}

void waitAllgatherWithCudaBackend(
    Communication* communication,
    SymMemForAllgather* allgather_handle,
    CUstream stream) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();
  const int64_t world_size = communicator.size();

  // Wait for all other ranks to complete using batch operations
  std::vector<CUstreamBatchMemOpParams> wait_complete_ops(world_size - 1);
  int op_idx = 0;
  for (int64_t rank = 0; rank < world_size; ++rank) {
    if (rank == my_device_index)
      continue;
    wait_complete_ops[op_idx].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    wait_complete_ops[op_idx].waitValue.address = reinterpret_cast<CUdeviceptr>(
        allgather_handle->semaphoreUnicastPtr(rank, my_device_index));
    wait_complete_ops[op_idx].waitValue.value =
        static_cast<cuuint32_t>(IpcSemaphore::kIdle);
    wait_complete_ops[op_idx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    op_idx++;
  }
  NVFUSER_CUDA_SAFE_CALL(
      cuStreamBatchMemOp(stream, world_size - 1, wait_complete_ops.data(), 0));
}

} // anonymous namespace

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream) {
  P2pProtocol protocol = getP2pProtocol();
  switch (protocol) {
    case P2pProtocol::Get: {
      // wait for sender to be ready
      NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
          stream,
          reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
          (cuuint32_t)(IpcSemaphore::kInProgress),
          CU_STREAM_WAIT_VALUE_EQ));
      // Get the data from the sender
      NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
          ipc_handles.local().ptr(),
          ipc_handles.peer().ptr(),
          count,
          cudaMemcpyDeviceToDevice,
          stream));
      // Signals completion
      WriteValue32ToLocalAndPeer(stream, ipc_handles, IpcSemaphore::kIdle);
      break;
    }
    case P2pProtocol::Put: {
      WriteValue32ToLocalAndPeer(
          stream, ipc_handles, IpcSemaphore::kInProgress);
      break;
    }
    default:
      NVF_ERROR("Invalid P2P protocol: ", protocol);
  }
}

void recvWait(const P2pIpcHandle& ipc_handles, CUstream stream) {
  P2pProtocol protocol = getP2pProtocol();
  switch (protocol) {
    case P2pProtocol::Put:
      NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
          stream,
          reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
          (cuuint32_t)(IpcSemaphore::kIdle),
          CU_STREAM_WAIT_VALUE_EQ));
      break;
    case P2pProtocol::Get:
      break;
    default:
      NVF_ERROR("Invalid P2P protocol: ", protocol);
  }
}

void sendPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream) {
  P2pProtocol protocol = getP2pProtocol();
  switch (protocol) {
    case P2pProtocol::Get:
      // signal to self and peer that transfer is in progress
      WriteValue32ToLocalAndPeer(
          stream, ipc_handles, IpcSemaphore::kInProgress);
      break;
    case P2pProtocol::Put: {
      // wait for receiver to be ready
      NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
          stream,
          reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
          (cuuint32_t)(IpcSemaphore::kInProgress),
          CU_STREAM_WAIT_VALUE_EQ));
      // Put the data to the receiver
      NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
          ipc_handles.peer().ptr(),
          ipc_handles.local().ptr(),
          count,
          cudaMemcpyDeviceToDevice,
          stream));
      WriteValue32ToLocalAndPeer(stream, ipc_handles, IpcSemaphore::kIdle);
      break;
    }
    default:
      NVF_ERROR("Invalid P2P protocol: ", protocol);
  }
}

void sendWait(const P2pIpcHandle& ipc_handles, CUstream stream) {
  P2pProtocol protocol = getP2pProtocol();
  switch (protocol) {
    case P2pProtocol::Get:
      NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
          stream,
          reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
          (cuuint32_t)(IpcSemaphore::kIdle),
          CU_STREAM_WAIT_VALUE_EQ));
      break;
    case P2pProtocol::Put:
      break;
    default:
      NVF_ERROR("Invalid P2P protocol: ", protocol);
  }
}

void postWithCudaBackend(
    Communication* communication,
    at::Tensor input,
    SymmetricMemoryHandle* symmetric_memory_handle,
    CUstream stream) {
  NVF_ERROR(
      communication->backend() == CommunicatorBackend::kCuda,
      "Invalid backend, expected Cuda, got: ",
      communication->backend());

  Communicator& communicator = Communicator::getInstance();
  const int64_t world_size = communicator.size();
  NVF_ERROR(
      communication->team().size() == (size_t)world_size,
      "Only support world size team for broadcast with cuda backend, expected ",
      world_size,
      " got: ",
      communication->team().size());

  switch (communication->type()) {
    case CommunicationType::Broadcast: {
      auto* broadcast_handle =
          dynamic_cast<SymMemForBroadcast*>(symmetric_memory_handle);
      NVF_ERROR(broadcast_handle != nullptr, "Invalid broadcast handle");
      postBroadcastWithCudaBackend(
          communication, input, broadcast_handle, stream);
      break;
    }
    case CommunicationType::Allgather: {
      auto* allgather_handle =
          dynamic_cast<SymMemForAllgather*>(symmetric_memory_handle);
      NVF_ERROR(allgather_handle != nullptr, "Invalid allgather handle");
      postAllgatherWithCudaBackend(
          communication, input, allgather_handle, stream);
      break;
    }
    default:
      NVF_ERROR(
          false,
          "Unsupported communication type for CUDA backend: ",
          communication->type());
  }
}

void waitWithCudaBackend(
    Communication* communication,
    SymmetricMemoryHandle* symmetric_memory_handle,
    CUstream stream) {
  NVF_ERROR(
      communication->backend() == CommunicatorBackend::kCuda,
      "Invalid backend, expected Cuda, got: ",
      communication->backend());

  Communicator& communicator = Communicator::getInstance();
  const int64_t world_size = communicator.size();
  NVF_ERROR(
      communication->team().size() == (size_t)world_size,
      "Only support world size team for broadcast with cuda backend, expected ",
      world_size,
      " got: ",
      communication->team().size());

  switch (communication->type()) {
    case CommunicationType::Broadcast: {
      auto* broadcast_handle =
          dynamic_cast<SymMemForBroadcast*>(symmetric_memory_handle);
      NVF_ERROR(broadcast_handle != nullptr, "Invalid broadcast handle");
      waitBroadcastWithCudaBackend(communication, broadcast_handle, stream);
      break;
    }
    case CommunicationType::Allgather: {
      auto* allgather_handle =
          dynamic_cast<SymMemForAllgather*>(symmetric_memory_handle);
      NVF_ERROR(allgather_handle != nullptr, "Invalid allgather handle");
      waitAllgatherWithCudaBackend(communication, allgather_handle, stream);
      break;
    }
    default:
      NVF_ERROR(
          false,
          "Unsupported communication type for CUDA backend: ",
          communication->type());
  }
}

} // namespace nvfuser
