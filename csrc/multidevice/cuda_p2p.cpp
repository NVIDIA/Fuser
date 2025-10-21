// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_utils.h>
#include <multidevice/cuda_p2p.h>

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

} // anonymous namespace

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream) {
  P2pProtocol protocol = getP2pProtocol();
  switch (protocol) {
    case P2pProtocol::Get: {
      // wait for sender to be ready
      NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
          stream,
          reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
          (cuuint32_t)(IpcSemaphore::kInUse),
          CU_STREAM_WAIT_VALUE_EQ));
      // Get the data from the sender
      NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
          ipc_handles.local().ptr(),
          ipc_handles.peer().ptr(),
          count,
          cudaMemcpyDeviceToDevice,
          stream));
      // Signals completion
      WriteValue32ToLocalAndPeer(stream, ipc_handles, IpcSemaphore::kReady);
      break;
    }
    case P2pProtocol::Put: {
      WriteValue32ToLocalAndPeer(stream, ipc_handles, IpcSemaphore::kInUse);
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
          (cuuint32_t)(IpcSemaphore::kReady),
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
      WriteValue32ToLocalAndPeer(stream, ipc_handles, IpcSemaphore::kInUse);
      break;
    case P2pProtocol::Put: {
      // wait for receiver to be ready
      NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
          stream,
          reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
          (cuuint32_t)(IpcSemaphore::kInUse),
          CU_STREAM_WAIT_VALUE_EQ));
      // Put the data to the receiver
      NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
          ipc_handles.peer().ptr(),
          ipc_handles.local().ptr(),
          count,
          cudaMemcpyDeviceToDevice,
          stream));
      WriteValue32ToLocalAndPeer(stream, ipc_handles, IpcSemaphore::kReady);
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
          (cuuint32_t)(IpcSemaphore::kReady),
          CU_STREAM_WAIT_VALUE_EQ));
      break;
    case P2pProtocol::Put:
      break;
    default:
      NVF_ERROR("Invalid P2P protocol: ", protocol);
  }
}

} // namespace nvfuser
