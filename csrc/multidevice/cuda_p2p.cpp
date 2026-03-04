// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "multidevice/cuda_p2p.h"
#include "nvfuser_resources/alltoallv.h"
#include "nvfuser_resources/multicast.h"

#include "cuda_utils.h"
#include "multidevice/ipc_handle.h"
#include "multidevice/ipc_utils.h"
#include "multidevice/symmetric_tensor.h"
#include "multidevice/utils.h"
#include "options.h"

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
void launchAlltoallvKernel(
    const void* send,
    const uint64_t* recv_ptrs,
    const int64_t* send_offsets,
    const int64_t* send_sizes,
    const int64_t* recv_offsets,
    int64_t world_size,
    int64_t elem_size,
    int64_t max_send_bytes,
    CUstream stream) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    nvrtcProgram prog;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcCreateProgram(
        &prog,
        nvfuser_resources::alltoallv_cu,
        "alltoallv.cu",
        0,
        nullptr,
        nullptr));

    int major = 0;
    int minor = 0;
    int device = 0;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&device));
    cudaDeviceProp prop;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
    major = prop.major;
    minor = prop.minor;

    std::string arch_arg = "--gpu-architecture=compute_" +
        std::to_string(major) + std::to_string(minor);
    std::vector<const char*> opts = {arch_arg.c_str(), "--std=c++17"};
    // NVRTC needs CUDA headers to compile alltoallv.cu.
    opts.push_back("-I/usr/local/cuda/include");
    opts.push_back("-I/usr/local/cuda/include/cccl");

    nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (res != NVRTC_SUCCESS) {
      size_t logSize;
      NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
      std::vector<char> log(logSize);
      NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
      NVF_ERROR(false, "Alltoallv kernel compilation failed:\n", log.data());
    }

    size_t ptxSize;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
    NVFUSER_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    CUresult load_result = cuModuleLoadData(&module, ptx.data());
    if (load_result != CUDA_SUCCESS) {
      constexpr size_t kLogSize = 8192;
      char error_log[kLogSize];
      char info_log[kLogSize];
      CUjit_option options[] = {
          CU_JIT_ERROR_LOG_BUFFER,
          CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
          CU_JIT_INFO_LOG_BUFFER,
          CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
          CU_JIT_LOG_VERBOSE};
      void* option_values[] = {
          (void*)error_log,
          (void*)kLogSize,
          (void*)info_log,
          (void*)kLogSize,
          (void*)1};
      cuModuleLoadDataEx(&module, ptx.data(), 5, options, option_values);
      NVF_ERROR(
          false,
          "Alltoallv kernel module load failed with error: ",
          load_result,
          "\nInfo Log:\n",
          info_log,
          "\nError Log:\n",
          error_log);
    }

    NVFUSER_CUDA_SAFE_CALL(
        cuModuleGetFunction(&kernel, module, "alltoallv_kernel"));
  }

  if (max_send_bytes == 0) {
    return;
  }

  constexpr int kThreads = 256;
  const int64_t blocks_x = (max_send_bytes + kThreads - 1) / kThreads;
  void* args_kernel[] = {
      const_cast<void*>(static_cast<const void*>(&send)),
      const_cast<void*>(static_cast<const void*>(&recv_ptrs)),
      const_cast<void*>(static_cast<const void*>(&send_offsets)),
      const_cast<void*>(static_cast<const void*>(&send_sizes)),
      const_cast<void*>(static_cast<const void*>(&recv_offsets)),
      &world_size,
      &elem_size,
      &max_send_bytes};
  NVFUSER_CUDA_SAFE_CALL(cuLaunchKernel(
      kernel,
      blocks_x,
      static_cast<unsigned int>(world_size),
      1,
      kThreads,
      1,
      1,
      0,
      stream,
      args_kernel,
      nullptr));
}

std::vector<uint8_t> serializeInt64Vector(const std::vector<int64_t>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(int64_t));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::vector<int64_t> deserializeInt64Vector(const std::vector<uint8_t>& bytes) {
  NVF_CHECK(
      bytes.size() % sizeof(int64_t) == 0, "Invalid int64 byte buffer size.");
  const size_t count = bytes.size() / sizeof(int64_t);
  std::vector<int64_t> values(count);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::string alltoallvCountsKey(const std::string& tag, int64_t rank) {
  return "nvfuser_alltoallv_counts_" + tag + "_" + std::to_string(rank);
}

std::string alltoallvBarrierKey(const std::string& tag, int64_t rank) {
  return "nvfuser_alltoallv_barrier_" + tag + "_" + std::to_string(rank);
}

void launchMulticastKernel(
    void* dst,
    const void* src,
    size_t size,
    CUstream stream) {
  static CUmodule module = nullptr;
  static CUfunction kernel = nullptr;

  if (module == nullptr) {
    nvrtcProgram prog;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcCreateProgram(
        &prog,
        nvfuser_resources::multicast_cu,
        "multicast.cu",
        0,
        nullptr,
        nullptr));

    int major = 0;
    int minor = 0;
    int device = 0;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&device));
    cudaDeviceProp prop;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
    major = prop.major;
    minor = prop.minor;

    NVF_CHECK(
        major >= 9,
        "Multicast kernel using 'multimem' protocol requires Compute "
        "Capability >= 9.0 (Hopper+). ",
        "Current device ",
        device,
        " is Compute Capability ",
        major,
        ".",
        minor);

    std::string arch_arg = "--gpu-architecture=compute_" +
        std::to_string(major) + std::to_string(minor);
    std::vector<const char*> opts = {arch_arg.c_str(), "--std=c++17"};

    // st.multimem requires PTX ISA 8.0+
    if (major >= 9) {
      opts.push_back("--ptx-isa-version=8.0");
    }

    nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (res != NVRTC_SUCCESS && major >= 9) {
      // If 8.0 is not supported (e.g. older NVRTC), try without it
      opts.pop_back();
      res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    }

    if (res != NVRTC_SUCCESS) {
      size_t logSize;
      NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
      std::vector<char> log(logSize);
      NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
      NVF_ERROR(false, "Multicast kernel compilation failed:\n", log.data());
    }

    size_t ptxSize;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
    NVFUSER_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    CUresult load_result = cuModuleLoadData(&module, ptx.data());

    if (load_result != CUDA_SUCCESS) {
      // Fallback to extensive logging only on failure
      constexpr size_t kLogSize = 8192;
      char error_log[kLogSize];
      char info_log[kLogSize];
      CUjit_option options[] = {
          CU_JIT_ERROR_LOG_BUFFER,
          CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
          CU_JIT_INFO_LOG_BUFFER,
          CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
          CU_JIT_LOG_VERBOSE};
      void* option_values[] = {
          (void*)error_log,
          (void*)kLogSize,
          (void*)info_log,
          (void*)kLogSize,
          (void*)1};

      // Reload to capture logs
      cuModuleLoadDataEx(&module, ptx.data(), 5, options, option_values);

      NVF_ERROR(
          false,
          "Multicast kernel module load failed with error: ",
          load_result,
          "\nInfo Log:\n",
          info_log,
          "\nError Log:\n",
          error_log);
    }

    NVFUSER_CUDA_SAFE_CALL(
        cuModuleGetFunction(&kernel, module, "multimem_copy_kernel"));
  }

  // Ensure data is 16-byte aligned
  NVF_CHECK(
      (uintptr_t)dst % 16 == 0,
      "Multicast dst must be 16-byte aligned. ptr=",
      dst);
  NVF_CHECK(
      (uintptr_t)src % 16 == 0,
      "Multicast src must be 16-byte aligned. ptr=",
      src);
  // Also assume size is a multiple of 16 for simplicity in the kernel
  NVF_CHECK(
      size % 16 == 0, "Multicast size must be a multiple of 16. size=", size);

  int threads = 128;
  int blocks = 1;

  int device;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&device));
  int num_sms;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  // Maximize occupancy
  int max_blocks_per_sm;
  NVFUSER_CUDA_SAFE_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm, kernel, threads, 0));

  blocks = num_sms * max_blocks_per_sm;

  // Limit number of blocks so that we don't launch more threads than needed
  // to cover the message size (vectorized 16 bytes per thread).
  size_t vec_size = 16;
  size_t total_work_units = (size + vec_size - 1) / vec_size;
  size_t max_needed_blocks = (total_work_units + threads - 1) / threads;

  if ((size_t)blocks > max_needed_blocks) {
    blocks = std::max(1, (int)max_needed_blocks);
  }
  const auto& args = getEnableOptionArguments(EnableOption::MulticastProtocol);
  if (args.size() >= 2) {
    try {
      threads = std::stoi(args[1]);
    } catch (...) {
    }
  }
  if (args.size() >= 3) {
    try {
      blocks = std::stoi(args[2]);
    } catch (...) {
    }
  }

  void* args_kernel[] = {&dst, &src, &size};
  NVFUSER_CUDA_SAFE_CALL(cuLaunchKernel(
      kernel, blocks, 1, 1, threads, 1, 1, 0, stream, args_kernel, nullptr));
}

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
    CUstream stream,
    int64_t root) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();
  const int64_t world_size = communicator.size();

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

    MulticastProtocol protocol = getMulticastProtocol();
    if (protocol == MulticastProtocol::Multimem) {
      launchMulticastKernel(
          multicast_handle->bufferMulticastPtr(), src_ptr, count, stream);
    } else if (protocol == MulticastProtocol::BatchMemcpy) {
#if CUDA_VERSION < 12080
      NVF_THROW(
          "cudaMemcpyBatchAsync backend is not supported for CUDA version < "
          "12.8");
#else
      std::vector<void*> dsts(world_size);
      std::vector<const void*> srcs(world_size, src_ptr);
      std::vector<size_t> counts(world_size, count);
      std::vector<cudaMemcpyAttributes> attributes(world_size);
      std::vector<size_t> attrsIdxs(world_size);
      size_t numAttrs = world_size;
      for (int64_t rank = 0; rank < world_size; ++rank) {
        dsts[rank] = multicast_handle->bufferUnicastPtr(rank);
        attrsIdxs[rank] = rank;
        struct cudaMemLocation dst_location = {
            .type = cudaMemLocationTypeDevice, .id = (int)rank};
        struct cudaMemLocation src_location = {
            .type = cudaMemLocationTypeDevice, .id = (int)root};
        unsigned int flags = cudaMemcpyFlagPreferOverlapWithCompute;
        attributes[rank].dstLocHint = dst_location;
        attributes[rank].srcLocHint = src_location;
        attributes[rank].flags = flags;
        attributes[rank].srcAccessOrder = cudaMemcpySrcAccessOrderAny;
      }
      NVF_CHECK(
          stream != 0, "cudaMemcpyBatchAsync does not support default stream");
#if CUDA_VERSION >= 13000
      NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyBatchAsync(
          dsts.data(),
          srcs.data(),
          counts.data(),
          world_size,
          attributes.data(),
          attrsIdxs.data(),
          numAttrs,
          (cudaStream_t)stream));
#else
      size_t failIdx = 0;
      NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyBatchAsync(
          dsts.data(),
          srcs.data(),
          counts.data(),
          world_size,
          attributes.data(),
          attrsIdxs.data(),
          numAttrs,
          &failIdx,
          (cudaStream_t)stream));
#endif
#endif
    } else {
      NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
          multicast_handle->bufferMulticastPtr(),
          src_ptr,
          count,
          cudaMemcpyDeviceToDevice,
          stream));
    }

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
    CUstream stream,
    int64_t root) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();

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
  MulticastProtocol protocol = getMulticastProtocol();
  const void* src_ptr = input.data_ptr();
  const int64_t count = input.numel() * input.element_size();

  if (protocol == MulticastProtocol::Multimem) {
    launchMulticastKernel(
        allgather_handle->bufferMulticastPtr(my_device_index),
        src_ptr,
        count,
        stream);
  } else if (protocol == MulticastProtocol::BatchMemcpy) {
#if CUDA_VERSION < 12080
    NVF_THROW(
        "cudaMemcpyBatchAsync backend is not supported for CUDA version < "
        "12.8");
#else
    std::vector<void*> dsts(world_size);
    std::vector<const void*> srcs(world_size, src_ptr);
    std::vector<size_t> counts(world_size, count);
    std::vector<cudaMemcpyAttributes> attributes(world_size);
    std::vector<size_t> attrsIdxs(world_size);
    size_t numAttrs = world_size;
    for (int64_t rank = 0; rank < world_size; ++rank) {
      dsts[rank] = allgather_handle->bufferUnicastPtr(my_device_index, rank);
      attrsIdxs[rank] = rank;
      struct cudaMemLocation dst_location = {
          .type = cudaMemLocationTypeDevice, .id = (int)rank};
      struct cudaMemLocation src_location = {
          .type = cudaMemLocationTypeDevice, .id = (int)my_device_index};
      unsigned int flags = cudaMemcpyFlagPreferOverlapWithCompute;
      attributes[rank].dstLocHint = dst_location;
      attributes[rank].srcLocHint = src_location;
      attributes[rank].flags = flags;
      attributes[rank].srcAccessOrder = cudaMemcpySrcAccessOrderAny;
    }
    NVF_CHECK(
        stream != 0, "cudaMemcpyBatchAsync does not support default stream");
#if CUDA_VERSION >= 13000
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyBatchAsync(
        dsts.data(),
        srcs.data(),
        counts.data(),
        world_size,
        attributes.data(),
        attrsIdxs.data(),
        numAttrs,
        (cudaStream_t)stream));
#else
    size_t failIdx = 0;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyBatchAsync(
        dsts.data(),
        srcs.data(),
        counts.data(),
        world_size,
        attributes.data(),
        attrsIdxs.data(),
        numAttrs,
        &failIdx,
        (cudaStream_t)stream));
#endif
#endif
  } else {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
        allgather_handle->bufferMulticastPtr(my_device_index),
        src_ptr,
        count,
        cudaMemcpyDeviceToDevice,
        stream));
  }

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
    CUstream stream,
    int64_t root) {
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
          communication, input, broadcast_handle, stream, root);
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
    CUstream stream,
    int64_t root) {
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
      waitBroadcastWithCudaBackend(
          communication, broadcast_handle, stream, root);
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

AlltoallvMetadata prepareAlltoallvMetadata(
    const at::Tensor& send_counts,
    const std::string& tag) {
  Communicator& comm = Communicator::getInstance();
  const int64_t world_size = comm.size();
  const int64_t my_rank = comm.deviceId();
  NVF_CHECK(
      send_counts.is_cuda(), "alltoallv send_counts must be CUDA tensor.");
  NVF_CHECK(
      send_counts.dim() == 1 && send_counts.numel() == world_size,
      "alltoallv send_counts must be 1D [R].");

  auto store = comm.getTcpStore();
  auto send_counts_cpu = send_counts.to(at::kCPU);
  auto* send_ptr = send_counts_cpu.data_ptr<int64_t>();
  std::vector<int64_t> send_counts_vec(send_ptr, send_ptr + world_size);

  store->set(
      alltoallvCountsKey(tag, my_rank), serializeInt64Vector(send_counts_vec));

  std::vector<std::vector<int64_t>> counts_matrix(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    auto bytes = store->get(alltoallvCountsKey(tag, rank));
    counts_matrix[rank] = deserializeInt64Vector(bytes);
    NVF_CHECK(
        (int64_t)counts_matrix[rank].size() == world_size,
        "Invalid alltoallv counts size.");
  }
  comm.barrier();
  for (int64_t rank = 0; rank < world_size; ++rank) {
    store->deleteKey(alltoallvCountsKey(tag, rank));
  }

  std::vector<int64_t> recv_counts_vec(world_size, 0);
  for (int64_t sender = 0; sender < world_size; ++sender) {
    recv_counts_vec[sender] = counts_matrix[sender][my_rank];
  }

  std::vector<int64_t> send_offsets_vec(world_size, 0);
  int64_t prefix = 0;
  for (int64_t rank = 0; rank < world_size; ++rank) {
    send_offsets_vec[rank] = prefix;
    prefix += send_counts_vec[rank];
  }

  std::vector<int64_t> recv_offsets_vec(world_size, 0);
  for (int64_t peer = 0; peer < world_size; ++peer) {
    int64_t offset = 0;
    for (int64_t sender = 0; sender < my_rank; ++sender) {
      offset += counts_matrix[sender][peer];
    }
    recv_offsets_vec[peer] = offset;
  }

  int64_t total_recv = 0;
  for (auto value : recv_counts_vec) {
    total_recv += value;
  }

  int64_t max_recv = 0;
  int64_t max_send_total = 0;
  for (int64_t rank = 0; rank < world_size; ++rank) {
    int64_t total = 0;
    for (int64_t sender = 0; sender < world_size; ++sender) {
      total += counts_matrix[sender][rank];
    }
    if (total > max_recv) {
      max_recv = total;
    }
  }

  for (int64_t rank = 0; rank < world_size; ++rank) {
    int64_t total = 0;
    for (int64_t dest = 0; dest < world_size; ++dest) {
      total += counts_matrix[rank][dest];
    }
    if (total > max_send_total) {
      max_send_total = total;
    }
  }

  int64_t max_send = 0;
  for (auto value : send_counts_vec) {
    if (value > max_send) {
      max_send = value;
    }
  }

  auto cpu_options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto send_offsets_cpu = at::empty({world_size}, cpu_options);
  std::memcpy(
      send_offsets_cpu.data_ptr<int64_t>(),
      send_offsets_vec.data(),
      world_size * sizeof(int64_t));
  auto recv_offsets_cpu = at::empty({world_size}, cpu_options);
  std::memcpy(
      recv_offsets_cpu.data_ptr<int64_t>(),
      recv_offsets_vec.data(),
      world_size * sizeof(int64_t));
  auto recv_counts_cpu = at::empty({world_size}, cpu_options);
  std::memcpy(
      recv_counts_cpu.data_ptr<int64_t>(),
      recv_counts_vec.data(),
      world_size * sizeof(int64_t));

  AlltoallvMetadata metadata;
  metadata.send_counts = send_counts;
  metadata.recv_counts = recv_counts_cpu.to(send_counts.device());
  metadata.send_offsets = send_offsets_cpu.to(send_counts.device());
  metadata.recv_offsets = recv_offsets_cpu.to(send_counts.device());
  metadata.total_recv = total_recv;
  metadata.max_recv = max_recv;
  metadata.max_send_total = max_send_total;
  metadata.max_send_bytes = max_send;
  metadata.world_size = world_size;
  return metadata;
}

void alltoallvWithCudaBackend(
    const at::Tensor& send,
    const at::Tensor& recv,
    const AlltoallvMetadata& metadata,
    const std::vector<void*>& recv_ptrs,
    CUstream stream) {
  NVF_CHECK(send.is_cuda(), "alltoallv send must be CUDA.");
  NVF_CHECK(recv.is_cuda(), "alltoallv recv must be CUDA.");
  NVF_CHECK(
      (int64_t)recv_ptrs.size() == metadata.world_size,
      "recv_ptrs size must match world size.");

  auto cpu_options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto recv_ptrs_cpu = at::empty({metadata.world_size}, cpu_options);
  auto* ptrs = recv_ptrs_cpu.data_ptr<int64_t>();
  for (int64_t rank = 0; rank < metadata.world_size; ++rank) {
    ptrs[rank] =
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(recv_ptrs[rank]));
  }
  auto recv_ptrs_cuda = recv_ptrs_cpu.to(send.device());

  const int64_t elem_stride =
      metadata.max_send_total > 0 ? send.numel() / metadata.max_send_total : 1;
  NVF_CHECK(
      metadata.max_send_total == 0 ||
          send.numel() % metadata.max_send_total == 0,
      "alltoallv send numel must be divisible by max_send_total.");
  NVF_CHECK(
      metadata.max_recv == 0 || recv.numel() % metadata.max_recv == 0,
      "alltoallv recv numel must be divisible by max_recv.");

  auto send_offsets = metadata.send_offsets;
  auto send_counts = metadata.send_counts;
  auto recv_offsets = metadata.recv_offsets;
  int64_t max_send_bytes = metadata.max_send_bytes;
  if (elem_stride > 1) {
    send_offsets = metadata.send_offsets * elem_stride;
    send_counts = metadata.send_counts * elem_stride;
    recv_offsets = metadata.recv_offsets * elem_stride;
    max_send_bytes = metadata.max_send_bytes * elem_stride;
  }

  launchAlltoallvKernel(
      send.data_ptr(),
      reinterpret_cast<const uint64_t*>(recv_ptrs_cuda.data_ptr<int64_t>()),
      send_offsets.data_ptr<int64_t>(),
      send_counts.data_ptr<int64_t>(),
      recv_offsets.data_ptr<int64_t>(),
      metadata.world_size,
      send.element_size(),
      max_send_bytes * send.element_size(),
      stream);
}

void alltoallvBarrier(const std::string& tag) {
  Communicator& comm = Communicator::getInstance();
  comm.barrier();
}

} // namespace nvfuser
