// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "multidevice/dispatch_combine.h"

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/bincount.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/floor_divide.h>
#include <c10/cuda/CUDAGuard.h>

#include "exceptions.h"
#include "multidevice/communicator.h"
#include "multidevice/cuda_p2p.h"
#include "multidevice/symmetric_tensor.h"
#include "utils.h"

namespace nvfuser {
namespace {

std::vector<int64_t> toSplitSizes(const at::Tensor& sizes_tensor) {
  auto cpu_sizes = sizes_tensor.to(at::kCPU);
  auto* ptr = cpu_sizes.data_ptr<int64_t>();
  return std::vector<int64_t>(ptr, ptr + cpu_sizes.numel());
}

int64_t sumSplitSizes(const std::vector<int64_t>& splits) {
  int64_t total = 0;
  for (auto value : splits) {
    total += value;
  }
  return total;
}

void waitWork(const c10::intrusive_ptr<c10d::Work>& work) {
  if (work) {
    work->wait();
  }
}

} // namespace

DispatchResult doMoeDispatch(
    const at::Tensor& x,
    const at::Tensor& topk_idx,
    const at::Tensor& topk_weights,
    int64_t num_experts,
    Communicator* communicator,
    CommunicatorBackend backend) {
  NVF_CHECK(communicator != nullptr, "Dispatch requires a valid communicator.");
  NVF_CHECK(x.is_cuda(), "Dispatch input x must be on CUDA.");
  NVF_CHECK(topk_idx.is_cuda(), "Dispatch topk_idx must be on CUDA.");
  NVF_CHECK(topk_weights.is_cuda(), "Dispatch topk_weights must be on CUDA.");
  NVF_CHECK(
      topk_weights.is_floating_point(),
      "Dispatch topk_weights must be floating point.");
  NVF_CHECK(
      x.device() == topk_idx.device(),
      "Dispatch expects x and topk_idx on the same device.");
  NVF_CHECK(
      x.device() == topk_weights.device(),
      "Dispatch expects x and topk_weights on the same device.");
  NVF_CHECK_EQ(x.dim(), 2, "Dispatch expects x to be 2D [tokens, hidden].");

  const int64_t num_tokens = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t world_size = communicator->size();
  NVF_CHECK_EQ(num_experts % world_size, 0, "num_experts must be divisible.");
  const int64_t experts_per_rank = num_experts / world_size;

  NVF_CHECK(
      topk_idx.dim() == 2 && topk_idx.size(0) == num_tokens &&
          topk_idx.size(1) == 1,
      "Only topk=1 supported. topk_idx must be shape [T, 1], got: ",
      topk_idx.sizes());
  auto topk_idx_flat = topk_idx.reshape({num_tokens});
  NVF_CHECK(
      topk_weights.dim() == 2 && topk_weights.size(0) == num_tokens &&
          topk_weights.size(1) == 1,
      "Only topk=1 supported. topk_weights must be shape [T, 1], got: ",
      topk_weights.sizes());

  // Assume contiguous expert placement: rank = expert_id / experts_per_rank.
  auto topk_idx_long = topk_idx_flat.to(at::kLong);
  auto rank_for_token = at::floor_divide(topk_idx_long, experts_per_rank);
  // Sorting by expert id groups tokens by rank and by expert within rank.
  auto sorted_indices = at::argsort(topk_idx_long);

  // Reorder payloads so alltoall can send contiguous chunks per rank.
  auto send_x = x.index_select(0, sorted_indices);
  auto send_topk_idx = topk_idx.index_select(0, sorted_indices);
  auto send_topk_weights = topk_weights.index_select(0, sorted_indices);
  // Track original token indices for the combine step.
  auto send_src_idx = sorted_indices.to(at::kLong);

  // Split metadata is exchanged via CPU (TCPStore), so we sync/copy here.
  auto rank_for_token_cpu = rank_for_token.to(at::kCPU);
  auto n_tokens_to_rank_cpu =
      at::bincount(rank_for_token_cpu, {}, world_size).to(at::kLong);
  auto n_tokens_to_rank = n_tokens_to_rank_cpu.to(x.device());
  if (backend == CommunicatorBackend::kNccl) {
    NVF_CHECK(
        communicator->isBackendAvailable(backend),
        "Backend not available for dispatch: ",
        backend);
    auto* pg = communicator->getWorld(backend);
    NVF_CHECK(pg != nullptr, "Dispatch backend is null.");

    auto n_tokens_from_rank = at::empty_like(n_tokens_to_rank);
    std::vector<int64_t> one_split(world_size, 1);
    waitWork(pg->alltoall_base(
        n_tokens_from_rank, n_tokens_to_rank, one_split, one_split));

    auto input_splits = toSplitSizes(n_tokens_to_rank);
    auto output_splits = toSplitSizes(n_tokens_from_rank);
    auto total_recv = sumSplitSizes(output_splits);

    // Allocate receive buffers for payloads and metadata.
    // TODO: support preallocated buffers.
    auto recv_x = at::empty({total_recv, hidden}, x.options());
    auto recv_topk_idx =
        at::empty({total_recv, topk_idx.size(1)}, topk_idx.options());
    auto recv_topk_weights =
        at::empty({total_recv, topk_weights.size(1)}, topk_weights.options());
    auto recv_src_idx = at::empty({total_recv}, send_src_idx.options());

    // Alltoall exchange payloads with per-rank splits.
    waitWork(pg->alltoall_base(recv_x, send_x, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_topk_idx, send_topk_idx, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_topk_weights, send_topk_weights, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_src_idx, send_src_idx, output_splits, input_splits));

    return DispatchResult{
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        n_tokens_to_rank,
        n_tokens_from_rank};
  }

  NVF_CHECK_EQ(
      backend, CommunicatorBackend::kCuda,
      "Only CUDA and NCCL backends are supported for MoeDispatch.");

  auto metadata =
      prepareAlltoallvMetadata(n_tokens_to_rank, "moe_dispatch_counts");
  auto n_tokens_from_rank = metadata.recv_counts;
  const int64_t total_recv = metadata.total_recv;
  const int64_t max_recv = metadata.max_recv;

  // Allocate symmetric buffers for send/recv payloads.
  auto send_x_sym = SymmetricTensor::allocate(
      {metadata.max_send_total, hidden}, x.scalar_type(), x.device());
  send_x_sym.narrow(0, 0, num_tokens).copy_(send_x);
  auto send_topk_idx_sym = SymmetricTensor::allocate(
      {metadata.max_send_total, topk_idx.size(1)},
      topk_idx.scalar_type(),
      x.device());
  send_topk_idx_sym.narrow(0, 0, num_tokens).copy_(send_topk_idx);
  auto send_topk_weights_sym = SymmetricTensor::allocate(
      {metadata.max_send_total, topk_weights.size(1)},
      topk_weights.scalar_type(),
      x.device());
  send_topk_weights_sym.narrow(0, 0, num_tokens).copy_(send_topk_weights);
  auto send_src_idx_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, send_src_idx.scalar_type(), x.device());
  send_src_idx_sym.narrow(0, 0, num_tokens).copy_(send_src_idx);

  auto recv_x_sym = SymmetricTensor::allocate(
      {max_recv, hidden}, x.scalar_type(), x.device());
  auto recv_topk_idx_sym = SymmetricTensor::allocate(
      {max_recv, topk_idx.size(1)}, topk_idx.scalar_type(), x.device());
  auto recv_topk_weights_sym = SymmetricTensor::allocate(
      {max_recv, topk_weights.size(1)}, topk_weights.scalar_type(), x.device());
  auto recv_src_idx_sym = SymmetricTensor::allocate(
      {max_recv}, send_src_idx.scalar_type(), x.device());

  SymmetricTensor recv_x_handle(recv_x_sym);
  SymmetricTensor recv_topk_idx_handle(recv_topk_idx_sym);
  SymmetricTensor recv_topk_weights_handle(recv_topk_weights_sym);
  SymmetricTensor recv_src_idx_handle(recv_src_idx_sym);
  recv_x_handle.setupRemoteHandles("moe_dispatch_recv_x");
  recv_topk_idx_handle.setupRemoteHandles("moe_dispatch_recv_topk_idx");
  recv_topk_weights_handle.setupRemoteHandles("moe_dispatch_recv_topk_weights");
  recv_src_idx_handle.setupRemoteHandles("moe_dispatch_recv_src_idx");

  std::vector<void*> recv_x_ptrs(world_size);
  std::vector<void*> recv_topk_idx_ptrs(world_size);
  std::vector<void*> recv_topk_weights_ptrs(world_size);
  std::vector<void*> recv_src_idx_ptrs(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    recv_x_ptrs[rank] = recv_x_handle.remoteTensor(rank).data_ptr();
    recv_topk_idx_ptrs[rank] =
        recv_topk_idx_handle.remoteTensor(rank).data_ptr();
    recv_topk_weights_ptrs[rank] =
        recv_topk_weights_handle.remoteTensor(rank).data_ptr();
    recv_src_idx_ptrs[rank] = recv_src_idx_handle.remoteTensor(rank).data_ptr();
  }

  auto stream =
      static_cast<CUstream>(at::cuda::getDefaultCUDAStream().stream());
  alltoallvWithCudaBackend(
      send_x_sym, recv_x_sym, metadata, recv_x_ptrs, stream);
  alltoallvWithCudaBackend(
      send_topk_idx_sym,
      recv_topk_idx_sym,
      metadata,
      recv_topk_idx_ptrs,
      stream);
  alltoallvWithCudaBackend(
      send_topk_weights_sym,
      recv_topk_weights_sym,
      metadata,
      recv_topk_weights_ptrs,
      stream);
  alltoallvWithCudaBackend(
      send_src_idx_sym, recv_src_idx_sym, metadata, recv_src_idx_ptrs, stream);
  alltoallvBarrier("moe_dispatch_counts");

  auto recv_x = recv_x_sym.narrow(0, 0, total_recv);
  auto recv_topk_idx = recv_topk_idx_sym.narrow(0, 0, total_recv);
  auto recv_topk_weights = recv_topk_weights_sym.narrow(0, 0, total_recv);
  auto recv_src_idx = recv_src_idx_sym.narrow(0, 0, total_recv);

  return DispatchResult{
      recv_x,
      recv_topk_idx,
      recv_topk_weights,
      recv_src_idx,
      n_tokens_to_rank,
      n_tokens_from_rank};
}

CombineResult doMoeCombine(
    const at::Tensor& x,
    const at::Tensor& topk_weights,
    const at::Tensor& src_idx,
    const at::Tensor& n_tokens_to_rank,
    const at::Tensor& n_tokens_from_rank,
    Communicator* communicator,
    CommunicatorBackend backend) {
  NVF_CHECK(communicator != nullptr, "Combine requires a valid communicator.");
  NVF_CHECK(x.is_cuda(), "Combine input x must be on CUDA.");
  const bool has_topk_weights = topk_weights.numel() > 0;
  if (has_topk_weights) {
    NVF_CHECK(topk_weights.is_cuda(), "Combine topk_weights must be on CUDA.");
    NVF_CHECK(
        topk_weights.is_floating_point(),
        "Combine topk_weights must be floating point.");
    NVF_CHECK(
        topk_weights.dim() == 2 && topk_weights.size(0) == x.size(0) &&
            topk_weights.size(1) == 1,
        "topk_weights must be shape [T, 1], got: ",
        topk_weights.sizes());
  }
  NVF_CHECK(src_idx.is_cuda(), "Combine src_idx must be on CUDA.");
  NVF_CHECK(
      n_tokens_to_rank.is_cuda(), "Combine n_tokens_to_rank must be CUDA.");
  NVF_CHECK(
      n_tokens_from_rank.is_cuda(), "Combine n_tokens_from_rank must be CUDA.");
  NVF_CHECK_EQ(x.dim(), 2, "Combine expects x to be 2D [tokens, hidden].");
  NVF_CHECK_EQ(src_idx.dim(), 1, "src_idx must be 1D.");
  NVF_CHECK_EQ(
      src_idx.size(0), x.size(0), "src_idx size must match x first dimension.");
  NVF_CHECK_EQ(
      n_tokens_to_rank.numel(),
      communicator->size(),
      "n_tokens_to_rank must match world size.");
  NVF_CHECK_EQ(
      n_tokens_from_rank.numel(),
      communicator->size(),
      "n_tokens_from_rank must match world size.");

  // Reconstruct source ranks from per-rank counts. alltoall_base concatenates
  // received chunks in rank order, so this matches the receive layout.
  auto src_rank = at::arange(
                      n_tokens_from_rank.numel(),
                      at::TensorOptions().dtype(at::kLong).device(x.device()))
                      .repeat_interleave(n_tokens_from_rank.to(at::kLong));
  NVF_CHECK_EQ(
      src_rank.size(0),
      x.size(0),
      "Reconstructed src_rank must match x first dimension.");
  // Sort by source rank so alltoall can send contiguous chunks per rank.
  auto sorted_indices = at::argsort(src_rank);
  auto send_x = x.index_select(0, sorted_indices);
  auto send_src_idx = src_idx.index_select(0, sorted_indices);

  // Split sizes come from dispatch counts.
  if (backend == CommunicatorBackend::kNccl) {
    NVF_CHECK(
        communicator->isBackendAvailable(backend),
        "Backend not available for combine: ",
        backend);
    auto* pg = communicator->getWorld(backend);
    NVF_CHECK(pg != nullptr, "Combine backend is null.");

    auto input_splits = toSplitSizes(n_tokens_from_rank);
    auto output_splits = toSplitSizes(n_tokens_to_rank);
    auto total_recv = sumSplitSizes(output_splits);
    auto hidden = x.size(1);

    auto recv_x = at::empty({total_recv, hidden}, x.options());
    auto recv_src_idx = at::empty({total_recv}, src_idx.options());

    waitWork(pg->alltoall_base(recv_x, send_x, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_src_idx, send_src_idx, output_splits, input_splits));

    auto combined_x = at::empty({total_recv, hidden}, x.options());
    combined_x.index_copy_(0, recv_src_idx, recv_x);

    // topk_weights is reserved for future weighted combine.
    (void)topk_weights;
    return CombineResult{combined_x};
  }

  NVF_CHECK(
      backend == CommunicatorBackend::kCuda,
      "Only CUDA and NCCL backends are supported for MoECombine.");
  const int64_t world_size = communicator->size();

  auto metadata =
      prepareAlltoallvMetadata(n_tokens_from_rank, "moe_combine_counts");
  const int64_t total_recv = metadata.total_recv;
  const int64_t max_recv = metadata.max_recv;
  auto hidden = x.size(1);

  // Allocate symmetric buffers for send/recv payloads.
  auto send_x_sym = SymmetricTensor::allocate(
      {metadata.max_send_total, hidden}, x.scalar_type(), x.device());
  send_x_sym.narrow(0, 0, x.size(0)).copy_(send_x);
  auto send_src_idx_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, src_idx.scalar_type(), x.device());
  send_src_idx_sym.narrow(0, 0, x.size(0)).copy_(send_src_idx);

  auto recv_x_sym = SymmetricTensor::allocate(
      {max_recv, hidden}, x.scalar_type(), x.device());
  auto recv_src_idx_sym =
      SymmetricTensor::allocate({max_recv}, src_idx.scalar_type(), x.device());

  SymmetricTensor recv_x_handle(recv_x_sym);
  SymmetricTensor recv_src_idx_handle(recv_src_idx_sym);
  recv_x_handle.setupRemoteHandles("moe_combine_recv_x");
  recv_src_idx_handle.setupRemoteHandles("moe_combine_recv_src_idx");

  std::vector<void*> recv_x_ptrs(world_size);
  std::vector<void*> recv_src_idx_ptrs(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    recv_x_ptrs[rank] = recv_x_handle.remoteTensor(rank).data_ptr();
    recv_src_idx_ptrs[rank] = recv_src_idx_handle.remoteTensor(rank).data_ptr();
  }

  auto stream =
      static_cast<CUstream>(at::cuda::getDefaultCUDAStream().stream());
  alltoallvWithCudaBackend(
      send_x_sym, recv_x_sym, metadata, recv_x_ptrs, stream);
  alltoallvWithCudaBackend(
      send_src_idx_sym, recv_src_idx_sym, metadata, recv_src_idx_ptrs, stream);
  alltoallvBarrier("moe_combine_counts");

  auto recv_x = recv_x_sym.narrow(0, 0, total_recv);
  auto recv_src_idx = recv_src_idx_sym.narrow(0, 0, total_recv);

  // Scatter by original token index to restore local order.
  auto combined_x = at::empty({total_recv, hidden}, x.options());
  combined_x.index_copy_(0, recv_src_idx, recv_x);

  // topk_weights is reserved for future weighted combine.
  (void)topk_weights;
  return CombineResult{combined_x};
}

} // namespace nvfuser
