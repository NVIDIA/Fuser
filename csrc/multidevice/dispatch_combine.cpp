// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "multidevice/dispatch_combine.h"

#include <tuple>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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

at::Tensor flattenTopk(const at::Tensor& topk, int64_t num_tokens) {
  const bool is_1d = topk.dim() == 1 && topk.size(0) == num_tokens;
  const bool is_2d =
      topk.dim() == 2 && topk.size(0) == num_tokens && topk.size(1) == 1;
  NVF_CHECK(
      is_1d || is_2d,
      "Only topk=1 supported. topk_idx/weights must be shape [T] or [T, 1], "
      "got: ",
      topk.sizes());
  return topk.reshape({num_tokens});
}

} // namespace

DispatchResult doMoEDispatch(
    const at::Tensor& x,
    const at::Tensor& topk_idx,
    const at::Tensor& topk_weights,
    const at::Tensor& is_token_in_rank,
    int64_t num_experts,
    Communicator* communicator,
    CommunicatorBackend backend) {
  NVF_CHECK(communicator != nullptr, "Dispatch requires a valid communicator.");
  NVF_CHECK(x.is_cuda(), "Dispatch input x must be on CUDA.");
  NVF_CHECK(topk_idx.is_cuda(), "Dispatch topk_idx must be on CUDA.");
  NVF_CHECK(topk_weights.is_cuda(), "Dispatch topk_weights must be on CUDA.");
  NVF_CHECK(
      is_token_in_rank.is_cuda(), "Dispatch is_token_in_rank must be on CUDA.");
  NVF_CHECK(
      is_token_in_rank.dim() == 2,
      "is_token_in_rank must be 2D [tokens, ranks], got: ",
      is_token_in_rank.sizes());
  NVF_CHECK(x.dim() == 2, "Dispatch expects x to be 2D [tokens, hidden].");

  const int64_t num_tokens = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t world_size = communicator->size();
  const int64_t my_rank = communicator->deviceId();
  NVF_CHECK(
      is_token_in_rank.size(1) == world_size,
      "is_token_in_rank second dim must match world size.");
  NVF_CHECK(num_experts % world_size == 0, "num_experts must be divisible.");

  c10::cuda::CUDAGuard device_guard(x.device());
  NVF_CHECK(
      [&]() {
        auto token_counts = is_token_in_rank.to(at::kLong).sum(1);
        auto min_val = token_counts.min().item<int64_t>();
        auto max_val = token_counts.max().item<int64_t>();
        return min_val == 1 && max_val == 1;
      }(),
      "Only topk=1 is supported. Each token must be assigned to exactly one "
      "rank.");

  auto topk_idx_flat = flattenTopk(topk_idx, num_tokens);
  auto topk_weights_flat = flattenTopk(topk_weights, num_tokens);

  // Determine destination rank per token (topk=1).
  auto rank_for_token = is_token_in_rank.to(at::kLong).argmax(1).to(at::kLong);
  // Sort tokens by destination rank for contiguous alltoall slices.
  auto sorted = rank_for_token.sort();
  auto sorted_indices = std::get<1>(sorted);

  // Reorder payloads so alltoall can send contiguous chunks per rank.
  auto send_x = x.index_select(0, sorted_indices);
  auto send_topk_idx = topk_idx_flat.index_select(0, sorted_indices);
  auto send_topk_weights = topk_weights_flat.index_select(0, sorted_indices);
  // Track original token indices and source rank for the combine step.
  auto send_src_idx = sorted_indices.to(at::kLong);
  // All entries are identical, so no relayout is needed.
  auto send_src_rank = at::full(
      {num_tokens},
      my_rank,
      at::TensorOptions().dtype(at::kLong).device(x.device()));

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

    auto recv_x = at::empty({total_recv, hidden}, x.options());
    auto recv_topk_idx = at::empty({total_recv}, topk_idx_flat.options());
    auto recv_topk_weights =
        at::empty({total_recv}, topk_weights_flat.options());
    auto recv_src_idx = at::empty({total_recv}, send_src_idx.options());
    auto recv_src_rank = at::empty({total_recv}, send_src_rank.options());

    waitWork(pg->alltoall_base(recv_x, send_x, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_topk_idx, send_topk_idx, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_topk_weights, send_topk_weights, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_src_idx, send_src_idx, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_src_rank, send_src_rank, output_splits, input_splits));

    const int64_t experts_per_rank = num_experts / world_size;
    auto local_expert = recv_topk_idx - my_rank * experts_per_rank;
    auto expert_sorted = local_expert.sort();
    auto expert_order = std::get<1>(expert_sorted);
    recv_x = recv_x.index_select(0, expert_order);
    recv_topk_idx = recv_topk_idx.index_select(0, expert_order);
    recv_topk_weights = recv_topk_weights.index_select(0, expert_order);
    recv_src_idx = recv_src_idx.index_select(0, expert_order);
    recv_src_rank = recv_src_rank.index_select(0, expert_order);

    return DispatchResult{
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        recv_src_rank,
        n_tokens_to_rank,
        n_tokens_from_rank};
  }

  NVF_CHECK(
      backend == CommunicatorBackend::kCuda,
      "Only CUDA and NCCL backends are supported for MoEDispatch.");

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
      {metadata.max_send_total}, topk_idx_flat.scalar_type(), x.device());
  send_topk_idx_sym.narrow(0, 0, num_tokens).copy_(send_topk_idx);
  auto send_topk_weights_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, topk_weights_flat.scalar_type(), x.device());
  send_topk_weights_sym.narrow(0, 0, num_tokens).copy_(send_topk_weights);
  auto send_src_idx_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, send_src_idx.scalar_type(), x.device());
  send_src_idx_sym.narrow(0, 0, num_tokens).copy_(send_src_idx);
  auto send_src_rank_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, send_src_rank.scalar_type(), x.device());
  send_src_rank_sym.narrow(0, 0, num_tokens).copy_(send_src_rank);

  auto recv_x_sym = SymmetricTensor::allocate(
      {max_recv, hidden}, x.scalar_type(), x.device());
  auto recv_topk_idx_sym = SymmetricTensor::allocate(
      {max_recv}, topk_idx_flat.scalar_type(), x.device());
  auto recv_topk_weights_sym = SymmetricTensor::allocate(
      {max_recv}, topk_weights_flat.scalar_type(), x.device());
  auto recv_src_idx_sym = SymmetricTensor::allocate(
      {max_recv}, send_src_idx.scalar_type(), x.device());
  auto recv_src_rank_sym = SymmetricTensor::allocate(
      {max_recv}, send_src_rank.scalar_type(), x.device());

  SymmetricTensor recv_x_handle(recv_x_sym);
  SymmetricTensor recv_topk_idx_handle(recv_topk_idx_sym);
  SymmetricTensor recv_topk_weights_handle(recv_topk_weights_sym);
  SymmetricTensor recv_src_idx_handle(recv_src_idx_sym);
  SymmetricTensor recv_src_rank_handle(recv_src_rank_sym);
  recv_x_handle.setupRemoteHandles("moe_dispatch_recv_x");
  recv_topk_idx_handle.setupRemoteHandles("moe_dispatch_recv_topk_idx");
  recv_topk_weights_handle.setupRemoteHandles("moe_dispatch_recv_topk_weights");
  recv_src_idx_handle.setupRemoteHandles("moe_dispatch_recv_src_idx");
  recv_src_rank_handle.setupRemoteHandles("moe_dispatch_recv_src_rank");

  std::vector<void*> recv_x_ptrs(world_size);
  std::vector<void*> recv_topk_idx_ptrs(world_size);
  std::vector<void*> recv_topk_weights_ptrs(world_size);
  std::vector<void*> recv_src_idx_ptrs(world_size);
  std::vector<void*> recv_src_rank_ptrs(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    recv_x_ptrs[rank] = recv_x_handle.remoteTensor(rank).data_ptr();
    recv_topk_idx_ptrs[rank] =
        recv_topk_idx_handle.remoteTensor(rank).data_ptr();
    recv_topk_weights_ptrs[rank] =
        recv_topk_weights_handle.remoteTensor(rank).data_ptr();
    recv_src_idx_ptrs[rank] = recv_src_idx_handle.remoteTensor(rank).data_ptr();
    recv_src_rank_ptrs[rank] =
        recv_src_rank_handle.remoteTensor(rank).data_ptr();
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
  alltoallvWithCudaBackend(
      send_src_rank_sym,
      recv_src_rank_sym,
      metadata,
      recv_src_rank_ptrs,
      stream);
  alltoallvBarrier("moe_dispatch_counts");
  auto recv_x = recv_x_sym.narrow(0, 0, total_recv);
  auto recv_topk_idx = recv_topk_idx_sym.narrow(0, 0, total_recv);
  auto recv_topk_weights = recv_topk_weights_sym.narrow(0, 0, total_recv);
  auto recv_src_idx = recv_src_idx_sym.narrow(0, 0, total_recv);
  auto recv_src_rank = recv_src_rank_sym.narrow(0, 0, total_recv);

  // Locally reorder by expert id so each rank processes contiguous experts.
  const int64_t experts_per_rank = num_experts / world_size;
  auto local_expert = recv_topk_idx - my_rank * experts_per_rank;
  auto expert_sorted = local_expert.sort();
  auto expert_order = std::get<1>(expert_sorted);
  recv_x = recv_x.index_select(0, expert_order);
  recv_topk_idx = recv_topk_idx.index_select(0, expert_order);
  recv_topk_weights = recv_topk_weights.index_select(0, expert_order);
  recv_src_idx = recv_src_idx.index_select(0, expert_order);
  recv_src_rank = recv_src_rank.index_select(0, expert_order);

  return DispatchResult{
      recv_x,
      recv_topk_idx,
      recv_topk_weights,
      recv_src_idx,
      recv_src_rank,
      n_tokens_to_rank,
      n_tokens_from_rank};
}

CombineResult doMoECombine(
    const at::Tensor& x,
    const at::Tensor& topk_weights,
    const at::Tensor& src_idx,
    const at::Tensor& src_rank,
    const at::Tensor& n_tokens_to_rank,
    const at::Tensor& n_tokens_from_rank,
    Communicator* communicator,
    CommunicatorBackend backend) {
  NVF_CHECK(communicator != nullptr, "Combine requires a valid communicator.");
  NVF_CHECK(x.is_cuda(), "Combine input x must be on CUDA.");
  NVF_CHECK(topk_weights.is_cuda(), "Combine topk_weights must be on CUDA.");
  NVF_CHECK(src_idx.is_cuda(), "Combine src_idx must be on CUDA.");
  NVF_CHECK(src_rank.is_cuda(), "Combine src_rank must be on CUDA.");
  NVF_CHECK(
      n_tokens_to_rank.is_cuda(), "Combine n_tokens_to_rank must be CUDA.");
  NVF_CHECK(
      n_tokens_from_rank.is_cuda(), "Combine n_tokens_from_rank must be CUDA.");
  NVF_CHECK(x.dim() == 2, "Combine expects x to be 2D [tokens, hidden].");
  NVF_CHECK(
      src_idx.dim() == 1 && src_rank.dim() == 1,
      "src_idx and src_rank must be 1D.");
  NVF_CHECK(
      n_tokens_to_rank.numel() == communicator->size(),
      "n_tokens_to_rank must match world size.");
  NVF_CHECK(
      n_tokens_from_rank.numel() == communicator->size(),
      "n_tokens_from_rank must match world size.");

  const int64_t world_size = communicator->size();
  c10::cuda::CUDAGuard device_guard(x.device());

  // Sort by source rank so alltoall can send contiguous chunks per rank.
  auto sorted = src_rank.sort();
  auto sorted_indices = std::get<1>(sorted);
  auto send_x = x.index_select(0, sorted_indices);
  auto send_topk_weights = topk_weights.index_select(0, sorted_indices);
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
    auto recv_topk_weights = at::empty({total_recv}, topk_weights.options());
    auto recv_src_idx = at::empty({total_recv}, src_idx.options());

    waitWork(pg->alltoall_base(recv_x, send_x, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_topk_weights, send_topk_weights, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_src_idx, send_src_idx, output_splits, input_splits));

    auto combined_x = at::empty({total_recv, hidden}, x.options());
    combined_x.index_copy_(0, recv_src_idx, recv_x);
    auto combined_topk_weights =
        at::empty({total_recv}, topk_weights.options());
    combined_topk_weights.index_copy_(0, recv_src_idx, recv_topk_weights);

    return CombineResult{combined_x, combined_topk_weights};
  }

  NVF_CHECK(
      backend == CommunicatorBackend::kCuda,
      "Only CUDA and NCCL backends are supported for MoECombine.");

  auto metadata =
      prepareAlltoallvMetadata(n_tokens_from_rank, "moe_combine_counts");
  const int64_t total_recv = metadata.total_recv;
  const int64_t max_recv = metadata.max_recv;
  auto hidden = x.size(1);

  // Allocate symmetric buffers for send/recv payloads.
  auto send_x_sym = SymmetricTensor::allocate(
      {metadata.max_send_total, hidden}, x.scalar_type(), x.device());
  send_x_sym.narrow(0, 0, x.size(0)).copy_(send_x);
  auto send_topk_weights_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, topk_weights.scalar_type(), x.device());
  send_topk_weights_sym.narrow(0, 0, x.size(0)).copy_(send_topk_weights);
  auto send_src_idx_sym = SymmetricTensor::allocate(
      {metadata.max_send_total}, src_idx.scalar_type(), x.device());
  send_src_idx_sym.narrow(0, 0, x.size(0)).copy_(send_src_idx);

  auto recv_x_sym = SymmetricTensor::allocate(
      {max_recv, hidden}, x.scalar_type(), x.device());
  auto recv_topk_weights_sym = SymmetricTensor::allocate(
      {max_recv}, topk_weights.scalar_type(), x.device());
  auto recv_src_idx_sym =
      SymmetricTensor::allocate({max_recv}, src_idx.scalar_type(), x.device());

  SymmetricTensor recv_x_handle(recv_x_sym);
  SymmetricTensor recv_topk_weights_handle(recv_topk_weights_sym);
  SymmetricTensor recv_src_idx_handle(recv_src_idx_sym);
  recv_x_handle.setupRemoteHandles("moe_combine_recv_x");
  recv_topk_weights_handle.setupRemoteHandles("moe_combine_recv_topk_weights");
  recv_src_idx_handle.setupRemoteHandles("moe_combine_recv_src_idx");

  std::vector<void*> recv_x_ptrs(world_size);
  std::vector<void*> recv_topk_weights_ptrs(world_size);
  std::vector<void*> recv_src_idx_ptrs(world_size);
  for (int64_t rank = 0; rank < world_size; ++rank) {
    recv_x_ptrs[rank] = recv_x_handle.remoteTensor(rank).data_ptr();
    recv_topk_weights_ptrs[rank] =
        recv_topk_weights_handle.remoteTensor(rank).data_ptr();
    recv_src_idx_ptrs[rank] = recv_src_idx_handle.remoteTensor(rank).data_ptr();
  }

  auto stream =
      static_cast<CUstream>(at::cuda::getDefaultCUDAStream().stream());
  alltoallvWithCudaBackend(
      send_x_sym, recv_x_sym, metadata, recv_x_ptrs, stream);
  alltoallvWithCudaBackend(
      send_topk_weights_sym,
      recv_topk_weights_sym,
      metadata,
      recv_topk_weights_ptrs,
      stream);
  alltoallvWithCudaBackend(
      send_src_idx_sym, recv_src_idx_sym, metadata, recv_src_idx_ptrs, stream);
  alltoallvBarrier("moe_combine_counts");

  auto recv_x = recv_x_sym.narrow(0, 0, total_recv);
  auto recv_topk_weights = recv_topk_weights_sym.narrow(0, 0, total_recv);
  auto recv_src_idx = recv_src_idx_sym.narrow(0, 0, total_recv);

  // Scatter by original token index to restore local order.
  auto combined_x = at::empty({total_recv, hidden}, x.options());
  combined_x.index_copy_(0, recv_src_idx, recv_x);
  auto combined_topk_weights = at::empty({total_recv}, topk_weights.options());
  combined_topk_weights.index_copy_(0, recv_src_idx, recv_topk_weights);

  return CombineResult{combined_x, combined_topk_weights};
}

} // namespace nvfuser
