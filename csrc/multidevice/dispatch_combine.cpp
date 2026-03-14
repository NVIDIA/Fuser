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
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_utils.h"
#include "exceptions.h"
#include "multidevice/communicator.h"
#include "multidevice/cuda_p2p.h"
#include "multidevice/ipc_handle.h"
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

// Single cache for alltoallv contexts. Each context holds the sync
// state (semaphores for counts exchange + completion) and any number
// of named recv buffers. One rendezvous per context; steady-state is
// pure cache hits.
SymMemForAlltoallv& getOrCreateAlltoallv(
    const std::string& tag,
    at::Device device) {
  static auto* cache = new std::
      unordered_map<std::string, std::unique_ptr<SymMemForAlltoallv>>();
  auto& entry = (*cache)[tag];
  if (!entry) {
    entry = std::make_unique<SymMemForAlltoallv>(device, tag);
  }
  return *entry;
}

// max_send_total and max_send_bytes are separate: max_send_total
// sizes the send buffer, max_send_bytes sizes the kernel grid X.
AlltoallvMetadata prepareAlltoallvMetadataGpu(
    SymMemForAlltoallv& ctx,
    const at::Tensor& send_counts,
    int64_t max_send_total,
    int64_t max_send_bytes,
    int64_t max_recv,
    CUstream stream) {
  const int64_t W = ctx.worldSize();
  const int64_t my_rank = ctx.myRank();
  auto gpu_opts =
      at::TensorOptions().dtype(at::kLong).device(send_counts.device());

  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
      ctx.syncBuffer().data_ptr<int64_t>(),
      send_counts.data_ptr<int64_t>(),
      W * sizeof(int64_t),
      cudaMemcpyDeviceToDevice,
      reinterpret_cast<cudaStream_t>(stream)));

  ctx.signalCountsReady(stream);
  ctx.waitCountsReady(stream);

  auto counts_matrix = at::empty({W, W}, gpu_opts);
  for (int64_t r = 0; r < W; r++) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
        counts_matrix[r].data_ptr<int64_t>(),
        reinterpret_cast<void*>(ctx.syncRemotePtr(r)),
        W * sizeof(int64_t),
        cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(stream)));
  }

  ctx.resetCountsSem(stream);

  auto recv_counts =
      counts_matrix.select(1, my_rank).contiguous();

  auto send_offsets = at::zeros({W}, gpu_opts);
  if (W > 1) {
    send_offsets.narrow(0, 1, W - 1)
        .copy_(send_counts.cumsum(0).narrow(0, 0, W - 1));
  }

  at::Tensor recv_offsets = my_rank > 0
      ? counts_matrix.narrow(0, 0, my_rank).sum(0)
      : at::zeros({W}, gpu_opts);

  return AlltoallvMetadata{
      send_counts,
      recv_counts,
      send_offsets,
      recv_offsets,
      max_recv,
      max_recv,
      max_send_total,
      max_send_bytes,
      W};
}

void alltoallvGpuSync(
    SymMemForAlltoallv& ctx,
    CUstream stream) {
  ctx.doneBarrier(stream);
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
      x.device() == topk_idx.device() && x.device() == topk_weights.device(),
      "Dispatch expects all inputs on the same device.");
  NVF_CHECK_EQ(x.dim(), 2, "Dispatch expects x to be 2D [T, H].");

  const int64_t num_tokens = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t world_size = communicator->size();
  NVF_CHECK_EQ(num_experts % world_size, 0, "num_experts must be divisible.");
  const int64_t experts_per_rank = num_experts / world_size;

  NVF_CHECK(
      topk_idx.dim() == 2 && topk_idx.size(0) == num_tokens &&
          topk_idx.size(1) == 1,
      "Only topk=1 supported. topk_idx shape: ",
      topk_idx.sizes());
  NVF_CHECK(
      topk_weights.dim() == 2 && topk_weights.size(0) == num_tokens &&
          topk_weights.size(1) == 1,
      "Only topk=1 supported. topk_weights shape: ",
      topk_weights.sizes());

  auto topk_idx_long = topk_idx.reshape({num_tokens}).to(at::kLong);
  auto rank_for_token = at::floor_divide(topk_idx_long, experts_per_rank);
  auto sorted_indices = at::argsort(topk_idx_long);

  auto send_x = x.index_select(0, sorted_indices);
  auto send_topk_idx = topk_idx.index_select(0, sorted_indices);
  auto send_topk_weights = topk_weights.index_select(0, sorted_indices);
  auto send_src_idx = sorted_indices.to(at::kLong);

  // n_tokens_to_rank[r] = number of tokens this rank sends to rank r.
  // Uses scatter_add instead of bincount — bincount has an internal
  // CPU-GPU copy that breaks CUDA graph capture.
  auto rank_for_token_long = rank_for_token.to(at::kLong);
  auto gpu_long_opts = at::TensorOptions().dtype(at::kLong).device(x.device());
  auto n_tokens_to_rank =
      at::zeros({world_size}, gpu_long_opts)
          .scatter_add(
              0, rank_for_token_long, at::ones({num_tokens}, gpu_long_opts));

  // ---------- NCCL backend (not graph-capturable) ----------
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
    auto recv_topk_idx =
        at::empty({total_recv, topk_idx.size(1)}, topk_idx.options());
    auto recv_topk_weights =
        at::empty({total_recv, topk_weights.size(1)}, topk_weights.options());
    auto recv_src_idx = at::empty({total_recv}, send_src_idx.options());

    waitWork(pg->alltoall_base(recv_x, send_x, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_topk_idx, send_topk_idx, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_topk_weights, send_topk_weights, output_splits, input_splits));
    waitWork(pg->alltoall_base(
        recv_src_idx, send_src_idx, output_splits, input_splits));

    return {
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        n_tokens_to_rank,
        n_tokens_from_rank};
  }

  // ---------- CUDA backend (graph-capturable, zero CPU-GPU sync) ----------
  //
  // Recv buffers are [C, ...] where C = T*R. Over-allocating to this
  // worst-case capacity avoids reading the data-dependent receive count
  // to the CPU, which would break CUDA graph capture. Only the first
  // V = Σ n_tokens_from_rank rows contain valid data after the alltoallv.

  NVF_CHECK_EQ(
      backend,
      CommunicatorBackend::kCuda,
      "Only CUDA and NCCL backends are supported for MoeDispatch.");

  auto stream =
      static_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());
  const int64_t capacity = num_tokens * world_size;

  auto& ctx = getOrCreateAlltoallv("moe_dispatch", x.device());

  auto metadata = prepareAlltoallvMetadataGpu(
      ctx,
      n_tokens_to_rank,
      /*max_send_total=*/num_tokens,
      /*max_send_bytes=*/num_tokens,
      /*max_recv=*/capacity,
      stream);
  auto n_tokens_from_rank = metadata.recv_counts;

  auto& rx = ctx.recv("x", capacity, {hidden}, x.scalar_type(), x.device());
  auto& ri = ctx.recv(
      "topk_idx",
      capacity,
      {topk_idx.size(1)},
      topk_idx.scalar_type(),
      x.device());
  auto& rw = ctx.recv(
      "topk_weights",
      capacity,
      {topk_weights.size(1)},
      topk_weights.scalar_type(),
      x.device());
  auto& rs =
      ctx.recv("src_idx", capacity, {}, send_src_idx.scalar_type(), x.device());

  alltoallvWithCudaBackend(send_x, rx.buffer, metadata, rx.remote_ptrs, stream);
  alltoallvWithCudaBackend(
      send_topk_idx, ri.buffer, metadata, ri.remote_ptrs, stream);
  alltoallvWithCudaBackend(
      send_topk_weights, rw.buffer, metadata, rw.remote_ptrs, stream);
  alltoallvWithCudaBackend(
      send_src_idx, rs.buffer, metadata, rs.remote_ptrs, stream);
  alltoallvGpuSync(ctx, stream);

  return {
      rx.buffer,
      ri.buffer,
      rw.buffer,
      rs.buffer,
      n_tokens_to_rank,
      n_tokens_from_rank};
}

CombineResult doMoeCombine(
    const at::Tensor& x,
    const at::Tensor& topk_weights,
    const at::Tensor& src_idx,
    const at::Tensor& n_tokens_to_rank,
    const at::Tensor& n_tokens_from_rank,
    int64_t num_tokens,
    Communicator* communicator,
    CommunicatorBackend backend) {
  NVF_CHECK(communicator != nullptr, "Combine requires a valid communicator.");
  NVF_CHECK(x.is_cuda(), "Combine input x must be on CUDA.");
  NVF_CHECK(src_idx.is_cuda(), "Combine src_idx must be on CUDA.");
  NVF_CHECK(
      n_tokens_to_rank.is_cuda() && n_tokens_from_rank.is_cuda(),
      "Combine count tensors must be on CUDA.");
  NVF_CHECK_EQ(x.dim(), 2, "Combine expects x to be 2D.");
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

  // ---------- NCCL backend (not graph-capturable) ----------
  if (backend == CommunicatorBackend::kNccl) {
    NVF_CHECK(
        communicator->isBackendAvailable(backend),
        "Backend not available for combine: ",
        backend);
    auto* pg = communicator->getWorld(backend);
    NVF_CHECK(pg != nullptr, "Combine backend is null.");

    auto src_rank = at::arange(
                        n_tokens_from_rank.numel(),
                        at::TensorOptions().dtype(at::kLong).device(x.device()))
                        .repeat_interleave(n_tokens_from_rank.to(at::kLong));
    auto sorted_indices = at::argsort(src_rank);
    auto send_x = x.index_select(0, sorted_indices);
    auto send_src_idx = src_idx.index_select(0, sorted_indices);

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

    (void)topk_weights;
    return {combined_x};
  }

  // ---------- CUDA backend (graph-capturable, zero CPU-GPU sync) ----------
  //
  // Tokens in x [C, H] are already in sender-rank order from dispatch's
  // alltoallv, so no sort is needed (repeat_interleave has a hidden .item()
  // sync that would break graph capture). The alltoallv kernel reads only
  // the valid rows via send_counts.
  //
  // Recv is [T, H] — exact, since each rank gets back its original T tokens.
  // T = num_tokens is CPU-known, so no GPU read is needed for sizing.

  NVF_CHECK(
      backend == CommunicatorBackend::kCuda,
      "Only CUDA and NCCL backends are supported for MoECombine.");

  const int64_t capacity = x.size(0);
  const int64_t hidden = x.size(1);

  auto stream =
      static_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());

  auto& ctx = getOrCreateAlltoallv("moe_combine", x.device());
  auto metadata = prepareAlltoallvMetadataGpu(
      ctx,
      n_tokens_from_rank,
      /*max_send_total=*/capacity,
      /*max_send_bytes=*/num_tokens,
      /*max_recv=*/num_tokens,
      stream);

  auto& rx = ctx.recv("x", num_tokens, {hidden}, x.scalar_type(), x.device());
  auto& rs =
      ctx.recv("src_idx", num_tokens, {}, src_idx.scalar_type(), x.device());

  alltoallvWithCudaBackend(x, rx.buffer, metadata, rx.remote_ptrs, stream);
  alltoallvWithCudaBackend(
      src_idx, rs.buffer, metadata, rs.remote_ptrs, stream);
  alltoallvGpuSync(ctx, stream);

  auto combined_x = at::zeros({num_tokens, hidden}, x.options());
  combined_x.index_copy_(
      0,
      rs.buffer.narrow(0, 0, num_tokens),
      rx.buffer.narrow(0, 0, num_tokens));

  (void)topk_weights;
  return {combined_x};
}

} // namespace nvfuser
