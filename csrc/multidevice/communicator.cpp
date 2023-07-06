// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED

#include <multidevice/communicator.h>
#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#endif
#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif
#ifdef USE_C10D_UCC
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#endif

namespace nvfuser {

// Parse the environment to retrieve MPI rank and MPI world size and sets rank
// and size accordingly Returns 0 in case of success, 1 otherwise
int parseEnv(RankType& rank, int64_t& size) {
  char* env = nullptr;

  // retrieves the rank of the current process
  env = std::getenv("OMPI_COMM_WORLD_RANK");
  if (!env) {
    env = std::getenv("WORLD_RANK");
    if (!env) {
      return 1;
    }
  }
  rank = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_SIZE");
  if (!env) {
    env = std::getenv("WORLD_SIZE");
    if (!env) {
      return 1;
    }
  }
  size = std::atoi(env);
  return 0;
}

// creates and return a process group backend
c10::intrusive_ptr<c10d::Backend> createBackend(
    CommunicatorBackend backend,
    ::c10::intrusive_ptr<c10d::TCPStore> store,
    RankType rank,
    int64_t size) {
#ifdef USE_C10D_NCCL
  if (backend == CommunicatorBackend::nccl) {
    auto pg_opts = c10::make_intrusive<::c10d::ProcessGroupNCCL::Options>();
    return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
        store, rank, size, pg_opts);
  }
#endif

#ifdef USE_C10D_GLOO
  if (backend == CommunicatorBackend::gloo) {
    auto pg_opts = c10d::ProcessGroupGloo::Options::create();
    return c10::make_intrusive<::c10d::ProcessGroupGloo>(
        store, rank, size, pg_opts);
  }
#endif

#ifdef USE_C10D_UCC
  if (backend == CommunicatorBackend::ucc) {
    auto pg_opts = c10d::ProcessGroupUCC::Options::create();
    return c10::make_intrusive<::c10d::ProcessGroupUCC>(
        store, rank, size, pg_opts);
  }
#endif
  TORCH_CHECK(false, "no distributed backend available");
}

Communicator::Communicator(CommunicatorBackend backend, RankType server_rank)
    : rank_(0), size_(0) {
  // retrieves rank and communicator size
  int status = parseEnv(rank_, size_);
  TORCH_CHECK(status == 0, "distributed configuration is not available");

  // creates the backend
  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (rank_ == server_rank) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);
  pg_ = createBackend(backend, store, rank_, size_);
}

void Communicator::sendRecv(
    RankType receiver_rank,
    RankType sender_rank,
    std::vector<at::Tensor>& tensor,
    int tag) {
  if (sender_rank == receiver_rank) { // send-to-self
    return;
  }
  if (rank() == sender_rank) {
    // post send and wait for completion
    TORCH_INTERNAL_ASSERT(
        pg_->send(tensor, receiver_rank, tag)->wait(),
        "error during communication");
  } else if (rank() == receiver_rank) {
    // post receive and wait for completion
    TORCH_INTERNAL_ASSERT(
        pg_->recv(tensor, sender_rank, tag)->wait(),
        "error during communication");
  }
}

} // namespace nvfuser

#endif
