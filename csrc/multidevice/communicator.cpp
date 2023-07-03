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
#if defined(USE_C10D_UCC) && defined(NVFUSER_BUILD_WITH_UCC)
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#endif

namespace nvfuser {

// Parse the environment to retrieve MPI rank and MPI world size and sets rank
// and size accordingly Returns 0 in case of success, 1 otherwise
int parseEnv(
    RankType& rank,
    int64_t& size,
    RankType& local_rank,
    int64_t& local_size,
    std::string& master_addr,
    int& master_port) {
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

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) {
    env = std::getenv("WORLD_LOCAL_RANK");
    if (!env) {
      return 1;
    }
  }
  local_rank = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env) {
    env = std::getenv("WORLD_LOCAL_SIZE");
    if (!env) {
      return 1;
    }
  }
  local_size = std::atoi(env);

  // retrieves master address
  env = std::getenv("MASTER_ADDR");
  if (env) {
    master_addr = env;
  } else if (local_size == size) {
    master_addr = "localhost";
  } else {
    TORCH_WARN("the environment variable MASTER_ADDR "
    "must be specified.");
    return 1;
  }

  // retrieves master port
  env = std::getenv("MASTER_PORT");
  if (env) {
    master_port = std::atoi(env);
  } else {
    if (master_addr != "localhost") {
      TORCH_WARN("the environment variable MASTER_PORT "
      "has not been specified");
    }
    master_port = 0;
  }

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

#if defined(USE_C10D_UCC) && defined(NVFUSER_BUILD_WITH_UCC)
  if (backend == CommunicatorBackend::ucc) {
    return c10::make_intrusive<::c10d::ProcessGroupUCC>(store, rank, size);
  }
#endif
  TORCH_CHECK(false, "no distributed backend available");
}

Communicator::Communicator(CommunicatorBackend backend, RankType server_rank)
    : rank_(0), size_(0), local_rank_(0), local_size_(0) {
  // retrieves rank and communicator size
  int status = parseEnv(rank_, size_, local_rank_, local_size_, master_addr_, master_port_);
  TORCH_CHECK(status == 0, "distributed configuration is not available");

  if (rank_ == server_rank) {
    char hostname_c[HOST_NAME_MAX];
    gethostname(hostname_c, HOST_NAME_MAX);
    std::string hostname = hostname_c;
    if (hostname != master_addr_) {
      TORCH_WARN("server rank's hostname=" + hostname +
                      " but MASTER_ADDR=" + master_addr_);
    }
  }

  // creates the backend
  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (rank_ == server_rank) ? true : false;
  if (master_port_) {
    store_opts.port = master_port_;
  }
  auto store = c10::make_intrusive<c10d::TCPStore>(master_addr_, store_opts);
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
