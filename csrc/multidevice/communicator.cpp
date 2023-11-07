// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <netdb.h>

#include <multidevice/communicator.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
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

// Parse the environment to retrieve MPI rank, world size, local rank,
// local world size, and also master address and master port.
// Returns true if the distributed configuration is valid, false otherwise
bool parseEnv(
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
      return false;
    }
  }
  rank = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_SIZE");
  if (!env) {
    env = std::getenv("WORLD_SIZE");
    if (!env) {
      return false;
    }
  }
  size = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) {
    env = std::getenv("WORLD_LOCAL_RANK");
    if (!env) {
      return false;
    }
  }
  local_rank = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env) {
    env = std::getenv("WORLD_LOCAL_SIZE");
    if (!env) {
      return false;
    }
  }
  local_size = std::atoi(env);

  // retrieves master address
  env = std::getenv("MASTER_ADDR");
  if (env) {
    // replace the potential aliased hostname by the "official" name
    master_addr = gethostbyname(env)->h_name;
  } else if (local_size == size) {
    master_addr = "localhost";
  } else {
    TORCH_WARN(
        "the environment variable MASTER_ADDR "
        "must be specified in multi-node environment");
    return false;
  }

  // retrieves master port
  env = std::getenv("MASTER_PORT");
  if (env) {
    master_port = std::atoi(env);
  } else {
    TORCH_WARN(
        "the environment variable MASTER_PORT "
        "has not been specified. Set to default");
  }

  return true;
}

inline std::string getTeamKey(const Team& team, CommunicatorBackend backend) {
  std::string backend_str = (backend == CommunicatorBackend::ucc) ? "ucc" : "nccl";
  return std::accumulate(
      std::begin(team),
      std::end(team),
      std::string{backend_str},
      [](const std::string& a, const RankType& b) {
        return a.empty() ? std::to_string(b) : a + ',' + std::to_string(b);
      });
}

// creates and return a process group backend
c10::intrusive_ptr<c10d::Backend> createBackend(
    CommunicatorBackend backend,
    ::c10::intrusive_ptr<c10d::Store> store,
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
  NVF_CHECK(false, "no distributed backend available");
}

Communicator::Communicator(
    CommunicatorBackend backend,
    RankType server_local_rank)
    : is_available_(false),
      default_backend_(backend),
      rank_(0),
      size_(0),
      local_rank_(0),
      local_size_(0),
      master_port_(0) {
  // retrieves rank and communicator size
  is_available_ = parseEnv(
      rank_, size_, local_rank_, local_size_, master_addr_, master_port_);

  if (!is_available_) {
    return;
  }

  c10d::TCPStoreOptions store_opts;
  {
    char hostname[HOST_NAME_MAX]; // NOLINT (modernize-avoid-c-arrays)
    NVF_ERROR(
        gethostname(hostname, HOST_NAME_MAX) == 0,
        "error when retrieving hostname");
    // we define the server as the process at the master host with local rank 0
    store_opts.isServer = (master_addr_ == "localhost" ||
                           master_addr_ == gethostbyname(hostname)->h_name) &&
        local_rank_ == server_local_rank;
  }
  store_opts.port = master_port_ ? master_port_ : comm_master_port_default;
  store_ = c10::make_intrusive<c10d::TCPStore>(master_addr_, store_opts);

  addBackend(backend);
}

void Communicator::addBackend(CommunicatorBackend backend) {
  std::vector<RankType> all_ranks(size_);
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
  // create backend's world.
  world_[backend] = getBackendForTeam(all_ranks, backend);
}

c10::intrusive_ptr<c10d::Backend> Communicator::getBackendForTeam(
    const Team& team, CommunicatorBackend backend) {
  if (backend == CommunicatorBackend::none) 
    backend = default_backend_;
  std::string team_key = getTeamKey(team, backend);
  // check if backend associated with the team is present in the cache
  if (backends_.find(team_key) ==
      backends_.end()) { // create the backend and cache it
    // check that the caller's rank belongs to the requested team
    auto rank_it = std::find(team.begin(), team.end(), deviceId());
    NVF_ERROR(
        rank_it != team.end(),
        "only devices in the team should participate to its initialization");
    // retrieve the caller's rank index/position in the team
    RankType team_rank = std::distance(team.begin(), rank_it);
    // generate a string key which is unique to the team
    // create the team and cache it
    backends_[team_key] = createBackend(
        backend,
        c10::make_intrusive<c10d::PrefixStore>(team_key, store_),
        team_rank,
        static_cast<int64_t>(team.size()));
  }
  return backends_.at(team_key);
}

c10::intrusive_ptr<c10d::Work> Communicator::sendRecv(
    DeviceIdxType receiver,
    DeviceIdxType sender,
    std::vector<at::Tensor>& tensors,
    CommunicatorBackend backend,
    int tag) {
  NVF_ERROR(
      deviceId() == sender || deviceId() == receiver,
      "only sender or receiver should post the sendRecv");
  NVF_ERROR(sender != receiver, "cannot send to self");
  if (backend == CommunicatorBackend::none) {
    backend = default_backend_;
  }

  auto world = world_[backend];

  if (deviceId() == sender) {
    return world->send(tensors, static_cast<int>(dIdToRank(receiver)), tag);
  }
  return world->recv(tensors, static_cast<int>(dIdToRank(sender)), tag);
}

} // namespace nvfuser

#endif
