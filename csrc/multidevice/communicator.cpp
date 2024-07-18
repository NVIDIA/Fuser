// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/communicator.h>

#include <netdb.h>
#include <map>

#ifdef NVFUSER_DISTRIBUTED
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
#endif

namespace nvfuser {

std::ostream& operator<<(std::ostream& out, const CommunicatorBackend& cb) {
  switch (cb) {
    case CommunicatorBackend::nccl:
      out << "NCCL";
      break;
    case CommunicatorBackend::ucc:
      out << "UCC";
      break;
    case CommunicatorBackend::gloo:
      out << "GLOO";
      break;
  }
  return out;
}

namespace {
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
  std::string backend_str =
      (backend == CommunicatorBackend::ucc) ? "ucc" : "nccl";
  return std::accumulate(
      std::begin(team),
      std::end(team),
      std::string{backend_str},
      [](const std::string& a, const RankType& b) {
        return a.empty() ? std::to_string(b) : a + ',' + std::to_string(b);
      });
}

#ifdef NVFUSER_DISTRIBUTED
// creates and return a process group backend
c10::intrusive_ptr<c10d::Backend> createBackend(
    CommunicatorBackend backend,
    c10::intrusive_ptr<c10d::Store> store,
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
    constexpr auto timeout = std::chrono::milliseconds(30 * 60 * 1000);
    return c10d::ProcessGroupUCC::createProcessGroupUCC(
        store, static_cast<int>(rank), static_cast<int>(size), timeout);
  }
#endif
  NVF_ERROR(false, "no distributed backend available");
}
#endif
} // namespace

Communicator::Communicator(
    CommunicatorBackend backend,
    RankType server_local_rank)
    : is_available_(false),
      default_backend_(backend),
      rank_(0),
      size_(0),
      local_rank_(0),
      local_size_(0),
      master_port_(0),
      ucc_available_(false),
      nccl_available_(false) {
  // retrieves rank and communicator size
  is_available_ = parseEnv(
      rank_, size_, local_rank_, local_size_, master_addr_, master_port_);

  if (!is_available_) {
    return;
  }

#ifdef NVFUSER_DISTRIBUTED
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
  constexpr int comm_master_port_default =
      c10d::TCPStoreOptions::kDefaultPort; // 29500
  store_opts.port = master_port_ ? master_port_ : comm_master_port_default;
  store_ = c10::make_intrusive<c10d::TCPStore>(master_addr_, store_opts);
#endif

#if defined(USE_C10D_UCC) && defined(NVFUSER_BUILD_WITH_UCC)
  ucc_available_ = true;
#endif

#ifdef USE_C10D_NCCL
  nccl_available_ = true;
#endif
}

void Communicator::cleanup() {
  if (!is_available_) {
    TORCH_WARN(
        "The singleton Communicator isn't available. "
        "This is likely because Communicator::cleanup was called more than "
        "once or the instance wasn't successfully initialized.");
    return;
  }

  store_ = nullptr;

#if defined(NVFUSER_DISTRIBUTED) && defined(USE_C10D_NCCL)
  for (auto& [key, backend] : backends_) {
    // Call shutdown before destructing a ProcessGroupNCCL as instructed by
    // https://github.com/pytorch/pytorch/blob/e62073d7997c9e63896cb5289ffd0874a8cc1838/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1164-L1170.
    if (auto* pg_nccl = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get())) {
      pg_nccl->shutdown();
    }
  }
#endif
  backends_.clear();

  is_available_ = false;
}

c10d::Backend* Communicator::getBackendForTeam(
    const Team& team,
    std::optional<CommunicatorBackend> backend,
    const std::string& prefix) {
  NVF_ERROR(
      is_available(),
      "The singleton Communicator isn't available. "
      "This is likely because Communicator::cleanup has been called "
      "or the instance wasn't successfully initialized.");

  CommunicatorBackend b = getBackend(backend);
  // generate a string key which is unique to the team
  // create the team and cache it
  std::string team_key = prefix + getTeamKey(team, b);
  // check if backend associated with the team is present in the cache
  if (backends_.find(team_key) ==
      backends_.end()) { // create the backend and cache it
#ifdef NVFUSER_DISTRIBUTED
    backends_[team_key] = [&]() -> c10::intrusive_ptr<c10d::Backend> {
      // check that the caller's rank belongs to the requested team
      auto rank_it = std::find(team.begin(), team.end(), deviceId());
      if (rank_it == team.end()) {
        return nullptr;
      }
      // retrieve the caller's rank index/position in the team
      RankType team_rank = std::distance(team.begin(), rank_it);
      return createBackend(
          b,
          c10::make_intrusive<c10d::PrefixStore>(team_key, store_),
          team_rank,
          static_cast<int64_t>(team.size()));
    }();
#else
    backends_[team_key] = nullptr;
#endif
  }
  return backends_.at(team_key).get();
}

c10d::Backend* Communicator::getWorld(
    std::optional<CommunicatorBackend> backend) {
  std::vector<RankType> all_ranks(size_);
  std::iota(all_ranks.begin(), all_ranks.end(), 0);

  return getBackendForTeam(all_ranks, backend);
}

void Communicator::barrier(std::optional<CommunicatorBackend> backend) {
  // Explicitly specify the (local) device ID to avoid a warning. Without this,
  // ProcessGroupNCCL::barrier may guess the wrong mapping and failed to block
  // CPU properly:
  // https://github.com/pytorch/pytorch/blob/7e4329c258306cc14303895e5f1e6036b009e74f/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3905-L3912.
  c10d::BarrierOptions options{.device_ids = {local_rank()}};
  getWorld(backend)->barrier(options)->wait();
}

} // namespace nvfuser
