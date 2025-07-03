// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_utils.h>
#include <multidevice/communicator.h>
#include <options.h>
#include <utils.h>

#include <netdb.h>
#include <map>

#ifdef NVFUSER_DISTRIBUTED
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
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
    case CommunicatorBackend::kNccl:
      out << "NCCL";
      break;
    case CommunicatorBackend::kUcc:
      out << "UCC";
      break;
    case CommunicatorBackend::kCuda:
      out << "CUDA";
      break;
  }
  return out;
}

namespace {

// Iterate through a list of environmental variables and stop at the first one
// that succeeds. If none of the variables are available, returns nullptr.
char* tryReadEnv(const std::vector<std::string>& envs) {
  for (const auto& env : envs) {
    if (char* ret = std::getenv(env.c_str())) {
      return ret;
    }
  }
  return nullptr;
}

// Parses the environment to retrieve MPI rank, world size, local rank,
// local world size, and also master address and master port.
//
// We intend to support mpirun, torchrun
// (https://docs.pytorch.org/docs/stable/elastic/run.html#environment-variables)
// and slurm. However, only mpirun is tested in CI at this moment, so I
// wouldn't be surprised the other launchers don't work out of the box.
//
// Returns true if the distributed configuration is valid, false otherwise.
bool parseEnv(
    RankType& rank,
    int64_t& size,
    RankType& local_rank,
    int64_t& local_size,
    std::string& master_addr,
    int& master_port) {
  // retrieves the rank of the current process
  char* env = tryReadEnv({"OMPI_COMM_WORLD_RANK", "RANK", "SLURM_PROCID"});
  if (env == nullptr) {
    return false;
  }
  rank = std::atoi(env);

  // retrieves the size of the communicator
  env = tryReadEnv({"OMPI_COMM_WORLD_SIZE", "WORLD_SIZE", "SLURM_NTASKS"});
  if (env == nullptr) {
    return false;
  }
  size = std::atoi(env);

  // retrieves the size of the communicator
  env =
      tryReadEnv({"OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK", "SLURM_LOCALID"});
  if (env == nullptr) {
    return false;
  }
  local_rank = std::atoi(env);

  // retrieves the size of the communicator
  env = tryReadEnv(
      {"OMPI_COMM_WORLD_LOCAL_SIZE",
       "LOCAL_WORLD_SIZE",
       "SLURM_NTASKS_PER_NODE"});
  if (env == nullptr) {
    return false;
  }
  local_size = std::atoi(env);

  // retrieves master address
  env = std::getenv("NVFUSER_MASTER_ADDR");
  if (env) {
    // replace the potential aliased hostname by the "official" name
    master_addr = gethostbyname(env)->h_name;
  } else if (local_size == size) {
    master_addr = "localhost";
  } else {
    TORCH_WARN(
        "the environment variable NVFUSER_MASTER_ADDR "
        "must be specified in multi-node environment");
    return false;
  }

  // retrieves master port
  if ((env = std::getenv("NVFUSER_MASTER_PORT")) != nullptr) {
    master_port = std::atoi(env);
  } else {
    LOG(INFO) << "The environment variable NVFUSER_MASTER_PORT has not been "
                 "specified. "
              << "Set the master port to default: " << master_port;
  }

  return true;
}

inline std::string getTeamKey(const Team& team, CommunicatorBackend backend) {
  std::string backend_str =
      (backend == CommunicatorBackend::kUcc) ? "ucc" : "nccl";
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
  if (backend == CommunicatorBackend::kNccl) {
    auto pg_opts = c10::make_intrusive<::c10d::ProcessGroupNCCL::Options>();
    return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
        store, rank, size, pg_opts);
  }
#endif

#if defined(USE_C10D_UCC) && defined(NVFUSER_BUILD_WITH_UCC)
  if (backend == CommunicatorBackend::kUcc) {
    constexpr auto timeout = std::chrono::milliseconds(30 * 60 * 1000);
    return c10d::ProcessGroupUCC::createProcessGroupUCC(
        store, static_cast<int>(rank), static_cast<int>(size), timeout);
  }
#endif
  NVF_THROW("no distributed backend available");
}
#endif
} // namespace

Communicator::Communicator(
    CommunicatorBackend backend,
    RankType server_local_rank)
    : is_available_(false),
      default_backend_(backend),
      rank_(0),
      size_(1),
      local_rank_(0),
      local_size_(1),
      master_port_(
          c10d::TCPStoreOptions::kDefaultPort + 42), // to avoid collision
      ucc_available_(false),
      nccl_available_(false) {
  if (isOptionDisabled(DisableOption::Multidevice)) {
    return;
  }

  // retrieves rank and communicator size
  is_available_ = parseEnv(
      rank_, size_, local_rank_, local_size_, master_addr_, master_port_);

  if (!is_available_) {
    return;
  }

  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(local_rank_));

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
  store_opts.port = master_port_;
  store_ = c10::make_intrusive<c10d::TCPStore>(master_addr_, store_opts);
#endif

#if defined(USE_C10D_UCC) && defined(NVFUSER_BUILD_WITH_UCC)
  ucc_available_ = true;
#endif

#ifdef USE_C10D_NCCL
  nccl_available_ = true;
#endif
}

namespace {
void waitForDebuggerAtRanks(
    Communicator* communicator,
    const std::vector<DeviceIdxType>& ranks) {
  if (std::count(ranks.begin(), ranks.end(), communicator->deviceId()) > 0) {
    volatile bool waiting = true;
    auto pid = getpid();
    std::cerr << "Process " << pid
              << " is waiting for the debugger. To continue debugging, "
              << "start gdb, `attach " << pid
              << "`, `set var waiting=false`, and `fini`." << std::endl;
    while (waiting) { // Please change `waiting` in the debugger.
    }
    std::cerr << "Process " << getpid() << " finished waiting." << std::endl;
  }

  if (communicator->is_available()) {
    communicator->barrier();
  }
}
} // namespace

Communicator& Communicator::getInstance() {
  // This isn't the best practice to use singleton. Ideally, we'd like to
  // ```
  // static Communicator communicator;
  // ```
  // and let the destructor clean it up at program exit after `main` returns.
  // This however would cause a "driver shutting down" error, likely because
  // another static variable destructor shuts down the CUDA driver before
  // ~Communicator. Note that the order of static variable destruction
  // across translation units is undefined.
  //
  // Therefore, we `new Communicator()` as a raw pointer and let the user
  // call Communicator::getInstance().cleanup() to clean up the Communicator
  // explicitly before the end of `main`. For example, the cleanup method is
  // called via MultiDeviceTestEnvironment::TearDown in C++ unit tests and
  // nvfuser._cleanup() in Python.
  static auto* communicator = new Communicator();

  // EnableOption::WaitDebugger can be used to attach gdb to one of the
  // processes for debugging. For example,
  //
  // ```
  // mpirun -np 2 -x NVFUSER_ENABLE='wait_debugger(1)' bin/test_multidevice
  // --gtest_filter=*ReduceScatter
  // ```
  //
  // When an mpirun fails, it usually prints out something like
  // ```
  // mpirun detected that one or more processes exited with non-zero status,
  // thus causing the job to be terminated. The first process to do so was:
  //
  //   Process name: [[17665,1],0]
  //   Exit code:    1
  // ```
  // The last bit of the process name (0 in this case) is the rank of the first
  // failing process, and usually the rank to debug.
  //
  // Sometimes, multiple processes fail, and a failed, non-gdb'ed process can
  // cause `mpirun` to terminate the entire job including the process being
  // gdb'ed. For that, I use `mpirun -continuous` so `mpirun` keeps running the
  // process being gdb'ed.
  if (isOptionEnabled(EnableOption::WaitDebugger)) {
    static std::once_flag once;
    std::call_once(once, [&]() {
      // Catch exceptions so call_once always flips `once` and executes this
      // functor only once.
      try {
        const std::vector<std::string>& ranks_as_str =
            getEnableOptionArguments(EnableOption::WaitDebugger);
        std::vector<DeviceIdxType> ranks;
        for (const auto& rank_as_str : ranks_as_str) {
          const DeviceIdxType rank = std::stol(rank_as_str);
          NVF_CHECK(
              rank >= 0 && rank < communicator->size(),
              "rank=",
              rank,
              " must be in the range of [0,",
              communicator->size(),
              ").");
          ranks.push_back(rank);
        }
        waitForDebuggerAtRanks(communicator, ranks);
      } catch (const std::exception& e) {
        TORCH_WARN("Failed to wait for debugger: ", e.what());
      }
    });
  }

  return *communicator;
}

void Communicator::cleanup() {
  static bool cleaned_up = false;
  NVF_CHECK(
      !cleaned_up,
      "The singleton Communicator has already been cleaned up. This is "
      "likely because Communicator::cleanup was called more than once");
  cleaned_up = true;

  // Without this, the TCPStore server can be cleaned up before TCPStore
  // clients are created, causing an hang. This happened with
  // test_multidevice.py::test_sizes_and_ranks.
  if (is_available()) {
    barrier();
  }

  store_ = nullptr;

#if defined(NVFUSER_DISTRIBUTED) && defined(USE_C10D_NCCL)
  // Sort backends to work around a NCCL bug (nvbugs/4889623). Closing backends
  // in different orders between ranks have been causing a hang.
  std::vector<std::pair<std::string, c10::intrusive_ptr<c10d::Backend>>>
      keyed_backends(backends_.begin(), backends_.end());
  std::sort(keyed_backends.begin(), keyed_backends.end());
  for (auto& [key, backend] : keyed_backends) {
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
  NVF_CHECK(
      is_available(),
      "The singleton Communicator isn't available. "
      "This is most likely because the instance wasn't successfully "
      "initialized due to lack of a multi-process running (e.g. mpirun or "
      "torchrun). Sometimes, this is because Communicator::cleanup has been "
      "accidentally called before this function.");

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
