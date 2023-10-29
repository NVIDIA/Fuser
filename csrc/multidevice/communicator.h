// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#ifdef USE_DISTRIBUTED

#include <exceptions.h>
#include <multidevice/multidevice.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace nvfuser {

/*
   This file implements the class Communicator which sets up the inter-process
   Backend. This class contains inter-process information, such as the rank, the
   world size, as well as the Process Group that can be called to perform
   inter-process communications.

   Each process is associated with a unique deviceId and device. The actual MPI
   rank remains private to the class and should not be used by the user. The
   communicator class holds privately the mappings ranks <-> device IDs <->
   device.

*/

using RankType = DeviceIdxType;

// Supported backends. TODO: only tested with nccl for now
enum class CommunicatorBackend { nccl, ucc, gloo };

constexpr CommunicatorBackend comm_backend_default = CommunicatorBackend::nccl;
constexpr int comm_server_local_rank_default = 0;
constexpr int comm_master_port_default =
    c10d::TCPStoreOptions::kDefaultPort; // 29500

class Communicator {
 public:
  Communicator(
      CommunicatorBackend backend = comm_backend_default,
      RankType server_local_rank = comm_server_local_rank_default);

  Communicator(const Communicator&) = delete;
  Communicator& operator=(const Communicator&) = delete;

  // returns if distributed config is available
  auto is_available() const {
    return is_available_;
  }

  // returns the number of processes in the communicator
  auto size() const {
    return size_;
  }

  // returns the local number of processes in the communicator (within the node)
  auto local_size() const {
    return local_size_;
  }

  // performs a send/receive p2p data transfer
  c10::intrusive_ptr<c10d::Work> sendRecv(
      DeviceIdxType receiver,
      DeviceIdxType sender,
      std::vector<at::Tensor>& tensor,
      int tag = 0);

  // performs a blocking barrier in the communicator
  void barrier() const {
    world_->barrier()->wait();
  }

  // returns the backend associated with a team
  c10::intrusive_ptr<c10d::Backend> getBackendForTeam(const Team& team);

  // returns the device associated with the current process
  auto device() const {
    return at::Device("cuda:" + std::to_string(local_rank_));
  }

  // returns the device Id associated with the current process
  DeviceIdxType deviceId() const {
    return rankToDiD(rank_);
  }

 private:
  // returns the rank corresponding to a device index
  RankType dIdToRank(DeviceIdxType d_id) const {
    return static_cast<RankType>(d_id);
  }

  // returns the device index corresponding to a rank
  DeviceIdxType rankToDiD(RankType rank) const {
    return static_cast<DeviceIdxType>(rank);
  }

  bool is_available_;
  CommunicatorBackend backend_type_;
  RankType rank_;
  int64_t size_;
  RankType local_rank_;
  int64_t local_size_;
  std::string master_addr_;
  int master_port_;
  // stores the world's store used for the backend init
  c10::intrusive_ptr<c10d::TCPStore> store_;
  // stores the world's backend
  c10::intrusive_ptr<c10d::Backend> world_;
  // cache for the created backends. The keys are strings generated from Teams
  std::unordered_map<std::string, c10::intrusive_ptr<c10d::Backend>> backends_;
};

} // namespace nvfuser

#endif
