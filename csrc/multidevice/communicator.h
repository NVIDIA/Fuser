#pragma once
#ifdef USE_DISTRIBUTED

#include <multidevice/multidevice.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace nvfuser {

/*
   This file implements the class Communicator which sets up the inter-process
   Backend. This class contains inter-process information, such as the rank, the
   world size, as well as the Process Group that can be called to perform
   inter-process communications

   Only one node configuration is supported for now.
   TODO: extend to multinode.
*/

// Supported backends. TODO: only tested with nccl for now
enum class CommunicatorBackend { nccl, ucc, gloo };

class Communicator {
 public:
  Communicator(CommunicatorBackend backend, RankType server_rank);

  // returns the rank of the current process
  auto rank() const {
    return rank_;
  }

  // returns the number of processes in the communicator
  auto size() const {
    return size_;
  }

  // returns the local rank of the current process (within the node)
  auto local_rank() const {
    return local_rank_;
  }

  // returns the local number of processes in the communicator (within the node)
  auto local_size() const {
    return local_size_;
  }

  // returns the flattenend list of ranks of the communicator
  auto ranks() const {
    std::vector<RankType> ret(size());
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
  }

  // performs a (blocking) send/receive p2p data transfer
  void sendRecv(
      RankType receiver_rank,
      RankType sender_rank,
      std::vector<at::Tensor>& tensor,
      int tag = 0);

  // performs a barrier in the communicator
  void barrier() const {
    pg_->barrier();
  }

  // stores the process group backend
 private:
  RankType rank_;
  int64_t size_;
  RankType local_rank_;
  int64_t local_size_;
  c10::intrusive_ptr<c10d::Backend> pg_;
};

} // namespace nvfuser

#endif
