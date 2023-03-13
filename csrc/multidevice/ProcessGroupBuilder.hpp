#pragma once
#ifdef USE_DISTRIBUTED

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace nvfuser {

// Class to construct the ProcessGroup interface for communication
class TORCH_CUDA_CU_API ProcessGroupBuilder {
 public:
  // returns the backend used for inter-device communications
  c10::intrusive_ptr<c10d::Backend> getProcessGroup(
      std::string backend, // can be "nccl" or "gloo". TODO: add "ucc"
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size);
};

// Parse the environment to retrieve MPI rank and MPI world size
// returns 0 in case of success, 1 otherwise
int parseEnv(int& grank, int& gsize);

} // namespace nvfuser

#endif
