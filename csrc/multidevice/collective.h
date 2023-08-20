// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#ifdef USE_DISTRIBUTED

#include <multidevice/communicator.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

/*
  This struct gathers all the parameters necessary for the
  construction a collective
*/
struct TORCH_CUDA_CU_API CollectiveParams {
  DeviceIdxType root = -1;
  std::vector<at::Tensor> src_bufs;
  std::vector<at::Tensor> dst_bufs;
  std::vector<DeviceIdxType> team; // should not have duplicate
};

/*
The class "Collective" represents a MPI-style collective
communication operation to be executed on the network. The base class Collective
should not be used directly but through its derived classes:
Broadcast, Gather, Scatter, Allgather, and SendRecv. Other collectives will be
added later.

Later, Collective could be made a derived class of Expr and be thought
as a kernel IRs resulting of the lowering of a PipelineCommunication.

CollectiveParams contains the arguments for the collective constructors.
Note that each rank (associated with a device index through communicator.deviceId())
will fill CollectiveParams with different arguments, depending on the role
they play in this collective. For example, the root of a Gather collective will
have <team_size> destination buffers, whereas non-root will have no destination
buffers. Also, the ranks not participating in the collective should not
instantiate it.

The method "post" triggers the execution of the collective. This call is
non-blocking. The collective can be posted multiple times.
It is assumed that the current device_index (given by
communicator.deviceId()) belongs to the team of the collective,
otherwise an error is thrown.
*/

class TORCH_CUDA_CU_API Collective {
 public:
  virtual ~Collective() = default;

  std::string toString(int indent = 0) const;

  const auto& params() const {
    return params_;
  }

  // Triggers the execution of the collective. This is a non-blocking call.
  // The collective can be posted multiple times
  virtual c10::intrusive_ptr<c10d::Work> post(Communicator& comm) {
        TORCH_INTERNAL_ASSERT(false, "not implemented");
  };

 protected:
  // argument "name" is only used for printing
  // argument "has_root" indicates is the collective is rooted
  Collective(CollectiveParams params, std::string name, bool has_root = true);

  void post_common(Communicator& comm);

  // store the arguments of the collective
  CollectiveParams params_;
  // stores the index of the root in the team
  DeviceIdxType root_rank_ = -1;
  // utility buffer used in Gather and Scatter derived classes
  std::vector<std::vector<at::Tensor>> buf_list_;

 private:
  // used for printing
  std::string collective_type_;
  // indicates if the collective is rooted
  bool has_root_ = true;
};

/*
Copies the root's src buffer to each device's dst buffer

Requirements:
  - the root is set and belongs to the team
  - the root has one src buffer, and no dst buffer
  - non-roots have no src buffer and one dst buffer
  - all buffers have the same size
*/
class TORCH_CUDA_CU_API Broadcast : public Collective {
 public:
  Broadcast(CollectiveParams params);
  c10::intrusive_ptr<c10d::Work> post(Communicator& comm) override;
};

/*
Copies each device's source buffer to the root's respective src
buffer. The order of the sender devices matches the order of the
root's buffers.

Requirements:
  - the root is set and belongs to the team
  - the root has one src buffer and <team_size> dst buffers
  - non-roots have one src buffer and no dst buffer
  - all buffers have the same size
*/
class TORCH_CUDA_CU_API Gather : public Collective {
 public:
  Gather(CollectiveParams params);
  c10::intrusive_ptr<c10d::Work> post(Communicator& comm) override;
};

/*
Copies each device's src buffer to each device's respective src
buffer. The order of the devices matches the order of the
buffers

Requirements:
  - all device have one src buffer and <team_size> dst buffers
  - all buffers have the same size
*/
class TORCH_CUDA_CU_API Allgather : public Collective {
 public:
  Allgather(CollectiveParams params);
  c10::intrusive_ptr<c10d::Work> post(Communicator& comm) override;
};

/*
Copies each root's src buffer to each device's dst buffer.
The order of the buffers matches the order of the receiver devices

Requirements:
  - the root is set and belongs to the team
  - the root has <team_size> src buffers and one dst buffer
  - non-roots have no src buffer and one dst buffer
  - all buffers have the same size
*/
class TORCH_CUDA_CU_API Scatter : public Collective {
 public:
  Scatter(CollectiveParams params);
  c10::intrusive_ptr<c10d::Work> post(Communicator& comm) override;
};

/*
Copies the sender's src buffers to the receiver's dst buffer
It is equivalent to a Broadcast with a team of size == 2

Requirements:
  - the team must be of size 2
  - the root is set and belongs to the team. The "root" corresponds to the
sender
  - the root has one src buffers and no dst buffer
  - the unique non-root have no src buffer and one dst buffer
  - all buffers have the same size
*/
class TORCH_CUDA_CU_API SendRecv : public Collective {
 public:
  SendRecv(CollectiveParams params);
  c10::intrusive_ptr<c10d::Work> post(Communicator& comm) override;
};

} // namespace nvfuser

#endif
