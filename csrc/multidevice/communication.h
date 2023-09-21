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
  construction a communication
*/
struct CommParams {
  DeviceIdxType root = -1;
  std::vector<at::Tensor> src_bufs;
  std::vector<at::Tensor> dst_bufs;
  Team team; // should not have duplicate
};

/*
The class "Communication" represents a MPI-style communication
communication operation to be executed on the network. The base class
Communication should not be used directly but through its derived classes:
Broadcast, Gather, Scatter, Allgather, and SendRecv. Other collectives will be
added later.

Later, Communication could be made a derived class of Expr and be thought
as a kernel IRs resulting of the lowering of a PipelineCommunication.

CommParams contains the arguments for the communication constructors.
Note that each process (associated with a device index given by
communicator.deviceId()) will fill CommParams with different arguments,
depending on the role they play in this communication. For example, the root of
a Gather communication will have <team_size> destination buffers, whereas
non-root will have no destination buffers. Also, the ranks not participating in
the communication should not instantiate it.

The method "post" triggers the execution of the communication. This call is
non-blocking. The communication can be posted multiple times.
It is assumed that the current device_index (given by
communicator.deviceId()) belongs to the team of the communication,
otherwise an error is thrown.

NOTE: pytorch's NCCL process group API needs <team_size> buffers on root for
scatter/gather operation.
*/

class Communication {
 public:
  virtual ~Communication() = default;

  std::string toString(int indent = 0) const;

  const auto& params() const {
    return params_;
  }

  // Triggers the execution of the communication. This is a non-blocking call.
  // The communication can be posted multiple times
  virtual c10::intrusive_ptr<c10d::Work> post(Communicator& comm) = 0;

 protected:
  // argument "name" is only used for printing
  // argument "has_root" indicates if the communication is rooted
  Communication(CommParams params, std::string name, bool has_root = true);

  // store the arguments of the communication
  CommParams params_;
  // stores the relative index of the root in the team
  DeviceIdxType root_relative_index_ = -1;

 private:
  // used for printing
  std::string collective_type_;
  // indicates if the communication is rooted
  bool has_root_ = true;
};

/*
Copies the root's src buffer to each device's dst buffer

Requirements:
  - the root is set and belongs to the team
  - the root has one src buffer, and no or one dst buffer
  - non-roots have no src buffer and one dst buffer
  - all buffers have the same size
*/
class Broadcast : public Communication {
 public:
  Broadcast(CommParams params);
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
class Gather : public Communication {
 public:
  Gather(CommParams params);
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
class Allgather : public Communication {
 public:
  Allgather(CommParams params);
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
class Scatter : public Communication {
 public:
  Scatter(CommParams params);
  c10::intrusive_ptr<c10d::Work> post(Communicator& comm) override;
};

/*
Copies the sender's src buffers to the receiver's dst buffer
It is equivalent to a Broadcast with a team of size == 2

Requirements:
  - the team must be of size 2 or 1 (in which case the SendRecv reduces to a
local copy)
  - all buffers have the same size
  - the root is set and belongs to the team. The "root" corresponds to the
sender
  - If the team size the root has one src buffers and no dst buffer (or one in
case of a local copy)
  - If team is of size 2, the unique non-root have no src buffer and one dst
buffer
*/
class SendRecv : public Communication {
 public:
  SendRecv(CommParams params);
  c10::intrusive_ptr<c10d::Work> post(Communicator& comm) override;
};

} // namespace nvfuser

#endif
