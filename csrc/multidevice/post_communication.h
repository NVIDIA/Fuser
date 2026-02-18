// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include "multidevice/communication.h"

namespace nvfuser {

// The method "post" triggers the execution of the communication. This call is
// non-blocking. The communication can be posted multiple times.
// It is assumed that the current device_index (given by
// communicator.deviceId()) belongs to the team of the communication,
// otherwise an error is thrown.
//
// NOTE: pytorch's NCCL process group API needs <team_size> buffers on root for
// scatter/gather operation.
// (*) Broadcast
// Copies the root's src buffer to each device's dst buffer
// Requirements:
//   - the root is set and belongs to the team
//   - the root has one src buffer, and no or one dst buffer
//   - non-roots have no src buffer and one dst buffer
//   - all buffers have the same size
// (*) Gather
// Copies each device's source buffer to the root's respective src
// buffer. The order of the sender devices matches the order of the
// root's buffers.
// Requirements:
//   - the root is set and belongs to the team
//   - the root has one src buffer and <team_size> dst buffers
//   - non-roots have one src buffer and no dst buffer
//   - all buffers have the same size
// (*) Allgather
// Copies each device's src buffer to each device's respective src
// buffer. The order of the devices matches the order of the
// buffers
// Requirements:
//   - all device have one src buffer and <team_size> dst buffers
//   - all buffers have the same size
// (*) Scatter
// Copies each root's src buffer to each device's dst buffer.
// The order of the buffers matches the order of the receiver devices
// Requirements:
//   - the root is set and belongs to the team
//   - the root has <team_size> src buffers and one dst buffer
//   - non-roots have no src buffer and one dst buffer
//   - all buffers have the same size
// (*) Reduce
// Reduce the src buffers to the root's dst buffer.
// Requirements:
//   - the root is set and belongs to the team
//   - the root has one src buffers and one dst buffer
//   - non-roots have one src buffer and no dst buffer
//   - all buffers have the same size
// (*) Allreduce
// Reduce the src buffers to the dst buffer.
// Requirements:
//   - all devices have one src buffer and one dst buffer
//   - all buffers have the same size
// (*) ReduceScatter
// Reduce all the src buffers and shard the result to the dst buffers.
// Requirements:
//   - all devices have <team_size> src buffer and one dst buffer
//   - all buffers have the same size
// (*) SendRecv
// Copies the sender's src buffers to the receiver's dst buffer
// It is equivalent to a Broadcast with a team of size == 2
c10::intrusive_ptr<c10d::Work> postSingleCommunication(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    DeviceIdxType root_index = -1);

c10::intrusive_ptr<c10d::Work> postSingleCommunication(
    P2PCommunication* communication,
    DeviceIdxType my_device_index,
    DeviceIdxType peer,
    c10d::Backend* backend,
    at::Tensor buffer);

} // namespace nvfuser
