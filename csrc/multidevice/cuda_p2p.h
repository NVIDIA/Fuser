// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <cuda.h>
#include <multidevice/ipc_handle.h>

namespace nvfuser {

namespace getZcopy {

void RecvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);
void SendPost(const P2pIpcHandle& ipc_handles, CUstream stream);
void SendWait(const P2pIpcHandle& ipc_handles, CUstream stream);

} // namespace getZcopy

} // namespace nvfuser
