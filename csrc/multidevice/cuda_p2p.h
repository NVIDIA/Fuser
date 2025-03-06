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

namespace get_zcopy {

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream);
void sendPost(const P2pIpcHandle& ipc_handles, CUstream stream);
void sendWait(const P2pIpcHandle& ipc_handles, CUstream stream);

} // namespace get_zcopy

} // namespace nvfuser
