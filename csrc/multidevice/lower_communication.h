// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <multidevice/communication.h>
#include <multidevice/multidevice.h>
#include <multidevice/pipeline_ir.h>

namespace nvfuser {

// Lower a PipelineCommunication into a series of Communication, given a
// device_index.
std::vector<std::shared_ptr<Communication>> lowerCommunication(
    DeviceIdxType device_index,
    PipelineCommunication* c,
    at::Tensor input_tensor,
    at::Tensor output_tensor);

} // namespace nvfuser

#endif
