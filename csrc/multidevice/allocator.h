// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <multidevice/multidevice.h>
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>

namespace nvfuser {

// Allocate only the necessary tensors in a pipeline, given a device index and
// concrete inputs. returns a map associating the allocated buffer with the
// corresponding symbolic Val. The allocations correspond to intermediate
// tensors that will be received from an inter-device communication and will be
// used as a subsequent stage's input.
std::unordered_map<Val*, c10::IValue> allocatePipelineIntermediateBuffers(
    Pipeline* pipeline,
    DeviceIdxType my_device_index,
    std::vector<c10::IValue> global_inputs_IValues);

} // namespace nvfuser

#endif
