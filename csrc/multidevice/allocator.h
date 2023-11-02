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
#include <multidevice/pipeline_ir.h>
#include <multidevice/pipeline.h>

namespace nvfuser {

std::unordered_map<Val*, c10::IValue> allocatePipelineIntermediateBuffers(Pipeline* pipeline, DeviceIdxType dId, std::vector<c10::IValue> global_inputs_IValues);

} // namespace nvfuser

#endif
