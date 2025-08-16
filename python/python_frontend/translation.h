// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_record.h>

namespace nvfuser::python_frontend {

// Translate a CPP Fusion into a Python FusionDefinition.
NVF_API std::unordered_map<const nvfuser::Val*, size_t> translate(
    Fusion* fusion,
    FusionDefinition* fd);

} // namespace nvfuser::python_frontend
