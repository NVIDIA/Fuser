// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace nvfuser {

// Registers all external C++ runtime functions with the LLVM JIT
// These functions are callable from JIT-compiled code
// Includes: tensor metadata accessors, allocation, kernel launch, ATen ops,
// NVTX profiling
void registerExternalFunctionsImpl(
    llvm::orc::LLJIT* jit,
    llvm::orc::JITDylib& dest_dynamic_lib);

} // namespace nvfuser
