// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>

namespace nvfuser {

namespace matmul_heuristic_plugin {

//! This is intended as a minimal interface for enabling matmul heuristics.
//! In order to plug in your own custom heuristic, create a dynamic library
//! that exports a function whose signature matches HeuristicFunc (see below).
//! If that library is located at /path/to/libfoo.so you can set
//! NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/libfoo.so to use the plugin to
//! determine matmul parameters automatically.

//! This is an opaque type representing the information available to the plugin
//! to determine the kernel configuration. It can be queried through the getter
//! functions below
struct ProblemDescription;

//! Getters for ProblemDescription
uint32_t getProblemM(const ProblemDescription* problem);
uint32_t getProblemN(const ProblemDescription* problem);
uint32_t getProblemK(const ProblemDescription* problem);
uint32_t getProblemBatchSize(const ProblemDescription* problem);
//! Explicit mapping for layout:
//!  0 = NN
//!  1 = NT
//!  2 = TN
//!  3 = TT
uint8_t getProblemLayout(const ProblemDescription* problem);
//! String like HSH or TSS indicating input, compute, and accumulate precision
//! where the letters are mapped to types using the following mapping (case
//! insensitive):
//!  B = Int8
//!  I = Int32
//!  Q = FP8 (E4M3)
//!  R = FP8 (E5M2)
//!  T = BFloat16
//!  H = Float16
//!  F = TensorFloat32
//!  S = Float32
//!  D = Float64
//!  C = complex<float>
//!  Z = complex<double>
const char* getProblemPrecision(const ProblemDescription* problem);

//! This is the return type of a HeuristicFunc (defined below) implemented in a
//! plugin. This is used to set values in MatmulParams
struct KernelConfig;

//! Setters for KernelConfig
void setCtaTile(KernelConfig* config, uint16_t m, uint16_t n, uint16_t k);
void setWarpTile(KernelConfig* config, uint16_t m, uint16_t n, uint16_t k);
void setInstructionTile(
    KernelConfig* config,
    uint16_t m,
    uint16_t n,
    uint16_t k);
void setSplitKFactor(KernelConfig* config, uint16_t f);
void setLoadStages(KernelConfig* config, uint8_t s);
void setGridSwizzleFactor(KernelConfig* config, uint8_t g);
void setCtaOrder(KernelConfig* config, uint8_t o);
void setDoubleBufferSmemRead(KernelConfig* config, bool b);
void setRotateLdMatrixOutOfMainLoop(KernelConfig* config, bool b);

} // namespace matmul_heuristic_plugin

} // namespace nvfuser
