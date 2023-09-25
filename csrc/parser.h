// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <fusion.h>

/*
 * This file handles Parsing PyTorch jit ir;
 *
 * It is used in two places:
 *   1. When partitioning PyTorch jit ir to create prim::CudaFusionGroup, each
 *      node is queried by `isNodeParsible` to determine whether the node could
 *      be handled by the fuser (whether a given PyTorch jit operator should be
 *      merged);
 *   2. lowering PyTorch jit ir to CUDA codegen ir.
 *      creates a `Fusion` by traversing a PyTorch jit graph.
 *
 * TODO: we could consider exposing API to allow custom registration of parsing
 * rules for a given PyTorch jit operator.
 */

namespace nvfuser {

constexpr int kPwThreadX = 128;
constexpr int kFcdReductionThreadX = 128;
constexpr int kNonFcdReductionThreadX = 32;
constexpr int kNonFcdReductionThreadY = 32;

bool hasReductionNode(const torch::jit::Block* block);
bool isReductionToSizeNode(const torch::jit::Node* node);
bool isReductionNode(const torch::jit::Node* node);

bool hasNormalizationNode(const torch::jit::Block* block);
bool isNormalizationNode(const torch::jit::Node* node);

bool isElementWiseNode(const torch::jit::Node* node);

// returns whether or not a parsing function exists for the given node type.
bool isNodeParsible(const torch::jit::Node* node);
bool shouldProfileNode(const torch::jit::Node* node);

bool skipNodeKind(const std::string& symbol_str, bool flip);

void InsertProfileNodes(torch::jit::ProfilingRecord* pr);

// lowers PyTorch jit graph to `Fusion`.
std::unique_ptr<Fusion> parseJitIR(
    const std::shared_ptr<torch::jit::Graph>& graph);

} // namespace nvfuser
