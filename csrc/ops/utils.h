// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <scheduler/matmul_utils.h>
#include <type.h>
#include <visibility.h>

#include <vector>

namespace nvfuser {
namespace ops {

TensorView* maybe_broadcast_inner_to_rank(TensorView* t, size_t rank);

TensorView* maybe_broadcast_index_tv(TensorView* t, size_t dim, size_t rank);

Val* simplifiedInt(Val* val);

// If one size is nullptr, return the other. If both symbolic just return v1. If
// one's concrete, prefer that one (simplified). If both concrete make sure
// they're the same size.
Val* promoteSize(Val* v1, Val* v2);

// Will return a new value of type val with the DataType dtype.
Val* newScalar(ValType vtype, DataType dtype);

IterType promoteIterType(IterType type1, IterType type2);

// For MatmulOp, the input iterdomains at a given index do not necessarily map
// to the output iterdomain at that index This function aligns the input
// iterdomain to the output and returns a vector where each element is the input
// iterdomain corresponding to the output iterdomain at that index. If the
// element is nullptr, there is no mapping between input-output at that index.
// Based on the input dimensions following cases are possible:
// 1. A/B is 1D: [M, K] x [K] -> [M]
// Mapping A: {id_M}, Mapping B: {nullptr}
// 2. A and B are 2D: [M, K] x [K, N] -> [M, N]
// Mapping A: {id_M, nullptr}, Mapping B: {nullptr, id_N}
// 3. A/B are atleast 1D and one of them is > 2D: [B, M, K] x [K, N] -> [B, M,
// N] Mapping A: {id_B, id_M, nullptr}, Mapping B: {nullptr, nullptr, id_N}
std::vector<IterDomain*> mapMatmulOpIterDomains(
    const std::vector<IterDomain*>& input_domain,
    MatmulRole input_role,
    size_t out_size);

IterDomain* newOutputIterDomain(const std::vector<IterDomain*>& ids);

std::vector<IterDomain*> newOutputDomain(const std::vector<Val*>& vals);

TensorView* newOutputTV(const std::vector<Val*>& vals, DataType dtype);

std::vector<Val*> maybeBroadcast(const std::vector<Val*>& vals);

NVF_API Val* newValLike(Val* val, DataType dtype);

// returns the minimum init value for reduction:
//   -inf for floating type;
//   lowest value for integer type;
//   false for bool.
Val* getMinimumValue(DataType v);

// returns the maximum init value for reduction:
//   inf for floating type;
//   highest value for integer type;
//   true for bool.
Val* getMaximumValue(DataType v);

std::vector<unsigned int> canonicalizeAxes(
    const std::vector<int64_t>& axes,
    int64_t ndims);

} // namespace ops
} // namespace nvfuser
