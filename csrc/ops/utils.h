// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <scheduler/matmul_utils.h>
#include <type.h>
#include <visibility.h>

#include <vector>

namespace nvfuser {

enum class AttnRole { Q = 0, K, V, Mask };

struct ScaledTensorView {
  TensorView* tv;
  TensorView* block_scaling_factor = nullptr;
  TensorView* global_scaling_factor = nullptr;
};

namespace ops {

TensorView* maybe_broadcast_inner_to_rank(TensorView* t, size_t rank);

// A utility function that broadcasts index TensorView to the rank of the other
// TensorView.
TensorView* maybeBroadcastIndexTv(TensorView* t, size_t dim, size_t rank);

// A utility function that checks if index tv is already broadcasted to correct
// shape for index_select
bool isIndexAlreadyBroadcast(
    const std::vector<IterDomain*>& index_domain,
    size_t dim,
    size_t rank);

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
// 1. A/B is 1D: [M, K] x [K] -> [M] (Mapping A: {id_M}, Mapping B: {nullptr})
// or [K] x [N, K] -> [N] (Mapping A: {nullptr}, Mapping B: {id_N})
// 2. A and B are 2D: [M, K] x [K, N] -> [M, N] (Mapping A: {id_M, nullptr},
// Mapping B: {nullptr, id_N})
// 3. A/B are atleast 1D and one of them is > 2D: [B, M, K] x [K, N] -> [B, M,
// N] (Mapping A: {id_B, id_M, nullptr}, Mapping B: {nullptr, nullptr, id_N})
// Args:
// 1. input_domain: root/logical domain without reductions for any input to
// MatmulOp
// 2. input_position: Specifies if the input is A / B (0 or 1)
// 3: out_size: MatmulOp output dimension (input and output may not be the same
// size).
std::vector<IterDomain*> mapMatmulOpIterDomains(
    const std::vector<IterDomain*>& input_domain,
    int64_t input_position,
    size_t out_size);

// For LinearOp, the output is the same as the first input (A[*,
// in_features])for all but the last dimension. If the second input is 2D
// (B[out_features, in_features]), the last dimension of output is out_features.
// If bias is 1D (bias[out_features]) it maps to the last dimension of the
// output. Args:
// 1. input_domain: root/logical domain without reductions for any input to
// LinearOp
// 2. input_position: Specifies if the input is A / B / Bias (0, 1, or 2)
// (MatmulTensorRole::Input_A/Input_B/Input_C) 3: out_size: LinearOp output
// dimension (input and output may not be the same size).
std::vector<IterDomain*> mapLinearOpIterDomains(
    const std::vector<IterDomain*>& input_domain,
    int64_t input_position,
    size_t out_size,
    bool k_bcast);

// Takes a vector of aligned input iterdomains to create the output iterdomain.
// This is used if the input iterdomains are not trivially mapped to the output
// iterdomains. For eg: MatmulOp. If given, the forced_iter_type argument will
// be the output IterType regardless of the inputs; otherwise the output
// IterType is inferred from ids.
IterDomain* newOutputIterDomain(
    const std::vector<IterDomain*>& ids,
    const std::optional<IterType> force_iter_type = std::nullopt);

// Takes a vector of `Val*`s and assumes they are all aligned to create the
// output tensorview, e.g., for BinaryOp. `vals` can contain scalars, e.g, when
// creating the output TensorView for `tv0+scalar`. This is for convenience and
// scalars will be ignored.
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

std::vector<int64_t> canonicalizeAxes(
    const std::vector<int64_t>& axes,
    int64_t ndims);

// Returns a scalar which is a two-sided identity element for the given binary
// operator
Val* binOpIdentity(BinaryOpType op_type, DataType dtype);

} // namespace ops
} // namespace nvfuser
