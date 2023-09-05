// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/interface_nodes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <type.h>

namespace nvfuser {

//!
//! The TypePromotionConfig flags are derived from Aten/TensorIterator.h
//!
//! 1) check_all_same_dtype_ flag checks that all inputs and defined outputs
//! have the same dtype. Default = False
//!
//! 2) promote_inputs_to_common_dtype flag will cast the inputs to the common
//! dtype. Default = True
//!
//! 3) promote_integer_inputs_to_float flag will cast the common dtype to the
//! default float scalar type if it is an integral type (including bool).
//!
struct TypePromotionConfig {
  bool promote_integer_inputs_to_float = false;
  bool require_full_precision_promoted = false;
};

namespace TypePromotion {

static const TypePromotionConfig comparison_op_config;
static const TypePromotionConfig default_op_config;
static const TypePromotionConfig float_op_config{
    /* promote_integer_inputs_to_float */ true,
    /* require_full_precision_promoted */ false};
static const TypePromotionConfig float_only_op_config{
    /* promote_integer_inputs_to_float */ false,
    /* require_full_precision_promoted */ true};

} // namespace TypePromotion

// Implements the the behavior of the following flags:
//   - promote_inputs_to_common_dtype
//   - promote_integer_inputs_to_float
DataType computeTypes(
    const TypePromotionConfig& config,
    const std::vector<torch::jit::TypePtr>& operands);

DataType computeTypes(
    const TypePromotionConfig& config,
    const std::vector<Val*>& operands,
    const bool cast_half_to_float = true);

// Computes the common dtype for the given operands
// Casts operands to common dtype if necessary
// Automatically cast FP16/BF16 dtype to Float
std::vector<Val*> promoteValues(
    const TypePromotionConfig& config,
    const std::vector<Val*>& operands);

std::vector<Val*> promoteValues(
    const std::vector<Val*>& operands,
    DataType common_type);

// Casts value to common dtype if necessary
// Avoid cast if value's dtype matches its dtype class
Val* optionalCast(DataType dtype, Val* v);

// Casts value to common dtype if necessary
Val* optionalCastStrict(DataType dtype, Val* v);

} // namespace nvfuser
