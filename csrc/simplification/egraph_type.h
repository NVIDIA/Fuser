// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

namespace egraph {

// Id is used internally to identify ASTNodes, ENodes and EClasses. The total
// number of ASTNodes or ENodes supported is bounded by the capacity of this
// integer. We will probably not surpass 65536 of these elements, but for safety
// we currently use 32 bits.
using Id = uint32_t;

// We support the following subset of Exprs.
//
// NOTE: although we support BinaryOp, we also _flatten_ expressions like
//
//   u + (v + (w + (x + (y + z))))
//
// using the CommutativeBinaryOp symbol. ENodes with this symbol might have
// more than 2 arguments and their order is arbitrary; two ENodes with this
// symbol and the same op_type, with the same collection of arguments but in
// permutated order should always map to the same EClass ID.
enum ENodeFunctionSymbol {
  NoDefinition,
  LoadStoreOp,
  CastOp,
  UnaryOp,
  BinaryOp,
  TernaryOp,
  CommutativeBinaryOp,
};

} // namespace egraph

} // namespace nvfuser
