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

namespace egraph {

using Id = uint32_t;
/* This is a sketch of an approach where we wrap the Id in order to implement
 * operators on it that can be used to quickly create new ENodes and return
 * their Ids. This is motivated by a desire to make it convenient to express
 * substitutions and relations to prove.
 *
 * Given Ids a, b, c for nodes na and nb, represent (na < nb) || nc
 *
 *  Id n1 = eg->add({{FunctionSymbol::BinaryOp, BinaryOpType::LT}, {a, b}});
 *  Id n2 =
 *      eg->add({{FunctionSymbol::BinaryOp, BinaryOpType::LogicalOr}, {n1, c}});
 *
 * Since this is so cumbersome, the Id type SHOULD IMPLEMENT operators for
 * common operations, so you may instead do
 *
 *   Id n2 = (a < b) || c;
 *
 * Downsides:
 *  - Id as an integer already has a bunch of operators. We should never want
 *    to use them directly, but if we did then we would return a new
 *    integer-equivalent that is not the simple integer sum we were looking for
 *    but rather the Id of an ENode representing a symbolic sum.
 *  - This also complicates the interface. We require actual integer Ids in
 *    some places like in the UnionFind for example so we will need to directly
 *    refer to the wrapped value or its type sometimes.
 *
 * TODO: (see above) Instead, a middle ground may be to provide a wrapper that
 * can be _explicitly_ converted to and from Id with short utilities:
 *
 *   Id foo(Id a, Id b) {
 *     return enode(a) < enode(b) || enode(c);
 *   }
 *
 * The wrapped types would have no accessors and would need to be converted
 * back to an Id to be useful.

struct Id {
  // Id::IntType is used internally to identify ENodes and EClasses. The total
  // number of ENodes supported is bounded by the capacity of this integer. We
  // will probably not surpass 65536 of these elements, but for safety we
  // currently use 32 bits.
  using IntType = uint32_t;
  uint32_t id;

 public:
  // Enable implicit conversion to and from IntType
  Id(const IntType& id_=0U) : id(id_) {}
  operator IntType&() { return id; }
  operator IntType() const { return id; }

  Id operator+(const Id& other);
  Id operator-(const Id& other);
  // ...
};
*/

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
enum FunctionSymbol {
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
