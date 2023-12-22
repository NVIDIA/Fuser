// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nvfuser {

class Statement;
class Expr;
class Val;
class IterDomain;
class TensorView;
class IrContainer;

class IrSerde {
 public:
  IrSerde(const IrContainer* container);

  int64_t map(Statement* v) const;
  int64_t map(const Statement* v) const;

  int64_t map(Val* v) const;
  int64_t map(const Val* v) const;

  int64_t map(Expr* e) const;
  int64_t map(const Expr* e) const;

  std::vector<int64_t> map(const std::vector<Statement*>& vals) const;
  std::vector<int64_t> map(const std::vector<Val*>& vals) const;
  std::vector<int64_t> map(const std::vector<Expr*>& exprs) const;

  std::vector<int64_t> map(const std::vector<IterDomain*>& vals) const;
  std::vector<int64_t> map(const std::vector<TensorView*>& vals) const;

 private:
  const IrContainer* container_;
  const std::unordered_map<Val*, int64_t> vals_to_id_map_;
  const std::unordered_map<Expr*, int64_t> exprs_to_id_map_;
};

} // namespace nvfuser
