// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/all_nodes.h>
#include <ir/container.h>
#include <ir/serde.h>

namespace {
// This helper function converts selected keys to their corresponding values.
// During serialization, we map pointers to an integer. For deserialization,
// we reverse the mapping from integers to pointers.
template <typename K, typename V, typename ContainerV, typename ContainerK>
std::vector<V> convertContainer(
    const ContainerV& all_values,
    const ContainerK& selected_keys) {
  std::vector<V> result;
  result.reserve(selected_keys.size());
  std::transform(
      selected_keys.begin(),
      selected_keys.end(),
      std::back_inserter(result),
      [&](K key) { return all_values.at(key); });
  return result;
}

} // namespace

namespace nvfuser {

IrSerde::IrSerde(const IrContainer* container)
    : container_{container},
      vals_to_id_map_{container->deterministic_vals_map(
          /*include_persistent_values=*/true)},
      exprs_to_id_map_{container->deterministic_exprs_map()} {}

int64_t IrSerde::map(Statement* stmt) const {
  if (stmt->isVal()) {
    return map(stmt->asVal());
  } else {
    return map(stmt->asExpr());
  }
}

int64_t IrSerde::map(const Statement* stmt) const {
  return map((Statement*)stmt);
}

int64_t IrSerde::map(Val* v) const {
  if (v == nullptr) {
    return -1;
  }
  NVF_ERROR(
      vals_to_id_map_.count(v) > 0,
      "Missing value: ",
      v->toString(),
      " from vals_to_id_map");
  return vals_to_id_map_.at(v);
}

int64_t IrSerde::map(const Val* v) const {
  // TODO use const Val* key with unordered_map to avoid const cast to Val*
  return map((Val*)v);
}

int64_t IrSerde::map(Expr* e) const {
  if (e == nullptr) {
    return -1;
  }
  NVF_ERROR(
      exprs_to_id_map_.count(e) > 0,
      "Missing expr: ",
      e->toString(),
      " from exprs_to_id_map");
  return exprs_to_id_map_.at(e);
}

int64_t IrSerde::map(const Expr* e) const {
  // TODO use const Expr* key with unordered_map to avoid const cast to Expr*
  return map((Expr*)e);
}

std::vector<int64_t> IrSerde::map(const std::vector<Statement*>& stmts) const {
  std::vector<int64_t> result;
  result.reserve(stmts.size());
  std::transform(
      stmts.begin(),
      stmts.end(),
      std::back_inserter(result),
      [&](Statement* key) { return map(key); });
  return result;
}

std::vector<int64_t> IrSerde::map(const std::vector<Val*>& vals) const {
  return convertContainer<Val*, int64_t>(vals_to_id_map_, vals);
}

std::vector<int64_t> IrSerde::map(const std::vector<Expr*>& exprs) const {
  return convertContainer<Expr*, int64_t>(exprs_to_id_map_, exprs);
}

std::vector<int64_t> IrSerde::map(const std::vector<IterDomain*>& vals) const {
  return convertContainer<Val*, int64_t>(vals_to_id_map_, vals);
}

std::vector<int64_t> IrSerde::map(const std::vector<TensorView*>& vals) const {
  return convertContainer<Val*, int64_t>(vals_to_id_map_, vals);
}

} // namespace nvfuser
