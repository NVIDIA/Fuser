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
      [&](K key) {
        if (key == nullptr) {
          return -1l;
        }
        return all_values.at(key);
      });
  return result;
}

} // namespace

namespace nvfuser {

IrSerde::IrSerde(const IrContainer* container)
    : container_{container},
      toposorted_vals_{topologicalSortValues(container->deterministic_vals())},
      vals_to_id_map_{createToposortValuesMap()},
      exprs_to_id_map_{container->deterministic_exprs_map()} {}

// A generic utility then ensures that all of a value's dependencies should have
// indicies less than its index.
std::vector<Val*> IrSerde::topologicalSortValues(
    const std::deque<Val*>& values) const {
  std::unordered_set<Val*> to_sort;
  std::copy(
      values.begin(), values.end(), std::inserter(to_sort, to_sort.end()));
  std::unordered_set<const Val*> visited;
  std::vector<Val*> sorted;

  // Insert special values immediately
  visited.insert(nullptr);
  visited.insert(container_->getZeroVal());
  visited.insert(container_->getOneVal());
  visited.insert(container_->getFalseVal());
  visited.insert(container_->getTrueVal());
  visited.insert(container_->getMagicZeroVal());

  int64_t deleted = 1;

  // Topological Sort
  while (!to_sort.empty()) {
    --deleted;
    for (auto top_val : to_sort) {
      if (visited.count(top_val)) {
        to_sort.erase(top_val);
        ++deleted;
        break;
      } else {
        bool ready_to_pop = true;
        for (const auto producer : top_val->inputs()) {
          if (!visited.count(producer)) {
            ready_to_pop = false;
          }
        }

        if (ready_to_pop) {
          visited.insert(top_val);
          sorted.push_back(top_val);
        }
      }
    }

    NVF_ERROR(
        deleted >= 0,
        "Failed to remove value from to_sort, so we are stopping to break infinite loop.");
  }
  return sorted;
}

std::unordered_map<Val*, int64_t> IrSerde::createToposortValuesMap()
    const noexcept {
  std::unordered_map<Val*, int64_t> vals_map;
  int64_t count = 0;

  vals_map.emplace(nullptr, -1);
  vals_map.try_emplace(container_->getZeroVal(), -2);
  vals_map.try_emplace(container_->getOneVal(), -3);
  vals_map.try_emplace(container_->getFalseVal(), -4);
  vals_map.try_emplace(container_->getTrueVal(), -5);
  vals_map.try_emplace(container_->getMagicZeroVal(), -6);

  std::transform(
      toposorted_vals_.begin(),
      toposorted_vals_.end(),
      std::inserter(vals_map, vals_map.end()),
      [&count](Val* val) { return std::make_pair(val, count++); });
  return vals_map;
}

int64_t IrSerde::map(Statement* stmt) const {
  if (stmt == nullptr) {
    return -1;
  }
  if (stmt->isVal()) {
    return map(stmt->asVal());
  } else {
    return map(stmt->asExpr());
  }
}

int64_t IrSerde::map(const Statement* stmt) const {
  if (stmt == nullptr) {
    return -1;
  }
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
  if (v == nullptr) {
    return -1;
  }
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
  if (e == nullptr) {
    return -1;
  }
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
