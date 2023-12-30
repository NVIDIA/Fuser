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
      toposorted_stmts_{topologicalSortStatements(
          container->deterministic_vals(),
          container->deterministic_exprs())},
      vals_to_id_map_{createToposortValuesMap()},
      exprs_to_id_map_{createToposortExpressionsMap()} {}

// A generic utility then ensures that all of a value's dependencies should have
// indicies less than its index.
//
// For a given Value, all its Val dependencies require a valid definition
// Expr. Initially, we create a Val without definition expression. This Val
// is only valid for Expr. Then, we create an Expr that requires inputs,
// outputs, and attribute Vals. After creating an Expr, we assign the Expr to
// output Vals' definition. Now, output Val are valid for other Vals.
std::vector<Statement*> IrSerde::topologicalSortStatements(
    const std::deque<Val*>& values,
    const std::deque<Expr*>& exprs) {
  std::unordered_set<Statement*> to_sort;

  std::copy(
      values.begin(), values.end(), std::inserter(to_sort, to_sort.end()));
  std::copy(exprs.begin(), exprs.end(), std::inserter(to_sort, to_sort.end()));

  std::vector<Statement*> sorted;

  // valid_value_dependencies holds all statements available for value
  // dependencies. Expressions require output values in their constructor, so
  // output values are created without their definition expression. All values
  // must have their definition expression before they are available as a value
  // dependency.
  std::unordered_set<const Statement*> valid_value_dependencies;

  // Insert special values immediately
  valid_value_dependencies.insert(nullptr);
  valid_value_dependencies.insert(container_->getZeroVal());
  valid_value_dependencies.insert(container_->getOneVal());
  valid_value_dependencies.insert(container_->getFalseVal());
  valid_value_dependencies.insert(container_->getTrueVal());
  valid_value_dependencies.insert(container_->getMagicZeroVal());

  // created_statements holds all statements available for expr dependencies.
  std::unordered_set<const Statement*> created_statements(
      valid_value_dependencies);
  NVF_ERROR(valid_value_dependencies.size() == created_statements.size());

  bool removed_any_statment_from_to_sort = false;
  bool any_ready_to_pop = false;
  // Topological Sort
  while (!to_sort.empty()) {
    removed_any_statment_from_to_sort = false;
    for (auto top_stmt : to_sort) {
      if (created_statements.count(top_stmt) > 0) {
        to_sort.erase(top_stmt);
        removed_any_statment_from_to_sort = true;
        break;
      } else {
        // Check if a statements dependencies are satisfied.
        bool ready_to_pop = true;

        if (top_stmt->isVal()) {
          for (const auto producer : top_stmt->serdeDependencies()) {
            if (valid_value_dependencies.count(producer) == 0) {
              ready_to_pop = false;
            }
          }
        } else {
          // expression input values must be valid.
          for (const auto producer : top_stmt->asExpr()->inputs()) {
            if (valid_value_dependencies.count(producer) == 0) {
              ready_to_pop = false;
            }
          }
          // serdeDependencies == outputs and attributes
          for (const auto producer : top_stmt->serdeDependencies()) {
            if (created_statements.count(producer) == 0) {
              ready_to_pop = false;
            }
          }
        }


        any_ready_to_pop |= ready_to_pop;
        if (ready_to_pop) {
          if (top_stmt->isVal()) {
            // 1) Create Val without definition expression.
            // It is only valid for expressions.
            created_statements.insert(top_stmt);
            if (top_stmt->asVal()->definition() == nullptr) {
              valid_value_dependencies.insert(top_stmt);
            }
          } else {
            // 2) Create Expr that requires inputs, outputs, and attribute Vals.
            // Expr is valid for both expressions and vals.
            created_statements.insert(top_stmt);
            valid_value_dependencies.insert(top_stmt);

            // 3) After creating Expr, assign Expr to output definition.
            // Output Val are now valid.
            for (const auto output_val : top_stmt->asExpr()->outputs()) {
              valid_value_dependencies.insert(output_val);
            }
          }
          sorted.push_back(top_stmt);
        }
      }
    }

    NVF_ERROR(
        removed_any_statment_from_to_sort || any_ready_to_pop,
        "Failed to remove any statement from to_sort",
        " and none of the statements are ready to be removed in the next iteration,"
        " so we are stopping here to break infinite loop.");
  }
  NVF_ERROR(valid_value_dependencies.size() == created_statements.size());
  NVF_ERROR(sorted.size() == (values.size() + exprs.size()));
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

  std::vector<Statement*> vals_stmts;
  std::copy_if(
      toposorted_stmts_.begin(),
      toposorted_stmts_.end(),
      std::back_inserter(vals_stmts),
      [](Statement* s) { return s != nullptr && s->isVal(); });
  std::transform(
      vals_stmts.begin(),
      vals_stmts.end(),
      std::inserter(vals_map, vals_map.end()),
      [&count](Statement* stmt) {
        return std::make_pair(stmt->asVal(), count++);
      });
  return vals_map;
}

std::unordered_map<Expr*, int64_t> IrSerde::createToposortExpressionsMap()
    const noexcept {
  std::unordered_map<Expr*, int64_t> exprs_map;
  int64_t count = 0;

  exprs_map.emplace(nullptr, -1);
  std::vector<Statement*> exprs_stmts;
  std::copy_if(
      toposorted_stmts_.begin(),
      toposorted_stmts_.end(),
      std::back_inserter(exprs_stmts),
      [](Statement* s) { return s != nullptr && s->isExpr(); });
  std::transform(
      exprs_stmts.begin(),
      exprs_stmts.end(),
      std::inserter(exprs_map, exprs_map.end()),
      [&count](Statement* stmt) {
        return std::make_pair(stmt->asExpr(), count++);
      });
  return exprs_map;
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
  return map((Statement*)stmt);
}

int64_t IrSerde::map(Val* v) const {
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
