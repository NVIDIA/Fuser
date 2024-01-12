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

// This helper function checks if all the items are present in a set of
// nvfuser::Statements.
template <typename Container>
bool checkSetContainsItems(
    const std::unordered_set<const nvfuser::Statement*>& set,
    const Container& items) {
  return std::all_of(items.begin(), items.end(), [&](const auto& a) {
    return set.count(a) > 0;
  });
}

int64_t getStatementId(nvfuser::Statement* stmt) {
  if (stmt == nullptr) {
    return -1l;
  }
  return stmt->id();
}

} // namespace

namespace nvfuser {

IrSerde::IrSerde(const IrContainer* container)
    : container_{container},
      toposorted_stmts_{topologicalSortStatements(container)},
      vals_to_id_map_{createToposortValuesMap()},
      exprs_to_id_map_{createToposortExpressionsMap()} {}

std::vector<int64_t> IrSerde::update(const std::vector<Statement*>& new_stmts) {
  std::unordered_set<int64_t> new_stmts_set;
  for (Statement* stmt : new_stmts) {
    new_stmts_set.insert(getStatementId(stmt));
  }

  std::vector<Statement*> updated_stmts;
  std::vector<int64_t> statement_ids;
  std::unordered_set<int64_t> this_stmts_set;
  // Remove statments that are missing from new_stmts
  for (Statement* stmt : toposorted_stmts_) {
    int64_t stmt_id = getStatementId(stmt);
    if (new_stmts_set.count(stmt_id) == 0) {
      continue;
    }
    statement_ids.push_back(stmt_id);
    this_stmts_set.insert(stmt_id);
    updated_stmts.push_back(stmt);
  }

  // Add statements that are missing from toposorted_stmts_
  for (Statement* stmt : new_stmts) {
    int64_t stmt_id = getStatementId(stmt);
    if (this_stmts_set.count(stmt_id) > 0) {
      continue;
    }
    statement_ids.push_back(stmt_id);
    updated_stmts.push_back(stmt);
  }

  toposorted_stmts_.swap(updated_stmts);
  return statement_ids;
}

std::vector<Statement*> IrSerde::topologicalSortStatements(
    const IrContainer* container) {
  if (container->validSerializationState()) {
    return container->deterministic_stmts();
  }
  return topologicalSortStatements(
      container->deterministic_vals(), container->deterministic_exprs());
}

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
  std::vector<Statement*> sorted;
  sorted.reserve(values.size() + exprs.size());

  // During segmentation, intermediate TensorViews can become new global
  // TensorViews. A new TensorDomain is created for the new global TensorView
  // but its original TensorDomain remains in container. The values of the
  // original TensorDomain are altered and reused making the original
  // TensorDomain invalid. A solution is to prune the unused TensorDomain from
  // the Container.
  std::vector<Val*> all_tensorviews;
  std::copy_if(
      values.begin(),
      values.end(),
      std::back_inserter(all_tensorviews),
      [](Val* v) { return v->isA<TensorView>(); });

  // Collect TensorDomains that are used in a TensorView
  std::unordered_set<Val*> invalid_tensor_domains;
  std::copy_if(
      values.begin(),
      values.end(),
      std::inserter(invalid_tensor_domains, invalid_tensor_domains.end()),
      [&all_tensorviews](Val* v) {
        if (!v->isA<TensorDomain>()) {
          return false;
        }
        return std::all_of(
            all_tensorviews.begin(), all_tensorviews.end(), [&v](Val* tv) {
              return tv->as<TensorView>()->domain() != v->as<TensorDomain>();
            });
      });

  std::unordered_set<Val*> to_sort_values;
  std::copy(
      values.begin(),
      values.end(),
      std::inserter(to_sort_values, to_sort_values.end()));

  std::unordered_set<Expr*> to_sort_exprs;
  std::copy(
      exprs.begin(),
      exprs.end(),
      std::inserter(to_sort_exprs, to_sort_exprs.end()));

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

  std::vector<Val*> ready_values;
  std::vector<Expr*> ready_exprs;
  ready_values.reserve(exprs.size());
  ready_exprs.reserve(values.size());
  // Topological Sort
  while (!to_sort_values.empty() || !to_sort_exprs.empty()) {
    ready_values.clear();
    ready_exprs.clear();

    // Find any available values and expressions to add to sorted vector
    std::copy_if(
        to_sort_values.begin(),
        to_sort_values.end(),
        std::back_inserter(ready_values),
        [&valid_value_dependencies](Val* stmt) {
          return checkSetContainsItems(
              valid_value_dependencies, stmt->serdeDependencies());
        });

    std::copy_if(
        to_sort_exprs.begin(),
        to_sort_exprs.end(),
        std::back_inserter(ready_exprs),
        [&created_statements, &valid_value_dependencies](Expr* stmt) {
          return checkSetContainsItems(
                     valid_value_dependencies, stmt->inputs()) &&
              checkSetContainsItems(
                     created_statements, stmt->serdeDependencies());
        });

    NVF_ERROR(
        !ready_values.empty() || !ready_exprs.empty(),
        "Failed to find any statements from to_sort_values or to_sort_exprs ",
        "that are ready to be removed in the next iteration, so we are ",
        "stopping here to break infinite loop.");

    // Add all statements to sorted vector except TensorDomain statements that
    // are unused by any TensorView.
    std::copy_if(
        ready_values.begin(),
        ready_values.end(),
        std::back_inserter(sorted),
        [&invalid_tensor_domains](Val* stmt) {
          return invalid_tensor_domains.count(stmt) == 0;
        });
    std::copy(
        ready_exprs.begin(), ready_exprs.end(), std::back_inserter(sorted));

    // Add all statements to created_statements
    std::copy(
        ready_values.begin(),
        ready_values.end(),
        std::inserter(created_statements, created_statements.end()));
    std::copy(
        ready_exprs.begin(),
        ready_exprs.end(),
        std::inserter(created_statements, created_statements.end()));

    // Erase all statements from to_sort_values and to_sort_exprs
    for (auto stmt : ready_values) {
      to_sort_values.erase(stmt->asVal());
    }
    for (auto stmt : ready_exprs) {
      to_sort_exprs.erase(stmt);
      // After creating an expression, its output values are now valid
      // because the definition for those values is this expression.
      std::copy(
          stmt->outputs().begin(),
          stmt->outputs().end(),
          std::inserter(
              valid_value_dependencies, valid_value_dependencies.end()));
    }

    // Any Expression or Val without a definition expression is immediately
    // valid.
    std::copy_if(
        ready_values.begin(),
        ready_values.end(),
        std::inserter(valid_value_dependencies, valid_value_dependencies.end()),
        [](Val* stmt) { return stmt->definition() == nullptr; });
    std::copy(
        ready_exprs.begin(),
        ready_exprs.end(),
        std::inserter(
            valid_value_dependencies, valid_value_dependencies.end()));
  }
  NVF_ERROR(valid_value_dependencies.size() == created_statements.size());
  NVF_ERROR(
      (sorted.size() + invalid_tensor_domains.size()) ==
      (values.size() + exprs.size()));
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

std::vector<int64_t> IrSerde::map(const std::unordered_set<Val*>& vals) const {
  return convertContainer<Val*, int64_t>(vals_to_id_map_, vals);
}

} // namespace nvfuser
