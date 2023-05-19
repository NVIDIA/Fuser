// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <serde/expr_evaluator_serde.h>

namespace nvfuser::serde {

namespace {

template <typename VALTYPE>
std::vector<VALTYPE*> getImmediateProducers(VALTYPE* val) {
  if (val->definition()) {
    auto expr = val->definition();
    return expr->inputs();
  } else {
    return {};
  }
}

//! IR-Generic utility, collects all the producers required for the
//!  given list of IR values and returns them along with the original
//!  list in topological order.
template <typename VALTYPE>
std::vector<VALTYPE*> makeSortedEvaluationList(std::vector<VALTYPE*> input) {
  // Deduplicate
  std::vector<VALTYPE*> to_sort;
  std::unordered_set<VALTYPE*> visited;
  for (auto val : input) {
    if (!visited.count(val)) {
      to_sort.push_back(val);
      visited.insert(val);
    }
  }

  std::vector<VALTYPE*> sorted;
  visited.clear();

  // Topological Sort
  //  Note: didn't explicitly exclude producers that are not in the original
  //   list. This should be acceptable for the intended use.
  while (!to_sort.empty()) {
    auto top_val = to_sort.back();
    if (visited.count(top_val)) {
      to_sort.pop_back();
    } else {
      bool ready_to_pop = true;
      for (auto producer : getImmediateProducers(top_val)) {
        if (!visited.count(producer)) {
          ready_to_pop = false;
          to_sort.push_back(producer);
        }
      }
      if (ready_to_pop) {
        visited.insert(top_val);
        sorted.push_back(top_val);
        to_sort.pop_back();
      }
    }
  }

  return sorted;
}

} // namespace

void ExpressionSerde::generate() {
    auto list = makeSortedEvaluationList(all_values_);
    for (auto v : list) {
        if (NamedScalar* ns = dynamic_cast<NamedScalar*>(v)) {
            std::cout << "named scalar\t" << ns->toString() << std::endl;
        } else {
            std::cout << "es input\t" << v->toString() << std::endl;
        }
    }
}

//! Kernel IR utility, collects all the symbolic values used in allocation nodes.
std::vector<kir::Allocate*> collectBufferSizes(const std::vector<Expr*>& exprs) {
  std::vector<kir::Allocate*> buffers;
  std::vector<Expr*> to_visit(exprs);
  while(!to_visit.empty()) {
    auto expr = to_visit.back();
    to_visit.pop_back();
    if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      buffers.push_back(allocate);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      auto for_loop_exprs = for_loop->body().exprs();
      to_visit.insert(to_visit.end(), for_loop_exprs.begin(), for_loop_exprs.end());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      auto ite_then_exprs = ite->thenBody().exprs();
      auto ite_else_exprs = ite->elseBody().exprs();
      to_visit.insert(to_visit.end(), ite_then_exprs.begin(), ite_then_exprs.end());
      to_visit.insert(to_visit.end(), ite_else_exprs.begin(), ite_else_exprs.end());
    }
  }
  return buffers;
}

void ExpressionSerde::bind(kir::Kernel* kernel) {
  // Collect allocation sizes:
  for (auto allocate : collectBufferSizes(kernel->topLevelExprs())) {
    if (TensorView* tv = dynamic_cast<TensorView*>(allocate->buffer())) {
        if (tv->getMemoryType() == MemoryType::Global) {
            bind(tv);
        }
    }
  }
  bindInputs(kernel);
}

void ExpressionSerde::bind(TensorView* tv, bool is_input) {
    if (tv->getMemoryType() != MemoryType::Global) {
        return;
    }
    if (is_input) {
        bind(tv->getMaybeRFactorDomain(), is_input);
        return;
    }
    bind(tv->getRootDomain());
    bind(tv->getRFactorDomain());
    bind(tv->getAllocationDomain());
    bind(tv->getLeafDomain());
}

void ExpressionSerde::bindInputs(kir::Kernel* kernel) {
    for (auto input : kernel->inputs()) {
        if (TensorView* tv = dynamic_cast<TensorView*>(input)) {
            bind(tv, true /* is_inputs */);
        } else {
            bind(input, true /* is_inputs */);
        }
    }
    auto list = makeSortedEvaluationList(input_values_);
    for (auto v : list) {
        std::cout << "es input\t" << v->toString() << std::endl;
    }
}

} // namespace nvfuser::serde
