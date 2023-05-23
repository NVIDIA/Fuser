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

//! Kernel IR utility, collects all the symbolic values used in allocation
//! nodes.
std::vector<kir::Allocate*> collectBufferSizes(
    const std::vector<Expr*>& exprs) {
  std::vector<kir::Allocate*> buffers;
  std::vector<Expr*> to_visit(exprs);
  while (!to_visit.empty()) {
    auto expr = to_visit.back();
    to_visit.pop_back();
    if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      buffers.push_back(allocate);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      auto for_loop_exprs = for_loop->body().exprs();
      to_visit.insert(
          to_visit.end(), for_loop_exprs.begin(), for_loop_exprs.end());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      auto ite_then_exprs = ite->thenBody().exprs();
      auto ite_else_exprs = ite->elseBody().exprs();
      to_visit.insert(
          to_visit.end(), ite_then_exprs.begin(), ite_then_exprs.end());
      to_visit.insert(
          to_visit.end(), ite_else_exprs.begin(), ite_else_exprs.end());
    }
  }
  return buffers;
}

void bind(std::vector<Val*>& all_values, Val* v) {
  all_values.push_back(v);
}

void bind(std::vector<Val*>& all_values, std::vector<IterDomain*> domain) {
  for (auto d : domain) {
    bind(all_values, d->extent());
  }
}

void bind(std::vector<Val*>& all_values, TensorView* tv) {
  if (tv->getMemoryType() != MemoryType::Global) {
    return;
  }
  bind(all_values, tv->getRootDomain());
  bind(all_values, tv->getRFactorDomain());
  bind(all_values, tv->getAllocationDomain());
  bind(all_values, tv->getLeafDomain());
}

} // namespace

flatbuffers::Offset<serde::NaiveValueGenerator> ExpressionSerde::serialize(flatbuffers::FlatBufferBuilder& builder, kir::Kernel* kernel) {
  // Collect allocation sizes:
  std::vector<Val*> all_values;
  for (auto allocate : collectBufferSizes(kernel->topLevelExprs())) {
    if (TensorView* tv = dynamic_cast<TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
    }
  }
  auto list = makeSortedEvaluationList(all_values);

  for (auto v : list) {
    if (v->definition() == nullptr) {
      if (NamedScalar* ns = dynamic_cast<NamedScalar*>(v)) {
        named_scalar_values_.insert(ns->name());
      } else if (v->isConstScalar()) {
        const_values_.insert(v->evaluateInt());
      } else {
        symbolic_values_.insert(v);
      }
    } else {
      derived_values_.push_back(v);
    }
  }

  /*
  table NaiveValueGenerator {
    instructions : [Instruction];
  }

  table Instruction {
    instruction : InstructionType;
    unary_type : UnaryOpType;
    binary_type : BinaryOpType;
    data_type : DataType;
    src0 : int;
    src1 : int;
    dest : int;
    name : string;
  }
  */

  using fb_instruction = flatbuffers::Offset<serde::Instruction>;
  std::vector<fb_instruction> instructions_fb;

  for(const auto& name : named_scalar_values_) {
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_NamedString,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_None,
        0,
        0,
        0,
        name.c_str());
    instructions_fb.push_back(inst);
  }

  for(const auto& v : const_values_) {
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_Scalar,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_Int,
        v,
        0,
        0,
        nullptr /* name */);
    instructions_fb.push_back(inst);
  }
  for(auto& val : symbolic_values_) {
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_Symbolic,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_Int,
        val->name(),
        0,
        0,
        val->toString().c_str());
    instructions_fb.push_back(inst);
  }

  for(auto& val : derived_values_) {
    auto def = val->definition();
    TORCH_INTERNAL_ASSERT(def, "Expected definition with derived value.");
    if (auto uop = dynamic_cast<UnaryOp*>(def)) {
      // TODO
      // auto inst = makeUnaryOp(uop);
      // instructions_fb.push_back(inst);
      std::cout << uop->toString() << std::endl;
    } else if (auto bop = dynamic_cast<BinaryOp*>(def)) {
      // TODO
      // auto inst = makeBinaryOp(bop);
      // instructions_fb.push_back(inst);
      std::cout << bop->toString() << std::endl;
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unknown Expression.");
    }
  }

  return serde::CreateNaiveValueGeneratorDirect(builder, &instructions_fb);
}

} // namespace nvfuser::serde
