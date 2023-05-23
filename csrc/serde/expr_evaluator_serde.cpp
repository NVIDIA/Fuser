// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <serde/expr_evaluator_serde.h>
#include <serde/utils.h>

namespace nvfuser::serde {

namespace {

template <typename VALTYPE>
std::vector<VALTYPE*> getImmediateProducers(VALTYPE* val) {
  return (val->definition()) ? val->definition()->inputs()
                             : std::vector<VALTYPE*>();
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

// 1. Generate extents for IterDomains that compose root domain
// 2. Create new extents using split, merge, reorder operations for rfactor,
// allocation, and leaf domains
void bind(std::vector<Val*>& all_values, nvfuser::TensorView* tv) {
  if (tv->getMemoryType() != MemoryType::Global) {
    return;
  }
  bind(all_values, tv->getRootDomain());
}

} // namespace

flatbuffers::Offset<Instruction> ExpressionSerde::serializeUnaryOp(
    flatbuffers::FlatBufferBuilder& builder,
    UnaryOp* uop) const {
  serde::DataType dtype = (uop->getUnaryOpType() == nvfuser::UnaryOpType::Cast)
      ? mapToSerdeDtype(uop->out()->getDataType().value())
      : serde::DataType_None;

  auto inst = serde::CreateInstructionDirect(
      builder,
      serde::InstructionType_Unary,
      mapToSerdeUnaryOp(uop->getUnaryOpType()),
      serde::BinaryOpType_None,
      dtype,
      operation_stack_.at(uop->inputs().front()),
      0,
      (int64_t)operation_stack_.size(),
      uop->toString().c_str());
  return inst;
}

flatbuffers::Offset<Instruction> ExpressionSerde::serializeBinaryOp(
    flatbuffers::FlatBufferBuilder& builder,
    BinaryOp* bop) const {
  auto inst = serde::CreateInstructionDirect(
      builder,
      serde::InstructionType_Unary,
      serde::UnaryOpType_None,
      mapToSerdeBinaryOp(bop->getBinaryOpType()),
      serde::DataType_None,
      operation_stack_.at(bop->inputs().front()),
      operation_stack_.at(bop->inputs().back()),
      (int64_t)operation_stack_.size(),
      bop->toString().c_str());
  return inst;
}

flatbuffers::Offset<serde::NaiveValueGenerator> ExpressionSerde::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    kir::Kernel* kernel) {
  // Collect allocation sizes
  std::vector<Val*> all_values;
  for (auto allocate : collectBufferSizes(kernel->topLevelExprs())) {
    if (TensorView* tv = dynamic_cast<TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
    }
  }
  auto list = makeSortedEvaluationList(all_values);

  std::unordered_set<nvfuser::NamedScalar*> named_scalar_values;
  std::unordered_set<nvfuser::Int*> const_int_values;
  std::unordered_set<nvfuser::Val*> symbolic_values;
  std::vector<nvfuser::Val*> derived_values;
  for (auto v : list) {
    if (v->definition() == nullptr) {
      if (NamedScalar* ns = dynamic_cast<NamedScalar*>(v)) {
        named_scalar_values.insert(ns);
      } else if (v->isConstInt()) {
        const_int_values.insert(v->as<nvfuser::Int>());
      } else {
        symbolic_values.insert(v);
      }
    } else {
      derived_values.push_back(v);
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

  using fb_instruction = flatbuffers::Offset<Instruction>;
  std::vector<fb_instruction> instructions_fb;
  for (const auto& ns : named_scalar_values) {
    std::cout << ns->name() << std::endl;
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_NamedString,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_None,
        0,
        0,
        0,
        ns->name().c_str());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(ns, operation_stack_.size());
  }

  for (const auto& int_val : const_int_values) {
    auto inst = serde::CreateInstructionDirect(
        builder,
        serde::InstructionType_Scalar,
        serde::UnaryOpType_None,
        serde::BinaryOpType_None,
        serde::DataType_Int,
        int_val->evaluateInt(),
        0,
        0,
        nullptr /* name */);
    instructions_fb.push_back(inst);
    operation_stack_.emplace(int_val, operation_stack_.size());
  }
  for (auto& val : symbolic_values) {
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
    operation_stack_.emplace(val, operation_stack_.size());
  }

  for (auto& val : derived_values) {
    auto def = val->definition();
    TORCH_INTERNAL_ASSERT(def, "Expected definition with derived value.");
    if (auto uop = dynamic_cast<UnaryOp*>(def)) {
      auto inst = serializeUnaryOp(builder, uop);
      instructions_fb.push_back(inst);
      operation_stack_.emplace(val, operation_stack_.size());
    } else if (auto bop = dynamic_cast<BinaryOp*>(def)) {
      auto inst = serializeBinaryOp(builder, bop);
      instructions_fb.push_back(inst);
      operation_stack_.emplace(val, operation_stack_.size());
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unknown Expression.");
    }
  }
  return serde::CreateNaiveValueGeneratorDirect(builder, &instructions_fb);
}

std::vector<flatbuffers::Offset<AllocateBuffer>> ExpressionSerde::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<const kir::Allocate*>& allocations) {
  using fb_allocate = flatbuffers::Offset<serde::AllocateBuffer>;
  std::vector<fb_allocate> fb_global_allocations;

  for (auto alloc : allocations) {
    auto alloc_buffer_tv = alloc->buffer()->as<nvfuser::TensorView>();
    TORCH_INTERNAL_ASSERT(alloc_buffer_tv);

    auto fb_alloc = serde::CreateAllocateBuffer(
        builder, serialize(builder, alloc_buffer_tv), alloc->zeroInit());
    fb_global_allocations.push_back(fb_alloc);
  }
  return fb_global_allocations;
}

// TODO create separate functions for TensorDomain and IterDomain
flatbuffers::Offset<serde::SymbolicTensor> ExpressionSerde::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::TensorView* tv) {
  std::vector<flatbuffers::Offset<IterationDomain>> fb_root_domain;
  for (auto id : tv->getRootDomain()) {
    TORCH_INTERNAL_ASSERT(
        operation_stack_.count(id->extent()),
        "Missing value in NaiveValueGenerator stack.");
    auto extent_id = operation_stack_.at(id->extent());
    fb_root_domain.push_back(serde::CreateIterationDomain(builder, extent_id));
  }

  return serde::CreateSymbolicTensor(
      builder, serde::CreateDomainDirect(builder, &fb_root_domain));
}

} // namespace nvfuser::serde
