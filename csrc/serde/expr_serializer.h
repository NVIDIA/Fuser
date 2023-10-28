// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/all_nodes.h>
#include <kernel.h>
#include <vector>

namespace nvfuser::serde {

//! kir::Allocate nodes are required at runtime to calculate the size of global
//! intermediate tensors and dynamic shared memory buffers. We avoid lowering
//! the entire kernel during deserialization by using ExpressionBuilder to make
//! the kir::Allocate nodes.
//!
//! ExpressionSerializer serializes values and operations necessary to generate
//! the kir::Allocate nodes in a Kernel.
class ExpressionSerializer {
 public:
  ExpressionSerializer() = default;

  flatbuffers::Offset<NaiveValueGenerator> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      kir::Kernel* kernel,
      const std::vector<const kir::Allocate*>& allocations);

  std::vector<flatbuffers::Offset<AllocateBuffer>> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<const kir::Allocate*>& allocations);

 private:
  flatbuffers::Offset<Instruction> serializeAttribute(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::Val* val);

  flatbuffers::Offset<Instruction> serializeBinaryOp(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::BinaryOp* bop);

  flatbuffers::Offset<Instruction> serializeGetAttr(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::GetAttr* attr);

  flatbuffers::Offset<Instruction> serializeGetItem(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::GetItem* item);

  flatbuffers::Offset<Instruction> serializeGetMetaData(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::GetMetaData* metadata);

  flatbuffers::Offset<Instruction> serializeMerge(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::Merge* merge);

  std::array<flatbuffers::Offset<Instruction>, 3> serializeResize(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::Resize* resize);

  std::array<flatbuffers::Offset<Instruction>, 2> serializeSplit(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::Split* split);

  flatbuffers::Offset<Instruction> serializeSwizzle2D(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::Swizzle2D* swizzle);

  flatbuffers::Offset<Instruction> serializeUnaryOp(
      flatbuffers::FlatBufferBuilder& builder,
      nvfuser::UnaryOp* uop);

  flatbuffers::Offset<SymbolicTensor> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const nvfuser::TensorView* tv);

  flatbuffers::Offset<flatbuffers::Vector<int64_t>> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      std::vector<Val*> domain);

  flatbuffers::Offset<IterDomain> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const nvfuser::IterDomain* id);

  // Bind the value to all_values_ container
  void bind(nvfuser::Val* v);

  // Bind iterDomain
  void bind(nvfuser::IterDomain* id);

  // Bind the iterDomain's extent for the given domain
  void bindDomain(const std::vector<nvfuser::IterDomain*>& domain);

  // 1. Generate extents for IterDomains that compose root domain
  // 2. Create new extents using split, merge, reorder operations for rfactor,
  // allocation, and leaf domains
  void bind(const nvfuser::TensorView* tv);

  // Bind kir::Allocate nodes
  void bind(const std::vector<const kir::Allocate*>& allocations);

  void printStack() const {
    std::vector<nvfuser::Val*> ordered_stack(operation_stack_.size());
    for (auto item : operation_stack_) {
      ordered_stack.at(item.second) = item.first;
    }
    std::cout << "================ ExpressionSerializer Stack ================"
              << std::endl;
    for (auto idx : c10::irange(ordered_stack.size())) {
      auto item = ordered_stack.at(idx);
      std::cout << idx << " ptr: " << ((void*)item) << "\t" << item
                << std::endl;
    }
    std::cout << "============================================================"
              << std::endl;
  }

  std::unordered_map<Val*, long> operation_stack_;
  std::vector<nvfuser::Val*> all_values_;
  std::vector<nvfuser::NamedScalar*> named_scalar_values_;
  std::vector<nvfuser::Val*> const_int_values_;
  std::vector<nvfuser::Val*> symbolic_values_;
  std::deque<nvfuser::Val*> derived_values_;
};

} // namespace nvfuser::serde
