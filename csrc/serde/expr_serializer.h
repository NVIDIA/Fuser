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
  ExpressionSerializer(kir::Kernel* kernel) : kernel_{kernel} {}

  flatbuffers::Offset<NaiveValueGenerator> serializeNaiveValueGenerator(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<const kir::Allocate*>& allocations);

  std::vector<flatbuffers::Offset<AllocateBuffer>> serializeAllocations(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<const kir::Allocate*>& allocations);

 private:
  flatbuffers::Offset<SymbolicTensor> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const nvfuser::TensorView* tv);

  template <typename T>
  flatbuffers::Offset<flatbuffers::Vector<int64_t>> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<T*>& values);

  void printStack() const {
    std::vector<const nvfuser::Val*> ordered_stack(operation_stack_.size());
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

  kir::Kernel* kernel_;
  std::unordered_map<const Val*, long> operation_stack_;
};

} // namespace nvfuser::serde
