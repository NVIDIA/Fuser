
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

//! ExpressionBuilder is similar to ExpressionEvaluator, but instead of
//! calculating concrete values, it creates Fusion IR expressions given
//! serialized NaiveValueGenerator.
//!
//! Supported: kir::Allocate
//! TODO: Val* and TensorView*
class ExpressionBuilder {
  using Allocations = flatbuffers::Vector<flatbuffers::Offset<AllocateBuffer>>;

 public:
  ExpressionBuilder(kir::Kernel* kernel);
  void deserialize(const NaiveValueGenerator* buffer);
  std::vector<const kir::Allocate*> deserialize(const Allocations* buffers);

 private:
  void deserialize(const Instruction* buffer);
  nvfuser::Val* buildUnaryOp(const UnaryOp* buffer);
  nvfuser::Val* buildBinaryOp(const BinaryOp* buffer);
  nvfuser::IterDomain* buildIterDomain(const IterDomain* buffer);

  void printStack() const {
    std::cout << "================ ExpressionBuilder Stack ================"
              << std::endl;
    for (auto idx : c10::irange(operation_stack_.size())) {
      auto item = operation_stack_.at(idx);
      std::cout << idx << " ptr: " << ((void*)item) << "\t" << item
                << std::endl;
    }
    std::cout << "========================================================="
              << std::endl;
  }

  kir::Kernel* kernel_;
  std::vector<Val*> operation_stack_;
};

} // namespace nvfuser::serde
