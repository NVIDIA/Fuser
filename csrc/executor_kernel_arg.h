// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <exceptions.h>
#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <serde/fusion_cache_generated.h>
#include <torch/csrc/jit/ir/ir.h>
#include <type.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace nvfuser {

class TORCH_CUDA_CU_API KernelArgumentHolder {
 public:
  static KernelArgumentHolder createKernelArgumentHolder(
      const c10::ArrayRef<c10::IValue>& inputs,
      std::optional<int8_t> device = std::nullopt);

  KernelArgumentHolder() = default;

  KernelArgumentHolder(const KernelArgumentHolder& self) = default;

  //! Computes the smallest index type for the currently held
  //! arguments. It does not consider any other tensors used in a kernel.
  PrimDataType getSmallestIndexTypeOfArguments() const;

  // Push a tensor proxy to the arguments
  void pushTensorProxy(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides,
      at::ScalarType dtype);

  void push(const c10::ArrayRef<c10::IValue>& args);

  void push(const std::vector<at::Tensor>& tensors);

  void erase(const PolymorphicValue* arg_to_delete);

  void push(PolymorphicValue val) {
    arguments_.push_back(std::make_shared<PolymorphicValue>(std::move(val)));
  }

  PolymorphicValue* back() {
    return arguments_.back().get();
  }

  PolymorphicValue* operator[](size_t ind) const {
    return arguments_.at(ind).get();
  };

  auto cbegin() const {
    return arguments_.cbegin();
  }

  auto cend() const {
    return arguments_.cend();
  }

  auto getBackInserter() {
    return std::back_inserter(arguments_);
  }

  size_t size() const {
    return arguments_.size();
  }

  bool empty() const {
    return arguments_.empty();
  }

  void setDeviceIndex(int8_t index) {
    device_index_ = index;
  }

  int8_t getDeviceIndex() const {
    return device_index_;
  }

  void setCacheId(size_t id) {
    cache_id_ = id;
  }

  std::optional<size_t> getCacheId() const {
    return cache_id_;
  }

  std::string toString() const;

  //! Serialize Kernel Argument Holder using flatbuffers
  flatbuffers::Offset<serde::KernelArgumentHolder> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize Kernel Argument Holder using flatbuffers
  void deserialize(const serde::KernelArgumentHolder* buffer);

  // In principle, this should return a `const std::vector<void*>&`, but
  // unfortunately cuLaunchKernel requires a non-const pointer `void**` instead
  // of a const pointer `void* const*`
  std::vector<void*>& getArgumentPointers(
      kir::Kernel* kernel,
      ExpressionEvaluator& expr_eval);

 private:
  std::vector<std::shared_ptr<PolymorphicValue>> arguments_;
  std::vector<std::vector<std::byte>> kernel_argument_bytes_;
  std::vector<void*> kernel_argument_pointers_;

  int8_t device_index_ = 0;
  std::optional<size_t> cache_id_ = std::nullopt;
  kir::Kernel* kernel_ = nullptr;
};

std::vector<std::byte> polymorphicValueToBytes(
    const PolymorphicValue& argument,
    const DataType& dtype,
    PrimDataType index_type);

} // namespace nvfuser
