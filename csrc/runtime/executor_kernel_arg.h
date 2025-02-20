// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <exceptions.h>
#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <serde/fusion_cache_generated.h>
#include <torch/csrc/jit/ir/ir.h>
#include <type.h>
#include <visibility.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace nvfuser {

//! KernelArgumentHolder copies meta information from kernel inputs, including
//! tensor sizes/shapes/dtype/memory_ptr and copies scalar inputs. It is used
//! for both compilation as well as kernel execution. The important thing is to
//! strip ownership of tensor from KernelArgumentHolder, so that during async
//! compilation, we are not unnecessarily holding memory that is not needed.
class KernelArgumentHolder {
 public:
  KernelArgumentHolder() = default;

  KernelArgumentHolder(const KernelArgumentHolder& self) = default;

  NVF_API KernelArgumentHolder(
      const c10::ArrayRef<c10::IValue>& inputs,
      std::optional<int8_t> device = std::nullopt);

  //! Computes the smallest index type for the currently held
  //! arguments. It does not consider any other tensors used in a kernel.
  NVF_API PrimDataType getSmallestIndexTypeOfArguments() const;

  // Push a tensor proxy to the arguments
  void pushTensorProxy(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides,
      at::ScalarType dtype);

  NVF_API void push(const c10::ArrayRef<c10::IValue>& args);

  NVF_API void push(const std::vector<at::Tensor>& tensors);

  void erase(const PolymorphicValue& arg_to_delete);

  c10::ArrayRef<c10::IValue> toArrayRef() const;

  void push(PolymorphicValue val) {
    arguments_.emplace_back(PolymorphicValue(std::move(val)));
  }

  PolymorphicValue& back() {
    return arguments_.back();
  }

  const PolymorphicValue& back() const {
    return arguments_.back();
  }

  PolymorphicValue& operator[](size_t ind) {
    return arguments_.at(ind);
  }

  const PolymorphicValue& operator[](size_t ind) const {
    return arguments_.at(ind);
  }

  // Returns iterator pointing to the beginning of vector container
  auto begin() const {
    return arguments_.begin();
  }

  // Returns iterator pointing to the end of vector container
  auto end() const {
    return arguments_.end();
  }

  // Returns iterator pointing to the beginning of vector container
  auto begin() {
    return arguments_.begin();
  }

  // Returns iterator pointing to the end of vector container
  auto end() {
    return arguments_.end();
  }

  auto rbegin() const {
    return arguments_.rbegin();
  }

  auto rend() const {
    return arguments_.rend();
  }

  auto rbegin() {
    return arguments_.rbegin();
  }

  auto rend() {
    return arguments_.rend();
  }

  auto cbegin() const {
    return arguments_.cbegin();
  }

  auto cend() const {
    return arguments_.cend();
  }

  auto cbegin() {
    return arguments_.cbegin();
  }

  auto cend() {
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

 private:
  std::vector<PolymorphicValue> arguments_;

  int8_t device_index_ = 0;
  std::optional<size_t> cache_id_ = std::nullopt;
};

std::vector<std::byte> polymorphicValueToBytes(
    const PolymorphicValue& argument,
    const DataType& dtype,
    PrimDataType index_type);

std::vector<std::byte> getKernelArgument(
    ExpressionEvaluator& ee,
    Val* parameter,
    PrimDataType index_type);

int64_t computeBytes(const KernelArgumentHolder& args);

int64_t computeBytes(const std::vector<at::Tensor>& outputs);

} // namespace nvfuser
