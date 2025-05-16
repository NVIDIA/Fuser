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

struct GlobalBufferInfo;

//! KernelArgumentHolder copies meta information from kernel inputs, including
//! tensor sizes/shapes/dtype/memory_ptr and copies scalar inputs. It is used
//! for both compilation as well as kernel execution. It takes ownership of
//! at::Tensors so care should be taken when using it relative to tensor.
class NVF_API KernelArgumentHolder {
 public:
  KernelArgumentHolder() = default;

  KernelArgumentHolder(const KernelArgumentHolder& self) = default;
  KernelArgumentHolder(KernelArgumentHolder& self) = default;
  KernelArgumentHolder(KernelArgumentHolder&& self) = default;
  KernelArgumentHolder& operator=(const KernelArgumentHolder& other) = default;
  KernelArgumentHolder& operator=(KernelArgumentHolder&& other) = default;

  // Constructor using std::enable_if_t for C++17 compatibility to prevent
  // implicit conversion to KernelArgumentHolder which can cause a recursive
  // call to the constructor.
  //
  // Previously this constructor took in an optional device but I couldn't
  // figure out how to get that to work with the variadic template.
  template <typename... Args>
  NVF_API KernelArgumentHolder(Args&&... args) {
    (push(std::forward<Args>(args)), ...);
    if (arguments_.empty()) {
      return;
    }
    device_index_ = getCommonDeviceCUDA(*this, std::nullopt);
  }

  //! Computes the smallest index type for the currently held
  //! arguments. It does not consider any other tensors used in a kernel.
  NVF_API PrimDataType getSmallestIndexTypeOfArguments() const;

  // Push a tensor proxy to the arguments
  void pushTensorProxy(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides,
      at::ScalarType dtype);

  // KernelArgumentHolder specific push to disambiguate from PolymorphicValue
  // push
  template <typename T>
  std::enable_if_t<std::is_same_v<std::decay_t<T>, KernelArgumentHolder>, void>
  push(const T& args) {
    arguments_.reserve(arguments_.size() + args.size());
    for (const auto& arg : args) {
      arguments_.emplace_back(arg);
    }
  }

  void reserve(size_t size) {
    arguments_.reserve(size);
  }

  void resize(size_t size) {
    arguments_.resize(size);
  }

  void push(const std::vector<at::Tensor>& tensors);
  void push(const c10::ArrayRef<c10::IValue>& args);
  void push(std::initializer_list<c10::IValue> args) {
    push(c10::ArrayRef<c10::IValue>(args));
  }

  void push(std::initializer_list<at::Tensor> args) {
    push(std::vector<at::Tensor>(args));
  }
  void push(std::vector<PolymorphicValue> args) {
    for (const auto& arg : args) {
      push(arg);
    }
  }

  void push(const std::vector<PolymorphicValue>& args);
  void push(const std::vector<c10::IValue>& args);
  // Needed to disambiguate from std::optional<at::Tensor> push when using
  // at::Tensor
  void push(at::Tensor tensor);
  void push(PolymorphicValue val);
  void push(std::optional<at::Tensor> tensor);

  void erase(const PolymorphicValue& arg_to_delete);

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

  int64_t size() const {
    return std::ssize(arguments_);
  }

  bool empty() const {
    return arguments_.empty();
  }

  void setDeviceIndex(std::optional<int8_t> index = std::nullopt);

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
  void setCommonDevice();

 private:
  std::vector<PolymorphicValue> arguments_;

  int8_t device_index_ = 0;
  std::optional<size_t> cache_id_ = std::nullopt;
};

// Used to convert a polymorphic value to a byte vector. Do not use for CUDA
// Tensors, use tensorToBytes instead.
std::vector<std::byte> polymorphicValueToBytes(
    const PolymorphicValue& argument,
    const DataType& dtype,
    PrimDataType index_type);

// Used to convert a CUDA tensor to a byte vector.
std::vector<std::byte> tensorToBytes(
    const PolymorphicValue& argument,
    const std::vector<int64_t>& logical_sizes,
    const std::vector<int64_t>& allocation_strides,
    PrimDataType idx_type,
    const std::vector<int64_t>& unsharded_logical_sizes = {});

int64_t computeBytes(const KernelArgumentHolder& args);

} // namespace nvfuser
