// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <polymorphic_value.h>
#include <type.h>

namespace nvfuser {

struct TensorMetaData : public Struct {
  PrimDataType dtype;
  void* data;
  c10::IntArrayRef logical_size;
  c10::IntArrayRef logical_stride;
  c10::IntArrayRef alloc_size;
  c10::IntArrayRef alloc_stride;
  std::vector<int64_t> logical_size_data;
  std::vector<int64_t> logical_stride_data;
  std::vector<int64_t> alloc_size_data;
  std::vector<int64_t> alloc_stride_data;

  // bool operator==(const TensorMetaData& other) const {
  //   return data == other.data && logical_size == other.logical_size &&
  //       logical_stride == other.logical_stride &&
  //       alloc_size == other.alloc_size && alloc_stride == other.alloc_stride;
  // }

  std::function<PolymorphicValue()> getter(
      const std::string& key) const override {
    if (key == "data") {
      return [this]() { return PolymorphicValue(Pointer(data, dtype)); };
    } else if (key == "logical_size") {
      if (!logical_size_data.empty()) {
        return [this]() { return PolymorphicValue(logical_size_data); };
      } else {
        return [this]() { return PolymorphicValue(logical_size.vec()); };
      }
    } else if (key == "logical_stride") {
      if (!logical_stride_data.empty()) {
        return [this]() { return PolymorphicValue(logical_stride_data); };
      } else {
        return [this]() { return PolymorphicValue(logical_stride.vec()); };
      }
    } else if (key == "alloc_size") {
      if (!alloc_size_data.empty()) {
        return [this]() { return PolymorphicValue(alloc_size_data); };
      } else {
        return [this]() { return PolymorphicValue(alloc_size.vec()); };
      }
    } else if (key == "alloc_stride") {
      if (!alloc_stride_data.empty()) {
        return [this]() { return PolymorphicValue(alloc_stride_data); };
      } else {
        return [this]() { return PolymorphicValue(alloc_stride.vec()); };
      }
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unknown key ", key);
    }
  }

  std::function<void(const PolymorphicValue&)> setter(
      const std::string& key) override {
    if (key == "data") {
      return [this](const PolymorphicValue& value) { data = (void*)value; };
    } else if (key == "logical_size") {
      return [this](const PolymorphicValue& value) {
        logical_size_data = (std::vector<int64_t>)value;
        logical_size = c10::makeArrayRef(logical_size_data);
      };
    } else if (key == "logical_stride") {
      return [this](const PolymorphicValue& value) {
        logical_stride_data = (std::vector<int64_t>)value;
        logical_stride = c10::makeArrayRef(logical_stride_data);
      };
    } else if (key == "alloc_size") {
      return [this](const PolymorphicValue& value) {
        alloc_size_data = (std::vector<int64_t>)value;
        alloc_size = c10::makeArrayRef(alloc_size_data);
      };
    } else if (key == "alloc_stride") {
      return [this](const PolymorphicValue& value) {
        alloc_stride_data = (std::vector<int64_t>)value;
        alloc_stride = c10::makeArrayRef(alloc_stride_data);
      };
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unknown key ", key);
    }
  }

  StructType type() const override {
    TORCH_INTERNAL_ASSERT(logical_size.size() == logical_stride.size());
    TORCH_INTERNAL_ASSERT(alloc_size.size() == alloc_stride.size());
    return globalTensorMetaData(dtype, logical_size.size(), alloc_size.size());
  }
};

} // namespace nvfuser
