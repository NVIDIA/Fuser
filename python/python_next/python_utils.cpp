// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <python_utils.h>

#include <polymorphic_value.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <algorithm>
#include <ranges>

namespace nvfuser::python {

struct DimInfo {
  int64_t index;
  int64_t size;
  int64_t stride;
  int64_t stride_order;
  std::optional<bool> contiguity = std::nullopt;

  bool isBroadcast() {
    return stride == 0 || size == 1;
  }
};

std::pair<std::vector<std::optional<bool>>, std::vector<int64_t>>
computeTensorDescriptor(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  NVF_CHECK(
      sizes.size() == strides.size(),
      "compute_tensor_descriptor: "
      "Sizes and strides must have the same number of dimensions");
  std::vector<DimInfo> non_broadcast_dim_info_vec;
  std::vector<DimInfo> stride_zero_dims;
  for (auto i : std::ranges::iota_view(0u, sizes.size())) {
    // NOTE: not supporting negative stride yet, but we can probably allow it on
    // broadcast dims
    NVF_CHECK(
        strides[i] >= 0,
        "negative stride on tensor is not supported: strides[",
        i,
        "]=",
        strides[i]);
    DimInfo dim_info{(int64_t)i, sizes[i], strides[i]};
    if (strides[i] != 0) {
      non_broadcast_dim_info_vec.push_back(dim_info);
    } else {
      stride_zero_dims.push_back(dim_info);
    }
  }
  // sort non-broadcast dimensions by stride
  std::stable_sort(
      non_broadcast_dim_info_vec.begin(),
      non_broadcast_dim_info_vec.end(),
      [](const auto& l, const auto& r) { return l.stride > r.stride; });

  // combine dimensions while preserving the semantical position of broadcast
  // dimensions
  for (const auto& dim_info : stride_zero_dims) {
    non_broadcast_dim_info_vec.insert(
        non_broadcast_dim_info_vec.begin() + dim_info.index, dim_info);
  }

  // Dimensions are marked contiguous by inspecting the current dimension and
  // one to the right towards the inner dimension while skipping over broadcast
  // dimensions.
  // The innermost dimension, that is not broadcasted, does not have any
  // dimension to it's right and needs to have stride equal to 1 in order to be
  // marked contiguous.
  for (int64_t i = 0; i < (int64_t)sizes.size();) {
    non_broadcast_dim_info_vec[i].stride_order = (int64_t)sizes.size() - 1 - i;
    if (!non_broadcast_dim_info_vec[i].isBroadcast()) {
      auto l = i++;
      int64_t expected = 1;
      for (; i < (int64_t)sizes.size(); i++) {
        non_broadcast_dim_info_vec[i].stride_order =
            (int64_t)sizes.size() - 1 - i;
        if (!non_broadcast_dim_info_vec[i].isBroadcast()) {
          expected = non_broadcast_dim_info_vec[i].stride *
              non_broadcast_dim_info_vec[i].size;
          break;
        }
      }
      non_broadcast_dim_info_vec[l].contiguity =
          (non_broadcast_dim_info_vec[l].stride == expected);
    } else {
      i++;
    }
  }

  std::vector<int64_t> stride_order_vec(sizes.size(), -1);
  for (const auto& dim_info : non_broadcast_dim_info_vec) {
    stride_order_vec[dim_info.index] = dim_info.stride_order;
  }
  std::vector<std::optional<bool>> contiguity_vec;
  std::transform(
      non_broadcast_dim_info_vec.begin(),
      non_broadcast_dim_info_vec.end(),
      std::back_inserter(contiguity_vec),
      [](const DimInfo& val) { return val.contiguity; });

  return std::make_pair(contiguity_vec, stride_order_vec);
}

void verifyShape(const std::vector<int64_t>& shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    NVF_CHECK(
        shape[i] >= -1,
        "The value ",
        shape[i],
        " at index ",
        i,
        " was neither symbolic(-1), zero_element(0), broadcast(1), or static(>1).");
  }
}

nvfuser::KernelArgumentHolder from_pyiterable(
    const py::iterable& iter,
    std::optional<int64_t> device) {
  nvfuser::KernelArgumentHolder args;
  for (py::handle obj : iter) {
    // Allows for a Vector of Sizes to be inputed as a list/tuple
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
      for (py::handle item : obj) {
        args.push(torch::jit::toIValue(item, c10::AnyType::get()));
      }
    } else {
      args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
    }
  }

  // Transform int64_t device to int8_t
  std::optional<int8_t> selected_device = std::nullopt;
  if (device.has_value()) {
    NVF_CHECK(device.value() < 256, "Maximum device index is 255");
    selected_device = (int8_t)device.value();
  }
  args.setDeviceIndex(selected_device);
  return args;
}

std::vector<at::Tensor> to_tensor_vector(
    const nvfuser::KernelArgumentHolder& outputs) {
  // Convert outputs KernelArgumentHolder to std::vector<at::Tensor>
  std::vector<at::Tensor> out_tensors;
  out_tensors.reserve(outputs.size());
  std::transform(
      outputs.begin(),
      outputs.end(),
      std::back_inserter(out_tensors),
      [](const nvfuser::PolymorphicValue& out) {
        return out.as<at::Tensor>();
      });
  return out_tensors;
}

const char* dtypeToPyString(nvfuser::PrimDataType t) {
  using namespace nvfuser;
  switch (t) {
    case DataType::Bool:
      return "DataType.Bool";
    case DataType::Double:
      return "DataType.Double";
    case DataType::Float:
      return "DataType.Float";
    case DataType::Half:
      return "DataType.Half";
    case DataType::BFloat16:
      return "DataType.BFloat16";
    case DataType::Float8_e4m3fn:
      return "DataType.Float8_e4m3fn";
    case DataType::Float8_e5m2:
      return "DataType.Float8_e5m2";
    case DataType::Int:
      return "DataType.Int";
    case DataType::Int32:
      return "DataType.Int32";
    case DataType::ComplexFloat:
      return "DataType.ComplexFloat";
    case DataType::ComplexDouble:
      return "DataType.ComplexDouble";
    case DataType::Null:
      return "DataType.Null";
    case DataType::UInt64:
      return "DataType.UInt64";
    default:
      break;
  }
  NVF_THROW("No string found for data type.");
  return nullptr;
}

} // namespace nvfuser::python
