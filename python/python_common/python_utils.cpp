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

namespace nvfuser {

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

std::vector<std::optional<bool>> getContiguityVec(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride_order,
    bool contiguity) {
  const auto rank = static_cast<int64_t>(shape.size());
  std::vector<std::optional<bool>> contiguity_vec(rank);
  for (const auto index : arange(rank)) {
    const auto contig_index =
        stride_order.empty() ? index : rank - 1 - stride_order[index];
    if (shape[index] == 1) {
      contiguity_vec[contig_index] = std::nullopt;
    } else {
      contiguity_vec[contig_index] = contiguity;
    }
  }
  return contiguity_vec;
}

std::vector<int64_t> getTensorViewBuilderSizes(
    const std::vector<int64_t>& sizes,
    bool static_sizes) {
  // TensorViewBuilder assumes any dim with a compile time constant
  // size == 1 is a broadcast axis, symbolic sizes are
  // identified by -1, and size == 0 is not supported.
  std::vector<int64_t> dim_sizes;
  dim_sizes.reserve(sizes.size());
  for (size_t i : arange(sizes.size())) {
    NVF_ERROR(
        sizes[i] >= 0,
        "Size of ",
        sizes[i],
        " is not supported in nvFuser. Expected size >= 0.");
    if (static_sizes) {
      dim_sizes.push_back(sizes[i]);
    } else { // Symbolic defined tensor for dynamic shape usage
      if (sizes[i] == 1) {
        dim_sizes.push_back(1);
      } else {
        dim_sizes.push_back(-1);
      }
    }
  }
  return dim_sizes;
}

} // namespace nvfuser
