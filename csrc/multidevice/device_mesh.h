// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once
#include <multidevice/multidevice.h>

namespace nvfuser {

/*
   The class DevishMesh represents a set of devices on which a Pipeline Stage
   will be executed. This can be simply seen as a list (or n-array) of device
   indices (i.e. ints). Here, "dimensions" stands for the shape of the Mesh seen
   as an array. By default, we assume a flat Mesh, i.e., dimensions =
   (number_of_device_indices)
*/
class DeviceMesh final {
 public:
  // return the flatten vector of device indices
  const auto& deviceIndices() const {
    return device_indices_;
  }

  // return the shape of the mesh
  const auto& dimensions() const {
    return dimensions_;
  }

  // return the total length of the mesh
  int64_t size() const {
    int64_t ret = 1;
    for (auto dimension : dimensions_) {
      ret *= dimension;
    }
    return ret;
  }

  // set the attributes of the mesh
  void set(
      std::vector<DeviceIdxType> device_indices,
      std::vector<DimensionType> dimensions) {
    device_indices_ = std::move(device_indices);
    dimensions_ = std::move(dimensions);

    TORCH_INTERNAL_ASSERT(
        validate(device_indices_), "invalid parameters for Mesh Device");
  }

  void set(std::vector<DeviceIdxType> device_indices) {
    DimensionType length = static_cast<DimensionType>(device_indices.size());
    set(std::move(device_indices), {length});
  }

 private:
  /*
    returns whether the mesh is valid, i.e., its size is strictly positive and
    the size matches the number of device indices.
  */
  bool validate(const std::vector<DeviceIdxType>& device_indices) {
    return (
        size() == static_cast<int64_t>(device_indices.size()) && size() > 0);
  }

  // stores the device indices
  std::vector<DeviceIdxType> device_indices_;
  // stores the mesh shape
  std::vector<DimensionType> dimensions_;
};

} // namespace nvfuser
