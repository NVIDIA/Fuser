// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <vector>

#include <exceptions.h>
#include <multidevice/multidevice.h>
#include <visibility.h>

namespace nvfuser {

// DeviceMesh represents a set of (unique) devices arranged in a dense tensor
// with a concrete shape.
class DeviceMesh final {
 public:
  // https://google.github.io/styleguide/cppguide.html#Implicit_Conversions
  //
  // Not using `explicit` for the constructor that takes a vector would lead
  // to contention between operator<<(std::vector) defined in c10/util/Logging.h
  // and operator<<(DeviceMesh) defined later in this file, which would be
  // resolved arbitrarily by the compiler.
  //
  // There are no such contention for std::initializer_list so I chose to
  // allow implicit conversion for that. This allows users to write `DeviceMesh
  // mesh = {1, 2};`, which is more concise.
  // When no shape is specified, a 1D DeviceMesh is created by default.
  explicit DeviceMesh(
      std::vector<DeviceIdxType> devices = {},
      std::vector<int64_t> shape = {});
  DeviceMesh(std::initializer_list<DeviceIdxType> devices);
  DeviceMesh(const DeviceMesh&) = default;
  DeviceMesh(DeviceMesh&&) = default;
  DeviceMesh& operator=(const DeviceMesh&) = default;
  DeviceMesh& operator=(DeviceMesh&&) = default;

  // Creates a device mesh of [0 ... num_devices-1]. I didn't make it a
  // constructor because single-element initializer lists would be directed to
  // use that instead of the constructor for vectors.
  static DeviceMesh createForNumDevices(int64_t num_devices);

  // Creates a device mesh with the specified shape made of devices from
  // [0 ... size], where size is the total number of devices specified by the
  // shape.
  static DeviceMesh createForShape(std::vector<int64_t>& shape);
  static DeviceMesh createForShape(std::initializer_list<int64_t> shape);

  // Returns the total number of devices in the mesh
  int64_t size() const {
    return static_cast<int64_t>(vector_.size());
  }

  // Returns size of an axis in the mesh
  int64_t size(int64_t axis) {
    return shape_.at(axis);
  }

  // Returns the shape of the device mesh
  const std::vector<int64_t>& shape() const {
    return shape_;
  }

  // Returns a vector containing the device indices of the mesh
  const std::vector<DeviceIdxType>& vector() const {
    return vector_;
  }

  // Returns whether a device is present in the mesh
  bool has(const DeviceIdxType device) const {
    return std::find(vector_.begin(), vector_.end(), device) != vector_.end();
  }

  // Returns the global index of device in the mesh, or -1 if device is not
  // present.
  int64_t idxOf(const DeviceIdxType device) const {
    auto it = std::find(vector_.begin(), vector_.end(), device);
    if (it != vector_.end()) {
      return std::distance(vector_.begin(), it);
    }
    return -1;
  }

  // Returns the indices of a multi-dimensional mesh
  std::vector<int64_t> getLocalIndices(const DeviceIdxType device) const;

  // Returns the device at a particular global index in the mesh
  DeviceIdxType at(int64_t index) const {
    return vector_.at(index);
  }

  bool operator==(const DeviceMesh& other) const {
    return vector_ == other.vector() && shape_ == other.shape();
  }

  // Returns the max device index in the DeviceMesh.
  DeviceIdxType maxDeviceIdx() const;

  // Returns devices of this device's team given which axis of the DeviceMesh
  // Ex: [[0 1 2]
  //      [3 4 5]]
  // getTeam(4, 0) = {3, 4, 5}
  // getTeam(4, 1) = {1, 4}
  // TODO: calculate this once and save it.
  std::vector<DeviceIdxType> getTeam(DeviceIdxType device, int64_t axis = 0)
      const;

 private:
  void setDevices(std::vector<DeviceIdxType> devices);
  // Calculates offset and stride used to iterate over
  std::pair<int64_t, int64_t> getTeamOffsetStride(
      int64_t global_idx,
      int64_t axis) const;

  // stores the flattened list of device indices
  std::vector<DeviceIdxType> vector_;
  // stores the shape of the mesh
  std::vector<int64_t> shape_;
};

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh);

} // namespace nvfuser
