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

// The class DeviceMesh represents a set of (unique) devices on which a Pipeline
// Stage will be executed. For now, we only support flat meshes, but later we
// will add support for n-dimensional meshes.
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
  explicit DeviceMesh(std::vector<DeviceIdxType> devices = {});
  DeviceMesh(std::initializer_list<DeviceIdxType> devices);
  DeviceMesh(const DeviceMesh&) = default;
  DeviceMesh(DeviceMesh&&) = default;
  DeviceMesh& operator=(const DeviceMesh&) = default;
  DeviceMesh& operator=(DeviceMesh&&) = default;

  // Creates a device mesh of [0 .. num_devices-1]. I didn't make it a
  // constructor because single-element initializer lists would be directed to
  // use that instead of the constructor for vectors.
  static DeviceMesh createForNumDevices(int64_t num_devices);

  // Returns the number of devices in the mesh
  int64_t size() const {
    return static_cast<int64_t>(vector_.size());
  }

  // Returns a vector containing the device indices of the mesh
  const std::vector<DeviceIdxType>& vector() const {
    return vector_;
  }

  // Returns whether a device is present in the mesh
  bool has(const DeviceIdxType device) const {
    return std::find(vector_.begin(), vector_.end(), device) != vector_.end();
  }

  // Returns the index of device in the mesh, or -1 if device is not present.
  int64_t idxOf(const DeviceIdxType device) const {
    auto it = std::find(vector_.begin(), vector_.end(), device);
    if (it != vector_.end()) {
      return std::distance(vector_.begin(), it);
    }
    return -1;
  }

  // Returns the device at a particular index in the mesh
  DeviceIdxType at(int64_t index) const {
    return vector_.at(index);
  }

  bool operator==(const DeviceMesh& other) const {
    return vector_ == other.vector();
  }

 private:
  void setDevices(std::vector<DeviceIdxType> devices);

  // stores the list of device indices
  std::vector<DeviceIdxType> vector_;
};

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh);

} // namespace nvfuser
