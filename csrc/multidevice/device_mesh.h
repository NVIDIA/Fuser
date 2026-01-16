// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <vector>

#include <ATen/ATen.h>

#include "exceptions.h"
#include "multidevice/multidevice.h"
#include "type.h"
#include "visibility.h"

namespace nvfuser {

// DeviceMesh represents a set of unique devices arranged as a dense
// n-dimensional tensor. DeviceMesh and device parallel types determine
// how a tensorview is sharded among devices.
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
  NVF_API DeviceMesh();
  explicit NVF_API DeviceMesh(at::Tensor devices);
  NVF_API DeviceMesh(std::initializer_list<DeviceIdxType> devices);
  NVF_API DeviceMesh(
      const std::vector<int64_t>& devices,
      const std::vector<int64_t>& shape);

  DeviceMesh(const DeviceMesh&) = default;
  DeviceMesh(DeviceMesh&&) = default;
  DeviceMesh& operator=(const DeviceMesh&) = default;
  DeviceMesh& operator=(DeviceMesh&&) = default;

  // Creates a device mesh of [0 ... num_devices-1]. I didn't make it a
  // constructor because single-element initializer lists would be directed to
  // use that instead of the constructor for vectors.
  static DeviceMesh createForNumDevices(int64_t num_devices);
  // Creates a device mesh with the specified shape with devices numbered
  // [0 ... num_devices-1].
  static DeviceMesh createForShape(const std::vector<int64_t>& shape);

  // Returns the number of devices in the mesh
  int64_t size() const {
    return devices_.numel();
  }

  // Return the size of an axis in the mesh
  int64_t size(int64_t axis) const {
    return devices_.size(axis);
  }

  // Returns the shape of the device mesh
  const at::IntArrayRef shape() const {
    return devices_.sizes();
  }

  int64_t size(ParallelType parallel_type) const;

  // Returns a vector containing the device indices of the mesh
  std::vector<DeviceIdxType> vector() const {
    auto* data = devices_.data_ptr<DeviceIdxType>();
    return std::vector<DeviceIdxType>(data, data + devices_.numel());
  }

  // Returns whether a device is present in the mesh
  bool has(const DeviceIdxType device) const {
    return (devices_ == device).any().item<bool>();
  }

  // Returns the linear index of the given device in the mesh, or -1 if device
  // is not present.
  int64_t linearIndexOf(const DeviceIdxType device) const;

  // Returns the multi-dimensional index of the given device, or an undefined
  // tensor if device is not present.
  at::Tensor multiDimensionalIndexOf(const DeviceIdxType device) const;

  // Returns the device at a particular index in the mesh
  DeviceIdxType at(int64_t index) const {
    return devices_.flatten()[index].item<DeviceIdxType>();
  }

  // Returns the rank (number of dimensions) of the mesh.
  int64_t rank() const {
    return devices_.dim();
  }

  bool operator==(const DeviceMesh& other) const {
    return at::equal(devices_, other.devices_);
  }

  bool operator!=(const DeviceMesh& other) const {
    return !(*this == other);
  }

  // Returns the max device id in the DeviceMesh.
  DeviceIdxType maxDeviceId() const;

  // Maps a parallel type to axis. Returns -1 if the parallel type is
  // not in the device mesh.
  int64_t parallelTypeToAxis(ParallelType parallel_type) const;

  // Returns true if the DeviceMesh has the specified parallel type
  bool hasParallelType(ParallelType parallel_type) const;

  // Returns a slice of the DeviceMesh according to the device parallel type
  // that contains the device
  // Ex: [[0 1 2]
  //      [3 4 5]]
  // getSlice(4, ParallelType::DIDx) = {3, 4, 5}
  // getSlice(4, ParallelType::DIDy) = {1, 4}
  //
  // These might be worth caching per TV.
  std::vector<DeviceIdxType> getSlice(DeviceIdxType device, ParallelType ptype)
      const;

 private:
  at::Tensor devices_;
};

NVF_API std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh);

} // namespace nvfuser
