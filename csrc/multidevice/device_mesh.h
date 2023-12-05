// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once
#include <exceptions.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

/*
   The class DeviceMesh represents a set of (unique) devices on which a Pipeline
   Stage will be executed. For now, we only support flat meshes, but later we
   will add support for n-dimensional meshes.
*/
class DeviceMesh final {
 public:
  DeviceMesh(std::vector<DeviceIdxType> devices = {0}) {
    setDevices(devices);
  }

  std::string toString() const;

  DeviceMesh& operator=(const std::vector<DeviceIdxType>& devices) {
    setDevices(devices);
    return *this;
  }

  // returns a vector containing the device indices of the mesh
  const auto& vector() const {
    return vector_;
  }

  // returns whether a device is present in the mesh
  bool has(const DeviceIdxType device) const {
    return std::find(vector_.begin(), vector_.end(), device) != vector_.end();
  }

  bool operator==(const DeviceMesh& other) const {
    return vector() == other.vector();
  }

 private:
  void setDevices(std::vector<DeviceIdxType> devices) {
    vector_ = devices;
    NVF_ERROR(!devices.empty(), "empty device mesh");
    NVF_ERROR(
        std::unique(vector_.begin(), vector_.end()) == vector_.end(),
        "device mesh has duplicates");
  }

  // stores the list of device indices
  std::vector<DeviceIdxType> vector_;
};

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh);

} // namespace nvfuser
