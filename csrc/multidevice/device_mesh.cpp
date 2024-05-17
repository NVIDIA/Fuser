// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <multidevice/device_mesh.h>

#include <numeric>
#include <unordered_set>

// for operator<<(std::ostream&, const std::vector<T>&)
#include <c10/util/Logging.h>

namespace nvfuser {

DeviceMesh::DeviceMesh(std::vector<DeviceIdxType> devices) {
  setDevices(std::move(devices));
}

DeviceMesh::DeviceMesh(std::initializer_list<DeviceIdxType> devices) {
  setDevices(std::vector<DeviceIdxType>(devices));
}

void DeviceMesh::setDevices(std::vector<DeviceIdxType> devices) {
  vector_ = std::move(devices);

  std::unordered_set<DeviceIdxType> unique_devices(
      vector_.begin(), vector_.end());
  NVF_ERROR(
      unique_devices.size() == vector_.size(),
      "Device mesh has duplicates: ",
      vector_);
}

/*static*/ DeviceMesh DeviceMesh::createForNumDevices(
    const int64_t num_devices) {
  std::vector<DeviceIdxType> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  return DeviceMesh(devices);
}

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh) {
  out << "DeviceMesh{" << mesh.vector() << "}";
  return out;
}

} // namespace nvfuser
