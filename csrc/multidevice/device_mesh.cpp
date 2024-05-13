// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <multidevice/device_mesh.h>

#include <numeric>

namespace nvfuser {

/*static*/ DeviceMesh DeviceMesh::createForNumDevices(
    const int64_t num_devices) {
  std::vector<DeviceIdxType> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  return DeviceMesh(devices);
}

std::string DeviceMesh::toString() const {
  std::stringstream ss;
  ss << "DeviceMesh{";
  for (auto i : vector_) {
    ss << i << ", ";
  }
  ss << "}";
  return ss.str();
}

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh) {
  return out << mesh.toString();
}

} // namespace nvfuser
