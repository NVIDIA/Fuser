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
#include <c10/util/irange.h>

namespace nvfuser {

DeviceMesh::DeviceMesh(
    std::vector<DeviceIdxType> devices,
    std::vector<int64_t> shape) {
  if (shape.empty()) {
    shape_ = {(int64_t)devices.size()};
  } else {
    int64_t num_devices = std::accumulate(
        shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    NVF_ERROR(
        (int64_t)devices.size() == num_devices,
        "Specified a list of device with ",
        devices.size(),
        " elements ",
        " but shape contains ",
        num_devices);
    shape_ = std::move(shape);
  }
  setDevices(std::move(devices));
}

DeviceMesh::DeviceMesh(std::initializer_list<DeviceIdxType> devices) {
  setDevices(std::vector<DeviceIdxType>(devices));
}

std::vector<int64_t> DeviceMesh::getIndices(const DeviceIdxType device) const {
  auto global_idx = idxOf(device);
  if (global_idx == -1) {
    return {};
  }
  std::vector<int64_t> indices(shape_.size());
  int64_t accumulated_size = 1;
  for (int64_t i = (int64_t)shape_.size() - 1; i >= 0; i--) {
    indices[i] = (global_idx / accumulated_size) % shape_[i];
    accumulated_size *= shape_[i];
  }
  return indices;
}

std::pair<int64_t, int64_t> DeviceMesh::getTeamOffsetStride(
    int64_t device,
    int64_t axis) const {
  int64_t offset = 0;
  int64_t stride = 1;
  int64_t accumulated_size = 1;
  auto indices = getIndices(device);
  NVF_ERROR(!indices.empty(), "Device is not in DeviceMesh");
  for (int64_t i = (int64_t)shape_.size() - 1; i >= 0; i--) {
    if (i > axis) {
      stride *= shape_[i];
    }
    if (i != axis) {
      offset += indices[i] * accumulated_size;
    }
    accumulated_size *= shape_[i];
  }
  return std::make_pair(offset, stride);
}

std::vector<DeviceIdxType> DeviceMesh::getTeam(
    DeviceIdxType device,
    int64_t axis) const {
  NVF_ERROR(
      axis < (int64_t)shape_.size(),
      "DeviceMesh has ",
      shape_.size(),
      " dimensions, but requesting team for ",
      axis);

  auto [offset, stride] = getTeamOffsetStride(device, axis);
  std::vector<DeviceIdxType> team(shape_[axis]);
  for (auto i : c10::irange(team.size())) {
    team.at(i) = vector_.at(i * stride + offset);
  }
  return team;
}

DeviceIdxType DeviceMesh::maxDeviceIdx() const {
  return *std::max_element(vector_.begin(), vector_.end());
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

/*static*/ DeviceMesh DeviceMesh::createForShape(std::vector<int64_t> shape) {
  int64_t num_devices = std::accumulate(
      shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  std::vector<DeviceIdxType> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  return DeviceMesh(devices, shape);
}

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh) {
  out << "DeviceMesh{" << mesh.vector() << "}";
  return out;
}

} // namespace nvfuser
