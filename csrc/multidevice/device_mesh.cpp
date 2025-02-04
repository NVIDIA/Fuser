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

#include <type.h>

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
  if (mesh.shape().empty()) {
    out << "DeviceMesh{" << mesh.vector() << "}";
    return out;
  }

  out << "DeviceMesh";
  size_t nDevices = std::accumulate(
        mesh.shape().begin(), mesh.shape().end(), 1, std::multiplies<size_t>());
  size_t nDim = mesh.shape().size();
  std::vector<int64_t> strides = mesh.shape();
  for (int i = nDim - 2; i >= 0; --i) {
      strides[i] *= strides[i + 1];
  }

  for (size_t i = 0; i < nDevices; i++) {
    for (size_t axis = 0; axis < nDim; axis++) {
      if (i % strides[axis] == 0) {
        out << "{";
      }
    }
    out << mesh.vector().at(i);
    if ((i+1) % strides[nDim-1] != 0) {
      out << " ";
    }
    for (size_t axis = 0; axis < nDim; axis++) {
      if ((i+1) % strides[axis] == 0) {
        out << "}";
      }
    }
  }

  return out;
}

int64_t DeviceMesh::size(const ParallelType parallel_type) const {
  NVF_ERROR(
      parallel_type == ParallelType::DIDx,
      "We support only 1-D sharding for now.");
  return size();
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

DeviceIdxType DeviceMesh::maxDeviceId() const {
  return *std::max_element(vector_.begin(), vector_.end());
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

  std::vector<DeviceIdxType> team(shape_[axis]);
  for (auto i : c10::irange(team.size())) {
    team.at(i) = vector_.at(i * stride + offset);
  }
  return team;
}

} // namespace nvfuser
