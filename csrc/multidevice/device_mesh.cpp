// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <multidevice/device_mesh.h>

#include <utils.h>
#include <numeric>
#include <unordered_set>

// for operator<<(std::ostream&, const std::vector<T>&)
#include <c10/util/Logging.h>

#include <type.h>

namespace nvfuser {

DeviceMesh::DeviceMesh(
    std::vector<DeviceIdxType> devices,
    std::vector<int64_t> shape) {
  setDevices(std::move(devices));
  if (shape.empty()) {
    shape = {(int64_t)vector_.size()};
  } else {
    int64_t num_devices =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    NVF_ERROR(
        (int64_t)vector_.size() == num_devices,
        "Specified a list of device with ",
        vector_.size(),
        " elements ",
        " but shape contains ",
        num_devices);
  }
  shape_ = std::move(shape);
}

DeviceMesh::DeviceMesh(std::initializer_list<DeviceIdxType> devices) {
  setDevices(std::vector<DeviceIdxType>(devices));
  shape_ = {(int64_t)vector_.size()};
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

/*static*/ DeviceMesh DeviceMesh::createForShape(
    const std::vector<int64_t>& shape) {
  int64_t num_devices =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  std::vector<DeviceIdxType> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  return DeviceMesh(devices, shape);
}

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh) {
  out << "DeviceMesh";
  int64_t ndevices = std::accumulate(
      mesh.shape().begin(), mesh.shape().end(), 1, std::multiplies<>());
  int64_t ndims = mesh.rank();
  std::vector<int64_t> strides = mesh.shape();
  for (auto i = ndims - 2; i >= 0; --i) {
    strides[i] *= strides[i + 1];
  }

  for (auto i = 0; i < ndevices; i++) {
    for (auto axis = 0; axis < ndims; axis++) {
      if (i % strides[axis] == 0) {
        out << "{";
      }
    }
    out << mesh.vector().at(i);
    if ((i + 1) % strides[ndims - 1] != 0) {
      out << " ";
    }
    for (auto axis = 0; axis < ndims; axis++) {
      if ((i + 1) % strides[axis] == 0) {
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

namespace {
int64_t ptypeToAxis(ParallelType ptype, int64_t ndims) {
  NVF_ERROR(
      isParallelTypeDeviceDim(ptype),
      "Attempting to index into DeviceMesh with a non-device parallel type",
      ptype);
  int64_t offset =
      static_cast<int64_t>(ptype) - static_cast<int64_t>(ParallelType::DIDx);

  NVF_ERROR(
      offset < ndims,
      "DeviceMesh has ",
      ndims,
      " dimensions, but requesting slice for ",
      ptype);
  return ndims - 1 - offset;
}
} // namespace

std::vector<DeviceIdxType> DeviceMesh::getSlice(
    DeviceIdxType deviceId,
    ParallelType ptype) const {
  int64_t axis = ptypeToAxis(ptype, rank());
  auto indices = getIndices(deviceId);
  NVF_ERROR(
      !indices.empty(), "Device ", deviceId, " is not in DeviceMesh ", vector_);

  int64_t offset = 0;
  int64_t stride = 1;
  int64_t accumulated_size = 1;
  for (auto i = rank() - 1; i >= 0; i--) {
    if (i > axis) {
      stride *= shape_[i];
    }
    if (i != axis) {
      offset += indices[i] * accumulated_size;
    }
    accumulated_size *= shape_[i];
  }

  std::vector<DeviceIdxType> devices(shape_[axis]);
  for (auto i : arange(devices.size())) {
    devices.at(i) = vector_.at(i * stride + offset);
  }
  return devices;
}

} // namespace nvfuser
