// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/ops/unique_dim.h>
#include <multidevice/device_mesh.h>

#include <utils.h>
#include <numeric>

// for operator<<(std::ostream&, const std::vector<T>&)
#include <c10/util/Logging.h>

#include <type.h>

namespace nvfuser {

DeviceMesh::DeviceMesh() : DeviceMesh(at::empty({0}, at::dtype(at::kLong))) {}

DeviceMesh::DeviceMesh(at::Tensor devices) : devices_(devices.to(at::kLong)) {
  NVF_ERROR_EQ(
      devices_.numel(),
      std::get<0>(at::unique_dim(devices_.flatten(), 0)).numel(),
      "`devices_` contains duplicates: ",
      devices_);
}

DeviceMesh::DeviceMesh(std::initializer_list<DeviceIdxType> devices)
    : DeviceMesh(at::tensor(devices)) {}

DeviceMesh::DeviceMesh(
    const std::vector<int64_t>& devices,
    const std::vector<int64_t>& shape)
    : DeviceMesh(at::tensor(devices).view(shape)) {}

/*static*/ DeviceMesh DeviceMesh::createForNumDevices(
    const int64_t num_devices) {
  return DeviceMesh(at::arange(num_devices));
}

/*static*/ DeviceMesh DeviceMesh::createForShape(
    const std::vector<int64_t>& shape) {
  int64_t numel = at::empty(shape, at::device(at::kMeta)).numel();
  at::Tensor t = at::arange(numel).view(shape);
  return DeviceMesh(t);
}

std::ostream& operator<<(std::ostream& out, const DeviceMesh& mesh) {
  out << "DeviceMesh";
  int64_t ndims = mesh.rank();
  std::vector<int64_t> strides = mesh.shape().vec();
  for (auto i = ndims - 2; i >= 0; --i) {
    strides[i] *= strides[i + 1];
  }

  for (auto i : arange(mesh.size())) {
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

int64_t DeviceMesh::linearIndexOf(const DeviceIdxType device) const {
  at::Tensor indices = at::nonzero(devices_.flatten() == device);
  if (indices.numel() == 0) {
    return -1;
  }

  NVF_ERROR_EQ(
      indices.numel(),
      1,
      "Expect device ",
      device,
      " to be present in the mesh at most once: ",
      *this);
  return indices.item<int64_t>();
}

namespace {
template <typename T>
std::vector<T> flattenToVector(at::Tensor t) {
  t = t.flatten().contiguous();
  const auto* data = t.data_ptr<T>();
  return std::vector<T>(data, data + t.numel());
}
} // namespace

at::Tensor DeviceMesh::multiDimensionalIndexOf(
    const DeviceIdxType device) const {
  at::Tensor indices = at::nonzero(devices_ == device);
  if (indices.numel() == 0) {
    return at::Tensor();
  }

  NVF_ERROR_EQ(
      indices.size(0),
      1,
      "Expect device ",
      device,
      " to be present in the mesh at most once: ",
      *this);

  at::Tensor index = indices[0];
  NVF_ERROR_EQ(index.dim(), 1);
  NVF_ERROR_EQ(index.numel(), rank());
  return index;
}

DeviceIdxType DeviceMesh::maxDeviceId() const {
  return devices_.max().item<DeviceIdxType>();
}

int64_t DeviceMesh::parallelTypeToAxis(ParallelType parallel_type) const {
  NVF_ERROR(
      isParallelTypeDeviceDim(parallel_type),
      "Attempting to index into DeviceMesh with a non-device parallel type: ",
      parallel_type);
  int64_t offset = static_cast<int64_t>(parallel_type) -
      static_cast<int64_t>(ParallelType::DIDx);
  const int64_t ndims = rank();
  if (offset >= ndims) {
    return -1;
  }
  return ndims - 1 - offset;
}

bool DeviceMesh::hasParallelType(ParallelType parallel_type) const {
  return parallelTypeToAxis(parallel_type) != -1;
}

int64_t DeviceMesh::size(const ParallelType parallel_type) const {
  int64_t axis = parallelTypeToAxis(parallel_type);
  NVF_ERROR(
      axis != -1,
      "DeviceMesh of rank ",
      rank(),
      " does not have parallel type ",
      parallel_type);
  return size(axis);
}

std::vector<DeviceIdxType> DeviceMesh::getSlice(
    DeviceIdxType deviceId,
    ParallelType ptype) const {
  int64_t axis = parallelTypeToAxis(ptype);
  NVF_ERROR(
      axis != -1,
      "DeviceMesh of rank ",
      rank(),
      " does not have parallel type ",
      ptype);

  at::Tensor index = multiDimensionalIndexOf(deviceId);
  NVF_ERROR(index.defined(), "Device ", deviceId, " is not in ", *this);

  std::vector<at::indexing::TensorIndex> indices;
  indices.reserve(rank());
  for (int64_t i : arange(rank())) {
    if (i == axis) {
      indices.push_back(at::indexing::Slice());
    } else {
      indices.push_back(index[i]);
    }
  }
  at::Tensor slice = devices_.index(indices);
  return flattenToVector<DeviceIdxType>(slice);
}

} // namespace nvfuser
