// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <ATen/core/TensorBody.h>

#include <multidevice/device_mesh.h>
#include <type.h>

namespace nvfuser::python_frontend {

// A class that represents a distributed tensor. It wraps a local tensor, a
// mesh, and a mapping from mesh axes to tensor axes. If the mesh is empty,
// it degenerates into a non-distributed tensor.
class DistributedTensor {
 public:
  explicit DistributedTensor(
      at::Tensor local_tensor,
      DeviceMesh mesh = DeviceMesh())
      : local_(std::move(local_tensor)), mesh_(std::move(mesh)) {}
  DistributedTensor(const DistributedTensor&) = delete;
  DistributedTensor& operator=(const DistributedTensor&) = delete;
  DistributedTensor(DistributedTensor&&) = default;
  DistributedTensor& operator=(DistributedTensor&&) = default;

  const DeviceMesh& mesh() const {
    return mesh_;
  }

  at::Tensor local() const {
    return local_;
  }

  void setAxisIsShardedOn(int64_t axis, ParallelType parallel_type);

  int64_t axisShardedOn(ParallelType parallel_type) const;

 private:
  at::Tensor local_;
  DeviceMesh mesh_;
  std::unordered_map<ParallelType, int64_t> axis_sharded_on_;
};

} // namespace nvfuser::python_frontend
