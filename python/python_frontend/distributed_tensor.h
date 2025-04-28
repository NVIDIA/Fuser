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

class Sharding {
 public:
  explicit Sharding(DeviceMesh mesh = DeviceMesh()) : mesh_(std::move(mesh)) {}
  Sharding(const Sharding&) = delete;
  Sharding& operator=(const Sharding&) = delete;
  Sharding(Sharding&&) = default;
  Sharding& operator=(Sharding&&) = default;

  const DeviceMesh& mesh() const {
    return mesh_;
  }

  void setAxisIsShardedOn(int64_t axis, ParallelType parallel_type);

  int64_t axisShardedOn(ParallelType parallel_type) const;

 private:
  DeviceMesh mesh_;
  std::unordered_map<ParallelType, int64_t> axis_sharded_on_;
};

} // namespace nvfuser::python_frontend
