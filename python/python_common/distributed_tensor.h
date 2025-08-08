// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <ATen/core/TensorBody.h>

#include <fusion.h>
#include <multidevice/device_mesh.h>
#include <multidevice/utils.h>
#include <type.h>

namespace nvfuser {

class Sharding {
 public:
  explicit NVF_API Sharding(DeviceMesh mesh = DeviceMesh())
      : mesh_(std::move(mesh)) {}
  Sharding(const Sharding&) = delete;
  Sharding& operator=(const Sharding&) = delete;
  Sharding(Sharding&&) = default;
  Sharding& operator=(Sharding&&) = default;

  const DeviceMesh& mesh() const {
    return mesh_;
  }

  NVF_API void setAxisIsShardedOn(int64_t axis, ParallelType parallel_type);

  NVF_API int64_t axisShardedOn(ParallelType parallel_type) const;

 private:
  DeviceMesh mesh_;
  std::unordered_map<ParallelType, int64_t> axis_sharded_on_;
};

// Returns the output shardings of the given fusion. As a short cut, if none of
// the outputs have a device mesh, returns an empty vector indicating single-GPU
// execution.
NVF_API std::vector<Sharding> getOutputShardings(Fusion* fusion);

} // namespace nvfuser
