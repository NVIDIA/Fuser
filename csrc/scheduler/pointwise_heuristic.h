// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/heuristic.h>

#include <sstream>

namespace nvfuser {

// Parameters of the pointwise heuristic to describe the optimial schedule.
// Warning: equal operator is intended for use in caching the kernel associated
// with these pointwise parameters. It does not check if the launch parameters
// are equivelent!
class PointwiseParams : public HeuristicParams {
 public:
  PointwiseParams() : HeuristicParams(SchedulerType::PointWise) {};

  // Treat pointwise operation as 2-Dimensional, this is the location where we
  // split from left side of the domain to right. i.e. 0 means problem is
  // treated as 1-D, 1 of 3 would mean we treat the first dimension as the outer
  // dimension, and all the others as an inner dimension.
  int64_t break_point = 0;

  // Split block across left and right dimension
  bool split_block = false;

  // Split grid y dimension, if otherwise it would be too large
  bool split_grid_y_dim = false;

  // For many instances having BIDx on the inner most dimension is the most
  // performant parallel binding. However, if we're broadcasting the outer
  // dimension with a large inner dimension, it can be more performant to bind
  // BIDy on the inner most dimension.
  bool flip_grid_binding = false;

  // vectorization factor
  int64_t vectorization_factor = 1;

  // Unroll on top of vectorization
  // In the 2D scheduler, unroll the outer dimension to reuse loaded data across
  // rows, reducing loaded bytes by the unroll factor.
  // Always equals 1 for 1D scheduler.
  int64_t unroll_factor_outer = 1;

  // In the 2D scheduler, unroll the inner dimension to reuse loaded data across
  // cols, reducing loaded bytes by the unroll factor.
  // Also used in 1D scheduler.
  int64_t unroll_factor_inner = 1;

  using HeuristicParams::HeuristicParams;

  // Warning: Does not check launch parameters!
  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const PointwiseParams*>(other_base);
    if (other == nullptr) {
      return false;
    }
    bool attr_equal = other->cparams == cparams &&
        other->vectorization_factor == vectorization_factor &&
        other->break_point == break_point &&
        other->split_block == split_block &&
        other->split_grid_y_dim == split_grid_y_dim &&
        other->unroll_factor_outer == unroll_factor_outer &&
        other->unroll_factor_inner == unroll_factor_inner &&
        other->flip_grid_binding == flip_grid_binding;
    return attr_equal;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Pointwise Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << " Pointwise Characteristics:\n"
       << " Gridx: " << lparams.gdimx() << " BlckY: " << lparams.bdimy()
       << " BlckX: " << lparams.bdimx() << "\n";
    if (break_point) {
      ss << "2D Schedule\n"
         << "  Bcast break point: " << break_point << "\n";
      if (split_block) {
        ss << "Split block into y-dim\n";
      }
      if (split_grid_y_dim) {
        ss << "  Split y grid dim\n";
      }
    }
    ss << "vectorization_factor: " << vectorization_factor << "\n";
    ss << "unroll_factor_outer: " << unroll_factor_outer << "\n";
    ss << "unroll_factor_inner: " << unroll_factor_inner << "\n";
    if (flip_grid_binding) {
      ss << "Flip BIDx/BIDy bindings\n";
    }
    ss << "====================================\n";
    return ss.str();
  }

  // Warning: Hash is not based on launch parameters!
  size_t hash() const override {
    size_t attr_hash = static_cast<size_t>(vectorization_factor) ^
        static_cast<size_t>(break_point) << 4 ^
        static_cast<size_t>(split_block) << 5 ^
        static_cast<size_t>(split_grid_y_dim) << 6 ^
        static_cast<size_t>(unroll_factor_outer) << 7 ^
        static_cast<size_t>(unroll_factor_inner) << 9 ^
        static_cast<size_t>(flip_grid_binding) << 10;
    return attr_hash;
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<PointwiseParams>(*this);
  }
};

} // namespace nvfuser
