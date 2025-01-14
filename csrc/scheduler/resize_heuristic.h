// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/hash.h>
#include <ir/interface_nodes.h>
#include <scheduler/heuristic.h>
#include <utils.h>

#include <sstream>

namespace nvfuser {

class ResizeParams : public HeuristicParams {
 public:
  ResizeParams() : HeuristicParams(SchedulerType::Resize) {};

  // Split grid x dimension
  bool split_grid_x_dim = false;

  static constexpr int64_t max_gdimx = (1L << 31) - 1L;

  using HeuristicParams::HeuristicParams;

  // Warning: Does not check launch parameters!
  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const ResizeParams*>(other_base);
    if (other == nullptr) {
      return false;
    }
    bool attr_equal = other->cparams == cparams &&
        other->split_grid_x_dim == split_grid_x_dim;
    return attr_equal;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Resize Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << " Resize Characteristics:\n"
       << " split grid x dim: " << split_grid_x_dim << "\n";
    ss << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    return c10::get_hash(split_grid_x_dim);
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<ResizeParams>(*this);
  }
};

} // namespace nvfuser
