// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/heuristic.h>
#include <scheduler/normalization_inner.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>

namespace nvfuser {

// Parameters for TMA inner persistent normalization scheduler
class InnerNormTmaParams : public HeuristicParams {
 public:
  explicit InnerNormTmaParams(
      SchedulerType scheduler_type = SchedulerType::InnerPersistent)
      : HeuristicParams(scheduler_type) {}

  // Vectorization factor for memory access
  int64_t vectorization_factor = 1;

  // Project persistent buffers back to inputs to reduce persistent buffer size
  bool project_persistent_buffers = false;

  bool vectorize_load_smem_to_regs = false;

  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const InnerNormTmaParams*>(other_base);
    if (other == nullptr) {
      return false;
    }
    return other->cparams == cparams &&
        other->vectorization_factor == vectorization_factor &&
        other->project_persistent_buffers == project_persistent_buffers;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Inner Norm TMA Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << "\n"
       << "Vectorization factor: " << vectorization_factor << "\n"
       << (project_persistent_buffers ? "Project Persistent Buffers\n" : "")
       << lparams.toString() << cparams.toString() << "\n"
       << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(vectorization_factor) << (bits - 1) ^
        static_cast<size_t>(project_persistent_buffers) << (bits - 2);
    return attr_hash;
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<InnerNormTmaParams>(*this);
  }
};

namespace normalization_inner {
namespace tma {
std::unique_ptr<InnerNormTmaParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache);

void scheduleInnerPersistent(
    Fusion* fusion,
    const InnerNormTmaParams* tma_params);
} // namespace tma
} // namespace normalization_inner
} // namespace nvfuser
