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
#include <scheduler/normalization_utils.h>

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

  // Register persistent buffer size in inner dimension
  int64_t persistent_batch_size = 1;

  // Pre-load non-persistent buffers (ldg_tvs) before computing
  bool pre_load_ldg_tvs = true;

  // Use TMA to load non-persistent buffers
  bool tma_load_non_persistent_buffers = false;

  // Number of rows per block
  int64_t rows_per_block = 1;

  // Circular buffer options
  CircularBufferOptions circular_buffer_options;

  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const InnerNormTmaParams*>(other_base);
    if (other == nullptr) {
      return false;
    }
    return other->cparams == cparams &&
        other->vectorization_factor == vectorization_factor &&
        other->project_persistent_buffers == project_persistent_buffers &&
        other->vectorize_load_smem_to_regs == vectorize_load_smem_to_regs &&
        other->persistent_batch_size == persistent_batch_size &&
        other->pre_load_ldg_tvs == pre_load_ldg_tvs &&
        other->tma_load_non_persistent_buffers ==
        tma_load_non_persistent_buffers &&
        other->rows_per_block == rows_per_block &&
        other->circular_buffer_options == circular_buffer_options;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Inner Norm TMA Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << "\n"
       << "Vectorization factor: " << vectorization_factor << "\n"
       << (project_persistent_buffers ? "Project Persistent Buffers\n" : "")
       << (vectorize_load_smem_to_regs ? "Vectorize load smem to regs\n" : "")
       << "Persistent batch size: " << persistent_batch_size << "\n"
       << (pre_load_ldg_tvs ? "Pre-load ldg tvs\n" : "")
       << (tma_load_non_persistent_buffers ? "TMA load non-persistent buffers\n"
                                           : "")
       << "Rows per block: " << rows_per_block << "\n";
    if (circular_buffer_options.isEnable()) {
      ss << circular_buffer_options << "\n";
    } else {
      ss << "Circular buffer: not used\n";
    }
    ss << lparams.toString() << cparams.toString() << "\n"
       << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(vectorization_factor) << (bits - 1) ^
        static_cast<size_t>(project_persistent_buffers) << (bits - 2) ^
        static_cast<size_t>(vectorize_load_smem_to_regs) << (bits - 3) ^
        static_cast<size_t>(persistent_batch_size) << (bits - 4) ^
        static_cast<size_t>(pre_load_ldg_tvs) << (bits - 5) ^
        static_cast<size_t>(tma_load_non_persistent_buffers) << (bits - 6) ^
        static_cast<size_t>(rows_per_block) << (bits - 7) ^
        static_cast<size_t>(circular_buffer_options.stage) << (bits - 8) ^
        static_cast<size_t>(circular_buffer_options.prefetch) << (bits - 9);
    return attr_hash;
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<InnerNormTmaParams>(*this);
  }
};

namespace normalization_inner {
namespace tma {

using PersistentKernelProperties =
    normalization_scheduler_utils::PersistentKernelProperties;

std::unique_ptr<InnerNormTmaParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    const PersistentKernelProperties& prop,
    HeuristicDataCache* data_cache);

void scheduleInnerPersistent(
    Fusion* fusion,
    const InnerNormTmaParams* tma_params);
} // namespace tma
} // namespace normalization_inner
} // namespace nvfuser
