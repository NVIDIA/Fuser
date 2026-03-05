// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <memory>
#include <sstream>

#include "fusion.h"
#include "scheduler/reduction.h"
#include "scheduler/reduction_utils.h"

namespace nvfuser {

class TmaOuterReductionParams : public HeuristicParams {
 public:
  TmaOuterReductionParams(
      SchedulerType scheduler_type = SchedulerType::Reduction)
      : HeuristicParams(scheduler_type) {};

  // Thread block dimensions for 2D thread block
  // bdimx covers the iteration dimension, bdimy covers the reduction dimension
  int64_t bdimx = 1;
  int64_t bdimy = 1;

  // TMA tile dimensions
  int64_t tma_tile_i = 1;
  int64_t tma_tile_r = 1;

  // Unroll factor for the iteration dimension (within TMA tile)
  int64_t iter_unroll_factor = 1;

  // Grid dimension for parallelizing the outer reduction across CTAs
  int64_t grdim = 1;

  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const TmaOuterReductionParams*>(other_base);
    if (other == nullptr) {
      return false;
    }
    return other->cparams == cparams && other->bdimx == bdimx &&
        other->bdimy == bdimy && other->tma_tile_i == tma_tile_i &&
        other->tma_tile_r == tma_tile_r &&
        other->iter_unroll_factor == iter_unroll_factor &&
        other->grdim == grdim;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Outer Reduction TMA Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << "\n"
       << "bdimx: " << bdimx << "\n"
       << "bdimy: " << bdimy << "\n"
       << "tma_tile_i: " << tma_tile_i << "\n"
       << "tma_tile_r: " << tma_tile_r << "\n"
       << "iter_unroll_factor: " << iter_unroll_factor << "\n"
       << "grdim: " << grdim << "\n"
       << lparams.toString() << cparams.toString() << "\n"
       << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(bdimx) << (bits - 1) ^
        static_cast<size_t>(bdimy) << (bits - 2) ^
        static_cast<size_t>(tma_tile_i) << (bits - 3) ^
        static_cast<size_t>(tma_tile_r) << (bits - 4) ^
        static_cast<size_t>(iter_unroll_factor) << (bits - 5) ^
        static_cast<size_t>(grdim) << (bits - 6);
    return attr_hash;
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<TmaOuterReductionParams>(*this);
  }
};

namespace reduction {
namespace outer_tma {
std::unique_ptr<TmaOuterReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props);

void scheduleReduction(Fusion* fusion, const TmaOuterReductionParams* rparams);
} // namespace outer_tma
} // namespace reduction
} // namespace nvfuser
