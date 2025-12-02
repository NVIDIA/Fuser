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

  // If we want to vectorize casts or not
  bool vectorize_casts = true;

  // Unroll on top of vectorization
  // In the 2D scheduler, unroll the outer dimension to reuse loaded data across
  // rows, reducing loaded bytes by the unroll factor.
  // Always equals 1 for 1D scheduler.
  int64_t unroll_factor_outer = 1;

  // In the 2D scheduler, unroll the inner dimension to reuse loaded data across
  // cols, reducing loaded bytes by the unroll factor.
  // Also used in 1D scheduler.
  int64_t unroll_factor_inner = 1;

  // ========== TMA (Tensor Memory Accelerator) Configuration ==========
  // Enable TMA for hardware-accelerated global <-> shared memory transfers
  bool use_tma_load = false; // Use TMA for loading inputs
  bool use_tma_store = false; // Use TMA for storing outputs

  // TMA Domain: Defines the 2D logical structure [tma_domain_outer,
  // tma_domain_inner] The problem is split as: [I0 = n_elems] ->
  // [tma_domain_outer, tma_domain_inner] Example: n_elems=2048,
  // tma_domain_inner=512 -> TMA domain [4, 512]
  int64_t tma_domain_inner = -1; // Size of inner (contiguous) dimension

  // TMA Tile: Defines the 2D box size [tma_tile_outer, tma_tile_inner] loaded
  // by each TMA operation The domain is tiled as: [tma_domain_outer,
  // tma_domain_inner] ->
  //   [tma_domain_outer/tma_tile_outer, tma_tile_outer,
  //    tma_domain_inner/tma_tile_inner, tma_tile_inner]
  // Example: TMA domain [4, 512] with tiles [2, 128] requires 2*4=8 TMA loads
  int64_t tma_tile_outer = -1; // Outer tile dimension size
  int64_t tma_tile_inner = -1; // Inner tile dimension size

  using HeuristicParams::HeuristicParams;

  // Warning: Does not check launch parameters!
  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const PointwiseParams*>(other_base);
    if (other == nullptr) {
      return false;
    }
    bool attr_equal = other->cparams == cparams &&
        other->break_point == break_point &&
        other->split_block == split_block &&
        other->split_grid_y_dim == split_grid_y_dim &&
        other->flip_grid_binding == flip_grid_binding &&
        other->vectorization_factor == vectorization_factor &&
        other->vectorize_casts == vectorize_casts &&
        other->unroll_factor_outer == unroll_factor_outer &&
        other->unroll_factor_inner == unroll_factor_inner &&
        other->use_tma_load == use_tma_load &&
        other->use_tma_store == use_tma_store &&
        other->tma_domain_inner == tma_domain_inner &&
        other->tma_tile_outer == tma_tile_outer &&
        other->tma_tile_inner == tma_tile_inner;
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
    if (use_tma_load || use_tma_store) {
      ss << "TMA Configuration:\n";
      ss << "  use_tma_load: " << (use_tma_load ? "true" : "false") << "\n";
      ss << "  use_tma_store: " << (use_tma_store ? "true" : "false") << "\n";
      ss << "  tma_domain_inner: " << tma_domain_inner << "\n";
      ss << "  tma_tile_outer: " << tma_tile_outer << "\n";
      ss << "  tma_tile_inner: " << tma_tile_inner << "\n";
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
        static_cast<size_t>(flip_grid_binding) << 10 ^
        static_cast<size_t>(use_tma_load) << 11 ^
        static_cast<size_t>(use_tma_store) << 12 ^
        static_cast<size_t>(tma_domain_inner) << 13 ^
        static_cast<size_t>(tma_tile_outer) << 15 ^
        static_cast<size_t>(tma_tile_inner) << 17;
    return attr_hash;
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<PointwiseParams>(*this);
  }
};

} // namespace nvfuser
