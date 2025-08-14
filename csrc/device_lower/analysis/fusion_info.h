// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <device_lower/analysis/fused_reduction.h>
#include <device_lower/analysis/padded_parallel_dimensions.h>
#include <device_lower/analysis/thread_predicate.h>
#include <device_lower/analysis/trivial_broadcast.h>
#include <id_model/id_model.h>
#include <parallel_dimension_map.h>

#include <memory>

namespace nvfuser {

class FusionInfo {
 public:
  auto& concretizedBroadcastDomains() {
    return concretized_broadcast_domains_;
  }

  const auto& concretizedBroadcastDomains() const {
    return concretized_broadcast_domains_;
  }

  auto& fusedReductions() {
    return fused_reductions_;
  }

  const auto& fusedReductions() const {
    return fused_reductions_;
  }

  auto& paddedParallelDimensions() {
    return padded_parallel_dimensions_;
  }

  const auto& paddedParallelDimensions() const {
    return padded_parallel_dimensions_;
  }

  auto& threadPredicateMap() {
    return thread_predicate_map_;
  }

  const auto& threadPredicateMap() const {
    return thread_predicate_map_;
  }

  auto& parallelDimensionMap() {
    return parallel_dimension_map_;
  }

  const auto& parallelDimensionMap() const {
    return parallel_dimension_map_;
  }

  void set(std::unique_ptr<ComputeAtMap> ca_map) {
    ca_map_ = std::move(ca_map);
  }

  bool hasCaMap() const {
    return ca_map_.get() != nullptr;
  }

  ComputeAtMap& caMap() {
    NVF_ERROR(hasCaMap());
    return *ca_map_;
  }

  const ComputeAtMap& caMap() const {
    NVF_ERROR(hasCaMap());
    return *ca_map_;
  }

  bool hasIdModel() const {
    return id_model_.get() != nullptr;
  }

  void set(std::unique_ptr<IdModel> id_model) {
    id_model_ = std::move(id_model);
  }

  IdModel& idModel() {
    NVF_ERROR(hasIdModel());
    return *id_model_;
  }

  const IdModel& idModel() const {
    NVF_ERROR(hasIdModel());
    return *id_model_;
  }

 private:
  std::shared_ptr<const ConcretizedBroadcastDomains>
      concretized_broadcast_domains_;

  std::shared_ptr<const FusedReductionInfo> fused_reductions_;

  std::shared_ptr<const PaddedParallelDimensions> padded_parallel_dimensions_;

  std::shared_ptr<const ThreadPredicateMap> thread_predicate_map_;

  std::shared_ptr<const ParallelDimensionMap> parallel_dimension_map_;

  std::shared_ptr<ComputeAtMap> ca_map_;

  std::unique_ptr<IdModel> id_model_;
};

class FusionInfoGuard {
 public:
  FusionInfoGuard(FusionInfo* fusion_info);
  ~FusionInfoGuard();

  static FusionInfo* current();
  static bool hasCurrent();

 private:
  FusionInfo* prev_fusion_info_ = nullptr;

  static thread_local FusionInfo* active_fusion_info_;
};

} // namespace nvfuser
