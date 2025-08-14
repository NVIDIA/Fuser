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

// Define query, setter and accessor methods. For example, in the case
// of IdModel:
//
// bool hasIdModel() const;
// void set(std::unique_ptr<IdModel> id_model);
// IdModel& idModel();
// const IdModel& idModel() const;
#define DEFINE_FUNCTIONS(type, field, method) \
  bool has##type() const {                    \
    return field##_.get() != nullptr;         \
  }                                           \
  void set(std::unique_ptr<type> field) {     \
    field##_ = std::move(field);              \
  }                                           \
  type& method() {                            \
    NVF_ERROR(has##type());                   \
    return *field##_;                         \
  }                                           \
  const type& method() const {                \
    NVF_ERROR(has##type());                   \
    return *field##_;                         \
  }

#define DEFINE_FIELD(type, field) std::unique_ptr<type> field##_;

class FusionInfo {
 public:
  DEFINE_FUNCTIONS(
      ConcretizedBroadcastDomains,
      concretized_broadcast_domains,
      concretizedBroadcastDomains);

  DEFINE_FUNCTIONS(
      FusedReductionInfo,
      fused_reduction_info,
      fusedReductionInfo);

  DEFINE_FUNCTIONS(
      PaddedParallelDimensions,
      padded_parallel_dimensions,
      paddedParallelDimensions);

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

  DEFINE_FUNCTIONS(ComputeAtMap, ca_map, caMap);

  DEFINE_FUNCTIONS(IdModel, id_model, idModel);

 private:
  DEFINE_FIELD(ConcretizedBroadcastDomains, concretized_broadcast_domains);

  DEFINE_FIELD(FusedReductionInfo, fused_reduction_info);

  DEFINE_FIELD(PaddedParallelDimensions, padded_parallel_dimensions);

  std::shared_ptr<const ThreadPredicateMap> thread_predicate_map_;

  std::shared_ptr<const ParallelDimensionMap> parallel_dimension_map_;

  DEFINE_FIELD(ComputeAtMap, ca_map);
  DEFINE_FIELD(IdModel, id_model);
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
