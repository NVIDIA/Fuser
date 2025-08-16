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
#define FUSION_INFO_DEFINE_FUNCTIONS(type, field, method) \
  bool has##type() const {                                \
    return field##_.get() != nullptr;                     \
  }                                                       \
  void set(std::unique_ptr<type> field) {                 \
    field##_ = std::move(field);                          \
  }                                                       \
  type& method() {                                        \
    NVF_ERROR(has##type());                               \
    return *field##_;                                     \
  }                                                       \
  const type& method() const {                            \
    NVF_ERROR(has##type());                               \
    return *field##_;                                     \
  }

// Define a std::unique_ptr member, e.g.,
//
// std::unique_ptr<IdModel> id_model_;
#define FUSION_INFO_DEFINE_FIELD(type, field) std::unique_ptr<type> field##_;

class FusionInfo {
 public:
  FUSION_INFO_DEFINE_FUNCTIONS(
      ConcretizedBroadcastDomains,
      concretized_broadcast_domains,
      concretizedBroadcastDomains);

  FUSION_INFO_DEFINE_FUNCTIONS(
      FusedReductionInfo,
      fused_reduction_info,
      fusedReductionInfo);

  FUSION_INFO_DEFINE_FUNCTIONS(
      PaddedParallelDimensions,
      padded_parallel_dimensions,
      paddedParallelDimensions);

  FUSION_INFO_DEFINE_FUNCTIONS(
      ThreadPredicateMap,
      thread_predicate_map,
      threadPredicateMap);

  FUSION_INFO_DEFINE_FUNCTIONS(
      ParallelDimensionMap,
      parallel_dimension_map,
      parallelDimensionMap);

  FUSION_INFO_DEFINE_FUNCTIONS(ComputeAtMap, ca_map, caMap);

  FUSION_INFO_DEFINE_FUNCTIONS(IdModel, id_model, idModel);

 private:
  FUSION_INFO_DEFINE_FIELD(
      ConcretizedBroadcastDomains,
      concretized_broadcast_domains);

  FUSION_INFO_DEFINE_FIELD(FusedReductionInfo, fused_reduction_info);

  FUSION_INFO_DEFINE_FIELD(
      PaddedParallelDimensions,
      padded_parallel_dimensions);

  FUSION_INFO_DEFINE_FIELD(ThreadPredicateMap, thread_predicate_map);

  FUSION_INFO_DEFINE_FIELD(ParallelDimensionMap, parallel_dimension_map);

  FUSION_INFO_DEFINE_FIELD(ComputeAtMap, ca_map);

  FUSION_INFO_DEFINE_FIELD(IdModel, id_model);
};

#undef FUSION_INFO_DEFINE_FUNCTIONS
#undef FUSION_INFO_DEFINE_FIELD

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
