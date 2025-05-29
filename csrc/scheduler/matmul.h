// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <exceptions.h>
#include <fusion.h>
#include <mma_type.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/mma_utils.h>
#include <scheduler/registry.h>
#include <visibility.h>

// This file contains:
// - MatmulScheduler: The entry point for scheduler registry
// - namespace schedule_matmul: Implementation detail for
//   MatmulScheduler::schedule

namespace nvfuser {

class MatmulScheduler : public SchedulerEntry {
 public:
  bool canScheduleCompileTime(Fusion* fusion) override;

  bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) override;

  void schedule(Fusion* fusion, const HeuristicParams* params) override;

  constexpr static SchedulerType schedulerType() {
    return SchedulerType::Matmul;
  }
};

namespace schedule_matmul {

// Base class for AmpereMinus and HopperPlus
class Common {
 public:
  Common(Fusion* fusion, const MatmulParams* params)
      : fusion_(fusion),
        params_(params),
        id_model_(fusion, /*build_graphs=*/false) {}
  virtual ~Common() = default;

  virtual void run() = 0;

 protected:
  void findPatterns();

  void translatePatterns();

  // Get tensor roles and id roles
  // When there are multiple matmul patterns, we can have conflicting roles.
  // For now we throw an error if this is the case.
  // TODO: This should be checked in canScheduleCompileTime
  void findRoles();

  void countDims();

  //! Rebuilds IdModel, then updates all ValGroups in abstract tensors to refer
  //! to the new IdModel. This is necessary whenever we perform an operation
  //! that creates a new TensorView, such as caching or rFactor
  void updateIdModel();

  //! Defines all cache tensors of inputs and outputs. Schedules intermediate
  //! global TensorViews for skipping metadata operations like permute and
  //! broadcast when loading operands. Defines as_, bs_, acw_smems_, bcw_smems_.
  //! Sets the mma macro.
  //!
  //! If skip_intermediates is true, we call
  //! scheduler_utils::scheduleInputToSkipIntermediates on each operand to avoid
  //! computing metadata expressions.
  void cacheInputsAndOutputs(bool skip_intermediates);

  virtual void setOperandSmemLoadAndCacheOps(
      TensorView* operand,
      int64_t vec_size) = 0;

  //! Update IdModel by adding domain2 to it and mapping it to domain1
  //! Unfortunately, IDs in domain1 might not be present in the IdModel because
  //! IdModel is not automatically updated when we do scheduling and rFactor.
  //! In this case, the corresponding IDs will be just skipped.
  void addAndMapDomain(
      const std::vector<IterDomain*>& domain1,
      const std::vector<IterDomain*>& domain2);

  //! This calls orig->cacheBefore() and also updates the broadcast graph to
  //! reflect the new IterDomain mappings
  TensorView* cacheBefore(
      TensorView* orig,
      LoadStoreOpType op_type = LoadStoreOpType::Set);

  //! This calls orig->cacheAfter() and also updates the broadcast graph to
  //! reflect the new IterDomain mappings
  TensorView* cacheAfter(
      TensorView* orig,
      LoadStoreOpType op_type = LoadStoreOpType::Set,
      CacheOp cache_op = CacheOp::AllLevels,
      bool propagate_allocation_domain = false);

 protected:
  Fusion* fusion_;
  const MatmulParams* params_;
  IdModel id_model_;

  // Broadcast graph of id_model_, which we modify at times using e.g.
  // AbstractTensor.split or by mapping vals in cacheAfter and rFactor
  ValGraph* graph_ = nullptr;
  std::vector<mma_utils::MatmulPattern> patterns_;
  mma_utils::DimRolesMap id_roles_;
  mma_utils::TensorRolesMap tensor_roles_;
  mma_utils::MatmulOperandInnerDims inner_dims_;

  std::vector<ValGroup> canonical_dim_ordering_;

  int64_t num_splitk_dims_ = 0;
  int64_t num_device_dims_ = 0;
  int64_t num_local_batch_dims_ = 0;
  int64_t num_device_and_batch_dims_ = 0;

  std::vector<std::pair<TensorView*, TensorView*>> cached_epilogue_inputs_;

  std::vector<TensorView*> as_, bs_, acw_smems_, bcw_smems_, mma_results_,
      splitk_sums_, smem_epilogues_;
};

} // namespace schedule_matmul

} // namespace nvfuser
