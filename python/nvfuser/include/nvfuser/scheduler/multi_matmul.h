// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/mma_utils.h>
#include <val_graph.h>
#include <val_graph_visitor.h>
#include <visibility.h>

namespace nvfuser {

// Base class for AmpereMultipleMatmulScheduler and
// HopperMultipleMatmulScheduler
class MultipleMatmulScheduler {
 public:
  MultipleMatmulScheduler(Fusion* fusion, const MatmulParams* params)
      : fusion_(fusion),
        params_(params),
        id_model_(fusion, /*build_graphs=*/false) {}
  virtual ~MultipleMatmulScheduler() = default;

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

NVF_API void scheduleMultipleMatmuls(
    Fusion* fusion,
    const MatmulParams* mparams);

} // namespace nvfuser
