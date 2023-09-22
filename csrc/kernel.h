// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <device_lower/analysis/sync_information.h>
#include <device_lower/pass/warp_reduce.h>
#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/builder.h>
#include <parallel_dimension_map.h>
#include <utils.h>
#include <vectorization_info.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nvfuser {
namespace kir {

//! Summary of interesting facts about the kernel
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct KernelSummary {
  //! Count of WAR (write-after-read) hazard barriers
  int war_hazard_syncs_count = 0;

  //! List of global buffers
  std::vector<const kir::Allocate*> global_allocations;

  //! List of dynamic shared memory buffers
  std::vector<const kir::Allocate*> dynamic_smem_allocations;

  //! List of static shared memory buffers
  std::vector<const kir::Allocate*> static_smem_allocations;

  //! Indicate the need to generate random numbers
  bool has_philox_op = false;

  //! Do we have any block reductions?
  bool has_block_reductions = false;

  //! Number of static grid reductions
  bool has_grid_reductions = false;

  //! Do we have any grid reduction in a loop, or grid reductions dependent on
  //! grid reductions
  bool has_cooperative_grid_reduction = false;

  //! Do we have any block broadcasts?
  bool has_block_broadcasts = false;

  //! Do we have any grid broadcasts?
  bool has_grid_broadcasts = false;

  //! Do we have any welford op?
  bool has_welford = false;

  //! Do we have any welford op?
  bool has_block_welford = false;

  //! Do we have any welford op?
  bool has_grid_welford = false;

  //! Do we have any outer grouped grid welford op?
  bool has_outer_grouped_grid_welford = false;

  //! Largest shared memory buffer size of outer grouped grid welford
  int outer_grouped_grid_welford_largest_smem_size = 0;

  //! Largest shared memory buffer base type
  DataType largest_smem_data_type = DataType::Null;

  //! Do we have allocations of dynamic local memory?
  bool has_dynamic_local_memory_allocations = false;

  //! List of dynamic local memory buffers.
  //! Only used for debugging.
  std::vector<const kir::Allocate*> dynamic_lmem_allocations;

  //! ceilDiv extents that must be divisible
  std::vector<std::pair<const Val*, const Val*>> splits_to_validate;

  //! Effective ParallelTypes of broadcast ops
  std::unordered_map<const BroadcastOp*, ParallelTypeBitmap>
      broadcast_parallel_types;

  //! Track which tensor views are inputs or outputs of a vectorized operation
  //! and their maximum vectorized access size
  std::unordered_map<TensorView*, int> vectorized_accesses;

  // Sync map is needed to figure out if global memory buffers need to be marked
  // as volatile because they're used for communication.
  std::shared_ptr<const SyncMap> sync_map;

  // Parallel dimension map needed to set the correct properties of grid buffers
  // (is a dim inactive)
  ParallelDimensionMap parallel_dimension_map_;

  //! Track information on vectorized set operations for runtime validation
  std::vector<VectorizedSetInfo> vectorized_set_info;
};

class KernelPerformanceProfile {
 public:
  //! Register an expression to profile
  void registerExpr(const Expr* expr);

  //! Query if an expression is profiled
  bool isProfiled(const Expr* expr) const;

  //! Get the number of profiled expressions
  int getNumberOfProfileEntries() const {
    return num_profile_entries_;
  }

  //! Set the backing buffer of profile.
  void setBuffer(TensorView* buffer) {
    buffer_ = buffer;
  }

  //! Get the backing buffer
  TensorView* getBuffer() const {
    return buffer_;
  }

  //! Get the indices of the profile of an expression in the backing buffer
  std::array<int, 2> getIndicesInProfileBuffer(const Expr* expr) const;

  std::string toString(const at::Tensor& buffer) const;

 private:
  //! Get the new profile index
  int getNewIndex();

  //! Get the profile index
  std::optional<int> getIndex(const Expr* expr) const;

 private:
  int num_profile_entries_ = 0;

  //! Backing buffer of Nx2 integer tensor, where N is the number of profiled
  //! regions. Each region has two integer values, one representing
  //! the cycles spent, and another the count.
  TensorView* buffer_ = nullptr;

  //! Map profiled expressions to profile entry offsets
  std::unordered_map<const Expr*, int> expr_entry_map_;

  // TODO: Allow profiling of ForLoops
  //! Map profiled ForLoop to profile entry offsets
  // std::unordered_map<const kir::ForLoop*, int> loop_entry_map_;
};

class KernelInternalProxy;

//! Container for a lowered Kernel IR
//!
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class Kernel final : public Fusion {
  friend KernelInternalProxy;

 public:
  // Kernel starts by grabbing all the nodes from the provided fusion.
  // Kernel is not SSA, if a definition is not set, we should update it, but
  // not remove previous definition if it is set. This is primarily because when
  // we do something like generate an initialization statement for a reduction
  // TV, we may want to continue to do fusion like analysis on the original
  // expression.
  // TODO: Assert index type is int or int32
  Kernel(Fusion* fusion, PrimDataType index_type = PrimDataType::Int)
      : Fusion(*fusion), index_type_(index_type) {
    // Index type must be resolved to either int32 or int64
    NVF_ERROR(
        index_type_ == PrimDataType::Int ||
            index_type_ == PrimDataType::Int32 || "Invalid index type: ",
        index_type_);
  }

  Kernel() = delete;

  // No move or copy semantics
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  //! Finalize a kernel definition
  //!
  //! At this point we have a complete kernel definition and we can
  //! run analysis passes to build a KernelSummary.
  void finalize(std::vector<Expr*> top_level_exprs);

  const std::vector<Expr*>& topLevelExprs() const {
    return top_level_exprs_;
  }

  const KernelSummary& summary() const {
    return summary_;
  }

  PrimDataType indexType() const {
    return index_type_;
  }

  //! Checks if parallel type is padded
  bool isParallelTypePadded(ParallelType ptype) const {
    return ptype == ParallelType::TIDx &&
        warp_padded_parallel_info_.is_tidx_padded;
  }

  const WarpPaddedParallelInfo& getWarpPaddedParallelInfo() const {
    return warp_padded_parallel_info_;
  }

  const KernelPerformanceProfile& profile() const {
    return profile_;
  }

  //! Debug dump of the Kernel IR
  void print() const;

  const std::vector<Val*>& parameters() const {
    return parameters_;
  }

 protected:
  using IrContainer::registerExpr;
  using IrContainer::registerVal;

  //! Register the Val with this fusion
  void registerVal(Val* val) override;

  //! Register expr with this fusion.
  //! When we register an expression, we want to update the dependency tracking
  //! of Vals. We add expr to our general expr_set_,
  void registerExpr(Expr* expr) override;

 private:
  // Analyze the kernel IR and caches the summary of interesting data
  void analyze();

  // Top level statements
  std::vector<Expr*> top_level_exprs_;

  // Summary of interesting kernel data
  KernelSummary summary_;

  // Is this kernel being compiled with int32 or int64 indexing. This
  // information is required to resolve DataType::Index
  PrimDataType index_type_ = PrimDataType::Int;

  WarpPaddedParallelInfo warp_padded_parallel_info_;

  KernelPerformanceProfile profile_;

  // Parameters of the kernel. The parameters contain the inputs and outputs of
  // the kernel, intermediate buffers, and special items such as RNG state and
  // tensor map for TMA support, etc. The parameters are not required to have no
  // definition. If a parameter has a definition, its definition will be
  // evaluated before the kernel is executed.
  std::vector<Val*> parameters_;
};

//! A special debugging proxy for Kernel.
//!
//! Should not be used for other than testing and debugging.
class KernelInternalProxy {
 public:
  KernelInternalProxy(Kernel* kernel) : kernel_(kernel) {}

  std::vector<Expr*>& topLevelExprs();

 private:
  Kernel* kernel_ = nullptr;
};

} // namespace kir
} // namespace nvfuser
