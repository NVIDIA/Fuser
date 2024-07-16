// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>

#include <compute_at_map.h>
#include <device_lower/analysis/fused_reduction.h>
#include <device_lower/analysis/predicate_elimination.h>
#include <device_lower/analysis/sync_information.h>
#include <device_lower/analysis/thread_predicate.h>
#include <device_lower/analysis/tma.h>
#include <device_lower/analysis/trivial_broadcast.h>
#include <device_lower/pass/allocation.h>
#include <device_lower/pass/circular_buffer.h>
#include <device_lower/pass/predicate.h>
#include <device_lower/pass/scalar_hoist.h>
#include <device_lower/pass/warp_reduce.h>
#include <exceptions.h>
#include <executor_params.h>
#include <expr_simplifier.h>
#include <id_model/id_model.h>
#include <id_model/indexing.h>
#include <ir/all_nodes.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <logical_domain_map.h>
#include <non_divisible_split.h>
#include <options.h>
#include <parallel_dimension_map.h>
#include <vectorization_info.h>
#include <visibility.h>

#include <functional>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

// TODO: we frequently use pairwise root mapping from consumers to producers.
// This information is implicitly in the computeAtMaps, but there's no isolated
// container for this information that we can reuse. Would be nice to generate
// such a structure and propagate it through lowering.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class GpuLower : public NonCopyable {
  class KernelIrMapper;

 public:
  GpuLower() = delete;

  using Pass = std::pair<
      std::string, // name of the pass
      std::function<std::vector<Expr*>(const std::vector<Expr*>&)>>;

  // GpuLower lowers the provided fusion into a kernel which can be translated
  // into cuda code. index_type allows to compile the kernel based on int32
  // indexing instead of int64 for additional performance.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  NVF_API explicit GpuLower(
      Fusion* fusion,
      const CompileParams& cparams = CompileParams());

  NVF_API kir::Kernel* kernel() const;

  //! Returns the currently active lowering object.
  //! It's an error if no lowering is in progress.
  static GpuLower* current();

  //! Query if lowering is in progress
  static bool hasCurrent();

  //! Actually run the lowering by executing the passes in the order given by
  //! passes_
  NVF_API kir::Kernel* run();

  const PrimDataType& indexType() const {
    return cparams_.index_type.value();
  }

  const auto& minDeviceVersion() const {
    return min_device_version_;
  }

  const std::string& minDeviceVersionReason() const {
    return min_device_version_reason_;
  }

  std::shared_ptr<const ConcretizedBroadcastDomains>
  concretizedBroadcastDomains() {
    return concretized_broadcast_domains_;
  }

  const ThreadPredicateMap& threadPredMap() const {
    return thread_pred_map_;
  }

  // Returns non-const reference. Necessary to reset a predicate flag
  // when a broadcast expression is fused into a reduction.
  ThreadPredicateMap& threadPredMap() {
    return thread_pred_map_;
  }

  std::shared_ptr<const ComputeAtMap> caMap() const {
    return std::const_pointer_cast<const ComputeAtMap>(compute_at_map_);
  }

  IdModel& idModel() {
    NVF_ERROR(id_model_.get());
    return *id_model_;
  }

  const IdModel& idModel() const {
    NVF_ERROR(id_model_.get());
    return *id_model_;
  }

  bool isTensorIndexerEnabled() const {
    return tensor_indexer_.get() != nullptr;
  }

  TensorIndexer& tensorIndexer() {
    NVF_ERROR(tensor_indexer_.get());
    return *tensor_indexer_;
  }

  const TensorIndexer& tensorIndexer() const {
    NVF_ERROR(tensor_indexer_.get());
    return *tensor_indexer_;
  }

  const ParallelDimensionMap& parallelDimensionMap() const {
    return parallel_dimension_map_;
  }

  ParallelDimensionMap& parallelDimensionMap() {
    return parallel_dimension_map_;
  }

  PredicateElimination& predicateElimination() {
    NVF_ERROR(pred_elimination_.get() != nullptr);
    return *pred_elimination_;
  }

  const PredicateElimination& predicateElimination() const {
    NVF_ERROR(pred_elimination_.get() != nullptr);
    return *pred_elimination_;
  }

  LocalAllocationInfoMap& localAllocationInfoMap() {
    return local_allocation_info_map_;
  }

  const WarpPaddedParallelInfo& getWarpPaddedParallelInfo() const {
    return warp_pad_info_;
  }

  auto& nonDivisibleSplitInfo() {
    return non_divisible_split_info_;
  }

  const auto& nonDivisibleSplitInfo() const {
    return non_divisible_split_info_;
  }

  const auto& divisibleSplitSet() const {
    return divisible_splits_;
  }

  CircularBufferInfo& circularBufferInfo() {
    return circular_buffer_info_;
  }

  CommonScalarMap& commonScalarMap() {
    return common_scalar_map_;
  }

  const auto& vectorizedAccesses() const {
    return vectorized_accesses_;
  }

  auto& vectorizedAccesses() {
    return vectorized_accesses_;
  }

  const auto& vectorizedSetInfo() const {
    return vectorized_set_info_;
  }

  auto& vectorizedSetInfo() {
    return vectorized_set_info_;
  }

  bool requiresIdModel() const {
    return requires_id_model_;
  }

  bool& requiresIdModel() {
    return requires_id_model_;
  }

  FusedReductionInfo& fusedReductionInfo() {
    return fused_reduction_info_;
  }

  std::shared_ptr<const SyncMap> syncMap() const {
    return sync_map_;
  }

  kir::KernelPerformanceProfile& profile() {
    return profile_;
  }

  std::unordered_map<const Expr*, TensorView*>& ldstMBarrierMap() {
    return ldst_mbarrier_map_;
  }

  const std::unordered_map<const Expr*, TensorView*>& ldstMBarrierMap() const {
    return ldst_mbarrier_map_;
  }

  bool isNvFuserZeroEnabled() {
    if (isOptionDisabled(DisableOption::MagicZero)) {
      return false;
    }
    return cparams_.enable_magic_zero;
  }

  // This is an interface to propagate information after expression
  //  replacement on the kernel IR. E.g.:
  //    for ...
  //       c = a + b   (expr 0)
  //  after any pass that does replacement:
  //    for ...
  //       c1 = a1 + b1 (expr1)
  //  The previous analysis that was performed on expr0 might still
  //    be valid on expr1 but that info would be lost after replacement.
  //  This function provides an interface to manually update the info
  //    in any pass that performs replacement.
  void propagateExprInfo(const Expr* old_expr, const Expr* new_expr);

  std::vector<Val*>& allKnownVals() {
    return all_known_vals_;
  }

  const std::vector<Val*>& allKnownVals() const {
    return all_known_vals_;
  }

  const std::vector<Pass>& passes() const {
    return passes_;
  }

  std::vector<Pass>& passes() {
    return passes_;
  }

  std::unordered_map<TensorView*, const TMAInfo>& consumerToTMAInfo() {
    return consumer_to_tma_info_;
  }

  const std::unordered_map<TensorView*, const TMAInfo>& consumerToTMAInfo()
      const {
    return consumer_to_tma_info_;
  }

  // Register a boolean Val as a predicate to validate at the run time. Optional
  // validation error messages can be given as args.
  template <typename... Args>
  void validate(Val* validation_condition, Args... args) {
    auto sv = simplifyExpr(validation_condition);
    if (sv->isTrue()) {
      // If validation_condition is simplified to true, we know that the
      // condition is always true regardless of the runtime values of the
      // inputs. We can skip the validation. For example, we are not interested
      // in validating that 3 < 4 or i % 8 < 8 every time we run the kernel.
      return;
    }
    std::string message = to_str(args...);
    NVF_ERROR(!sv->isFalse(), message);
    validations_.emplace_back(sv, message);
  }

  const std::vector<std::pair<const Val*, std::string>>& validations() const {
    return validations_;
  }

  std::vector<std::pair<const Val*, std::string>>& validations() {
    return validations_;
  }

 private:
  void analysis(Fusion* fusion);

  // Goes through the parallelized iterdomains of the used TVs and find
  //  the parallel dimensions that need to be padded to a multiples of
  //  warp size.
  void collectPaddedParallelDims();

  bool resolveComputeWith(Fusion* fusion);

 private:
  // Lowered Kernel IR
  std::unique_ptr<kir::Kernel> kernel_;

  // Passes to lower kernel, in order
  std::vector<Pass> passes_;

  // Some stateful information during lowering
  // TODO: A lot of this information uses a define class then call build. It
  // would be safer to wrap all of these in unique pointers and remove the build
  // interface and default constructor. That way they couldn't be accessed
  // without being initialized.
  std::pair<int64_t, int64_t> min_device_version_;
  std::string min_device_version_reason_;
  std::shared_ptr<const ConcretizedBroadcastDomains>
      concretized_broadcast_domains_;
  ThreadPredicateMap thread_pred_map_;
  std::unique_ptr<PredicateElimination> pred_elimination_;
  std::shared_ptr<ComputeAtMap> compute_at_map_;
  LocalAllocationInfoMap local_allocation_info_map_;
  WarpPaddedParallelInfo warp_pad_info_;
  ParallelDimensionMap parallel_dimension_map_;
  NonDivisibleSplitInfo non_divisible_split_info_;
  CircularBufferInfo circular_buffer_info_;
  CommonScalarMap common_scalar_map_;
  FusedReductionInfo fused_reduction_info_;
  std::shared_ptr<const SyncMap> sync_map_;
  kir::KernelPerformanceProfile profile_;
  std::unordered_set<Split*> divisible_splits_;
  CompileParams cparams_;
  std::unique_ptr<IdModel> id_model_;
  std::unique_ptr<TensorIndexer> tensor_indexer_;
  std::unordered_map<TensorView*, const TMAInfo> consumer_to_tma_info_;

  // Track which tensor views are inputs or outputs of a vectorized operation
  // and their maximum vectorized access size
  // std::unordered_map<TensorView*, VectorizationInfo> vectorized_accesses_;
  std::unordered_map<TensorView*, int64_t> vectorized_accesses_;
  // Info on each vectorized set op
  std::vector<VectorizedSetInfo> vectorized_set_info_;

  // All vals that are known to the kernel, including fusion inputs and
  // precomputed values
  std::vector<Val*> all_known_vals_;

  // Keep track of the mbarrier used for each load/store operation
  std::unordered_map<const Expr*, TensorView*> ldst_mbarrier_map_;

  // Keep track of validations needed at runtime. For example, a pair of
  //! "extent mod split_factor == 0" and an error message for divisibility check
  //! for vectorization.
  std::vector<std::pair<const Val*, std::string>> validations_;

  Fusion* fusion_ = nullptr;

  // A temporary flag which is true if the fusion uses any feature that requires
  // the new experimental id model
  bool requires_id_model_ = false;
};

} // namespace nvfuser
