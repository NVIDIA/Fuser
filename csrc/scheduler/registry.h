// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <executor_kernel_arg.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/compile_time_info.h>
#include <scheduler/heuristic.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/pointwise_heuristic.h>
#include <scheduler/reduction_heuristic.h>
#include <scheduler/utils.h>
#include <utils.h>

namespace nvfuser {

class SegmentedGroup;
class ExpressionEvaluator;

//!  SchedulerRuntimeInfo is the abstraction introduced in
//! this PR for passing runtime input dependent information
//! to the schedulers and kernel caches.
//!
//! Note:
//!  if any additional info needed,  or maybe just the inputs themselves it
//!    could just be added to this class, and they will be distributed to the
//!    segmenter and schedulers.
//!  It is important that input id encoding should be up to date with any change
//!   of this class to avoid launching compiled kernels with illegal inputs.

class TORCH_CUDA_CU_API SchedulerRuntimeInfo : public NonCopyable {
 public:
  // Max vector size we will consider, in bytes,
  //  currently set to 16B = 128b
  static constexpr size_t max_alignment_size_in_byte = 16;

  //! Create runtime info for given fusion and input. Creating and binding
  //! evaluator is optional. The evaluator is used to manage intermediate
  //! integers in the fusion. We need them for segmenter and schedulers,
  //! but we don't need them when we are just using this class to provide
  //! additional encoding for kernel cache lookup.
  //!
  //! The index type of forced_index_type is used if given, no matter
  //! how large the actual arguments and fusion tensors
  //! are. CORRECTNESS IS NOT GUARANTEED.
  SchedulerRuntimeInfo(
      Fusion* complete_fusion,
      KernelArgumentHolder args,
      PrecomputedValues* precomputed_values = nullptr,
      const std::vector<TensorView*>& all_tvs = {},
      std::optional<PrimDataType> forced_index_type = std::nullopt);

  SchedulerRuntimeInfo(
      Fusion* complete_fusion,
      const at::ArrayRef<c10::IValue>& aten_inputs);

  //! Lookup for the alignment sizes of the given tv. Currently only returns
  //!  actual alignment info for input tensors to the complete fusion,
  //!  and for other intermediate/fuser-allocated tensors will
  //!  return max_alignment_size_in_byte.
  size_t getAlignmentSize(TensorView* tv);

  // Gets maximum vectorizable width of tv, assumes we can merge across all
  // iteration domains if contiguous, unless contig_merge=false. Cannot permute
  // the dimensions to fix contiguity. Ignores dimensions that are broadcast or
  // reduction.
  size_t getMaxVectorizableWidth(TensorView* tv, bool contig_merge = true);

  // Computes alignment size in bytes for provided ptr address
  static size_t computeAlignmentSize(size_t ptr_address);

  // Return the runtime pointer value for provided tensor view
  size_t ptrOf(TensorView* tv) const;

  PrimDataType getIndexType() const {
    return index_type_;
  }

  Fusion* fusion() {
    return complete_fusion_;
  }

  ExpressionEvaluator& expressionEvaluator() {
    TORCH_INTERNAL_ASSERT(expression_evaluator_ != nullptr);
    return *expression_evaluator_;
  }

 private:
  // Build and bind full fusion inputs to an expression evaluator
  std::unique_ptr<ExpressionEvaluator> getExpressionEvaluator(
      const KernelArgumentHolder& inputs,
      PrecomputedValues* precomputed_values);

  bool isInputTv(TensorView* tv) {
    return std::find(
               complete_fusion_->inputs().begin(),
               complete_fusion_->inputs().end(),
               tv) != complete_fusion_->inputs().end();
  }

 private:
  // Returns the offset of tv in the inputs ignoring non tensor views. Used to
  // access input_sizes, input_strides, input_ptr
  int offsetTensorPos(TensorView* tv);

  // Expression evaluator used to probe sizes in the fusion IR
  std::unique_ptr<ExpressionEvaluator> expression_evaluator_ = nullptr;

  // Fusion reference that this runtime info is associated with
  Fusion* complete_fusion_ = nullptr;

  // Copy of aten input pointer addresses
  // TODO: Support output tensor pointers
  std::unordered_map<Val*, size_t> input_ptrs_;

  // Copy of aten input tensor strides (in bytes)
  std::unordered_map<Val*, std::vector<size_t>> input_discontig_strides_;

  // Cache for getAlignmentSize
  std::unordered_map<TensorView*, size_t> alignment_map_;
  // Cache for getMaxVectorizableWidth
  std::unordered_map<TensorView*, size_t> max_vectorword_map_;

  // Found index mode kernel needs to be run in
  PrimDataType index_type_ = PrimDataType::Int;

  // TODO: Remove
  std::unordered_map<TensorView*, size_t> vectorword_map_;
};

class HeuristicSummary;

//! Virtual base class for schedule heuristics
//!   heuristic implementations derive from this
//!   class and implement a schedule(Fusion*)
//!   and a bool canSchedule(Fusion*) interface
class TORCH_CUDA_CU_API SchedulerEntry {
 public:
  //! Fusion runtime facing API,
  //!   builds a new entry with the given heuristics
  //!   corresponding to the given fusion
  static std::unique_ptr<SchedulerEntry> makeEntry(
      ScheduleHeuristic sh,
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  virtual ~SchedulerEntry() = default;

  //! External access for canSchedule utilities through SchedulerEntry
  //!  to avoid exposing a single function to the namespace
  static bool canSchedule(
      ScheduleHeuristic sh,
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  //! Fusion segmenter facing API,
  //!   returns a schedule that applies in the given fusion, returns a nullopt
  //!   if no schedule in the registry can handle.
  static std::optional<ScheduleHeuristic> proposeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info);

  //! Fusion runtime facing API,
  //!   schedule the given fusion with heuristics owned
  //!   by this entry, for actual heuristics to override
  virtual void schedule(Fusion* fusion) = 0;

  //! Heuristic comparison
  bool sameAs(const SchedulerEntry* other);

  ScheduleHeuristic heuristic() const {
    return heuristic_;
  }

  const std::shared_ptr<HeuristicParams>& params() const {
    return params_;
  }

  const ReductionParams& reductionParams() const {
    auto rparams = std::dynamic_pointer_cast<ReductionParams>(params_);
    TORCH_INTERNAL_ASSERT(
        rparams != nullptr, "Heuristic parameter is not a reduction parameter");
    return *rparams;
  }

  const PointwiseParams& pointwiseParams() const {
    auto pparams = std::dynamic_pointer_cast<PointwiseParams>(params_);
    TORCH_INTERNAL_ASSERT(
        pparams != nullptr, "Heuristic parameter is not a pointwise parameter");
    return *pparams;
  }

  const TransposeParams& transposeParams() const {
    auto tparams = std::dynamic_pointer_cast<TransposeParams>(params_);
    TORCH_INTERNAL_ASSERT(
        tparams != nullptr, "Heuristic parameter is not a transpose parameter");
    return *tparams;
  }

  const MatmulParams& matmulParams() const {
    auto mparams = std::dynamic_pointer_cast<MatmulParams>(params_);
    TORCH_INTERNAL_ASSERT(
        mparams != nullptr, "Heuristic parameter is not a matmul parameter");
    return *mparams;
  }

  void updateLaunchConstraint(const LaunchParams& launch_params) {
    params_->lparams = launch_params;
  }

 protected:
  explicit SchedulerEntry(ScheduleHeuristic heuristic)
      : heuristic_(heuristic) {}

  //! Heuristic parameters if applicable
  std::shared_ptr<HeuristicParams> params_ = nullptr;

 private:
  //! What kind of heuristics does this entry have?
  const ScheduleHeuristic heuristic_;
};

//! Hash function for a scheduler entry
class TORCH_CUDA_CU_API SchedulerEntryHash {
 public:
  size_t operator()(const SchedulerEntry& se) const;
};

//! Debug print function for heuristics
TORCH_CUDA_CU_API std::string toString(ScheduleHeuristic sh);

//! Debug print function for heuristics
TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    ScheduleHeuristic sh);

} // namespace nvfuser
