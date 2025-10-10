// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <cstddef>
#include <cstdint>

#include <expr_evaluator.h>
#include <fusion.h>
#include <runtime/executor_kernel_arg.h>
#include <utils.h>
#include <visibility.h>

namespace nvfuser {

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

class SchedulerRuntimeInfo : public NonCopyable {
 public:
  //! Create runtime info for given fusion and input. Creating and binding
  //! evaluator is optional. The evaluator is used to manage intermediate
  //! integers in the fusion. We need them for segmenter and schedulers,
  //! but we don't need them when we are just using this class to provide
  //! additional encoding for kernel cache lookup.
  //!
  //! The index type of forced_index_type is used if given, no matter
  //! how large the actual arguments and fusion tensors
  //! are. CORRECTNESS IS NOT GUARANTEED.
  NVF_API SchedulerRuntimeInfo(
      Fusion* complete_fusion,
      KernelArgumentHolder args,
      PrecomputedValues* precomputed_values = nullptr,
      const std::vector<TensorView*>& all_tvs = {},
      std::optional<PrimDataType> forced_index_type = std::nullopt);

  //! Lookup for the alignment sizes of the given tv. Currently only returns
  //!  actual alignment info for input tensors to the complete fusion,
  //!  and for other intermediate/fuser-allocated tensors will
  //!  return max_alignment_size_in_bit.
  size_t getAlignmentSizeBit(TensorView* tv);

  //! Returns sizes of tensor dimensions in same order as allocation domain,
  //! ignoring any IterType::Reduction domains in the allocation domain. This
  //! only works for complete Fusion inputs whose allocation domain is a
  //! permutation of their root domain and will raise an exception otherwise.
  const std::vector<int64_t>& getInputAllocationSizes(TensorView* tv) const {
    NVF_ERROR(
        isInputTv(tv),
        "TensorView ",
        tv->toString(),
        " is not an input or its logical domain is not a permutation of its ",
        "allocation domain");
    auto sizes_it = input_sizes_.find(tv);
    NVF_ERROR(sizes_it != input_sizes_.end());
    return sizes_it->second;
  }

  //! Returns strides of tensor in same order as allocation domain, in elements
  //! instead of bytes/bits. Only works for complete Fusion inputs whose
  //! allocation domain is a permutation of their root domain and will raise an
  //! exception otherwise.
  const std::vector<int64_t>& getInputAllocationStrides(TensorView* tv) const {
    NVF_ERROR(
        isInputTv(tv),
        "TensorView ",
        tv->toString(),
        " is not an input or its logical domain is not a permutation of its ",
        "allocation domain");
    auto strides_it = input_strides_elements_.find(tv);
    NVF_ERROR(strides_it != input_strides_elements_.end());
    return strides_it->second;
  }

  // Computes alignment size in bits for provided ptr address
  static size_t computeAlignmentSizeBit(size_t ptr_address_in_bytes);

  // Return the runtime pointer value (in bytes) for provided tensor view
  size_t ptrOf(TensorView* tv) const;

  PrimDataType getIndexType() const {
    return index_type_;
  }

  Fusion* fusion() {
    return complete_fusion_;
  }

  ExpressionEvaluator& expressionEvaluator() {
    NVF_ERROR(expression_evaluator_ != nullptr);
    return *expression_evaluator_;
  }

 private:
  // Build and bind full fusion inputs to an expression evaluator
  std::unique_ptr<ExpressionEvaluator> getExpressionEvaluator(
      const KernelArgumentHolder& inputs,
      PrecomputedValues* precomputed_values);

  bool isInputTv(TensorView* tv) const {
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

  // Copy of aten input tensor sizes ordered like the TensorView's allocation
  // domain
  std::unordered_map<Val*, std::vector<int64_t>> input_sizes_;

  // Copy of aten input tensor strides (in elements) ordered like the
  // TensorView's allocation domain
  std::unordered_map<Val*, std::vector<int64_t>> input_strides_elements_;

  // Copy of aten input tensor strides (in bytes) for only discontiguous
  // dimensions
  std::unordered_map<Val*, std::vector<size_t>> input_discontig_strides_bytes_;

  // Cache for getAlignmentSize
  std::unordered_map<TensorView*, size_t> alignment_map_bit_;

  // Found index mode kernel needs to be run in
  PrimDataType index_type_ = PrimDataType::Int;

  // TODO: Remove
  std::unordered_map<TensorView*, size_t> vectorword_map_;
};

} // namespace nvfuser
