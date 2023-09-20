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

#include <fusion.h>
#include <ir/builder_passkey.h>
#include <ir/internal_base_nodes.h>
#include <ir/internal_nodes.h>
#include <mma_type.h>
#include <type.h>

#include <torch/csrc/jit/ir/ir.h>

#include <complex>
#include <limits>
#include <sstream>

//! Nodes in here are intended to be "user facing" users in this sense being
//! those that want to be able to generate CUDA code.

//! IR header hierarchy
//! 1. utils.h - PolymorphicBase and NonCopyable
//! 2. ir/base_nodes.h - Statement, Expr, and Val
//! 3. ir/internal_base_nodes.h - IterDomain and TensorDomain
//! 4. ** ir/interface_nodes.h ** - TensorView and Scalar
//! 5. ir/internal_nodes.h - Any internal-only IR nodes

namespace nvfuser {

class WelfordResult;
class ViewTransform;

class IrCloner;

namespace ir_utils {
std::string varName(const Val* val);
}

template <typename T>
T& Expr::attribute(size_t index) const {
  if constexpr (PolymorphicValue::is_candidate_type<T>) {
    return attributeVal(index)->value().as<T>();
  } else {
    return attributeVal(index)->value().as<Opaque>().as<T>();
  }
}

//! Mode during propagation of computeAt, standard will throw an error if
//! computeAt position provided can't be satisfied, best effort will lower the
//! computeAt position as needed during traversal, most inlined will increase
//! the compute at position to maximum possible through traversal.
enum class ComputeAtMode { Standard, BestEffort, MostInlined };

class TransformPropagator;
struct MostInlinedTransformPropagator;
class TransformIter;
class TransformReplay;
class OptOutMutator;
class TensorDomain;

class MaxPosCalculator;

namespace ir_utils {
class TVDomainGuard;
}

//! TensorView is our primitive Tensor Type used in code generation. It can be
//! thought of as representing physical memory, however, its dimensionality is
//! modifed as split/merge/computeAt functions are called. The history of
//! these transformations are kept and used for generating actual code
//! referncing physical memory. Generally when users are thinking of code
//! generation in reference to a Tensor, this is the class they should be
//! interacting with.
//!
//! The reason we need both TensorView and TensorDomain is that we need to have
//! a record of both what is being computed and how it is being computed. For
//! example we may have the operation:
//!
//!   TV3[I, J, K] = TV2[I, J, K] + TV1[I, J, K]
//!
//! The mathematical operations here are on the tensor views TV1, TV2, and
//! TV3. This operation is a pointwise operation. To compute this pointwise
//! operation we iterate over the 3D TensorDomain [I, J, K], where K is the
//! fastest changing dimension.
//!
//! \todo Need to work on the const model for TensorView, making all functions
//! that should be const, const. Gave this a try but expanded really quickly.
//! getComputeAtAxis not being const because it can return a TV that some expect
//! to be non-const is the biggest headache.
//!
class TensorView : public Val {
 public:
  TensorView(
      IrBuilderPasskey passkey,
      TensorDomain* domain,
      DataType dtype,
      MemoryType mtype = MemoryType::Local);

  explicit TensorView(
      IrBuilderPasskey passkey,
      const std::shared_ptr<c10::TensorType>& tensor_type);

  explicit TensorView(
      IrBuilderPasskey passkey,
      const std::shared_ptr<torch::jit::Value>& jit_value);

  TensorView(const TensorView* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  TensorDomain* domain() const {
    return domain_;
  }

  void setContiguity(const std::vector<std::optional<bool>>& contig) {
    domain()->setContiguity(contig);
  }

  void setContiguity(bool contig) {
    setContiguity(
        TensorDomain::getContiguityFilledWith(getMaybeRFactorDomain(), contig));
  }

  const std::vector<std::optional<bool>>& getContiguity() {
    return domain()->contiguity();
  }

  bool hasReduction() const {
    return domain()->hasReduction();
  }

  bool hasBlockReduction() const {
    return domain()->hasBlockReduction();
  }

  bool hasGridReduction() const {
    return domain()->hasGridReduction();
  }

  bool hasBroadcast() const {
    return domain()->hasBroadcast();
  }

  bool hasRFactor() const {
    return domain()->hasRFactor();
  }

  bool hasAllocation() const {
    return domain()->hasAllocation();
  }

  //! Returns true if this tensor is zero dimensional,
  //!  i.e. a wrapped scalar or an empty placeholder.
  bool isZeroDim() const {
    return nDims() == 0;
  }

  //! Returns true if this tensor does not contain
  //!  any value.
  bool isEmptyTensor() const;

  std::optional<unsigned int> getReductionAxis() const {
    return domain()->getReductionAxis();
  }

  const std::vector<IterDomain*>& getRootDomain() const {
    return domain()->root();
  };

  const std::vector<IterDomain*>& getRFactorDomain() const {
    return domain()->rfactor();
  };

  const std::vector<IterDomain*>& getAllocationDomain() const {
    return domain()->allocation();
  };

  const std::vector<IterDomain*>& getLeafDomain() const {
    return domain()->leaf();
  };

  // If rfactor domain exists in domain() return it, otherwise return root
  // domain.
  const std::vector<IterDomain*>& getMaybeRFactorDomain() const {
    return domain()->maybeRFactor();
  };

  // If allocation domain exists in domain() return it, otherwise return
  // getMaybeRFactorDomain()
  const std::vector<IterDomain*>& getMaybeAllocationDomain() const {
    return domain()->maybeAllocation();
  };

  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      std::vector<std::optional<bool>> new_contiguity) {
    domain()->setAllocationDomain(
        std::move(new_allocation_domain), std::move(new_contiguity));
  }

  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      bool new_contiguity) {
    domain()->setAllocationDomain(
        std::move(new_allocation_domain), new_contiguity);
  }

  IterDomain* axis(int pos) const;

  // Does it share outer axes with other tensors?
  bool hasComputeAt() const {
    return compute_at_pos_ > 0;
  }

  bool hasMaxProducerPosition() const {
    return max_producer_pos_ > 0;
  }

  size_t nDims() const {
    return domain()->nDims();
  }

  // sets cpu_scalar_ value, which is special handling for CPU based zero-dim
  // tensors (i.e. CPU Tensors that only have one value). This is only used if
  // on an input value, otherwise ignored. This is important as special handling
  // because these "scalars" should be type promoted as a tensor, but we want to
  // avoid explicit copying of the data, so we want to pass the data value as a
  // standard kernel argument value.
  void setCpuScalar(bool is_cpu_scalar);

  // returns cpu_scalar_ value, which is special handling for CPU based zero-dim
  // tensors (i.e. CPU Tensors that only have one value). This is only used if
  // on an input value, otherwise ignored. This is important as special handling
  // because these "scalars" should be type promoted as a tensor, but we want to
  // avoid explicit copying of the data, so we want to pass the data value as a
  // standard kernel argument value.
  bool isCpuScalar() const {
    return cpu_scalar_;
  }

  // Returns the position that this tensor is produced at relative to its axes.
  unsigned int getComputeAtPosition() const {
    return compute_at_pos_;
  }

  // Returns the maximum position of producers are being computed at relative to
  // this tensor. This position dictates the clear expectations of producers.
  unsigned int getMaxProducerPosition() const {
    return max_producer_pos_;
  }

  unsigned int getMaybeMaxProducerPosition() const {
    return maybe_max_producer_pos_;
  }

  //! This is used when we disconnect a tensorview from a reduction
  //!  operation and connect it to a non-reduction operator. We need
  //!  to remove the reduction ids on the tv in this case.
  //! Currently only used in translate welford, and this function may
  //!  be refactored or extended if any more use cases appear.
  void clearReductionIterDomains();

  //! Compute this TensorView relative to a consumer position, -1 will
  //! compute tensors inline with each other, 0 doesn't share
  //! any loop nests between the tensors. It's an error when the given
  //! position is not legally viable. Alternatively, when the mode
  //! parameter is ComputeAtMode::BestEffort, the position is lowered
  //! one by one until a valid position is found. When
  //! ComputeAtMode::MostInlined is given, the position parameter is
  //! ignored, and the deepest possible position is searched.
  TensorView* computeAt(
      TensorView* consumer,
      int position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  // Split "axis" into 2 axes
  //! inner_split dictates if the factor section of the split should be inside
  //! the
  //! remainer or outside.
  //! e.g. split(0, 4, inner_split = true) will result in:
  //! tv[id{extent}] -> tv[id{ceilDiv(extent, factor)}, id{factor}]
  //! e.g. split(0, 4, inner_split = false) will result in:
  //! tv[id{extent}] -> tv[id{factor}, id{ceilDiv(extent, factor)}]
  //!
  //! When trim_out_of_bounds is true, only the inner domain defined by the
  //! start and stop positions is split.
  TensorView* split(
      int axis,
      unsigned int factor,
      bool inner_split = true,
      bool trim_out_of_bounds = false);

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor. Factor can be a symbolic
  // value instead of constant. This requires setting the symbolic value as an
  // input, or using a parallel dim from NamedScalar::getParallelDim
  TensorView* split(
      int axis,
      Val* factor,
      bool inner_split = true,
      bool trim_out_of_bounds = false);

  // Merge axis_o and axis_i into 1 IterDomain
  TensorView* merge(int axis_o, int axis_i);

  // Merge axis and axis+1 into 1 IterDomain
  TensorView* merge(int axis) {
    return merge(axis, axis + 1);
  }

  // Reorder axes according to old2new[old_pos] = new_pos
  TensorView* reorder(const std::unordered_map<int, int>& old2new);

  //! Swizzle the rectangular tile defined by the iterdomains corresponding
  //!  to the 2 given indices.
  TensorView* swizzle(
      Swizzle2DType swizzle_type,
      int x,
      int y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  // WARNING: rFactor does not return this TensorView, ir returns a new
  //  tensorview consumed by this!
  //
  // Take reduction axes out of this domain, and create a new
  // domain. New domain will be used to create this domain.
  //
  // For example:
  //  TV1[I0, R1, R2, I3] = TV0[I0, I1, I2, I3]
  //
  // After:
  //  TV1->rfactor({1}), TV1 is transformed to -> TV1[I0, R2, I3]
  //
  // The TensorView returned is: TV2[I0, R1, I2, I3]
  //
  // The reduction will now beset as:
  //  TV2[I0, R1, I2, I3] = TV0[I0, I1, I2, I3]
  //  TV1[I0, R2, I3] = TV2[I0, R1, I2, I3]
  //
  TensorView* rFactor(const std::vector<int>& axes);

  //! Multi-output version of rFactor, semantically similar with
  //! the reduction version except that the rfactor is done
  //! for all outputs in a consistent way
  std::vector<TensorView*> rFactor(
      const std::vector<int>& axes,
      const std::vector<TensorView*>& tvs);

  //! Create a TensorView before the original tensor. A common use case is to
  //! write results into shared memory or registers before moving to global
  //! memory. Analogous to TVM Cache_Write
  //!
  //! @param op_type: memory operator to use for the inserted op between
  //!   the the data tensor and the cache tensor
  TensorView* cacheBefore(LoadStoreOpType op_type = LoadStoreOpType::Set);

  //! Create a TensorView after the original tensor. A common use case is to
  //! read tensor into shared memory or registers. Analogous to TVM Cache_Read
  //!
  //! @param op_type: memory operator to use for the inserted op between
  //!   the the data tensor and the cache tensor
  TensorView* cacheAfter(
      LoadStoreOpType op_type = LoadStoreOpType::Set,
      CacheOp cache_op = CacheOp::Unspecified);

  // For a fusion output with other uses, we want to avoid writing to global
  // memory and then reading the output again. We write to global memory
  // separately after an operation. We replace this fusion output with the
  // direct write TensorView.
  TensorView* cacheFork();

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  void setMemoryType(MemoryType mt);

  // Apply double buffering transformation
  void doubleBuffer();

  // Apply circular buffering transformation
  void circularBuffer(unsigned int number_of_stage);

  // Returns true if this tensor is double buffered.
  bool isDoubleBuffered() const {
    return is_double_buffered_;
  }

  // Returns true if this tensor is circular buffered.
  bool isCircularBuffered() const {
    return is_circular_buffered_;
  }

  // Returns the depth of circular buffering if applicable.
  unsigned int circularBufferDepth() const {
    NVF_ERROR(is_circular_buffered_, toString(), "not circular buffered");
    return circular_buffer_stage_;
  }

  //! Transforms the innermost iterdomains according to the given mma swizzle,
  //!  this should be used on the tvs that are either inputs/outputs of an
  //!  MmaOp, or any tv's that are involved in prolog/epilog fusions and need to
  //!  have a matching thread swizzle with the mma operand/result.
  //! More detail on usage see [WarpMmaSwizzler] in scheduler/mma_utils.h .
  void applyMmaSwizzle(MmaOptions options);

  //! Returns if this tensor view has swizzle operator on its tensor domain.
  //!  This is the temporary flag for indicating that the new swizzle
  //!  implementation is used and will be removed in follow ups.
  bool hasSwizzleOp() const {
    return has_swizzle_op_;
  }

  friend TransformPropagator;
  friend MostInlinedTransformPropagator;
  friend TransformReplay;
  friend OptOutMutator;
  friend class InlineBatchingGuard;
  friend class ir_utils::TVDomainGuard;

  // Inline the computation of this tensor into its consumer at the given
  // position. If this tensor is already inlined in a higher position, then this
  // call is a no-op. If the right most dimensions before `pos` are
  // broadcasting, then will not inline into these broadcastings. If
  // best_effort, then will inline into the highest allowed position that is <=
  // `pos`.
  void inlineAt(
      int64_t pos,
      bool best_effort = false,
      MaxPosCalculator* calc = nullptr);

  //! Inline the computation of this tensor into a consumer at the given
  //! position. The consumer to compute with is determined when the
  //! fusion is lowered. Specifically, it is the first consumer tensor
  //! in the topologically ordered dependency graph. Before the
  //! lowering, its compute-with consumer is considered unresolved,
  //! which is then resolved by resolveComputeWith below.
  //!
  //! The position is relative to its own domain. It is an
  //! error if the position is smaller than the compute-at position. If this
  //! tensor is already inlined in a higher position with the same
  //! consumer, then this call is a no-op. The actual position is
  //! computed in the same way as inlineAt, except that computeWith
  //! does not have the constraint of the persistent data-dependency pattern.
  void computeWith(int pos, bool best_effort = false);

  //! Set the actual consumer tensors that this tensor is
  //! computed with. Requires a topologically sorted list expressions,
  //! which can be obtained reorderExprsForComputeAt. Return true if
  //! resolution is actually done. This should only be done in the
  //! Kernel container.
  bool resolveComputeWith(const std::vector<Expr*>& sorted_exprs);

  bool hasComputeWith() const {
    return getComputeWithPosition() > getComputeAtPosition();
  }

  bool hasResolvedComputeWith() const {
    return !compute_with_consumers_.empty();
  }

  //! Query if this tensor is computed with a given consumer.
  bool isComputedWith(const TensorView* consumer) const;

  //! Return the tensors with which this tensor is computed. It is an
  //! error to use this function without first resolving computeWith.
  const std::vector<TensorView*>& getComputeWithConsumers() const;

  unsigned int getComputeWithPosition() const {
    return compute_with_pos_;
  }

  unsigned int getMaxComputePosition() const {
    return std::max(getComputeWithPosition(), getComputeAtPosition());
  }

  //! Returns the position that this tensor is produced at for a given
  //! consumer. If this tensor is computed with the given consumer,
  //! which also means its computeWith needs to have been resolved, the
  //! computeWith position is returned. Otherwise, the default computeAt
  //! position is retured.
  unsigned int getComputePosition(const TensorView* consumer) const;

  // Update the max producer position of the current tensor. This is required
  // when we modify producer-consumer relationship of a scheduled tensor, for
  // example, grouping multiple reductions.
  void updateMaxProducerPosition();

  // Commit the current changes in leaf domain into rFactor domain. This
  // function can be used to do implicit transpose and view, but today, only
  // implicit transpose is being tested. This function can be dangerous: it
  // changes the the semantics of the current tensor without updating its
  // consumers consistently, and there is no reliable way to detect this
  // inconsistency. It is the responsibility of the caller of this function to
  // ensure consistency.
  void commitLeafToRFactor();

  //! Request that we reclaim the memory of this tv before any subsequent
  //! tensors are allocated.
  //!
  //! This method influences the shared memory allocator that assigns shared
  //! memory addresses at lowering. It ensures that the proper synchronization
  //! is present in the kernel to reuse memory and inserts new block
  //! synchronizations if necessary.
  void promoteReuse(bool b = true) {
    NVF_CHECK(
        memory_type_ == MemoryType::Shared,
        "promoteReuse should only be called on shared memory tensors");
    promote_reuse_ = b;
  }

  //! Returns whether we should insert syncs if needed in order to reuse the
  //! memory of this tensor.
  bool shouldPromoteReuse() const {
    return promote_reuse_;
  }

 protected:
  void setDomain(TensorDomain* td) {
    domain_ = td;
  }

 private:
  int64_t normalizeAxisPos(int64_t pos) const {
    if (pos < 0) {
      pos += (int64_t)nDims();
    }
    return pos;
  }

  //! A helper function to maintain the consistency of schedules of
  //! multiple outputs wheen doing rfactor on multi-output reduction ops.
  TensorView* multiOutputRfactorHelper(
      TensorView* tv,
      const std::vector<int>& axes);

  void clearComputeWith();

 private:
  TensorDomain* domain_ = nullptr;
  unsigned int compute_at_pos_ = 0;
  unsigned int max_producer_pos_ = 0;
  MemoryType memory_type_ = MemoryType::Local;
  bool is_double_buffered_ = false;

  //! Indicates if the tensor is circular buffered.
  bool is_circular_buffered_ = false;

  //! Indicates the circular buffering stage depth if applicable.
  unsigned int circular_buffer_stage_ = 0;

  // special handling for CPU based zero-dim tensors (i.e. CPU Tensors that
  // only have one value). This is only used if on an input value, otherwise
  // ignored. This is important as special handling because these "scalars"
  // should be type promoted as a tensor, but we want to avoid explicit
  // copying of the data, so we want to pass the data value as a standard
  // kernel argument value.
  bool cpu_scalar_ = false;

  //! Indicates if this tensor view has swizzle operator on its tensor domain.
  //!  This is the temporary flag for indicating that the new swizzle
  //!  implementation is used and will be removed in follow ups.
  bool has_swizzle_op_ = false;

  //! Direct consumer tensors that this tensor is computed with
  std::vector<TensorView*> compute_with_consumers_;

  //! Position where this tensor is computed with the compute-with
  //! consumer tensors. It should be always be equal or greater than
  //! the computeAt position
  unsigned int compute_with_pos_ = 0;

  //! Maximum position where producers may be computed at, including
  //! unresolved computeWith. This is equal to max_producer_pos_ when
  //! no producer has unresolved computeWith. It is only used before
  //! resolving computeWith so that no IterDomain should never be
  //! transformed when there may actually be a producer tensor that
  //! may be computed at.
  unsigned int maybe_max_producer_pos_ = 0;

  //! When this is true, it indicates, if this is a shared memory tensor and
  //! there other shared memory tensors whose lifetimes do not overlap and come
  //! later than this tensor's lifetime, that we should ensure that thread
  //! blocks are synchronized such that all threads have performed their last
  //! read of this tensor (or any tensors aliasing in) before writing to the
  //! current tensor. This will then allow us to safely reuse the memory
  //! allocated to this tensor.
  bool promote_reuse_ = false;
};

//! A simple TensorView builder
//!
//! Example usage:
//!
//!   auto tv = TensorViewBuilder()
//!       .ndims(ndims)
//!       .dtype(dtype)
//!       .contiguity(contiguity)
//!       .build();
//!
class TensorViewBuilder {
 public:
  //! Set the number of dimensions of the tensor (default 0, meaning scalar)
  TensorViewBuilder& ndims(size_t ndims);

  //! Set the data type of the tensor (default DataType::Float)
  TensorViewBuilder& dtype(DataType dtype);

  //! Set the contiguity information (default non-contiguous)
  TensorViewBuilder& contiguity(std::vector<std::optional<bool>> contiguity);
  TensorViewBuilder& contiguity(bool contiguity);

  //! Set the shape (default 0 dimensional, ie. scalar)
  TensorViewBuilder& shape(std::vector<Val*> shape);
  TensorViewBuilder& shape(const std::vector<int64_t>& shape);

  //! Set if a dimension is expanded
  TensorViewBuilder& expanded(std::vector<bool> expanded);

  //! Creates a new TensorView with the specified options
  TensorView* build() const;

 private:
  size_t ndims_ = 0;
  DataType dtype_ = DataType::Float;

  // contiguity_ is the vector that you will pass to the constructor of
  // TensorDomain. However, constructing this vector can be non-trivial, because
  // it is required to be nullopt for broadcast dimensions. We often want to
  // create contiguity vector that represents all contiguous or all
  // discontiguous. uniform_contiguity_ is there to make this use case more
  // convenient. If set, then TensorViewBuilder will automatically fill the
  // contiguity with the value of uniform_contiguity_ where it is not required
  // to be nullopt. Note that you can only set one of contiguity_ or
  // uniform_contiguity_.
  std::vector<std::optional<bool>> contiguity_;
  std::optional<bool> uniform_contiguity_ = std::nullopt;

  std::vector<Val*> shape_;
  std::vector<bool> expanded_;
};

} // namespace nvfuser
