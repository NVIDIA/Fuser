// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <optional>
#include <ranges>

#include <exceptions.h>
#include <ir/base_nodes.h>

//! IR header hierarchy
//! 1. utils.h - PolymorphicBase and NonCopyable
//! 2. ir/base_nodes.h - Statement, Expr, and Val
//! 3. ** ir/internal_base_nodes.h ** - IterDomain and TensorDomain
//! 4. ir/interface_nodes.h - TensorView and Scalar
//! 5. ir/internal_nodes.h - Any internal-only IR nodes

namespace nvfuser {

// Friends for direct access to split
class TensorDomain;
class IterDomain;
class RaggedIterDomain;
class TensorView;
class ReplayTransformations;
class IndexReferenceReplay;
class ViewTransform;
class Scope;
class IrCloner;
struct AnalyzeViewResult;

// Convenience utility to initialize IterDomain's without having to sort through
// all the default values. Intended to be used with
// IterDomain::IterDomain(IrBuilderPasskey, IterDomainBuilder).
class IterDomainBuilder {
 public:
  // Match legacy constructor
  IterDomainBuilder(Val* _start, Val* _extent);

  // Grab all the parameters from id to set the IterDomainBuilder
  IterDomainBuilder(const IterDomain* id);

  // Resets defaults for rfactor, is padded dim, padded to size, and is mma
  // swizzle which should only be set during scheduling.
  IterDomainBuilder& resetSchedulingParams();

  // Resets is_rfactor_domain
  IterDomainBuilder& resetRfactor();

  IterDomainBuilder& start(Val* _start);
  IterDomainBuilder& extent(Val* _extent);
  IterDomainBuilder& expanded_extent(Val* _expanded_extent);
  IterDomainBuilder& stop_offset(Val* _stop_offset);
  IterDomainBuilder& parallel_type(ParallelType _parallel_type);
  IterDomainBuilder& iter_type(IterType _iter_type);
  IterDomainBuilder& is_rfactor_domain(bool _is_rfactor_domain);
  IterDomainBuilder& is_padded_dimension(bool _is_padded_dimension);
  IterDomainBuilder& padded_to_size(std::optional<int64_t> _padded_to_size);

  IterDomain* build() const;

  // Must have start and extent at least
  IterDomainBuilder() = delete;

  Val* start_ = nullptr;
  Val* extent_ = nullptr;
  Val* expanded_extent_ = nullptr;
  Val* stop_offset_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;

  // Only relevant at scheduling time or compile time.
  bool is_rfactor_domain_ = false;
  bool is_padded_dimension_ = false;
  bool is_clustered_dimension_ = false;
  std::optional<int64_t> padded_to_size_ = std::nullopt;
};

//! Simply a representation of an annotated 1D iterable from start to extent.
//! TensorDomains which represent how to iterate over a tensor is made up of
//! IterDomains to form an ND iterable. We directly set parallization strategies
//! on IterDomains.
class NVF_API IterDomain : public Val {
 public:
  IterDomain(IrBuilderPasskey, const IterDomainBuilder& args);

  // Legacy constructor, TODO: should start moving to use the IterDomainBuilder
  // constructor. Same as the above but can set the offset of the stop point.
  IterDomain(
      IrBuilderPasskey,
      Val* start,
      Val* extent,
      Val* expanded_extent,
      Val* stop_offset,
      ParallelType parallel_type,
      IterType iter_type,
      bool is_rfactor_domain,
      bool is_padded_dimension,
      bool is_clustered_blocks,
      std::optional<int64_t> padded_to_size);

  IterDomain(const IterDomain* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool sameDefinition(const Val* other) const override;

  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  //! Returns a new IterDomain matching properties of this
  //!
  //! This does NOT copy the is_rfactor_domain flag.
  //!
  //! When map_with_original is true, the clone of the original is
  //! mapped in the Exact graph.
  IterDomain* cloneWithoutRFactor(bool map_with_original = false);

  //! Clone a vector domains
  static std::vector<IterDomain*> clone(
      const std::vector<IterDomain*>& domains);

  //! The optional parameters of rfactor_domain and iter_type can be
  //! used to override the default behavior.
  static IterDomain* merge(
      IterDomain* outer,
      IterDomain* inner,
      std::optional<bool> rfactor_domain = std::nullopt,
      std::optional<IterType> iter_type = std::nullopt);

  //! The optional parameters of rfactor_domain, outer_iter_type and
  //! inner_iter_type can be used to override the default behavior.
  static std::pair<IterDomain*, IterDomain*> split(
      IterDomain* in,
      Val* factor,
      bool inner_split,
      std::optional<bool> rfactor_domain = std::nullopt,
      std::optional<IterType> outer_iter_type = std::nullopt,
      std::optional<IterType> inner_iter_type = std::nullopt);

  //! Resize an IterDomain by expanding both the left and right sides
  //! by given widths. The resulting IterDomain has an extent of
  //! (left_expansion + in->extent() + right_expansion). Note that the
  //! expansion factors can be negative, meaning the input IterDomain
  //! is shrunk. This is the case when resize is used to represent
  //! slice.
  //!
  //! When mark_as_rfactor is true, the output IterDomain
  //! is marked as an rfactor domain. For example, expressions such as
  //! PadOp and SliceOp resize IterDomains and generate rfactor
  //! resized domains.
  //!
  //! Usually, the IterType of the output IterDomain will be Symbolic. This is
  //! because unless the left and right expansions are known at Fusion
  //! definition we cannot be sure that the output will have an extent != 1. In
  //! case the output extent is in fact 1, we will set the IterType to
  //! Broadcast. If the left and right expansions are constant, and sum to at
  //! least two, then even an empty input will result in an Iteration IterType.
  //! In these cases, we will set the output IterType to Iteration at
  //! definition. Otherwise, it will be set to Symbolic and will be resolved
  //! when concretization is performed by FusionExecutorCache.
  //!
  //! The optional iter_type argument can be used to force the output IterType,
  //! but for safety its use should typically be confined to concretization.
  static IterDomain* resize(
      IterDomain* in,
      Val* left_expansion,
      Val* right_expansion,
      bool mark_as_rfactor = false,
      std::optional<IterType> iter_type = std::nullopt);

  bool isReduction() const {
    return getIterType() == IterType::Reduction;
  }

  bool isIteration() const {
    return getIterType() == IterType::Iteration;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return getIterType() == IterType::Broadcast;
  }

  bool isSymbolic() const {
    return getIterType() == IterType::Symbolic;
  }

  bool isGatherScatter() const {
    return getIterType() == IterType::GatherScatter;
  }

  bool isStride() const {
    return getIterType() == IterType::Stride;
  }

  bool isVectorComponent() const {
    return getIterType() == IterType::VectorComponent;
  }

  bool isParallelized() const {
    return getParallelType() != ParallelType::Serial;
  }

  //! Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return isParallelTypeBlockDim(getParallelType());
  }

  //! Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return isParallelTypeThreadDim(getParallelType());
  }

  //! Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  bool isDeviceDim() const {
    return isParallelTypeDeviceDim(getParallelType());
  }

  bool isStream() const {
    return getParallelType() == ParallelType::Stream;
  }

  NVF_API void parallelize(ParallelType t);

  ParallelType getParallelType() const {
    return parallel_type_;
  }

  IterType getIterType() const {
    return iter_type_;
  }

  Val* start() const {
    return start_;
  }

  Val* stop() const;

  Val* stopOffset() const;

  Val* extent() const {
    NVF_ERROR(extent_ != nullptr);
    return extent_;
  }

  bool hasExpandedExtent() const {
    return expanded_extent_ != nullptr;
  }

  // Returns the expanded extent of a strided broadcast entry.
  Val* expandedExtent() const {
    NVF_ERROR(
        hasExpandedExtent(),
        "Requested expanded extent, but none found on this dimension.");
    return expanded_extent_;
  }

  Val* getMaybeExpandedExtent() const {
    if (hasExpandedExtent()) {
      return expandedExtent();
    }
    return extent();
  }

  //! Dimension padding interface:
  //!  2 modes are currently supported:
  //!
  //!   - mode 1: if to_size is given as a positive number,
  //!      the dimension will be padded to the size so that
  //!      this iterdomain will be compile-time constant
  //!      size and it is the scheduler's responsibility
  //!      to ensure no input larger than the padded size
  //!      will be observed
  //!
  //!   - mode 2: if no to_size is given, this dimension
  //!      is "dynamically" padded to next smallest multiple
  //!      of a warp size, i.e. 17 padded to 32, 33 padded to 64
  //!      based on the given input.
  void padToMultipleOfWarp(std::optional<int64_t> maybe_to_size = {}) {
    // Currently only restricted to TIDx to generate warp reduce
    NVF_CHECK(
        parallel_type_ == ParallelType::TIDx,
        "padToMultipleOfWarp : warp padding only supported on TIDx parallel "
        "dimension");
    is_padded_dimension_ = true;
    if (maybe_to_size.has_value()) {
      if (maybe_to_size.value() > 0) {
        padded_to_size_ = maybe_to_size.value();
      }
    }
  }

  //! Indicates if this iterdomain had padding
  //!  dynamical or statical
  bool hasPaddingToMultipleOfWarp() const {
    return is_padded_dimension_;
  }

  //! Sets whether this IterDomain uses CUDA thread block clusters (Hopper+).
  void setClusteredBlocks() {
    NVF_CHECK(
        parallel_type_ == ParallelType::BIDx,
        "setClusteredBlocks: only support set BIDx parallel type");
    is_clustered_dimension_ = true;
  }

  //! Returns whether this IterDomain uses clustered blocks.
  bool isClusteredBlockDim() const {
    return is_clustered_dimension_;
  }

  //! Returns a concrete value if this iterdomain
  //!  has been padded to a statical size.
  std::optional<int64_t> getMaybeSizeAfterPadding() const {
    return padded_to_size_;
  }

  //! True if range of iteration domain isn't across the full extent
  bool maybePartial() const;

  //! Check if IterDomain is a broadcast axis with compile-time
  //! known extent. This is the case with all size-1 IterDomains on
  //! a TensorView's root domain when the TensorView is created.
  bool isImplicitBroadcast() const {
    return isBroadcast() && extent()->isOneInt();
  }

  //! Split for stride by a given factor. It effectively does an inner
  //! split by the factor and sets the inner domain as a Stride
  //! domain.
  std::pair<IterDomain*, IterDomain*> stridedSplit(int64_t factor);

  //! Marks that this id represents a
  //!  instruction loop, mma use only.
  //!
  //! An instruction loop can be considered a generalization of
  //!  vectorization. It also represents a loop that's implemented
  //!  by an instruction and should not be realized by codegen and
  //!  cannot be inlined with.
  //! As an example, if a mma macro, call it mma_eg implements:
  //!  for m in M
  //!    for n in N
  //!      for k in K
  //!         C[m,n] += A[m,k]*B[k,n],
  //! But the generated code should simply be:
  //!  mma_eg(C,A,B)
  //! without the 3 level loopnest, i.e. they're instruction loops.
  //!
  //! In the actual mma macros, the loopnests it implements is a
  //!  transformed version of above to match the mma swizzle.
  //!  So it's different implicit loopnest for different macros.
  //!  MmaSwizzler will label the instruction loops case-by-case.
  bool isMma() const {
    return parallel_type_ == ParallelType::Mma;
  }

  //! Marks that this id represents an instruction loop, cp.async.bulk use only.
  bool isBulk() const {
    return parallel_type_ == ParallelType::Bulk;
  }

  //! Applies 2D swizzle on a rectangular tile defined by
  //!  a pair of iterdomains.
  static std::pair<IterDomain*, IterDomain*> swizzle(
      SwizzleType swizzle_type,
      IterDomain* in_x,
      IterDomain* in_y);
  static std::pair<IterDomain*, IterDomain*> swizzle(
      Swizzle2DType swizzle_type,
      IterDomain* in_x,
      IterDomain* in_y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

 protected:
  friend TensorDomain;
  friend ReplayTransformations;
  friend IndexReferenceReplay;
  friend RaggedIterDomain;

  //! Protected constructor for derived classes (e.g., RaggedIterDomain)
  //! that need to override the ValType
  IterDomain(
      IrBuilderPasskey passkey,
      ValType vtype,
      Val* start,
      Val* extent,
      Val* expanded_extent,
      Val* stop_offset,
      ParallelType parallel_type,
      IterType iter_type,
      bool is_rfactor_domain,
      bool is_padded_dimension,
      bool is_clustered_blocks,
      std::optional<int64_t> padded_to_size);

 private:
  //! Valid range is defined as [start:-stop_offset]
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;

  // Broadcast dimensions are assumed to be size 1 for the sake of code
  // generation. If a user though calls `expand` on a tensor that dimension is
  // still considered a broadcast dimension. However if we ever output that
  // dimension it should be a size dictated by the `expand` operation, and have
  // a stride of zero. Since this extent is important to track, but not
  // necessarily generate code for (still want loops on broadcast to be of size
  // 0), we simply store it separately from extent_. Having an expanded_extent_
  // is only allowed with broadcasted dimsneions. Only in this instance does it
  // make sense to have an expanded_extent_, because it's used when users are
  // expecting return tensors to have a physical domain. If a user simply
  // "broadcasts" an operation
  Val* const expanded_extent_ = nullptr;

  //! Distance of stop from the end
  Val* const stop_offset_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;
  bool is_rfactor_domain_ = false;
  bool is_padded_dimension_ = false;
  bool is_clustered_dimension_ = false;
  std::optional<int64_t> padded_to_size_ = std::nullopt;
};

//! RaggedIterDomain represents a dimension with variable extents
//! (ragged/jagged dimension). Used for PyTorch nested tensors.
//! Unlike IterDomain, the extent varies per component
//! and is stored as a TensorView rather than a single Val.
//!
//! Key properties:
//! - extents_: TensorView containing extent for each component (1D, 2D, or N-D)
//! - Uniform execution properties: ParallelType, IterType apply to all
//! components
class NVF_API RaggedIterDomain : public IterDomain {
 public:
  //! \param extents TensorView containing component extents (must be integer
  //! type)
  //! \param iter_type Iteration type (Iteration, Reduction, etc.)
  //! Only Iteration is allowed ATM.
  //! \param parallel_type Parallelization strategy (applies
  //! uniformly)
  RaggedIterDomain(
      IrBuilderPasskey passkey,
      TensorView* extents,
      IterType iter_type = IterType::Iteration,
      ParallelType parallel_type = ParallelType::Serial);

  //! Cloning constructor for IR cloning
  RaggedIterDomain(const RaggedIterDomain* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  //! Accessor for the extents tensor
  TensorView* extents() const {
    return extents_;
  }

  //! Partition an IterDomain into component and ragged dimensions
  //! Creates a component IterDomain and a RaggedIterDomain based on extents
  //!
  //! \param in Input IterDomain to partition (must be regular IterDomain)
  //! \param extents Extents tensor defining the size of each component (must be
  //! 1D)
  //!        Shape: [num_components], values: [extent0, extent1, ...,
  //!        extent(n-1)]
  //! \return Pair of (component_id, ragged_id)
  //!         component_id: IterDomain with extent = num_components
  //!         ragged_id: RaggedIterDomain with the provided extents
  //!
  //! TODO: Support multi-dimensional extents for nested ragged structures
  static std::pair<IterDomain*, RaggedIterDomain*> partition(
      IterDomain* in,
      TensorView* extents);

 private:
  //! Extent tensor containing all component extents
  //! Can be 1D, 2D, or N-D depending on nesting structure
  TensorView* extents_ = nullptr;
};

//! TensorDomain holds a vector of IterDomains. It holds an IterDomain for every
//! logical axis in its associated tensor. TensorDomain does not directly hold
//! the Tensor it is associated with, and in theory could be associated with
//! multiple tensors. TensorDomain's primary responsibility is to provide a
//! mechanism to access history of transformations that were used to generate
//! it. This is done through the normal interaction of Expr/Val in Fusion. i.e.
//! if we want to know the previous operation generating a particular
//! TensorDomain we can simply call:
//!
//!     FusionGuard::getCurFusion()->definition(a_tensor_domain)
//!
//! which should give us an operation in the list [split, merge] or similar
//! operations that take in a TensorDomain, applies a transformation and outputs
//! a tensor domain.
class NVF_API TensorDomain : public Val {
 public:
  explicit TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> logical_domain,
      std::vector<std::optional<bool>> contiguity = {});

  // See notes [ Note stride order and contiguity vector ] in
  // python_bindings.cpp
  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> logical_domain,
      std::vector<int64_t> stride_order,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> logical_domain,
      std::vector<IterDomain*> loop_domain,
      std::vector<std::optional<bool>> contiguity = {},
      bool skip_loop_validation = false);

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> logical_domain,
      std::vector<IterDomain*> loop_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> logical_domain,
      std::vector<IterDomain*> allocation,
      std::vector<IterDomain*> loop_domain,
      std::vector<std::optional<bool>> contiguity = {},
      std::vector<IterDomain*> additional_ids = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> logical_domain,
      std::vector<IterDomain*> allocation,
      std::vector<IterDomain*> loop_domain,
      std::optional<std::vector<IterDomain*>> alternate_loop_domain,
      std::vector<std::optional<bool>> contiguity = {},
      std::vector<IterDomain*> additional_ids = {},
      bool skip_validation = false);

  TensorDomain(IrBuilderPasskey, const TensorDomain* src);

  TensorDomain(const TensorDomain* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool operator==(const TensorDomain& other) const;
  bool operator!=(const TensorDomain& other) const {
    return !(*this == other);
  }

  int64_t nDims() const {
    return static_cast<int64_t>(loop_domain_.size());
  }

  bool sameDefinition(const Val* other) const override;

  bool sameAs(const Statement* other) const override;

  static bool sameAs(
      const std::vector<IterDomain*>& lhs,
      const std::vector<IterDomain*>& rhs);

  // When `loop_only` is false, prints also the root, logical and allocation
  // domain if not empty.
  std::string toString(int indent_size, bool loop_only) const;
  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  // Note: [Contiguity]
  // Contiguity is a vector of optional<bool> which has the same number of
  // elements as logical_domain_. The contiguity of a broadcast dimension is
  // meaningless, so it has to be nullopt. The contiguity of a non-broadcasting
  // dimension is true if and only if it is memory dense with the next
  // non-broadcasting dimension.
  // For example, if I have a tensor torch.zeros(4, 1, 3).expand(-1, 10, -1),
  // the contiguity will be (true, nullopt, true), which means 4 is memory dense
  // with 3.
  const std::vector<std::optional<bool>>& contiguity() const {
    return contiguity_;
  }

  // The python frontend has a stride_order argument in the define_tensor
  // function. This argument allows the user to specify the allocation domain
  // for the TensorView. When translating the CPP Fusion into a Python
  // FusionDefinition, the stride_order argument is required if this
  // TensorDomain's allocation domain is a permutation of the logical domain.
  // This function generates the stride_order argument for this TensorDomain.
  std::vector<int64_t> strideOrder() const;

  void setContiguity(const std::vector<std::optional<bool>>& contig);

  std::string getContiguityString() const {
    return toDelimitedString(contiguity(), /*delim=*/" ");
  }

  bool hasReduction() const;

  bool hasBlockReduction() const;
  bool hasClusterReduction() const;
  bool hasGridReduction() const;
  bool hasBlockBroadcast() const;
  bool hasGridBroadcast() const;

  bool hasBroadcast() const;

  bool hasRoot() const {
    return !root_domain_.empty();
  }

  bool hasAllocation() const {
    return !allocation_domain_.empty();
  }

  // Returns if rfactor domain only consists of id's of iter type.
  bool hasViewLikeRFactor() const;

  bool hasVectorize() const;

  bool hasSymbolicAxis() const;

  std::optional<int64_t> getReductionAxis() const;

  // The input logical domain. The root domain of a consumer should equal the
  // logical domain of its producer ignoring reduction dimensions.
  const std::vector<IterDomain*>& root() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& maybeRoot() const {
    return root_domain_.empty() ? logical_domain_ : root_domain_;
  };

  // Check if id is a root ID. Always return false if there's no root
  // domain.
  bool isRoot(const IterDomain* id) const {
    return hasRoot() &&
        std::find(root().begin(), root().end(), id) != root().end();
  }

  bool isMaybeRoot(const IterDomain* id) const {
    return (hasRoot() && isRoot(id)) || (!hasRoot() && isLogical(id));
  }

  // The output logical domain.
  const std::vector<IterDomain*>& logical() const {
    return logical_domain_;
  };

  // Check if id is a logical ID.
  bool isLogical(const IterDomain* id) const {
    return std::find(logical().begin(), logical().end(), id) != logical().end();
  }

  // The allocation domain. This describes how data is stored in memory in
  // outer-to-inner order.
  const std::vector<IterDomain*>& allocation() const {
    return allocation_domain_;
  }

  // Check if id is an allocation ID. Always return false if there's
  // no allocation domain.
  bool isAllocation(const IterDomain* id) const {
    return hasAllocation() &&
        std::find(allocation().begin(), allocation().end(), id) !=
        allocation().end();
  }

  // The loop domain after scheduling. This defines loop nests and loop indices.
  const std::vector<IterDomain*>& loop() const {
    return loop_domain_;
  }

  const std::optional<std::vector<IterDomain*>>& alternateLoop() const {
    return alternate_loop_domain_;
  }

  const std::vector<IterDomain*>& initialLoop() const {
    return initial_loop_domain_;
  }

  // Check if id is a loop ID.
  bool isLoop(const IterDomain* id) const {
    return std::find(loop().begin(), loop().end(), id) != loop().end();
  }

  // Check if id is an intial loop ID.
  bool isInitialLoop(const IterDomain* id) const {
    return std::find(initialLoop().begin(), initialLoop().end(), id) !=
        loop().end();
  }

  // Get all IDs that is on the shortest path between any of the domains
  // (logical domain, root domain, loop domain, allocation domain) following
  // definition and uses path. Return values are topologically ordered and
  // unique.
  std::vector<IterDomain*> allIDs() const;

  std::vector<const std::vector<IterDomain*>*> allDomains() const;

  // Similar to allIDs but returns all ID expressions.
  std::vector<Expr*> allExprs() const;

  // Combine allIDs and allExprs
  std::vector<Statement*> allStatements() const;

  const std::vector<IterDomain*>& maybeAllocation() const {
    return hasAllocation() ? allocation_domain_ : logical();
  };

  // Additional IDs that are not on the path from one of
  // root/logical/allocation/loop domain to another. We need to keep track of
  // these IDs to ensure that we can find all paths/IDs of interest.
  const std::vector<IterDomain*>& additionalIDs() const {
    return additional_ids_;
  }

  // Set the loop domain of this TensorDomain.
  void setLoopDomain(std::vector<IterDomain*> new_loop_domain);

  // Set the alternate loop domain of this TensorDomain.
  void setAlternateLoopDomain(std::vector<IterDomain*> new_loop_domain);

  // Set the allocation domain of this TensorDomain. Because contiguity is
  // always defined w.r.t. the allocation domain, the contiguity must be updated
  // accordingly.
  NVF_API void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      std::vector<std::optional<bool>> new_contiguity,
      bool skip_validation = false);

  // Similar to the previous one, but with new contiguity filled with all true
  // or all false.
  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      bool new_contiguity,
      bool skip_validation = false) {
    auto contiguity_flags =
        getContiguityFilledWith(new_allocation_domain, new_contiguity);
    setAllocationDomain(
        std::move(new_allocation_domain),
        std::move(contiguity_flags),
        skip_validation);
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  IterDomain* axis(int64_t i) const;

  int64_t posOf(IterDomain* id) const;

  //! Returns a position of a root domain
  int64_t rootPosOf(IterDomain* id) const;

  //! Create a new broadcast IterDomain with the given extent in the loop domain
  void broadcast(int64_t axis, Val* extent);

  // Split "axis" into 2 axes
  //! inner_split dictates if the factor section of the split should be inside
  //! the
  //! remainer or outside.
  //! e.g. split(0, 4, inner_split = true) will result in:
  //! tv[id{extent}] -> tv[id{ceilDiv(extent, factor)}, id{factor}]
  //! e.g. split(0, 4, inner_split = false) will result in:
  //! tv[id{extent}] -> tv[id{factor}, id{ceilDiv(extent, factor)}]
  void split(int64_t axis_, Val* factor, bool inner_split);

  // Merge axis_o and axis_i. axis_i is the fast changing dimension. Resulting
  // axis is by default placed at original position axis_o
  void merge(int64_t axis_o, int64_t axis_i);

  // Partition axis into component and ragged dimensions based on extents
  void partition(int64_t axis, TensorView* extents);

  // Reorder axes according to map[old_pos] = new_pos
  void reorder(const std::unordered_map<int64_t, int64_t>& old2new);

  //! Applies 2D swizzle on a rectangular tile defined by
  //!  a pair of iterdomains contained in this domain.
  void swizzle(SwizzleType swizzle_type, int64_t x, int64_t y);
  void swizzle(
      Swizzle2DType swizzle_type,
      int64_t x,
      int64_t y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  // Resize an axis by left_expansion and right_expansion
  void resize(
      int64_t axis,
      Val* left_expansion,
      Val* right_expansion,
      std::optional<IterType> iter_type = std::nullopt);

  // Transform TensorView according to merge and split transformations
  TensorDomain* view(const AnalyzeViewResult& view_analysis);

  TensorDomain* flatten(int64_t start_dim, int64_t end_dim);

  static std::vector<IterDomain*> orderedAs(
      const std::vector<IterDomain*>& td,
      const std::unordered_map<int64_t, int64_t>& old2new);

  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noDevices(const std::vector<IterDomain*>&);
  // Usage example: `domain | TensorDomain::kNoDevices`. Unlike noDevices, this
  // returns a view so is more efficient. However, make sure `domain` outlives
  // the view.
  inline static constexpr auto kNoDevices =
      std::views::filter([](IterDomain* id) { return !id->isDeviceDim(); });
  inline static constexpr auto kNoReductions = std::views::filter(
      [](IterDomain* id) { return !id->isReduction() && !id->isStride(); });
  inline static constexpr auto kNoBroadcasts =
      std::views::filter([](IterDomain* id) { return !id->isBroadcast(); });

  static bool hasBroadcast(const std::vector<IterDomain*>&);
  static bool hasReduction(const std::vector<IterDomain*>&);

  // Get a vector whose size is the number of IDs in the given logical_domain
  // filled with fill_value or nullopt depending on whether its corresponding ID
  // is broadcast.
  static NVF_API std::vector<std::optional<bool>> getContiguityFilledWith(
      const std::vector<IterDomain*>& allocation_domain,
      bool fill_value);

  // pair is in order where second is the consumer of first
  std::pair<TensorDomain*, TensorDomain*> rFactor(
      const std::vector<int64_t>& axes);

 private:
  int64_t wrapDim(int64_t dim) const {
    return nvfuser::wrapDim(dim, nDims());
  }

 private:
  const std::vector<IterDomain*> root_domain_;
  const std::vector<IterDomain*> logical_domain_;
  std::vector<IterDomain*> allocation_domain_;
  std::vector<IterDomain*> loop_domain_;
  std::optional<std::vector<IterDomain*>> alternate_loop_domain_;
  // Initial loop domain. Loop domain is updated with transformations
  // such as split, but the initial loop domain can only change with
  // setLoopDomain
  std::vector<IterDomain*> initial_loop_domain_;
  std::vector<IterDomain*> additional_ids_;

  std::vector<std::optional<bool>> contiguity_;
};

} // namespace nvfuser
