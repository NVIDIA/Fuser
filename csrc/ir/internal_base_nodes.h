// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/base_nodes.h>
#include <optional>

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
class ReplayTransformations;
class IndexReferenceReplay;
class ViewTransform;
class Scope;
class IrCloner;
struct AnalyzeViewResult;

// Convenience utility to initialize IterDomain's without having to sort through
// all the default values. Intended to be used with
// IterDomain::IterDomain(IrBuilderPasskey IterDomainBuildArgs)
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
  IterDomainBuilder& is_mma_swizzled(bool _is_mma_swizzled);

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
  std::optional<int64_t> padded_to_size_ = std::nullopt;
  bool is_mma_swizzled_ = false;
};

//! Simply a representation of an annotated 1D iterable from start to extent.
//! TensorDomains which represent how to iterate over a tensor is made up of
//! IterDomains to form an ND iterable. We directly set parallization strategies
//! on IterDomains.
class IterDomain : public Val {
 public:
  IterDomain(IrBuilderPasskey, const IterDomainBuilder& args);

  // Legacy constructor, TODO: should start moving to use IterDomainBuildArgs
  // constructor Same as the above but can set the offset of the stop point
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
      std::optional<int64_t> padded_to_size_,
      bool is_mma_swizzled);

  IterDomain(const IterDomain* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  //! Returns a new IterDomain matching properties of this
  //!
  //! This does NOT copy the is_rfactor_domain flag.
  IterDomain* cloneWithoutRFactor() const;

  //! Clone a vector domains
  static std::vector<IterDomain*> clone(
      const std::vector<IterDomain*>& domains);

  static IterDomain* merge(IterDomain* outer, IterDomain* inner);

  //! start_offset and stop_offset defines partial split. Only root
  //! domains are allowed to have non-zero start and stop offsets.
  static std::pair<IterDomain*, IterDomain*> split(
      IterDomain* in,
      Val* factor,
      bool inner_split,
      Val* start_offset = nullptr,
      Val* stop_offset = nullptr);

  //! trim_out_of_bounds controls how the values outside start and stop
  //! positions are treated. The option is only valid with root
  //! domains as non-root domains do not have valid start and stop
  //! positions.
  //!
  //! \param trim_out_of_bounds Trims [0, start_] and [-stop_offset_, extent_]
  static std::pair<IterDomain*, IterDomain*> split(
      IterDomain* in,
      Val* factor,
      bool inner_split,
      bool trim_out_of_bounds);

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

  bool isGather() const {
    return getIterType() == IterType::Gather;
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

  void parallelize(ParallelType t);

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
        "padToMultipleOfWarp : warp padding only supported on TIDx parallel dimension");
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
  //!  WarpMmaSwizzler will label the instruction loops case-by-case.
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
      Swizzle2DType swizzle_type,
      IterDomain* in_x,
      IterDomain* in_y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  bool isMmaSwizzled() const {
    return is_mma_swizzled_;
  }

  //! Used by WarpMmaSwizzler, this is an utility for WarpMmaSwizzler
  //!  to lock the thread swizzled iterdomains.
  //! Only true for the iterdomains produced by WarpMmaSwizzler.
  //! Mma ops require specific swizzle patterns
  //!  and this label utility is to prevent any further transform on the
  //!  iterdomains involved in the swizzle so that the pattern remain correct in
  //!  generated code.
  //!
  //! Note:
  //!    Used only through WarpMmaSwizzler only and mma validation relies on
  //!    this
  //!  flag being set on the correct iterdomains.
  void toMmaSwizzled() {
    is_mma_swizzled_ = true;
  }

 protected:
  friend TensorDomain;
  friend ReplayTransformations;
  friend IndexReferenceReplay;

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
  std::optional<int64_t> padded_to_size_ = std::nullopt;

  //! Tracks if this id represents a thread swizzled loop or
  //!   models an implicit loop within instructions. Should not make
  //!   any changes once an id is warp mapped.
  bool is_mma_swizzled_ = false;
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
class TensorDomain : public Val {
 public:
  explicit TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> leaf_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> rfactor_domain,
      std::vector<IterDomain*> leaf_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> rfactor_domain,
      std::vector<IterDomain*> allocation,
      std::vector<IterDomain*> leaf_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(IrBuilderPasskey, const TensorDomain* src);

  TensorDomain(const TensorDomain* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool operator==(const TensorDomain& other) const;
  bool operator!=(const TensorDomain& other) const {
    return !(*this == other);
  }

  std::vector<IterDomain*>::size_type nDims() const {
    return leaf_domain_.size();
  }

  bool sameAs(const Statement* other) const override;

  static bool sameAs(
      const std::vector<IterDomain*>& lhs,
      const std::vector<IterDomain*>& rhs);

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  // Note: [Contiguity]
  // Contiguity is a vector of optional<bool> which has the same number of
  // elements as rfactor_domain_. The contiguity of a broadcast dimension is
  // meaningless, so it has to be nullopt. The contiguity of a non-broadcasting
  // dimension is true if and only if it is memory dense with the next
  // non-broadcasting dimension.
  // For example, if I have a tensor torch.zeros(4, 1, 3).expand(-1, 10, -1),
  // the contiguity will be (true, nullopt, true), which means 4 is memory dense
  // with 3.
  const std::vector<std::optional<bool>>& contiguity() const {
    return contiguity_;
  }

  void setContiguity(const std::vector<std::optional<bool>>& contig);

  std::string getContiguityString() const {
    std::stringstream ss;
    bool first = true;
    for (auto b : contiguity()) {
      if (!first) {
        ss << " ";
      }
      first = false;
      ss << (b.has_value() ? (*b ? "t" : "f") : "n");
    }
    return ss.str();
  }

  bool hasReduction() const {
    return has_reduction_;
  }

  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBlockBroadcast() const;
  bool hasGridBroadcast() const;

  bool hasBroadcast() const {
    return no_bcast_domain_.size() != leaf_domain_.size();
  }

  bool hasRFactor() const {
    return !rfactor_domain_.empty();
  }

  bool hasAllocation() const {
    return !allocation_domain_.empty();
  }

  // Returns if rfactor domain only consists of id's of iter type.
  bool hasViewLikeRFactor() const;

  bool hasVectorize() const;

  bool hasSymbolicAxis() const;

  std::optional<unsigned int> getReductionAxis() const;

  const std::vector<IterDomain*>& noReductions() const {
    return no_reduction_domain_;
  }

  const std::vector<IterDomain*>& noBroadcasts() const {
    return no_bcast_domain_;
  }

  const std::vector<IterDomain*>& root() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& rfactor() const {
    return rfactor_domain_;
  };

  const std::vector<IterDomain*>& allocation() const {
    return allocation_domain_;
  }

  const std::vector<IterDomain*>& leaf() const {
    return leaf_domain_;
  }

  // If rfactor domain exists in domain() return it, otherwise return root
  // domain.
  const std::vector<IterDomain*>& maybeRFactor() const {
    return hasRFactor() ? rfactor() : root();
  }

  const std::vector<IterDomain*>& maybeAllocation() const {
    return hasAllocation() ? allocation_domain_ : maybeRFactor();
  };

  // Set the allocation domain of this TensorDomain. The new allocation domain
  // must satisfy root <= allocation <= leaf, that is, it must be within the
  // history between root and leaf domain. Because contiguity is always defined
  // w.r.t. the allocation domain, the contiguity must be updated accordingly.
  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      std::vector<std::optional<bool>> new_contiguity);

  // Similar to the previous one, but with new contiguity filled with all true
  // or all false.
  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      bool new_contiguity) {
    auto contiguity_flags =
        getContiguityFilledWith(new_allocation_domain, new_contiguity);
    setAllocationDomain(
        std::move(new_allocation_domain), std::move(contiguity_flags));
  }

  void resetDomains() {
    no_reduction_domain_ = noReductions(leaf_domain_);
    no_bcast_domain_ = noBroadcasts(leaf_domain_);
    has_reduction_ = hasReduction(leaf_domain_);
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  IterDomain* axis(int i) const;

  int64_t posOf(IterDomain* id) const;

  //! Returns a position of a root domain
  int64_t rootPosOf(IterDomain* id) const;

  // Split "axis" into 2 axes
  //! inner_split dictates if the factor section of the split should be inside
  //! the
  //! remainer or outside.
  //! e.g. split(0, 4, inner_split = true) will result in:
  //! tv[id{extent}] -> tv[id{ceilDiv(extent, factor)}, id{factor}]
  //! e.g. split(0, 4, inner_split = false) will result in:
  //! tv[id{extent}] -> tv[id{factor}, id{ceilDiv(extent, factor)}]
  void split(
      int axis_,
      Val* factor,
      bool inner_split,
      bool trim_out_of_bounds = false);

  // Merge axis_o and axis_i. axis_i is the fast changing dimension. Resulting
  // axis is by default placed at original position axis_o
  void merge(int axis_o, int axis_i);

  // Reorder axes according to map[old_pos] = new_pos
  void reorder(const std::unordered_map<int, int>& old2new);

  //! Applies 2D swizzle on a rectangular tile defined by
  //!  a pair of iterdomains contained in this domain.
  void swizzle(
      Swizzle2DType swizzle_type,
      int x,
      int y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  // Transform TensorView according to merge and split transformations
  TensorDomain* view(const AnalyzeViewResult& view_analysis);

  TensorDomain* flatten(int64_t start_dim, int64_t end_dim);

  static std::vector<IterDomain*> orderedAs(
      const std::vector<IterDomain*>& td,
      const std::unordered_map<int, int>& old2new);

  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);

  static bool hasBroadcast(const std::vector<IterDomain*>&);
  static bool hasReduction(const std::vector<IterDomain*>&);

  // Get a vector whose size is the number of IDs in the given rfactor_domain
  // filled with fill_value or nullopt depending on whether its corresponding ID
  // is broadcast.
  static std::vector<std::optional<bool>> getContiguityFilledWith(
      const std::vector<IterDomain*>& rfactor_domain,
      bool fill_value);

  // pair is in order where second is the consumer of first
  std::pair<TensorDomain*, TensorDomain*> rFactor(const std::vector<int>& axes);

 private:
  const std::vector<IterDomain*> root_domain_;
  const std::vector<IterDomain*> rfactor_domain_;
  std::vector<IterDomain*> allocation_domain_;
  std::vector<IterDomain*> leaf_domain_;

  std::vector<IterDomain*> no_bcast_domain_;
  std::vector<IterDomain*> no_reduction_domain_;
  std::vector<std::optional<bool>> contiguity_;
  bool has_reduction_;
};

} // namespace nvfuser
