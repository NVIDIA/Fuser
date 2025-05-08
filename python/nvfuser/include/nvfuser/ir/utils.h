// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <exceptions.h>
#include <ir/all_nodes.h>
#include <type.h>
#include <visibility.h>

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace nvfuser::MmaOpUtils {

// The expected number of concrete domains for gemm
constexpr size_t expected_gemm_cdomains = 2;

void verifyMmaOpForEvaluation(MmaOp* mma_op, DataType expected_input_dtype);

struct MatmulInputs {
  Val* mma_lhs = nullptr;
  Val* mma_rhs = nullptr;
  Val* bias = nullptr;
  Val* alpha = nullptr;
  Val* beta = nullptr;
  // Ordering of dimensions M,N,K in MmaOp's output TensorView's root domain.
  // Determined based on position of iterdomains.
  // For addmm/matmul ([M,K] x [K,N]): M=0, N=2, K=1
  // For linear ([M,K] x [N,K]): M=0, N=1, K=2
  // mma_dims_pos = {m_pos, n_pos, k_pos}
  std::tuple<int, int, int> mma_dims_pos = {};
  // The elements denote if the corresponding iterdomain in the bias was a new
  // broadcast dimension. This is used to broadcast the bias for matmul/addmm
  // during evaluation.
  std::vector<bool> bias_bcast_flags = {};
};

} // namespace nvfuser::MmaOpUtils

namespace nvfuser::ir_utils {

// Replace values in fusion using ValReplacementMutator, it also updates fusion
// output according to the replacement_map. Returns the final
// replacement map, which includes the given replacement entries as
// well as those that are the results of the replacement.
std::unordered_map<Val*, Val*> replaceValue(
    Fusion*,
    const std::unordered_map<Val*, Val*>& replacement_map);

//! Checks whether this is a simple Set of a TensorView. If not, then this might
//! represent a scalar set, or a segment_set.
bool isSimpleTVSet(Expr* expr);

template <typename FilterType, typename Iterator>
class FilterIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = FilterType*;
  using pointer = value_type*;
  using reference = value_type&;

  FilterIterator(Iterator begin, Iterator end) : current_(begin), end_(end) {
    advance();
  }

  FilterType* operator*() const {
    return (*current_)->template as<FilterType>();
  }

  FilterType* operator->() const {
    return (*this);
  }

  FilterIterator& operator++() {
    ++current_;
    advance();
    return *this;
  }

  FilterIterator operator++(int) {
    const auto before_increment = *this;
    ++current_;
    advance();
    return before_increment;
  }

  bool operator==(const FilterIterator& other) const {
    NVF_ERROR(
        end_ == other.end_,
        "Comparing two FilteredViews that originate from different containers");
    return current_ == other.current_;
  }

  bool operator!=(const FilterIterator& other) const {
    return !(*this == other);
  }

 private:
  void advance() {
    current_ = std::find_if(current_, end_, [](const auto& val) {
      return dynamic_cast<const FilterType*>(val) != nullptr;
    });
  }

 private:
  Iterator current_;
  Iterator end_;
};

// An iterable view to a given container of Val pointers. Only returns
// Vals of a given Val type.
// NOTE: Add a non-const iterator if needed.
template <typename FilterType, typename InputIt>
class FilteredView {
 public:
  using value_type = FilterType*;
  using const_iterator = FilterIterator<FilterType, InputIt>;

  FilteredView(InputIt first, InputIt last) : input_it_(first), last_(last) {}

  const_iterator cbegin() const {
    return const_iterator(input_it_, last_);
  }

  const_iterator begin() const {
    return cbegin();
  }

  const_iterator cend() const {
    return const_iterator(last_, last_);
  }

  const_iterator end() const {
    return cend();
  }

  bool empty() const {
    return begin() == end();
  }

  std::vector<value_type> vector() const {
    return std::vector<value_type>(begin(), end());
  }

  size_t size() const {
    size_t s = 0;
    for (auto it = cbegin(); it != cend(); ++it) {
      ++s;
    }
    return s;
  }

 private:
  const InputIt input_it_;
  const InputIt last_;
};

template <typename FilterType, typename InputIt>
auto filterByType(InputIt first, InputIt last) {
  return FilteredView<FilterType, InputIt>(first, last);
}

template <typename FilterType, typename ContainerType>
auto filterByType(const ContainerType&& inputs) = delete;

template <typename FilterType, typename ContainerType>
auto filterByType(const ContainerType& inputs) {
  return filterByType<FilterType>(inputs.cbegin(), inputs.cend());
}

//! Returns a list of new-to-old mappings.
//!
//! This funcion canonicalizes the dimensions and validates that multiple old
//! dimension are mapped to the same new dimension.
std::vector<int64_t> normalizeNew2Old(
    const std::vector<int64_t>& new2old_in,
    int64_t ndims);

//! Returns a list of new-to-old mappings.
//!
//! The input map does not need to be complete. Missing axes are
//! assumed not to be affected.
//!
//! This is used to preprocess broadcast and transpose arguments.
//!
//! Example: (N := ndims)
//!   {{0, 1}} -> [1, 0, ...., N-1]
//!   Transposes the first two axes with no other change.
//!
//!   {{0, -1}} -> [N-1, ...., 0]
//!   Swaps the first and last axes.
std::vector<int64_t> normalizeOld2New(
    const std::unordered_map<int64_t, int64_t>& old2new_in,
    int64_t ndims);

//! Replaces reference Val with substitute in all Expr inputs and attributes.
//! Warning: Invalidates provided Expr.
//! Warning: Removes connection of reference through provided Expr.
//! Warning: Creates new Expr defining substitute.
NVF_API Expr* replaceValInExprInputs(
    Expr* expr,
    Val* reference,
    Val* substitute);

//! Replace old_val with new_val in all active uses as well as in fusion
//! outputs.
void replaceValInAllExprInputsAndFusionOutputs(Val* old_val, Val* new_val);

//! Removes the given expression and creates a new expression that is identical
//! to expr, but whose outputs are given by the new_outputs argument. It is an
//! error for Vals in new_outputs that are not equal to their old equivalents to
//! have a definition as these should be freshly-created Vals that are not yet
//! defined.
//!
//! Warning: Invalidates provided Expr.
//! Warning: Creates new Expr defining substitutes.
Expr* transferDefinitionToNewOutputs(
    Expr* expr,
    const std::vector<Val*>& new_outputs);

//! Recursively goes to the definition of the given Val and replace the Vals as
//! specified by replacement_map while cloning the given Val.
//!
//! This is similar to replaceValInExprInputs but is different as Vals are
//! cloned such that no other exprs using the same leaf Vals are not
//! modified. TODO: Consider cleaning up the multiple replacement
//! routines.
Val* replaceValRecursively(
    Val* val,
    const std::unordered_map<Val*, Val*>& replacement_map);

// Makes rfactor generic with reduction ops and Welford
NVF_API TensorView* rFactorHelper(
    TensorView* red_tv,
    const std::vector<int64_t>& axes);

// Return immediate producers of val, this function can be used on any Val and
// will return producers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<Val*> producerValsOf(const Val* val);

// Return immediate consumers of val, this function can be used on any Val and
// will return consumers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<Val*> consumerValsOf(const Val* val);

// Return immediate siblings of val, this function can be used on any Val and
// will return siblings through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<Val*> siblingValsOf(const Val* val);

// Return immediate producers of vals, this function can be used on any vals and
// will return producers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<Val*> producerValsOf(const std::vector<Val*>& vals);

// Return immediate consumers of vals, this function can be used on any vals and
// will return consumers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<Val*> consumerValsOf(const std::vector<Val*>& vals);

// Return immediate producers of tv, this function will return all immediate
// producers of tv through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
NVF_API std::vector<TensorView*> producerTvsOf(const TensorView* tv);

// Return immediate consumers of tv, this function will return all immediate
// consumers of tv through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
NVF_API std::vector<TensorView*> consumerTvsOf(const TensorView* tv);

// Return immediate siblings of tv, this function will return all immediate
// siblings of tv through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<TensorView*> siblingTvsOf(const TensorView* tv);

// Return immediate producers of tvs, this function will return all immediate
// producers of tvs through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<TensorView*> producerTvsOf(const std::vector<TensorView*>& tvs);

// Return immediate consumers of tvs, this function will return all immediate
// consumers of tvs through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<TensorView*> consumerTvsOf(const std::vector<TensorView*>& tvs);

// Returns producers of tv that are inputs of fusion
std::vector<TensorView*> inputTvsOf(TensorView* tv);

// Returns consumers of tv that are outputs of fusion
std::vector<TensorView*> outputTvsOf(TensorView* tv);

// Returns producers of tvs that are inputs of fusion
std::vector<TensorView*> inputTvsOf(std::vector<TensorView*> tvs);

// Returns consumers of tvs that are outputs of fusion
std::vector<TensorView*> outputTvsOf(std::vector<TensorView*> tvs);

// returns all tensor views used in the provided expressions
VectorOfUniqueEntries<TensorView*> allTvsOfExprs(
    const std::vector<Expr*>& exprs);

// returns all tensor views in fusion that are used between outputs and inputs
// except the specified set.
NVF_API std::vector<TensorView*> allTvsExcept(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& except);

// Returns the initialization value of tv or nullptr if not initialized.
Val* getReductionInitValOf(TensorView* tv);

// Returns if Expr is a reduction op
bool isReductionOp(const Expr*);

// Returns if Expr is a reduction op with TensorView or TensorIndex
NVF_API bool isReductionTvOp(const Expr*);

// Returns if Expr is a pointwise op op with TensorView or TensorIndex
bool isPointwiseTvOp(const Expr* expr);

bool isSegmentSet(const Expr* e);

// Returns all non-trivial view operations. We shouldn't have trivial view
// operations but this function is to simply make sure if we ever do we don't
// pull them in.
std::vector<ViewOp*> getViewOps(Fusion*);

template <typename T>
std::string toString(const T& nodes) {
  std::stringstream ss;
  for (auto stmt : nodes) {
    if (ss.tellp() != 0) {
      ss << ", ";
    }
    ss << stmt->toString();
  }
  return ss.str();
}

template <typename T>
std::string toInlineString(const T& nodes) {
  std::stringstream ss;
  for (auto stmt : nodes) {
    if (ss.tellp() != 0) {
      ss << ", ";
    }
    ss << stmt->toInlineString();
  }
  return ss.str();
}

// Test if the given tensor is an input of squeeze op
bool isSqueezeInput(const TensorView* tv);

// Test if the given ID in the given tensor is squeezed
bool isSqueezedID(const TensorView* tv, const IterDomain* id);

// Test if the given ID in the given tensor is indirectly accessed by,
// e.g., indexSelect, gather and scatter
bool isIndexedID(const TensorView* tv, const IterDomain* id);

// Test if the given ID in the given tensor is indirectly read by,
// e.g., indexSelect gather
bool isIndexedProducerID(const TensorView* tv, const IterDomain* id);

// Test if the given ID in the given tensor is indirectly written to by,
// e.g., scatter
bool isIndexedConsumerID(const TensorView* tv, const IterDomain* id);

// Return a producer ID, if any, that is indirectly accessed by, e.g.,
// indexSelect and gather.
IterDomain* getIndexedProducerID(const Expr* expr);

// Return the corresponding consumer if of a producer ID that is
// indirectly accessed.
IterDomain* getConsumerOfIndexedProducerID(const Expr* expr);

// Check if the given tv is first argment of indexSelect(lookup, dim, indices)
bool isIndexSelectLookupTv(const TensorView* tv);

// Check if the given tv is third argment of indexSelect(lookup, dim, indices)
bool isIndexSelectIndicesTv(const TensorView* tv);

bool isGatherLookupTv(const Val* tv);

std::string varName(const Val* val);

// Check if a tensor is resized as part of its root to logical transformations
bool hasResizedRfactor(const TensorView* tv);

// Returns tvs that have symbolic axes
std::vector<TensorView*> getTVsWithDynamicTransform(Fusion* fusion);

//! Check if dom0 and dom1 completely covers each other with no
//! redundancy. When they are equivalent, we can consider them as a different
//! view of the each other with affine transformations.
//!
//! For example, if we have
//!  I0  I1  I2  I3
//!   \  /    \  /
//!    I4      I5
//! then [I0, I1, I2, I3] is equivalent to [I4, I5], but [I1, I2, I3] is not
//! equivalent to [I4, I5].
//!
//! Another example, if we have
//!  I0  I1  I2  I3
//!   \  /    \  /
//!    I4      I5
//!   /  \    /  \.
//!  I6  I7  I8  I9
//! Then [I0, I1, I8, I9] is equivalent to [I6, I7, I2, I3]. [I0, I1, I2, I3] is
//! equivalent to [I6, I7, I8, I9]. But [I0, I1, I8, I3] is NOT equivalent to
//! [I6, I7, I2, I9]
//!
//! Broadcast IterDomains are ignored in this check, because we consider them as
//! placeholders and allow them to be created (and annihilated?) arbitrarily as
//! needed for convenience.
//!
//! Returns if each domain has unreachable IDs. It is an error if
//! redundant IDs are detected.
struct CompareDomainResult {
  bool dom0_has_unreachable_ids = false;
  bool dom1_has_unreachable_ids = false;
};

// TODO: Completely replace this with compareDomainWithReference
CompareDomainResult compareDomains(
    std::vector<IterDomain*> dom0,
    const std::vector<IterDomain*>& dom1,
    const std::vector<IterDomain*>& additional_ids = {},
    bool ignore_broadcast = true);

//! Validate dom0 and dom1 are equivalent
void validateDomainEquivalence(
    std::vector<IterDomain*> dom0,
    const std::vector<IterDomain*>& dom1,
    const std::vector<IterDomain*>& additional_ids = {});

struct CompareDomainWithReferenceResult {
  // Redundant IDs found in the given domain
  std::vector<IterDomain*> redundant_ids;
  // IDs found in the given domain that are not connected with the
  // reference domain
  std::vector<IterDomain*> additional_ids;
  // Reference IDs that are not reachable from the given domain
  std::vector<IterDomain*> unreachable_reference_ids;

  bool empty() const {
    return redundant_ids.empty() && additional_ids.empty() &&
        unreachable_reference_ids.empty();
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "{redundant_ids: " << toDelimitedString(redundant_ids)
       << ", additional_ids: " << toDelimitedString(additional_ids)
       << ", unreachable_reference_ids: "
       << toDelimitedString(unreachable_reference_ids) << "}";
    return ss.str();
  }
};

// Given a reference domain that has no redundancy, check if a given
// domain completely covers the reference domain with no
// redundancy. Redundant or extra IDs will be identified if
// any.
//
// Once caveat is that if any of the IDs of the reference domain is not
// reachable, it may not identify all of the redundant or extra IDs as
// it is unclear if they should be considered redundant or extra. For example,
// suppose we have a domain of {i0}, and there's an expession
// of `merge(i0, i1) -> i2`. Comparing {i0} with {i2} as a
// i2 will be returned as an unreachable ID. In this case, i0 is
// not used sicne i1 is missing, but it doesn't seem right to cnsider
// it's an extra ID since it would have been used if i1 were not
// missing. Similarly, it should not be considered redundant. This
// should not be a concern in practice since if any of the reference
// IDs is unreachable, it should be considered an error.
CompareDomainWithReferenceResult compareDomainWithReference(
    const std::vector<IterDomain*>& domain,
    const std::vector<IterDomain*>& reference);

//! Check if all the inputs required to compute needed_val are known
template <
    typename ValOrVectorOfVal,
    typename SetOfVal = std::unordered_set<const Val*>>
inline bool dependenciesSatisfied(
    // const Val*, Val*, std::vector<const Val*>, std::vector<Val*> or any other
    // container that has back(), pop_back(), empty() and emplace_back()
    ValOrVectorOfVal needed_vals,
    // std::unordered_set<const Val*>, std::unordered_map<const Val*, T> or any
    // other container that has count()
    const SetOfVal& known_vals = {}) {
  if constexpr (
      std::is_same_v<ValOrVectorOfVal, const Val*> ||
      std::is_same_v<ValOrVectorOfVal, Val*>) {
    // convert a single const Val* or Val* to a vector
    return dependenciesSatisfied(
        std::vector<const Val*>{needed_vals}, known_vals);
  } else {
    while (!needed_vals.empty()) {
      auto needed_val = needed_vals.back();
      needed_vals.pop_back();
      if (known_vals.count(needed_val) > 0 || needed_val->isConst()) {
        continue;
      }
      auto def = needed_val->definition();
      if (def == nullptr) {
        return false;
      }
      for (auto input : def->inputs()) {
        needed_vals.emplace_back(input);
      }
    }
  }
  return true;
}

//! Check if a conditional scope, i.e., ForLoop or IfThenElse, is
//! guaranteed not to cause thread divergence
bool isAlignedScopeExpr(const Expr* expr);

//! Get the only producer of a tensor view. If there are multiple producers,
//! then throw an error.
inline TensorView* getSoleProducerTv(const TensorView* tv) {
  auto producers = producerTvsOf(tv);
  NVF_ERROR(
      producers.size() == 1,
      "Expected only one producer of ",
      tv->toString(),
      ", but found ",
      producers.size(),
      " producers.");
  return producers[0];
}

//! Check and return a cycle found in fusion, search starts from `to` and ends
//! at `from`
NVF_API std::vector<Statement*> checkCycle(
    Fusion* fusion,
    const std::unordered_set<Statement*>& from,
    const std::vector<Val*>& to);

//! Check and return a cycle found in fusion
NVF_API std::vector<Statement*> checkCycle(Fusion* fusion);

//! Check if a Val is a tensor size;
NVF_API bool isTensorSize(const Val* val);

//! Check if a Val is a tensor stride;
bool isTensorStride(const Val* val);

//! Returns a vector of the given op type or exprs if multiple types are given.
template <typename... OpTypes>
auto getOpsOfType(Fusion* fusion) {
  using FirstOpType = std::tuple_element_t<0, std::tuple<OpTypes...>>;
  using ExprType =
      std::conditional_t<sizeof...(OpTypes) == 1, FirstOpType, Expr>;
  std::vector<ExprType*> ops;
  for (auto expr : fusion->exprs()) {
    if (expr->isOneOf<OpTypes...>()) {
      ops.push_back(expr->as<ExprType>());
    }
  }
  return ops;
}

//! Returns true if fusion has any ops of the given type.
template <typename... OpTypes>
bool hasOpsOfType(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (expr->isOneOf<OpTypes...>()) {
      return true;
    }
  }
  return false;
}

//! Returns true if tv is used by any ops of the given type.
template <typename... OpTypes>
bool isTvUsedByOpsOfType(TensorView* tv) {
  for (auto expr : tv->uses()) {
    if (expr->isOneOf<OpTypes...>()) {
      return true;
    }
  }
  return false;
}

//! Returns expressions that are of type ReductionOp, GroupedReductionOp, or
//! WelfordOp.
std::vector<Expr*> getAllTypesOfReductionOps(Fusion* fusion);

//! Returns true if fusion has any reduction ops.
bool hasAnyReductionOps(Fusion* fusion);

int64_t getVectorizeSize(const TensorView* tv);

// Returns the permutation from `in` to `out`, i.e., `out[i]==in[perm[i]]`. If
// `out` is not a permutation of `in`, returns nullopt.
template <typename T>
std::optional<std::vector<int64_t>> computePermutation(
    const std::vector<T>& in,
    const std::vector<T>& out) {
  // Both std::is_permutation and the rest of this function are O(n^2). This is
  // fine for the current use case of computing the root-to-rfactor
  // permutation. If needed, this can be improved by requiring T to be hashable
  // (leading to O(n)) and/or comparable (leading to O(nlogn)).
  if (!std::is_permutation(in.begin(), in.end(), out.begin(), out.end())) {
    return std::nullopt;
  }

  std::vector<int64_t> permutation;
  permutation.reserve(out.size());
  for (const T& out_element : out) {
    permutation.push_back(std::distance(
        in.begin(), std::find(in.begin(), in.end(), out_element)));
  }
  return permutation;
}

template <typename T>
std::vector<T> applyPermutation(
    const std::vector<T>& in,
    const std::vector<int64_t>& permutation) {
  NVF_CHECK(in.size() == permutation.size());

  std::vector<int64_t> identity(permutation.size());
  std::iota(identity.begin(), identity.end(), 0);
  NVF_CHECK(std::is_permutation(
      permutation.begin(), permutation.end(), identity.begin()));

  std::vector<T> out;
  out.reserve(permutation.size());
  for (auto i : permutation) {
    out.push_back(in[i]);
  }
  return out;
}

bool hasTrivialAllocationDomain(const TensorView* tv);

// Returns true if all expr outputs should be mapped unconditionally
bool hasUniformSiblings(Expr* expr);

// Returns true if memory_type is partitioned in parallel_type. See
// also isMemorySharedAcross. Specifically, isMemorySharedAcross == true does
// not imply isMemoryPartitionedAcross == false. For example, Local with no
// parallelization is not partitioned nor shared.
inline bool isMemoryPartitionedAcross(
    MemoryType memory_type,
    ParallelType parallel_type) {
  switch (memory_type) {
    case MemoryType::Local:
      return isParallelTypeThread(parallel_type) ||
          isParallelTypeDeviceDim(parallel_type);
    case MemoryType::Shared:
    case MemoryType::Tensor:
      return isParallelTypeBlockDim(parallel_type) ||
          isParallelTypeDeviceDim(parallel_type);
    case MemoryType::Global:
      return isParallelTypeDeviceDim(parallel_type);
    default:
      NVF_THROW("Unknown MemoryType: ", memory_type);
  }
}

// Returns true if memory_type is shared in parallel_type. See also
// isPartitionedMemory.
inline bool isMemorySharedAcross(
    MemoryType memory_type,
    ParallelType parallel_type) {
  switch (memory_type) {
    case MemoryType::Local:
      // Nothing is shared if it's Local
      return false;
    case MemoryType::Shared:
    case MemoryType::Tensor:
      // Only TID parallelized domains are shared if it's Shared or Tensor
      return isParallelTypeThreadDim(parallel_type);
    case MemoryType::Global:
      // Only TID and BID parallelized domains are shared if it's Global
      return isParallelTypeThreadDim(parallel_type) ||
          isParallelTypeBlockDim(parallel_type);
    default:
      NVF_THROW("Unknown MemoryType: ", memory_type);
  }
}

//! Check if the given tv has a root domain -> loop domain linear
//! transformation. This is a temporary check used to incrementally enable
//! IdModel. Eventually, this should be removed.
bool hasRootToLoopLinearTransformations(const TensorView* tv);

//! In addition to the above hasRootToLoopLinearTransformations, it
//! also checks the loop domain has any extra domain
bool isLoopDomainFullyDerivedFromLogicalDomain(TensorView* tv);

AsyncOpType getAsyncOpType(const Expr* expr);

//! If the given statement is nullptr, return "nullptr", otherwise return its
//! toString()
std::string nullOrToString(const Statement* stmt);

//! If the given statement is nullptr, return "nullptr", otherwise return its
//! toInlineString()
std::string nullOrToInlineString(const Statement* stmt);

//! Check if the given value is functional. A functional value is one that
//! always returns the same result when called with the same inputs.
bool isFunctional(const Val* v);

// Check if the given val is recursively defined, which is invalid in
// the Fusion IR but may not be necessarily the case in other IRs
// such as the Kernel IR
bool isRecursivelyDefined(Val* val);

// Return the number of operations that are used to define val. One
// instance of Expr is counted as a single operation.
int64_t getOperationCount(Val* val);

// Create a ForLoop IR node that represents:
//   for (int i = 0; i < size; i++)
ForLoop* createRangeLoop(int64_t size);

// Returns the first output of Expr that is a TensorView
TensorView* getTvOutput(const Expr*);

// Returns the first input of Expr that is a TensorView
TensorView* getTvInput(const Expr*);

// Generates the allocation domain for the given logical domain based on the
// stride order.
std::vector<IterDomain*> strideOrderToAllocation(
    const std::vector<IterDomain*>& logical_domain,
    const std::vector<int64_t>& stride_order);

// Returns the number of bytes of data types of the producer and
// consumer tensors of a cast unary op
std::optional<std::pair<int64_t, int64_t>> getPrecisionOfProducerConsumerTensors(
    UnaryOp* cast_op);

// Get the <size> in the PTX instruction of TMem load/store:
//   tcgen05.st.sync.aligned.32x32b.x<size>.b32
// The minimum unit of TMem load/store is 4 bytes, and the .x<size>
// in the PTX instruction is the number of this unit, not the number of items.
// For example, tcgen05.st.sync.aligned.32x32b.x4.b32 could mean 1 complex
// double, 2 doubles, 4 floats, 8 halfs, or 16 bytes.
int64_t getTMemLdStVectorizeSize(TensorView* consumer_tv);

} // namespace nvfuser::ir_utils
