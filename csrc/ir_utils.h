// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir_all_nodes.h>
#include <type.h>

#include <iterator>
#include <unordered_map>

namespace nvfuser {

namespace ir_utils {

// Replace values in fusion using ValReplacementMutator
void replaceValue(
    Fusion*,
    const std::unordered_map<Val*, Val*>& replacement_map);

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
    TORCH_INTERNAL_ASSERT(
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
    size_t ndims);

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
std::vector<int> normalizeOld2New(
    const std::unordered_map<int, int>& old2new_in,
    size_t ndims);

// Replace all uses of reference with substitute in expr. Return the Expr.
// Warning: Invalidates provided Expr.
// Warning: Removes connection of reference through provided Expr.
// Warning: Creates new Expr connecting substitue.
// Reference is found through direct pointer comparison.
TORCH_CUDA_CU_API Expr* replaceValInExpr(
    Expr* expr,
    Val* reference,
    Val* substitute);

//! Replace Vals in an index Val as specified by replacement_map while
//! cloning the given index Val. The index val is assumed to represent
//! a tensor index consisting of Ints and arithmetic expressions.
//!
//! This is similar to replaceValInExpr but is different as Vals are
//! cloned such that no other exprs using the same leaf Vals are not
//! modified. TODO: Consider cleaning up the multiple replacement
//! routines.
Val* replaceValInIndexVal(
    Val* index,
    const std::unordered_map<Val*, Val*>& replacement_map);

// Makes rfactor generic with reduction ops and Welford
TORCH_CUDA_CU_API TensorView* rfactorHelper(
    TensorView* red_tv,
    const std::vector<int>& axes);

// Return immediate producers of val, this function can be used on any Val and
// will return producers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<Val*> producerValsOf(const Val* val);

// Return immediate consumers of val, this function can be used on any Val and
// will return consumers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<Val*> consumerValsOf(const Val* val);

// Return immediate siblings of val, this function can be used on any Val and
// will return siblings through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<Val*> siblingValsOf(const Val* val);

// Return immediate producers of vals, this function can be used on any vals and
// will return producers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<Val*> producerValsOf(
    const std::vector<Val*>& vals);

// Return immediate consumers of vals, this function can be used on any vals and
// will return consumers through Exprs.
//
// Warning: returned val's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses val->definition() or val->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<Val*> consumerValsOf(
    const std::vector<Val*>& vals);

// Return immediate producers of tv, this function will return all immediate
// producers of tv through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<TensorView*> producerTvsOf(const TensorView* tv);

// Return immediate consumers of tv, this function will return all immediate
// consumers of tv through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<TensorView*> consumerTvsOf(const TensorView* tv);

// Return immediate siblings of tv, this function will return all immediate
// siblings of tv through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<TensorView*> siblingTvsOf(const TensorView* tv);

// Return immediate producers of tvs, this function will return all immediate
// producers of tvs through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<TensorView*> producerTvsOf(
    const std::vector<TensorView*>& tvs);

// Return immediate consumers of tvs, this function will return all immediate
// consumers of tvs through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
TORCH_CUDA_CU_API std::vector<TensorView*> consumerTvsOf(
    const std::vector<TensorView*>& tvs);

// Returns producers of tv that are inputs of fusion
TORCH_CUDA_CU_API std::vector<TensorView*> inputTvsOf(TensorView* tv);

// Returns consumers of tv that are outputs of fusion
TORCH_CUDA_CU_API std::vector<TensorView*> outputTvsOf(TensorView* tv);

// Returns producers of tvs that are inputs of fusion
TORCH_CUDA_CU_API std::vector<TensorView*> inputTvsOf(
    std::vector<TensorView*> tvs);

// Returns consumers of tvs that are outputs of fusion
TORCH_CUDA_CU_API std::vector<TensorView*> outputTvsOf(
    std::vector<TensorView*> tvs);

// returns all tensor views in fusion that are used between outputs and inputs.
TORCH_CUDA_CU_API std::vector<TensorView*> allTvs(Fusion* fusion);

// returns all tensor views in fusion that are used between outputs and inputs
// except the specified set.
TORCH_CUDA_CU_API std::vector<TensorView*> allTvsExcept(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& except);

TORCH_CUDA_CU_API std::vector<Expr*> getReductionOps(Fusion* fusion);

TORCH_CUDA_CU_API std::vector<IndexSelectOp*> getIndexSelectOps(Fusion* fusion);

TORCH_CUDA_CU_API std::vector<TorchGatherOp*> getTorchGatherOps(Fusion* fusion);

TORCH_CUDA_CU_API std::vector<MmaOp*> getMmaOps(Fusion* fusion);

TORCH_CUDA_CU_API std::vector<SelectOp*> getSelectOps(Fusion* fusion);

// Returns the initialization value of tv or nullptr if not initialized.
TORCH_CUDA_CU_API Val* getReductionInitValOf(TensorView* tv);

// Returns if Expr is a reduction op
TORCH_CUDA_CU_API bool isReductionOp(const Expr*);

// Returns if Expr is a reduction op with TensorView or TensorIndex
TORCH_CUDA_CU_API bool isReductionTvOp(const Expr*);

// Returns if Expr is a pointwise op op with TensorView or TensorIndex
TORCH_CUDA_CU_API bool isPointwiseTvOp(const Expr* expr);

// Returns all non-trivial view operations. We shouldn't have trivial view
// operations but this function is to simply make sure if we ever do we don't
// pull them in.
TORCH_CUDA_CU_API std::vector<ViewOp*> getViewOps(Fusion*);

template <typename T>
std::string toString(const T& nodes) {
  std::stringstream ss;
  for (const Statement* stmt : nodes) {
    if (ss.tellp() != 0) {
      ss << ", ";
    }
    ss << stmt->toString();
  }
  return ss.str();
}

// Test if the given tensor is an input of squeeze op
TORCH_CUDA_CU_API bool isSqueezeInput(const TensorView* tv);

// Test if the given ID in the given tensor is squeezed
TORCH_CUDA_CU_API bool isSqueezedID(const TensorView* tv, const IterDomain* id);

// Test if the given ID in the given tensor is indirectly accessed by,
// e.g., index_select, torch_gather and scatter
TORCH_CUDA_CU_API bool isIndexedID(const TensorView* tv, const IterDomain* id);

// Test if the given ID in the given tensor is indirectly read by,
// e.g., index_select and torch_gather
TORCH_CUDA_CU_API bool isIndexedProducerID(
    const TensorView* tv,
    const IterDomain* id);

// Test if the given ID in the given tensor is indirectly written to by,
// e.g., scatter
TORCH_CUDA_CU_API bool isIndexedConsumerID(
    const TensorView* tv,
    const IterDomain* id);

// Return a producer ID, if any, that is indirectly accessed by, e.g.,
// index_select and torch_gather.
TORCH_CUDA_CU_API IterDomain* getIndexedProducerID(const Expr* expr);

// Return the corresponding consumer if of a producer ID that is
// indirectly accessed.
TORCH_CUDA_CU_API IterDomain* getConsumerOfIndexedProducerID(const Expr* expr);

// Get all IDs of a tensor. Returned values are topologicaly ordered, and
// unique.
TORCH_CUDA_CU_API std::vector<IterDomain*> allIDsOf(const TensorView* tv);

// Check if the given tv is an input of SelectOp
TORCH_CUDA_CU_API bool isSelectInput(TensorView* tv);

// Check if the given tv is first argment of index_select(lookup, dim, indices)
TORCH_CUDA_CU_API bool isIndexSelectLookupTv(const TensorView* tv);

// Check if the given tv is third argment of index_select(lookup, dim, indices)
TORCH_CUDA_CU_API bool isIndexSelectIndicesTv(const TensorView* tv);

TORCH_CUDA_CU_API bool isTorchGatherLookupTv(const Val* tv);

TORCH_CUDA_CU_API std::string varName(const Val* val);

// Check if a tensor is resized as part of  its root to rfactor transformations
bool hasResizedRfactor(const TensorView* tv);

// Returns tvs that have symbolic axes
std::vector<TensorView*> getTVsWithDynamicTransform(Fusion* fusion);

//! Validate derived_domain completely covers initial_domain with no
//! redundancy. Consider derived_domains as a different view of the
//! same logical domain as initial_domain with affine
//! transformations. This validation makes sure both sets
//! of domains represent the same logical space.
//!
//! It is intended to be used to validate rfactor and leaf domains
//! of a tensor root domain.
//!
//! For example, it's an error if a initial ID is split and
//! only one of the outputs is included in the ids vector. It is
//! also an error if both a producer and consumer ID are included in
//! ids as they partially have the same dependency with the initial
//! domain.
void validateDomainEquivalence(
    const std::vector<IterDomain*>& initial_domain,
    const std::vector<IterDomain*>& derived_domain);

} // namespace ir_utils
} // namespace nvfuser
