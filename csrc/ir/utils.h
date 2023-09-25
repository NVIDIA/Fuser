// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <type.h>

#include <iterator>
#include <unordered_map>

namespace nvfuser::ir_utils {

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

//! Replaces reference Val with substitute in all Expr inputs and attributes.
//! Warning: Invalidates provided Expr.
//! Warning: Removes connection of reference through provided Expr.
//! Warning: Creates new Expr defining substitute.
Expr* replaceValInExprInputs(Expr* expr, Val* reference, Val* substitute);

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
TensorView* rfactorHelper(TensorView* red_tv, const std::vector<int>& axes);

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
std::vector<TensorView*> producerTvsOf(const TensorView* tv);

// Return immediate consumers of tv, this function will return all immediate
// consumers of tv through Exprs.
//
// Warning: returned tv's are not guaranteed to be between fusion inputs and
// outputs. This function simply uses tv->definition() or tv->uses() which is
// limited to not go through fusion inputs/outputs, but if on a path that isn't
// strictly between fusion inputs/outputs, it could effectively return dead
// code.
std::vector<TensorView*> consumerTvsOf(const TensorView* tv);

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

// returns all tensor views in fusion that are used between outputs and inputs.
std::vector<TensorView*> allTvs(Fusion* fusion);

// returns all tensor views in fusion that are used between outputs and inputs
// except the specified set.
std::vector<TensorView*> allTvsExcept(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& except);

std::vector<Expr*> getReductionOps(Fusion* fusion);

std::vector<IndexSelectOp*> getIndexSelectOps(Fusion* fusion);

std::vector<TorchGatherOp*> getTorchGatherOps(Fusion* fusion);

std::vector<MmaOp*> getMmaOps(Fusion* fusion);

std::vector<SelectOp*> getSelectOps(Fusion* fusion);

// Returns the initialization value of tv or nullptr if not initialized.
Val* getReductionInitValOf(TensorView* tv);

// Returns if Expr is a reduction op
bool isReductionOp(const Expr*);

// Returns if Expr is a reduction op with TensorView or TensorIndex
bool isReductionTvOp(const Expr*);

// Returns if Expr is a pointwise op op with TensorView or TensorIndex
bool isPointwiseTvOp(const Expr* expr);

// Returns all non-trivial view operations. We shouldn't have trivial view
// operations but this function is to simply make sure if we ever do we don't
// pull them in.
std::vector<ViewOp*> getViewOps(Fusion*);

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

template <typename T>
std::string toInlineString(const T& nodes) {
  std::stringstream ss;
  for (const Statement* stmt : nodes) {
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
// e.g., index_select, torch_gather and scatter
bool isIndexedID(const TensorView* tv, const IterDomain* id);

// Test if the given ID in the given tensor is indirectly read by,
// e.g., index_select and torch_gather
bool isIndexedProducerID(const TensorView* tv, const IterDomain* id);

// Test if the given ID in the given tensor is indirectly written to by,
// e.g., scatter
bool isIndexedConsumerID(const TensorView* tv, const IterDomain* id);

// Return a producer ID, if any, that is indirectly accessed by, e.g.,
// index_select and torch_gather.
IterDomain* getIndexedProducerID(const Expr* expr);

// Return the corresponding consumer if of a producer ID that is
// indirectly accessed.
IterDomain* getConsumerOfIndexedProducerID(const Expr* expr);

// Get all IDs of a tensor. Returned values are topologicaly ordered, and
// unique.
std::vector<IterDomain*> allIDsOf(const TensorView* tv);

// Check if the given tv is an input of SelectOp
bool isSelectInput(TensorView* tv);

// Check if the given tv is first argment of index_select(lookup, dim, indices)
bool isIndexSelectLookupTv(const TensorView* tv);

// Check if the given tv is third argment of index_select(lookup, dim, indices)
bool isIndexSelectIndicesTv(const TensorView* tv);

bool isTorchGatherLookupTv(const Val* tv);

std::string varName(const Val* val);

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
std::vector<Statement*> checkCycle(
    Fusion* fusion,
    const std::unordered_set<Statement*>& from,
    const std::vector<Val*>& to);

//! Check and return a cycle found in fusion
std::vector<Statement*> checkCycle(Fusion* fusion);

//! Check if a Val is a tensor size;
bool isTensorSize(const Val* val);

//! Check if a Val is a tensor stride;
bool isTensorStride(const Val* val);

} // namespace nvfuser::ir_utils
