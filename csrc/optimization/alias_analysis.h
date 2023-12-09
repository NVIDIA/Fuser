// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <unordered_map>

#include <fusion.h>
#include <ir/interface_nodes.h>

namespace nvfuser::optimization {

struct Layout {
  std::vector<IterDomain*> allocation_domain;
  std::vector<std::optional<bool>> contiguity;

  std::string toString(int indent_size = 0) const;
  bool isCompatibleWith(const Layout& other) const;
};

// Holds aliases found in a fusion. The expected user flow is
//
// ```
// AliasAnalysisResult analysis;
// analysis.add(...);
// ...
// analysis.add(...);
// analysis.finalize(fusion);
//
// // The user can now call const methods to retrieve information.
// ```
class AliasAnalysisResult {
 public:
  AliasAnalysisResult() = default;
  AliasAnalysisResult(const AliasAnalysisResult&) = delete;
  AliasAnalysisResult& operator=(const AliasAnalysisResult&) = delete;
  AliasAnalysisResult(AliasAnalysisResult&&) = default;
  AliasAnalysisResult& operator=(AliasAnalysisResult&&) = default;

  // Marks `source` as the immediate aliasing source of `alias` and sets the
  // preferred layout.
  void add(const TensorView* alias, const TensorView* source, Layout&& layout);

  void finalize(Fusion* fusion);

  // Returns the preferred layout. If `alias` is not in `preferred_layout_`,
  // returns the `TensorView`'s initial layout.
  Layout preferredLayout(const Val* alias) const;

  std::string toString(int indent_size) const;

  // Gets the aliased fusion input of a fusion output. Returns nullptr
  // when `fusion_out` is not a fusion output or does not alias a fusion input.
  const TensorView* getAliasedInput(const TensorView* fusion_out) const;

 private:
  // Walks up `alias_to_source_` to find the root of the chain. Returns itself
  // if `alias` doesn't alias anything.
  const Val* findRoot(const Val* alias) const;

  // Maps aliases (e.g. the output of a View) to their direct sources (e.g. the
  // input of the same View). Also stores the preferred output layout for the
  // alias. Consider path compression, a common optimization used in
  // disjoint-set data structure, so it's easy to figure out the root of an
  // alias.
  std::unordered_map<const TensorView*, std::pair<const TensorView*, Layout>>
      alias_to_source_;

  // Maps a fusion output to its aliased fusion input.
  std::unordered_map<const TensorView*, const TensorView*> out_to_root_;
};

// Finds aliases of the fusion inputs. The analysis should be conservative --
// when the analysis says B is an alias of input A and that B's layout
// (allocation domain and contiguity) is compatible with the preferred layout,
// `ExpressionEvaluator::evaluate(B)` should produce an `at::Tensor` that's an
// alias of the `at::Tensor` bound to A.
//
// Currently, for implementation convenience, AliasAnalysis ignores allocation
// domains of non-fusion-input TensorViews. It produces preferred layouts for
// these TensorViews and expects the user to resolve any incompatibility.
// MarkAliasPass, its only user at this moment, marks an output as an alias only
// when its allocation domain is empty. I'm happy to revisit this contract.
AliasAnalysisResult findAliases(Fusion* fusion);

} // namespace nvfuser::optimization
