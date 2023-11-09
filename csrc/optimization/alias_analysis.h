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
};

class AliasAnalysisResult {
 public:
  AliasAnalysisResult() = default;
  AliasAnalysisResult(const AliasAnalysisResult&) = delete;
  AliasAnalysisResult& operator=(const AliasAnalysisResult&) = delete;
  AliasAnalysisResult(AliasAnalysisResult&&) = default;
  AliasAnalysisResult& operator=(AliasAnalysisResult&&) = default;

  // Returns itself if `alias` doesn't alias anything.
  const Val* findRoot(const Val* alias) const;

  // Returns the preferred layout. If `alias` is not in `preferred_layout_`,
  // returns the `TensorView`'s initial layout.
  Layout preferredLayout(const Val* alias) const;

  // Marks `source` as the immediate aliasing source of `alias` and sets the
  // preferred layout.
  void add(const TensorView* alias, const TensorView* source, Layout&& layout);

 private:
  // Maps aliases (e.g. the output of a View) to their direct sources (e.g. the
  // input of the same View). Also stores the preferred output layout for the
  // alias. Consider path compression, a common optimization used in
  // disjoint-set data structure, so it's easy to figure out the root of an
  // alias.
  std::unordered_map<const TensorView*, std::pair<const TensorView*, Layout>>
      alias_to_source_;
};

// Finds aliases of the fusion inputs. The analysis should be conservative --
// when the analysis says B is an alias of input A,
// `ExpressionEvaluator::evaluate(B)` should produce an `at::Tensor` that's an
// alias of the `at::Tensor` bound to A.
AliasAnalysisResult findAliases(Fusion* fusion);

} // namespace nvfuser::optimization
