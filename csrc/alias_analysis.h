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

namespace nvfuser {

struct Layout {
  std::vector<IterDomain*> allocation_domain;
  std::vector<std::optional<bool>> contiguity;

  std::string toString(int indent_size = 0) const;

  // Returns whether this layout is compliant with `required`. This is
  // uni-directional. For example, `contiguity=[t,t]` is compliant with
  // `contiguity=[f,f]` but not vice versa. As a special case,
  // an empty `required.allocation` indicates no requirements, i.e., the method
  // always returns true.
  bool isCompliantWith(const Layout& required) const;
};

// Holds aliases found in a fusion. The expected user flow is
//
// ```
// AliasAnalysisResult analysis;
// analysis.add(...);
// ...
// analysis.add(...);
// analysis.finalize(fusion, ...);
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
  void add(const TensorView* alias, TensorView* source, Layout&& layout);

  // See `findAliases` for the meaning of
  // `can_override_empty_allocation_domain`.
  void finalize(Fusion* fusion, bool can_override_empty_allocation_domain);

  // Returns the preferred layout. If `alias` is not in `preferred_layout_`,
  // returns the `TensorView`'s initial layout.
  Layout preferredLayout(const Val* alias) const;

  std::string toString(int indent_size) const;

  // Gets the nearest aliased fusion input/output of a `fusion_out` other than
  // `fusion_out` itself. Returns null if that doesn't exist.
  TensorView* getNearestAliasedIo(const TensorView* fusion_out) const;

 private:
  // Same as `getNearestAliasedIo` except that the `get` method returns the
  // cached result.
  TensorView* findNearestAliasedIo(TensorView* fusion_out) const;

  // Maps aliases (e.g. the output of a View) to their direct sources (e.g. the
  // input of the same View). Also stores the preferred output layout for the
  // alias. Consider path compression, a common optimization used in
  // disjoint-set data structure, so it's easy to figure out the root of an
  // alias.
  std::unordered_map<const TensorView*, std::pair<TensorView*, Layout>>
      alias_to_source_;

  // Maps a fusion output to its nearest aliased fusion input/output.
  std::unordered_map<const TensorView*, TensorView*> out_to_root_;
};

// Finds aliases of the fusion inputs. The analysis should be conservative --
// when the analysis says B is an alias of input A and that B's preferred layout
// is compliant with the required layout, `ExpressionEvaluator::evaluate(B)`
// should produce an `at::Tensor` that's an alias of the `at::Tensor` bound to
// A.
//
// [Note on overriding empty allocation domains]
//
// We can override an empty allocation domain to any layout before segmentation
// but not after. For example,
// ```
// auto in = makeContigConcreteTensor({2, 3});
// auto slice_out = segment_set(slice(in, {0, 0}, {2, 2}));
// auto add_out = add(slice_out, slice_out)
// ```
// will be split at `slice_out` into two segments. `slice_out`'s contiguity
// needs to be [f,t] so it can be made an alias. If we were to let a scheduler
// (thus after segmentation) change its contiguity, we would have to change the
// input contiguity of the second segment as well. This is possible but hard to
// implement given the current infrastructure.
//
// Therefore, I chose to run alias analysis both before segmentation and in
// schedulers. The former, used by OptimizeLayoutPass, updates layouts to enable
// aliases; the latter, used by NoOpScheduler, calls Fusion::aliasOutputToInput
// to mark aliases.
AliasAnalysisResult findAliases(
    Fusion* fusion,
    bool can_override_empty_allocation_domain = true);

} // namespace nvfuser
