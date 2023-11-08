// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>

#include <fusion.h>
#include <ir/interface_nodes.h>

namespace nvfuser::optimization {

class AliasAnalysisResult {
 public:
  AliasAnalysisResult() = default;

  // Returns itself if `alias` doesn't alias anything.
  const Val* findRoot(const Val* alias) const;

  // Marks `source` as the immediate aliasing source of `alias`.
  void add(const TensorView* alias, const TensorView* source);

  AliasAnalysisResult(const AliasAnalysisResult&) = delete;
  AliasAnalysisResult& operator=(const AliasAnalysisResult&) = delete;
  AliasAnalysisResult(AliasAnalysisResult&&) = default;
  AliasAnalysisResult& operator=(AliasAnalysisResult&&) = default;

 private:
  // Maps aliases (e.g. the output of a View) to their direct sources (e.g. the
  // input of the same View). Consider path compression, a common optimization
  // used in disjoint-set data structure, so it's easy to figure out the root of
  // an alias.
  std::unordered_map<const TensorView*, const TensorView*> alias_to_source_;
};

// Finds aliases of the fusion inputs.
AliasAnalysisResult findAliases(Fusion* fusion);

} // namespace nvfuser::optimization
