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

#include <ir/all_nodes.h>

#include <memory>
#include <vector>

namespace nvfuser {

class ViewTransform;

//!
//! The goal of analyzeView is to find the minimum number of transformations
//! to convert from the original size to the new size. A naive view algorithm
//! would merge all axis together and then split according to the new sizes.
//!
//! This implementation will keep the original domains, if the domains are the
//! same size in the original and new shapes. If an original domain is not
//! evenly divisible by the new domain, we will merge the minimum number of
//! adjacent original domains.
//!
//! The view transformations are processed in the following order:
//! 1. Squeeze - Removes size-1 broadcast dimensions
//! 2. Keep, Merge, Split - Used to create new rfactor domain
//! 3. Broadcast - Inserts size-1 dimensions
//!
//! Broadcast is handled last because size-1 dimension can be inserted anywhere
//! in the new shape.
//!

struct AnalyzeViewResult {
  std::vector<bool> broadcast_axes;
  std::vector<bool> squeeze_axes;
  std::vector<std::shared_ptr<ViewTransform>> transforms;

  std::string toString() const;

  bool operator==(const AnalyzeViewResult& other) const;

  bool operator!=(const AnalyzeViewResult& other) const {
    return !(*this == other);
  }

  size_t hash() const;
};

struct AnalyzeViewConstraint {
  // 1 if size 1 dimension, otherwise 0;
  std::vector<int64_t> original_constraint;
  std::vector<int64_t> new_constraint;
  // Just the positions of true in AnalyzeViewResult::squeeze_axes
  std::vector<int64_t> squeeze_string;
  // Just the positions of true in AnalyzeViewResult:broadcast_axes
  std::vector<int64_t> broadcast_string;
  // A stringified version of the transformations:
  std::vector<int64_t> split_merge_string;

  std::vector<int64_t> conglomerateString() const {
    // Don't think this is necessary but just being safe. Using
    // -3 as a dilimeter between value groups.
    std::vector<int64_t> conglomerate = {
        (int64_t)original_constraint.size(),
        (int64_t)new_constraint.size(),
        -3};
    auto add_vec = [&conglomerate](const std::vector<int64_t>& vec) {
      conglomerate.insert(conglomerate.end(), vec.begin(), vec.end());
      conglomerate.push_back(-3);
    };
    add_vec(original_constraint);
    add_vec(new_constraint);
    add_vec(squeeze_string);
    add_vec(broadcast_string);
    add_vec(split_merge_string);
    return conglomerate;
  }

  bool operator==(const AnalyzeViewConstraint& other) const {
    return other.conglomerateString() == this->conglomerateString();
  }

  // Naive hashing function, likely has a lot of collisions, but may not matter
  // too much if we don't expact many types of views.
  size_t hash() const {
    size_t hash_value = 0;
    for (auto val : conglomerateString()) {
      if (val == std::numeric_limits<int64_t>::max()) {
        continue;
      }
      hash_value += val;
    }
    return hash_value;
  }
};

//! Infer -1 value in new view std::vector<int64_t> based on original view
//! std::vector<int64_t>. This shouldn't generally be used directly but is
//! useful for testing.
std::pair<std::vector<int64_t>, std::vector<int64_t>> inferViewShapes(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

// Find the transformations necessary to convert TensorView
// from original size to new size.
AnalyzeViewResult analyzeView(
    const TensorView* tv,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

// Find the constraints derived from the view transformations
AnalyzeViewConstraint analyzeViewConstraint(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

// Generate a new TensorDomain from the given view transformations.
// The original root domain is kept in the new TensorDomain,
// but a new rfactor domain is created from the view transformations.
TensorDomain* transformView(
    TensorDomain* original_domain,
    const AnalyzeViewResult& view_analysis);

//! Apply the reshape transformations of view_analysis to inp_tv
TensorView* reshape(TensorView* inp_tv, const AnalyzeViewResult& view_analysis);

} // namespace nvfuser
