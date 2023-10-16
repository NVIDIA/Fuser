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

//! There's three domains associated with performing a view operation:
//! 1) Original Domain:
//!   This view is the original input to the view operation. It has no
//!   transforms on it, it is however passed in without its reduction domains
//!   (as is expected since we're trying to generate the output of the
//!   operations).
//!
//! Squeezed domain:
//!   Predicting which operations are squeezed are not trivial. If a broadcast
//!   is between two iter domains in the original domain that must be merged for
//!   the view transform:
//!     - If the broadcast domain lines up with a broadcast domain in the final
//!       tensor domain keep it.
//!     - If the domain is size-1 but not marked as a broadcast domain (runtime
//!       size==1)
//!       Note: This isn't something we generally support consistently
//!     - If the broadcast domain is marked as a compile time broadcast domain,
//!       and doesn't line up with a broadcast domain in the final result.
//!       squeeze it.
//!   The index for these transformations is marked as the index of the original
//!   domain, as that's the input for the squeeze. This produces the squeezed
//!   domain.
//!
//! Post-view Domain:
//!   This domain is the original domain after the squeeze and all
//!   transformations. This domain holds the rfactor domains determined by
//!   merge/split operations of the find transformations pass. It is the final
//!   domain without all the broadcast operations (can have some that were
//!   preserved through the transformations).
//!       For example: {1, 2, 1, 4} -> {1, 2, 1, 2, 2} doesn't have any
//!         conflicts of the view transformation and the broadcast dimensions,
//!         so they won't be squeezed, they will simply be propagated
//!         through the view.
//!         {1, 2, 1, 4} -> {1, 8, 1} does have the second 1 dimension in
//!         between the 2 and 8 that have to be merged. The first broadcast axis
//!         will be propagated through the domains unafected, yet the second
//!         braodcast axis will be squeezed, then rebroadcasted.
//!  The transformation index marked for the splits/merges to produce this
//!  domain are done based on an "in progress" tensor view (called transform
//!  view index in the find transformation pass). This allows us to simply apply
//!  these transformations serially to produce this domain.
//!
//! Post-broadcast Domain:
//!    This domain finally matches the output of the view operation fully and
//!    can be used in further computations.
//!
//! View process at compute time:
//!   1) View takes in the input TensorView x, original runtime
//!      std::vector<int64_t>, and viewed runtime std::vector<int64_t>.
//!   2) AnalyzeView is called Which will figure out what series of
//!      transformations is required from the input tensor to the output tensor.
//!      These transformations are recorded.
//!   3) Squeeze operation is called on the squeezed axes from the analysis.
//!   4) applyViewTransforms will generate the output domain of the view
//!      operation.
//!        Calls TensorDomain::view(view_analysis) which returns the rfactored
//!        domain.
//!        Gets forwarded to transformView(TensorDomain, view_analysis)
//!        Gets forwarded to createViewDomain(TensorDomain, view_analysis)
//!        createViewDomain creates the new root domain, and calls
//!        createRfactorDomain on view_analysis.transforms().
//!   5) brooadcast will be called with view_analysis.broadcast_axes
//!
//! TODO: Caching assumes that all size-1 inputs are correctly marked as a
//! broadcast dimension. We should probably remove the runtime size-1 merge
//! support in find transformation.
//!
//! Simple abstract class to record transformation and the indices required to
//! apply it.
class Transform : public PolymorphicBase {
 public:
  virtual std::string toString() const = 0;

  int64_t index() const {
    return index_;
  }

  bool operator==(const Transform& other) const {
    return index() == other.index();
  }

  bool operator!=(const Transform& other) const {
    return !(*this == other);
  }

 protected:
  // Relevant location information for the transformation. Stored information is
  // related to when we have to apply that transformation (see long comment at
  // top of this file).
  Transform(int64_t index) : index_(index) {}

  const int64_t index_ = 0;
};

class ViewTransform : public Transform {
 public:
  // Function to apply the transformation. Transformation is applied on
  // current_transformed_domain. root_domain is required here to replace
  // IterDomains so we can flip the rfactor flag on the root domain if it's
  // involved in merge/split trasnforms to produce the rfactor domain.
  virtual void createRfactorDomain(
      std::vector<IterDomain*>& root_domain,
      std::vector<IterDomain*>& current_transformed_domain) = 0;

  // Convenience function to replace id in root_domain with an id that has
  // expand expanded, and rfactor flag turned on.
  static IterDomain* replaceRootIdWithRFactor(
      std::vector<IterDomain*>& root_domain,
      IterDomain* id);

  // Debugging utility to convert the transformation into a string.
  std::string toString() const override = 0;

  bool operator==(const ViewTransform& other) const {
    return Transform::operator==(other);
  }

  bool operator!=(const ViewTransform& other) const {
    return !(*this == other);
  }

 protected:
  ViewTransform(const int64_t& index) : Transform(index) {}
};

//! The merge tranformation either combines two root iterDomains together OR
//! the last rfactor iterDomain with a root iterDomain. Unlike the general
//! TensorView merge there's no merging across axes not placed in consecutive
//! positions for View.
class MergeTransform final : public ViewTransform {
 public:
  MergeTransform(int64_t index) : ViewTransform(index) {}

  std::string toString() const override;

  void createRfactorDomain(
      std::vector<IterDomain*>& root_domain,
      std::vector<IterDomain*>& current_transformed_domain) override;

  bool operator==(const MergeTransform& other) const {
    return ViewTransform::operator==(other);
  }

  bool operator!=(const MergeTransform& other) const {
    return !(*this == other);
  }
};

//! The split tranformation creates two new iterDomains via an outer split.
class SplitTransform final : public ViewTransform {
 public:
  SplitTransform(const int64_t index, int64_t split_factor);

  std::string toString() const override;

  void createRfactorDomain(
      std::vector<IterDomain*>& root_domain,
      std::vector<IterDomain*>& current_transformed_domain) override;

  int64_t split_factor() const {
    return split_factor_;
  }

  bool operator==(const SplitTransform& other) const {
    return ViewTransform::operator==(other) &&
        split_factor_ == other.split_factor_;
  }

  bool operator!=(const SplitTransform& other) const {
    return !(*this == other);
  }

 private:
  const int64_t split_factor_ = 0;
};

//! For any singleton dimensions in the new view, we create an implicit
//! broadcast dimension. We apply these transforms after the squeeze
//! and view transformation steps.
class BroadcastTransform final : public Transform {
 public:
  BroadcastTransform(int64_t index) : Transform(index) {}

  std::string toString() const override;
};

//! For any implicit broadcast dimensions in the original view, we remove
//! them using squeeze.
class SqueezeTransform final : public Transform {
 public:
  SqueezeTransform(int64_t index) : Transform(index) {}

  std::string toString() const override;
};

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
