// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <device_lower/analysis/divisible_split.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <maxinfo_propagator.h>
// TODO: Move to cpp file.
#include <ir/builder.h>

#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;

namespace vectorize_helper {

// Projects IterDomains through the fusion starting at provided reference. IDs
// in the reference are expected to be "contiguous", simply means dimensions
// that the iter domains are consecutive and next to eachother in the
// reference. This property is not enforced, but mapping can have some
// unpredictbale results if they are not. The reason we want contiguity here
// is this class is primarily used for vectorization analysis. Domains may be
// inserted or removed while propogating through the fusion and this class has
// to be senstitive to that.
//
// For example:
// Input: T0[i0, i2]
// Reference: T5[i0, i1, i2]
// If we want to base the vectorization size on the reference being contiguous
// in a 1D scheduler, we'd start the proces on the reference with {i0, i1,
// i2}. When we propogate to the input what we would still like is: {i0, i1,
// i2} to signify to us that the root domains in the input that map to the
// reference are not contiguous. So when we think of vector word, if we want
// the input to be included in the vectorized dimensions, we can only check
// multiples based on i2, not i0*i1*i2 like the reference would indicate.
//
// Another example:
// Input:[i1, i0, i2]
// Refrence [i0, i1, i2]
// Similarly as above when we propogate from the reference to the Input we'd
// like {i0, i1, i2}, which is the order of the reference, not the input. This
// is because we can compare that with the input domains to understand it's
// not ordered consistently, so once again we can only take into consideration
// vectorization based on i2.
//
// Another example:
// Input:[i0, i1, i2]
// Intermediate: [i1, i0, i2]
// Refrence [i0, i1, i2]
// Keeping the ordering relative to the reference also allows us to look
// though transpose operations without missing out in a case like this that
// the reference and input are consistently ordered so we can look at i0*i1*i2
// for our vector multiple even though there are transposes in between them.
//
// The tricky part of this class is what happens through combinations of view
// and transpose. IterDomains are projected for example:
//   tv0[2*3, 5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// With tv1 and 7*11 as the reference and ids. When we project from tv1 to
// tv0, we'd map the inner most 11, but we also want to map the 5*7 with an
// extent of 7. This can get tricky though as:
//   tv0[2, 3*5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// with tv1 and [2*3, 7*11] as the reference and ids. tv0's 2 and 11 dim are
// easily identified as being mapped. The 3*5*7 dimension however, is
// partially mapped on the left and right side. Since this class is  intended to
// line up "inner dimensions" of tensors through out the graph for the purpose
// of unrolling and vectorization, it only tracks partial dimensions as they are
// on the right hand side of iteration domains. For example in the last case we
// would only identify tv0's 3*5*7 dimension as being a mapping with extent 7.
// If we further had:
//   tv0[5*7*11]
//   tv1[5*7, 11] = view(tv0)
//   tv2[5, 7*11] = view(tv1)
// with tv2 and [7*11] as the reference and ids (this could be a valid example
// from the pointwise scheduler).
// (1) tv1 would:
//     map on 5*7 with extent 7
//     map on 11 with extent 11.
// (1) tv0 would:
//     map on 5*7*11 with size 7*11
//
// Finally if we have:
// tv0[3, 5, 7]
// tv1[7, 5, 3] = view(tv0)
// tv2[3, 5, 7] = view(tv1)
// with tv2 mapping on 5, 7
// We use fractional, symbolic, and conditional mappings so tv1:
//   maps on 3 with extent 3
//   maps on 5 with extent 5
//   maps on 7 with extent (5*7)/(5*3)
// Then tv0:
//   maps on 7 with extent 7
//   maps on 5 with extent 5
//
// This class is responsible for both computing the spanning tree and running
// the spanning tree.
//
// In other words this class implements:
//   MaxInfoSpanningTree::computeInfoC2P
//     and
//   MaxInfoSpanningTree::Propagator::propagateC2P
//
// The challenge here is the information we need for
// MaxInfoSpanningTree::computeInfoC2P is the same information we need to
// compute for MaxInfoSpanningTree::Propagator::propagateC2P
//
// We could compute both of these passes at the same time, only saving the
// result produced from processing the edge that's chosen from
// MaxInfoSpanningTree::computeInfoC2P while processing based on
// MaxInfoSpanningTree::Propagator::propagateC2P. However, this would require
// refactoring of MaxInfoSpanningTree so for right now this class just uses
// two passes.
//
// MaxInfoSpanningTree::computeInfoC2P runs first with recording_=false and
// will effectively compute the values of projected_root_ids_ and
// projected_rfactor_ids_. However it will compute these by running all edges
// between expressions. Therefore,
// MaxInfoSpanningTree::Propagator::propagateC2P later simply calls
// MaxInfoSpanningTree::computeInfoC2P with recording_=true where it will
// actually record the computed information since it will be then projected
// through the DAG maximizing saving information.
class TORCH_CUDA_CU_API ContiguousInnerDimensionsMapper
    : public MaxInfoSpanningTree,
      MaxInfoSpanningTree::Propagator {
 public:
  ContiguousInnerDimensionsMapper() = delete;

  static ContiguousInnerDimensionsMapper map(
      TensorView* reference,
      const std::vector<IterDomain*>& ids,
      std::shared_ptr<const ComputeAtMap> ca_map,
      const std::unordered_set<Split*>& divisible_splits);

  static ContiguousInnerDimensionsMapper map(
      TensorView* reference,
      const std::vector<IterDomain*>& ids) {
    auto ca_map = std::make_shared<ComputeAtMap>(reference->fusion());
    auto divisible_splits =
        getAllDivisibleSplits(reference->fusion(), ca_map.get());
    return ContiguousInnerDimensionsMapper::map(
        reference, ids, ca_map, divisible_splits);
  }

  bool hasMappedDims(TensorView* tv) const {
    return tv_infos_.find(tv) != tv_infos_.end();
  }

  const std::vector<IterDomain*>& mappedRootIds(TensorView* tv) const {
    TORCH_INTERNAL_ASSERT(
        tv_infos_.find(tv) != tv_infos_.end(),
        "TensorView not found: ",
        tv->toString());
    return std::dynamic_pointer_cast<const MappedDomain>(tv_infos_.at(tv))
        ->mapped_root_ids_;
  }

  const std::vector<IterDomain*>& mappedRFactorIds(TensorView* tv) const {
    TORCH_INTERNAL_ASSERT(
        tv_infos_.find(tv) != tv_infos_.end(),
        "TensorView not found: ",
        tv->toString());
    return std::dynamic_pointer_cast<const MappedDomain>(tv_infos_.at(tv))
        ->mapped_rfactor_ids_;
  }

  Val* getProjectedExtent(IterDomain* id) {
    if (projected_extent_.find(id) == projected_extent_.end()) {
      projected_extent_[id] = id->container()->oneVal();
    }
    return projected_extent_.at(id);
  }

  std::unordered_map<TensorView*, Val*> getTvToContigMergeOfInnerSizeMap();

 private:
  ContiguousInnerDimensionsMapper(
      TensorView* reference,
      const std::vector<IterDomain*>& reference_ids,
      std::shared_ptr<const ComputeAtMap> ca_map,
      const std::unordered_set<Split*>& divisible_splits);

  class MappedDomain : public Information {
   public:
    MappedDomain() = default;

    static std::shared_ptr<MappedDomain> build(
        std::vector<IterDomain*> root_ids,
        std::vector<IterDomain*> rfactor_ids,
        bool is_c2p) {
      auto ptr = std::make_shared<MappedDomain>();
      ptr->mapped_root_ids_ = root_ids;
      ptr->mapped_rfactor_ids_ = rfactor_ids;
      ptr->is_c2p_ = is_c2p;
      return ptr;
    }

    operator bool() const final {
      return !mapped_root_ids_.empty() || !mapped_rfactor_ids_.empty();
    }

    bool operator<(const Information& other_info) const final {
      auto other_mapped_domain = dynamic_cast<const MappedDomain&>(other_info);

      if (is_c2p_) {
        return mapped_rfactor_ids_.size() <
            other_mapped_domain.mapped_rfactor_ids_.size();
      }
      return mapped_root_ids_.size() <
          other_mapped_domain.mapped_root_ids_.size();
    }

    std::vector<IterDomain*> mapped_root_ids_;
    std::vector<IterDomain*> mapped_rfactor_ids_;
    // Information is not symmetric between c2p and p2c, track which direction
    // the computation is in for the < operator
    bool is_c2p_ = true;
  };

  // TODO: make pe a lanmda function so it is not evaluated if not needed
  void addProjectedExtent(IterDomain* id, Val* pe) {
    if (!recording_) {
      return;
    }
    projected_extent_[id] = pe;
  }

  // Return a boolean predicate indicating if the given ID is fully projected.
  Val* isFullyProjected(IterDomain* id);

  // From the projected extent (PE) of I1 and I2, update the PE of I1*I2.
  template <typename MergeOrSplit>
  void combinePE(const MergeOrSplit* merge_or_split, bool outer_maps);
  // From the projected extent (PE) of I1*I2, update the PE of I1 and I2.
  template <typename MergeOrSplit>
  void distributePE(const MergeOrSplit* merge_or_split);

  // Returns the projected inner size. Contiguous inner dimensions are merged.
  Val* getContigMergeOfInnerSize(TensorView* of_tv);

  // MaxInfoSpanningTree functions
  std::shared_ptr<Information> computeInfoC2P(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) final;

  std::shared_ptr<Information> computeInfoP2C(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) final;

  std::shared_ptr<Information> computeInfoSibling(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) final;

  // Projection from root<->rfactor domains
  std::vector<IterDomain*> projectId(
      const std::vector<IterDomain*>& from,
      const std::vector<IterDomain*>& to);

  // Propagator functions
  void propagateC2P(TensorView* from, TensorView* to) final;
  void propagateP2C(TensorView* from, TensorView* to) final;
  void propagateSibling(TensorView* from, TensorView* to) final;

  // Initialized to false, series of compute... calls will be performed to find
  // the spanning tree. Then propagate... calls will call the compute... calls.
  // recording_ starts as false, and stays that way during the first series of
  // compute... calls. As soon as the first propagate... calls are called,
  // recording_ will perpetually stay on.
  bool recording_ = false;

  std::shared_ptr<const ComputeAtMap> ca_map_;
  const std::unordered_set<Split*>& divisible_splits_;

  // Mapped root dimensions for each TensorView as we propogate. These
  // mappings are in the order of the reference.

  std::unordered_map<
      TensorView*,
      std::shared_ptr<MaxInfoSpanningTree::Information>>
      tv_infos_;

  std::unordered_map<IterDomain*, Val*> projected_extent_;
};

size_t getVectorizationFactor(
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference_tv,
    HeuristicSummary* data_cache,
    int break_point);

} // namespace vectorize_helper
} // namespace nvfuser
