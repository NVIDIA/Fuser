// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <scheduler/utils.h>

#include <unordered_set>
#include <vector>

namespace nvfuser {

class Fusion;
class TensorView;
class IterDomain;

namespace scheduler_tools {

// DomainMap uses the ComputeAtMap to find a reference TensorView
// that maps to all IterDomains in the fusion.
class DomainMap {
 public:
  DomainMap(Fusion* fusion);
  virtual ~DomainMap() = default;

  const ComputeAtMap& getComputeAtMap() const {
    return ca_map_;
  }

  // Determine if a TensorView is a valid reference tensor for this fusion.
  // The reference tensor must map to all the iterDomains in each input and
  // output.
  bool isValidReference(TensorView* tv, bool check_inputs = true);

 protected:
  // Determine if all IterDomains are mapped between input and the given tvs
  bool areAllInputIdsMappedTo(TensorView* input_tv, TensorView* output_tv);

  const scheduler_utils::CoveredDomainPropagator& getCoveredDomainPropagator(TensorView* reference_tv);

  // Determine if all source IterDomains in target_tv are contained by the
  // reference_tv, this ensures transformations from reference_tv can be
  // propagated to target_tv
  bool areAllTargetIdsCoveredBy(TensorView* target_tv, TensorView* reference_tv)
      const;

  virtual IterDomain* getMappedInputConcreteID(
      const std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const;

  // Erase input concrete ID if it is mapped to output ID
  bool eraseIfMapped(
      std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const;

  // Check if in_ids are mapped to ids through any root domain as
  // well as indirectly accessed domains with ops like torchGather
  void eraseifInputMappedThroughRootDomainAndIndexing(
      std::unordered_set<IterDomain*>& in_ids,
      const std::vector<IterDomain*>& ids) const;

  // Find any id in domain that maps with target id
  IterDomain* anyMapped(
      const std::vector<IterDomain*>& domain,
      IterDomain* target) const;

  Fusion* fusion_ = nullptr;
  ComputeAtMap ca_map_;
  std::vector<TensorView*> tvs_with_rfactor_;
  std::unordered_map<TensorView*, scheduler_utils::CoveredDomainPropagator> covered_domain_propagators_;
};

class PointwiseDomainMap : public scheduler_tools::DomainMap {
 public:
  using scheduler_tools::DomainMap::DomainMap;

  // The pointwise scheduler heuristics requires a minimum number of axes.
  // The output reference tensor should respect this requirement.
  TensorView* findReferenceTensor(int64_t minimum_num_axes = 0) const;

 private:
  bool hasMinimumSize(TensorView* tv, int64_t num_axes) const {
    NVF_ERROR(tv != nullptr);
    return (num_axes == 0 || (int64_t)tv->getLogicalDomain().size() > num_axes);
  }
};

// DomainMap uses the ComputeAtMap to find a reference TensorView
// that maps to all iterDomains in the fusion.
class TransposeDomainMap : public scheduler_tools::DomainMap {
 public:
  using scheduler_tools::DomainMap::DomainMap;

  // Note that this may not be able to find any reference if any
  // tensor in the group is only connected with an input through
  // rfactor or gather-like indexing ops. It is because
  // isValidReference is based a backward traversal, so there may not
  // be a traversal path to an input. This type of analysis is
  // expected to be possible much more easily with the new indexing
  // graph (#32), so we should revisit once it becomes available.
  TensorView* findReferenceFor(const std::vector<TensorView*>& group) const;

  IterDomain* getMappedAllocDimIn(TensorView* tv, IterDomain* root_dim) const;

  static bool hasAtLeastTwoValidGroups(Fusion* fusion);

  // scheduler assumes inner loop dimension on tv is an exact mapping, when the
  // mapping cannot be resolved, we'll return a `-1`
  int64_t getInnerLeafDim(TensorView* tv, IterDomain* root_dim) const;

  // Group inputs and outputs of a fusion by its inner most domain. For example
  //   inputs: t0, t1
  //   t2 = transpose(t1)
  //   t3 = t0 + t2
  //   t4 = sin(t0)
  //   t5 = cos(t1)
  //   outputs: t3, t4, t5
  //
  // Then we should have group {t0, t3, t4} and {t1, t5}
  //
  // The returned groups are sorted in descending size. If the sizes of two
  // group are equal, then we sort them by their members in the following order:
  //   output[0], output[1], ..., input[0], input[1], ...
  // That is, {ouput[0], output[2]} will be in front of {ouput[1], output[3]}
  // The order here must be deterministic, because in transpose heuristics, we
  // have `vectorize_factor1` and `vectorize_factor2` and we need to be sure
  // that `1` and `2` are assigned to the same group across runs.
  //
  // In the case where view is present in the graph, there are two cases: if the
  // view doesn't touch any inner dimension of any group, then the support of it
  // is trivial. In the case where view actually touches an inner-most dim, we
  // keep track of the inner-most dimension of view's split and merges.
  //
  // For example, if you have:
  //   T0 [2, 3, 5] <-- input
  //   T1 [2, 5, 3] <-- input
  //   T2 [2, 5, 3] = transpose(T0) + T1
  //   T3 [2, 15] = view(T2)
  //   output <-- T3
  //
  // Then T3 should be in the same group with T1, and T0 should have
  // different group with T1 and T3.
  std::vector<std::vector<TensorView*>> groupInputsOutputsByInnerDim() const;

  // In the transpose scheculing, unlike the pointwise scheduling, the
  // permissive map is required to find reference tensors. See also PR
  // #661
  IterDomain* getMappedInputConcreteID(
      const std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const override;
};

} // namespace scheduler_tools
} // namespace nvfuser
