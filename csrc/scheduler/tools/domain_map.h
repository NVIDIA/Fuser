// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>

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
  bool isValidReference(TensorView* tv) const;

 protected:
  // Determine if all IterDomains are mapped between input and the given tvs
  bool areAllInputIdsMappedTo(TensorView* input_tv, TensorView* output_tv)
      const;

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
};

} // namespace scheduler_tools
} // namespace nvfuser
