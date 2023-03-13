// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <ir_all_nodes.h>
#include <ir_utils.h>
#include <scheduler/utils.h>

namespace nvfuser {
namespace pointwise_utils {

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
  // The reference tensor must map to all the iterDomains in each input.
  bool isValidReference(TensorView* tv) const;

  // Determine if and ID is a selected ID of some SelectOp
  bool isSelectId(IterDomain* id) const {
    return select_ids_.count(id);
  }

 protected:
  // Determine if all IterDomains are mapped between input and the given tvs
  bool areAllInputIdsMappedTo(TensorView* input_tv, TensorView* output_tv)
      const;

  // Erase input concrete ID if it is mapped to output ID
  bool eraseIfMapped(
      std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* out_id) const;

  // Check if in_id is mapped to id through any view rfactor domain
  void eraseIfInputMappedThroughViewTo(
      std::unordered_set<IterDomain*>& in_concrete_ids,
      IterDomain* id) const;

  // Find any id in domain that maps with target id
  IterDomain* anyMapped(
      const std::vector<IterDomain*>& domain,
      IterDomain* target) const;

  Fusion* fusion_ = nullptr;
  ComputeAtMap ca_map_;
  std::vector<TensorView*> view_tvs_;
  std::unordered_set<IterDomain*> select_ids_;
};

// Returns number of non-reduction/non-broadcast dims in rfactor domain
inline size_t nRootDims(const TensorView* tv) {
  auto root_dom = tv->getMaybeRFactorDomain();
  size_t tv_n_dims = 0;
  for (auto dim : root_dom) {
    if (!dim->isReduction() && !dim->isBroadcast()) {
      tv_n_dims++;
    }
  }
  return tv_n_dims;
}

} // namespace pointwise_utils
} // namespace nvfuser
