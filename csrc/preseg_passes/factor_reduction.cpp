// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/factor_reduction.h>

#include <unordered_set>
#include <vector>

#include <ir/utils.h>

namespace nvfuser::preseg_passes {

void FactorReductionPass::runPass(Fusion* fusion) {
  // RFactor greatest common subset
  //
  // std::unordered_set<IterDomain*> id_subset;
  // std::unordered_set<TensorView*> tv_subset;
  // TensorView* reference_tv;
  //
  // std::vector<TensorView*> reduction_tvs =
  // scheduler_utils::getReductionTvs(fusion); 
  // if (reduction_tvs.size() < 1) {
  //   return;
  // }
  //
  // for (TensorView* tv : reduction_tvs) {
  //   const std::vector<IterDomain*>& tv_root_domain =
  //   reference_tv->getRootDomain(); if (reference_tv == nullptr) {
  //     reference_tv = tv;
  //     id_subset.insert(tv_root_domain.begin(), tv_root_domain.end());
  //     tv_subset.insert(tv);
  //     continue;
  //   }
  //
  //   std::unordered_set<IterDomain*> common_ids;
  //   size_t mismatch_ids = 0;
  //   for (IterDomain* id : id_subset) {
  //     map reference_id to this_tv_id
  //     if this_tv_id exists {
  //       common_ids.insert(id);
  //     } else {
  //       ++mismatch_ids;
  //     }
  //   }
  //
  //   if (mismatch_ids == id_subset.size()) {
  //     // rfactor greatest common subset
  //     if (tv_subset.size() == 1) {
  //       continue;
  //     }
  //
  //     // map common reduction axes to integer
  //     for (TensorView* tv : tv_subset) {
  //       for (IterDomain* id : id_subset) {
  //         map reduction id to this_tv_id
  //         get integer for this_tv_id
  //       }
  //       rfactor common reduction axes
  //     }
  //
  //     // reset state
  //     id_subset.clear();
  //     tv_subset.clear();
  //     reference_tv == nullptr;
  //
  //     continue;
  //   }
  //
  //   swap(id_subset, common_ids);
  //   tv_subset.append(tv);
  // }
}

} // namespace nvfuser::preseg_passes
