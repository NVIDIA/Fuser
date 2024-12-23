// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <scheduler/tools/domain_map.h>
#include <scheduler/utils.h>

namespace nvfuser {
namespace pointwise_utils {

// Returns number of non-reduction/non-broadcas/non-device dims in logical
// domain
inline int64_t nLogicalDims(const TensorView* tv) {
  auto logical_dom = tv->getLogicalDomain();
  int64_t tv_n_dims = 0;
  for (auto dim : logical_dom) {
    if (!dim->isReduction() && !dim->isBroadcast() && !dim->isDeviceDim()) {
      tv_n_dims++;
    }
  }
  return tv_n_dims;
}

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

// Return reference tensor view.
TensorView* getReferenceTensor(Fusion* fusion);

} // namespace pointwise_utils
} // namespace nvfuser
