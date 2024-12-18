// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/pointwise_utils.h>

namespace nvfuser {
namespace pointwise_utils {

TensorView* PointwiseDomainMap::findReferenceTensor(
    int64_t minimum_num_axes) const {
  TensorView* result = nullptr;
  int64_t max_dims = -1;
  for (auto output_tv :
       ir_utils::filterByType<TensorView>(fusion_->outputs())) {
    if (isValidReference(output_tv) &&
        hasMinimumSize(output_tv, minimum_num_axes) &&
        !output_tv->isFusionInput()) {
      int64_t n_dims = nLogicalDims(output_tv);
      if (n_dims > max_dims) {
        result = output_tv;
        max_dims = n_dims;
      }
    }
  }
  return result;
}

TensorView* getReferenceTensor(Fusion* fusion) {
  FusionGuard fg(fusion);
  PointwiseDomainMap domain_map(fusion);
  auto reference_tv = domain_map.findReferenceTensor();
  return reference_tv;
}

} // namespace pointwise_utils
} // namespace nvfuser
