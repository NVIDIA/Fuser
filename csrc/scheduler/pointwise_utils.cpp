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

TensorView* getReferenceTensor(Fusion* fusion) {
  FusionGuard fg(fusion);
  scheduler_tools::PointwiseDomainMap domain_map(fusion);
  auto reference_tv = domain_map.findReferenceTensor();
  return reference_tv;
}

} // namespace pointwise_utils
} // namespace nvfuser
