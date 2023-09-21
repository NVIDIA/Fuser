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
#include <transform_iter.h>

#include <algorithm>
#include <vector>

namespace nvfuser {

// TODO: Only replay dispatch is really borrowed from TransformIter, we should
// reevaluate the reuse of dispatch for classes that inherit TransformIter.
class TransformRFactor {
 public:
  // Transform the provided tensor domain to two domains, a producer and
  // consumer domain. These domains are created by taking axes and reducing them
  // in the producer domain, and taking the remaining reduction axes and
  // reducing them in the consumer domain.
  static std::pair<TensorDomain*, TensorDomain*> runReplay(
      TensorDomain*,
      std::vector<int> axes);
};

} // namespace nvfuser
