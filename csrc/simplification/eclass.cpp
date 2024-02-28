// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <simplification/eclass.h>
#include <simplification/egraph.h>

#include <optional>

namespace nvfuser {

namespace egraph {

void EClass::mergeFrom(const Id other_id) {
  EGraph* eg = EGraphGuard::getCurEGraph();

  EClass* other = eg->getEClassFromId(other_id);

  if (other == this) {
    return;
  }

  // See Fig. 9 of Willsey et al. 2021
  data.joinFrom(other->data);
}

} // namespace egraph

} // namespace nvfuser
