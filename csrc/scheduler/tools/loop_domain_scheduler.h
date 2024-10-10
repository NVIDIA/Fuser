// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

namespace nvfuser {

class TensorView;
class IterDomain;

namespace scheduler_tools {

// Create the loop domain of given tensors as specified by the
// reference. The new loop domain is connected to the existing IDs of
// each tensor by replaying exprs found in the Exact ValGraph.
void scheduleLoopDomainsLike(
    const std::vector<TensorView*>& tvs,
    const std::vector<IterDomain*>& ref_loop_dom);

} // namespace scheduler_tools
} // namespace nvfuser
