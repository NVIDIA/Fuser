// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dispatch.h>
#include <fusion.h>
#include <ir/all_nodes.h>

namespace nvfuser {

// TensorViews are all based on symbolic sizes. When we first initialize them
// we don't know if they're inputs or outputs which would mean that they have
// runtime shapes. Intermediate tensors (those not going to global memory) do
// not have this information. Since we need to have the correct information in
// the kernel being fetched for shapes, we want to replace input and output
// tensors to reference the runtime structure containing sizes.
void replaceSymbolicSizes(Fusion*);

// Going to generate a map of tensor view root domain extents to reduce the
// number used during lowering. For example if we have:
//
// T2[i0, i1] = T1[i0, i1] + T2[i2, i3]
//
// We know it would be safe to use:
//
// T2[i0, i1] = T1[i0, i1] + T2[i0, i1]
//
// And that way we don't generate T2.size[0] and T2.size[1], instead we will
// reuse T1.size[0] and T1.size[1]
// This is important when doing CSE as T2 and T1 would otherwise look like
// they're using different values, even though we know they're the same
//
// There's some duplicate logic here that's in computeAt map, but it's not so
// concice there to pull out. May want to consider making this mapping its own
// class especially as it may be useful during scheduling.
std::unordered_map<Val*, Val*> getSimplificationMap(Fusion* fusion);

} // namespace nvfuser
