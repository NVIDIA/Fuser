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

// GetMetaData and MetaDataAccessor nodes can be used to reference tensor sizes.
// This is important for corect coordination of sizes in multi-gpu scenarios.
// See MultiDeviceTest.Issue2758 for an example where a "size" will be different
// depending on the expression. i.e. even though T0 and T1 might both have i0 as
// a size. T0 might be non sharded for that dim and T1 sharded for that dim.
// Then the size could change if there's a segmentation between those tensors,
// being multipled or divided by the device dimension size.
void replaceMetaDataOps(Fusion*);

// TensorViews are all based on symbolic sizes. When we first initialize them
// we don't know if they're inputs or outputs which would mean that they have
// runtime shapes. Intermediate tensors (those not going to global memory) do
// not have this information. Since we need to have the correct information in
// the kernel being fetched for shapes, we want to replace input and output
// tensors to reference the runtime structure containing sizes.
void replaceSymbolicSizes(Fusion*);

} // namespace nvfuser
