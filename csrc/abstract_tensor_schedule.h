// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <abstract_tensor.h>
#include <ir/internal_base_nodes.h>
#include <val_graph.h>

namespace nvfuser {

//! Apply the transformations found in an AbstractTensor to a concrete
//! TensorView.
//!
//! Pre-existing loop transforms in the provided TensorView will not be
//! overwritten; the AbstractTensor must have all of the TensorView's loop
//! IterDomain ValGroups as producers of its AbstractIds.
void applyAbstractTransforms(
    const AbstractTensor& abstract_tensor,
    TensorView* tv,
    ValGraph* graph = nullptr);

} // namespace nvfuser
