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
void applyAbstractTransforms(
    const AbstractTensor& abstract,
    TensorView* concrete,
    ValGraph* graph = nullptr);

} // namespace nvfuser
