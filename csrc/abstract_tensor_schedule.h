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

//! Apply the transformations found in an AbstractTensor to concrete
//! TensorViews.
//!
//! Pre-existing loop transforms in the provided TensorViews will not be
//! overwritten; the AbstractTensor must have all of each TensorView's loop
//! IterDomain ValGroups as producers of its AbstractIds.
NVF_API void applyAbstractTransforms(
    const AbstractTensor& abstract_tensor,
    const std::vector<TensorView*>& tvs,
    ValGraph* graph = nullptr);

inline void applyAbstractTransforms(
    const AbstractTensor& abstract_tensor,
    TensorView* tv,
    ValGraph* graph = nullptr) {
  const std::vector<TensorView*> tvs{tv};
  applyAbstractTransforms(abstract_tensor, tvs, graph);
}

} // namespace nvfuser
