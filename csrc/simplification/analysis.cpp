// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <simplification/egraph_type.h>
#include <simplification/analysis.h>

#include <optional>

namespace nvfuser {

namespace egraph {

AnalysisData AnalysisData::join(const AnalysisData& other) const {
  NVF_ERROR(
      dtype == other.dtype,
      "Attempted to merge EClasses with different dtypes");

  PolymorphicValue joined_constant = constant;
  if (constant.hasValue() && other.constant.hasValue()) {
    NVF_ERROR(
        constant == other.constant,
        "Attempted to merge EClasses with differing constant values ",
        constant,
        " and ",
        other.constant);
  } else {
    joined_constant = constant.hasValue() ? constant other.constant;
  }

  // Select a representative ASTNode.
  // This step is important as it determines the form of the simplified Val* for
  // this EClass. Our implementation should seek to encourage simplicity by
  // estimating a cost for each ASTNode and propagating that information forward
  // during these join steps.
  Id joined_astnode_id = astnode_id;

  return {
      .dtype = dtype,
      .constant = joined_constant,
      .astnode_id = joined_astnode_id};
}

} // namespace egraph

} // namespace nvfuser
