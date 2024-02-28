// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <simplification/egraph_type.h>
#include <simplification/enode.h>
#include <simplification/union_find.h>

#include <optional>

namespace nvfuser {

namespace egraph {

//! An analysis holds properties of EClasses that are combined when EClasses are
//! merged.
//!
//! There is a mapping make(n) that maps an ENode to data (we call this function
//! AnalysisData::fromENode(n)).
//!
//! The analysis invariant must be preserved: the analysis data of an eclass
//! must always be equal to the result of
//!
//! For more details see Sec. 4.1 of Willsey et al. 2021.
struct AnalysisData {
  //! Each EClass must represent terms whose types match
  DataType dtype;

  //! EClasses can represent a single unique value (this is checked in join()).
  PolymorphicValue value;

 public:
  //! This is make(n) from Willsey et al. 2021.
  static AnalysisData fromENode(const ENode& a);

  //! Join this AnalysisData with data from another EClass to form data for
  //! their merged EClass.
  //!
  //! Here we check that dtypes and
  AnalysisData join(const AnalysisData& other) const {
    NVF_ERROR(
        dtype == other.dtype,
        "Attempted to merge EClasses with different dtypes");
    PolymorphicValue joined_value = value;
    if (value.hasValue() && other.value.hasValue()) {
      NVF_ERROR(
          value == other.value,
          "Attempted to merge EClasses with differing values ",
          value,
          " and ",
          other.value);
    } else if (value.hasValue()) {
      joined_value = value;
    } else {
      joined_value = other.value;
      return {.dtype = dtype, .value = joined_value};
    }
  }
};

} // namespace egraph

} // namespace nvfuser
