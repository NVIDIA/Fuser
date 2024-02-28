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
  PolymorphicValue constant;

  //! We perform extraction on the fly by selecting an ASTNode to represent this
  //! class every time we merge.
  //! See Section 4.3 of Willsey et al. 2021 for a more detailed description of
  //! how extraction is accomplished as an e-class analysis.
  std::optional<Id> astnode_id;

 public:
  //! This is make(n) from Willsey et al. 2021.
  static AnalysisData fromENode(const Id n_id);

  //! Join this AnalysisData with data from another EClass to form data for
  //! their merged EClass.
  //!
  //! Here we check that dtypes match for the given classes, we fold constants,
  //! and we also select between astnode_id and other.astnode_id to perform
  //! on-the-fly extraction.
  AnalysisData joinFrom(const AnalysisData& other) const;
};

} // namespace egraph

} // namespace nvfuser
