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

class EGraphSimplifier;

//! An EClass is simply an equivalence class of ENodes. This represents Vals
//! that are all proven to have exactly the same value in the generated kernel.
//! It holds a datatype, and a list of parent ENodes. The parents are ENodes
//! representing functions having a member of this EClass as one of the
//! arguments.
struct EClass {
  AnalysisData data;

  //! Holds pairs of ENodes and EClasses. Parent ENodes represent functions
  //! some of whose arguments are members of this EClass. The corresponding
  //! EClasses are the EClasses of those ENodes which might need to be merged
  //! during repair(). See the code listing in Fig. 4 of Willsey et al. 2021.
  std::list<std::pair<Id, Id>> parents;

  //! ENodes that are members of this class
  std::list<Id> members;

 protected:
  friend EGraphSimplifier;

  static EClass fromENode(const ENode& n) {
    // create new analysis data
    return {.data = AnalysisData::fromENode(n), .members = {n}, .parents = {}};
  }

  //! Merge in and drain another EClass. After this operation, the other EClass
  //! will be left with empty parents and members lists.
  void mergeFrom(EClass& other) {
    if (&other == this) {
      return;
    }

    // See Fig. 9 of Willsey et al. 2021
    data.joinFrom(other.data);

    // TODO: merge in union-find

    // TODO: add this EClass to work-list
  }
};

} // namespace egraph

} // namespace nvfuser
