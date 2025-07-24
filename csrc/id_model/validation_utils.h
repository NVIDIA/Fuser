// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <id_model/id_model.h>
#include <val_graph.h>

namespace nvfuser {

// Note that this class is a friend of ComputeAtMap as it needs to
// have private access
class IdModelValidator {
 public:
  IdModelValidator(Fusion* fusion, bool allow_self_mapping = false);

  // Validate a given exact graph of IdModel by comparing it with
  // ComputeAtMap. Their maps should
  // be almost the same but there are some differences.
  // - In ComputeAtMap, swizzles are just skipped no matter what swizzle
  // type is used, so only swizzle outputs are mapped. In IdModel,
  // only swizzle inputs are mapped, except for Loop swizzles where
  // their inputs and outputs are mapped.
  // - In ComputeAtMap, mappings are local. For example, if domain x0 is
  // split to x1 and x2, and also domain y0 is split to y1 and
  // y2. Suppose x0 and y0 are exactly mapped and the two splits are
  // also considered exactly the same, IdModel maps x1 and y1, and x2
  // and y2, respectively, whereas that doesn't happen with ComputeAtMap
  //
  // Accounting for the first difference doesn't seem trivial, so when
  // swizzle is used we give up validating the exact graph. The second
  // difference is whether mappings are propagated, which can be
  // accounted for by updating the ComputeAtMap as is done in IdModel.
  void checkExactGraphEquivalence(const ValGraph& exact_graph);

  void checkAlmostExactGraphEquivalence(const ValGraph& almost_exact_graph);

  void checkPermissiveGraphEquivalence(const ValGraph& permissive_graph);

 private:
  // Propagate mappings in a ComputeAtMap as is done in IdModel
  static void fullyPropagateMappings(DisjointSets<IterDomain*>& id_sets);

 private:
  ComputeAtMap ca_map_;
  // Validation is not enabled if swizzle is found. See the comment above
  bool has_swizzle_ = false;
};

} // namespace nvfuser
