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
  // ComputeAtMap.
  void checkExactGraphEquivalence(const ValGraph& exact_graph);

  void checkAlmostExactGraphEquivalence(const ValGraph& almost_exact_graph);

  void checkPermissiveGraphEquivalence(const ValGraph& permissive_graph);

 private:
  // Propagate mappings in a ComputeAtMap as is done in IdModel
  static void fullyPropagateMappings(DisjointSets<IterDomain*>& id_sets);

 private:
  ComputeAtMap ca_map_;
};

} // namespace nvfuser
