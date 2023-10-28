// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <compute_at_map.h>
#include <id_model/id_graph.h>
#include <id_model/id_model.h>

namespace nvfuser {

class IdModelValidator {
 public:
  static void checkExactMapEquivalence(const IdGraph& id_graph);
};

} // namespace nvfuser
