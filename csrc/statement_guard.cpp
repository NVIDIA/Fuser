// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <fusion.h>
#include <statement_guard.h>

namespace nvfuser {

StatementGuard::StatementGuard(Fusion* fusion)
    : fusion_([fusion] {
        // Trigger lazy initialization of axioms. Without this, we'd have to
        // remove axioms in the destructor, which no APIs can do at this
        // moment.
        fusion->axioms();
        return fusion;
      }()),
      prev_num_exprs_(fusion_->numExprs()),
      prev_num_vals_(fusion_->numVals(/*include_shortcuts=*/false)) {}

StatementGuard::~StatementGuard() {
  fusion_->removeStatementsCreatedAfter(prev_num_exprs_, prev_num_vals_);
}

} // namespace nvfuser
