// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

struct IndexFromIdGraph;

//! Insert magic zero definition at the begining of the kernel. Insert magic
//! zero update after every (outer most) loop nest with a compile time extent.
//!
//! This will make sure nvrtc does not aggressively save predicate and indices.
std::vector<Expr*> insertMagicZero(const std::vector<Expr*>& exprs);

//! Check if val is a reference to the magic zero variable
bool isMagicZero(const Val* val);

//! Check if val is protected with magic zero.
//!
//! Specifically, this returns true if val is defined as "x + magic_zero".
bool isProtectedWithMagicZero(const Val* val);

//! Check if val is protected with magic zero, if yes, unwrap it.
//! maybeUnwrapMagicZero(i1) -> i1
//! maybeUnwrapMagicZero(i2 + magic_zero) -> i2
Val* maybeUnwrapMagicZero(Val* val);

// Determine if we may run into over reuse of predicates or registers in the
// compiler. If the loop can be unrolled and the index and domain are not
// "simple" we likely want the loop protected.
//
// Magic zero protection should only be done for global memory and predicates.
// We should avoid use on registers. Shared memory does not require it, but
// likely wouldn't hurt.
bool needsMagicZero(
    kir::ForLoop* loop,
    IterDomain* reference_domain = nullptr,
    Val* ind = nullptr);

struct IndexMagicZeroInfo {
  //! Index that may be updated with magic zero
  Val* index = nullptr;
  //! Loop index that is protected by magic zero. nullptr if no loop
  //! is protected
  Val* original_loop_index = nullptr;
  //! Protected loop index. nullptr if no loop is protected
  Val* protected_loop_index = nullptr;
  //! Protected loop. nullptr if no loop is protected
  IterDomain* loop_id = nullptr;
};

//! Protect an index val of an IterDomain with magic zero
//!
//! This should be only used for predicate indexing.
//!
//! No protection is done if none of the loops is determined to require
//! protection by needsMagicZero.
IndexMagicZeroInfo protectPredicateIndexWithMagicZero(
    Val* index,
    const IndexFromIdGraph& id_graph,
    const std::vector<kir::ForLoop*>& loops);

//! Protect an index val of a tensor with magic zero
//!
//! This should be only used for non-predicate indexing.
//!
//! No protection is done if none of the loops is determined to require
//! protection by needsMagicZero.
void protectNonPredicateIndexWithMagicZero(
    const std::vector<kir::ForLoop*>& loops,
    const std::vector<IterDomain*>& loop_domains,
    std::unordered_map<IterDomain*, Val*>& concrete_loop_idx_map);

} // namespace nvfuser
