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

namespace nvfuser {

class ContigIDs;

void validateIr(Fusion* fusion);

//! Validate vectorization and collect information on vectorization
//! used in code generation as well as runtime validation.
void validateAndCollectVectorizeInfo(Fusion* fusion);

//! Find the contig allocation domains that a vectorized loop domain
//! of a consumer TV depends on. Required for runtime validation.
void fillConsumerVectorizedContigAllocationDomains(
    const TensorView* consumer_tv,
    const ContigIDs& contig_finder);

//! Find the contig allocation domains that a vectorized loop domain
//! of a producer TV depends on. Required for runtime validation.
//! Producer must be transformed as consumer.
void fillProducerVectorizedContigAllocationDomains(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const ContigIDs& contig_finder);

//! Validate data format and GPU arch compatibility of scheduled
//!  mma operators on the fusion.
void validateMma(Fusion* fusion);

//! Validates swizzle ops to ensure consistent indexing:
//!   - Currently only allow swizzle ops on the right of CA axis,
//!   - (Except ZShape) All swizzle ops have to be on const sized ids
//!   - Xor and Transpose swizzle have to have equal dimensions on the
//!       participating ids.
void validateSwizzle(Fusion* fusion);

//! Validate use of ParallelType::Group. It is currently only allowed
//! in ReductionOp and not in WelfordOp. Group has similar constraints
//! as Vectorize, e.g., it can only be used with IterDomains with
//! static extents. Differences are, e.g, it has no constraints on
//! alignments and predicates. Each individual reduction has its own
//! predicate, so it is possile for only part of grouped reductions to
//! be executed.
//!
//! Also, grouping is only enabled for persistent grid reductions, in
//! other words, grid allreduces. Note that no grid reduction without
//! broadcast is persistent anymore.
//!
//! Validated ReductionOp with ParallelType::Group is converted to
//! GroupedReductionOp.
void validateAndConvertIterDomainGrouping(Fusion* fusion);

//! Validate the number of grouped reductions is within the limit
void validateGroupedReductions(Fusion* fusion);

//! Validate all of the lookup TVs are ensured to be fusion inputs
void validateLookupTV(Fusion* fusion);

//! Check that there are no reductions over unexpanded broadcasts
void validateReductions(Fusion* fusion);

} // namespace nvfuser
