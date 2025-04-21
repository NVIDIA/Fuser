// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <ir/base_nodes.h>

#include <vector>

namespace nvfuser {

//! This pass finds trivial computations that can be avoided.
//!
//! For example, suppose we have the following fusion:
//!
//!   T0_g[ iS0{i1} ]
//!   T1_g[ iS1{i1}, bS2{1} ] = broadcast(T0_g)
//!   T2_l[ iS3{i1}, bS4{1} ] = set(T1_g)
//!
//! In a case like this, we would ordinarily allocate an intermediate global
//! tensor to hold T1, and it would be realized. However, under some conditions
//! we can avoid this allocation and copying by re-using T0_g in any uses of
//! T1_g. In the example above, it is clear that in the definition of T2_l, we
//! could simply find the appropriate element of T0_g and load that instead of
//! loading from T1_g. When we do this we call it a tensor producer alias to
//! distinguish it from other uses of the term "alias" in nvfuser such as smem
//! buffer aliasing and fusion input/output buffer reuse. Note that it is not
//! necessarily the case that a tensor producer alias points to a _direct_
//! producer of that tensor, since chains of trivial ops may be skipped by this
//! pass.
//!
//! Currently, we only allow aliasing across Exprs that satisfy these
//! conditions:
//!   1) The expr is a LoadStoreOp, BroadcastOp, or SqueezeOp
//!   2) It has input and output whose allocation IterDomains are exact mapped,
//!      ignoring Broadcast and Reduction axes.
//!   3) It has both input and output that are fully contiguous.
//! The first condition guarantees that we do not skip any important
//! computation, while the second and third condition allow us to re-use the
//! same linear index that we would have used without the alias but merely
//! changing the base address of the buffer. In the future, we may allow
//! relaxing the third condition, in which case the linear index would be
//! modified by replacing the intermediate buffer's strides with those of the
//! producer.
void findTensorProducerAliases(Fusion* fusion);

//! This is a lowering pass that actually removes top-level exprs whose outputs
//! have been marked as tensor producer aliases in the findTensorProducerAliases
//! analysis pass.
std::vector<Expr*> removeTensorProducerAliases(const std::vector<Expr*>& exprs);

} // namespace nvfuser
