// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <compute_at_map.h>
#include <ir/all_nodes.h>
#include <kernel_ir.h>
#include <parallel_type_bitmap.h>
#include <val_graph_nodes.h>

#include <bitset>
#include <map>

// Provides utilities for dealing with nested ForLoop and IfThenElse scopes

namespace nvfuser {

class ThreadPredicateMap;
class ValGraph;

namespace scope_utils {

//! Create an **empty** Forloop and copy the metadata.
kir::ForLoop* cloneForLoop(kir::ForLoop* for_loop);

//! Create an **empty** IfThenElse and copy the metadata.
kir::IfThenElse* cloneIfThenElse(kir::IfThenElse* ite);

} // namespace scope_utils

namespace ir_utils {

//! Returns true if the given Val is a schedule operation.
bool isScheduleOp(const Val* val);

// Create a TVDomainGuard that temporarily view a TensorView with specified
// all-true or all-false contiguity.
NVF_API ir_utils::TVDomainGuard overrideContiguityGuard(
    TensorView* tv,
    bool contiguity);

// Create a TVDomainGuard that temporarily setting allocation domain as
// getLogicalDomain() from a TensorView, contiguity are filled all true or
// all false
ir_utils::TVDomainGuard allocateToLogicalDomainGuard(
    TensorView* tv,
    bool contiguity);

//! Return inputs of provided IterDomains that are IterDomains. A list
//! of input IterDomain can be optionally given. Otherwise,
//! IterDomains with no defining expression are returned.
std::vector<IterDomain*> iterDomainInputsOf(
    const std::vector<IterDomain*>& input_ids,
    const std::vector<IterDomain*>& all_inputs = {});

// Return inputs of provided IterDomains that are IterDomains, order as the
// second provided vector.
std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order);

// Returns if Val is a TensorView or TensorIndex
bool isTV(const Val* const);

// Returns if Expr is a TensorView or TensorIndex Expr.
NVF_API bool isTvOp(const Expr*);

//! Returns the iterdomain that maps to the thread dimension grouped
//!  to warps. Returns nullopt if the reduction is not to be lowered to
//!  a warp reduction.
std::optional<std::pair<IterDomain*, IterDomain*>> getMaybeWarpReductionDim(
    const Val* output,
    const Val* input);

bool isScalarOp(const Expr*);

bool isIterDomainOp(const Expr*);

//! Get TensorView potentially via kir::TensorIndex. Returns nullptr if
//! cast fails.
TensorView* getTv(Val*);
const TensorView* getTv(const Val*);

//! Get only TensorView potentially via kir::TensorIndex.
std::vector<TensorView*> getTvs(const std::vector<Val*>& vals);

std::unordered_map<ParallelType, IterDomain*> getParallelDomains(
    const Val* val);

//! Returns true if the expression will be lowered to
//!  a ldmatrix intrinsic.
bool isLdMatrixOp(const Expr* expr);

bool isStMatrixOp(const Expr* expr);

//! Returns true if the expression will be lowered to
//!  a cp.async intrinsic.
bool isCpAsyncOp(const Expr* expr);

//! Returns true if the expression will be lowered to
//!  a cp.async.bulk or cp.async.bulk.tensor
bool isCpAsyncBulkLoad(const Expr* expr);
bool isCpAsyncBulkStore(const Expr* expr);
bool isCpAsyncBulk(const Expr* expr);

//! Returns true if the expression will be lowered to
//!  a cp.async.bulk.tensor intrinsic.
bool isCpAsyncBulkTensorTileLoad(const Expr* expr);
bool isCpAsyncBulkTensorTileStore(const Expr* expr);
bool isCpAsyncBulkTensorTile(const Expr* expr);

//! Returns true if the expression will be lowered to
//!  a cp.async.bulk intrinsic.
bool isCpAsyncBulk1DLoad(const Expr* expr);
bool isCpAsyncBulk1DStore(const Expr* expr);
bool isCpAsyncBulk1D(const Expr* expr);

//! Short-cut for detecting initialization for cpAsync op.
bool isCpAsyncInit(const Expr* expr);

//! Returns true if the expression will be lowered to
//!  a ld/st tmem intrinsic.
bool isLdStTMem(const Expr* expr);

//! Short-cut for matching a singleton expr in a if statement,
//!  which likely becomes a predicated instruction in ptx, eg.:
//!  if(...) {expr;}
//! Returns the expr if it is this pattern.
//! Returns nullptr if the pattern doesn't match.
std::optional<Expr*> getMaybePredicatedSingleton(Expr* expr);

//! Short-cut for checking if the expression loads from global memory.
bool isGlobalLoad(const Expr* expr);

//! Short-cut for checking if the given expression initializes buffers
//!  for global memory load.
bool isGlobalLoadInit(const Expr* expr);

//! Returns true if the given expression fills the output
//!  tensor with a single scalar.
bool isTensorScalarFillOp(const Expr* expr);

//! Flattens all the scoped exprs, i.e. ForLoop and IfThenElse,
//!  and returns all the exprs in all scopes in the original
//!  linear textural order.
NVF_API std::vector<Expr*> flattenScopedExprs(
    const std::vector<Expr*>& loop_nests);

NVF_API std::vector<Expr*> flattenScopedExprs(
    const Scope::ExprList& loop_nests);

//! Returns all swizzle ops between the set of iterdomains
//!  in `from` and `to`.
std::vector<Expr*> getAllSwizzlesBetween(
    std::vector<IterDomain*> from,
    std::vector<IterDomain*> to);

// Replace value pass on Kernel IR.
//  Replace each use of any Val* that apears in the given `replacement_map`
//  Keeps the predicate carried by each expr
//
// Warning: Blindly replaces all use based on pointer
// Warning: May invalidate indexing if replacing uses of allocated values
std::vector<Expr*> replaceInputsInExpr(
    const std::vector<Expr*>& exprs,
    const std::unordered_map<Val*, Val*>& replacement_map);

//! Returns true if the given TensorView is a smem tv of TMA load/store, or
//! an input of an MmaOp.
bool isTMAOrMMASmemTv(TensorView* tv);

//! Returns the swizzle mode of the given TensorView. The TensorView must be
//! an input of an MmaOp, or the smem tv of TMA load/store.
MmaInputSmemSwizzle getSwizzleMode(TensorView* tv);

//! Get the stage_slice_position if it is defined in the WarpSpecialized
//! circular buffer options struct.
std::optional<int64_t> getStageSlicePosition(const TensorView* tv);

// Returns true if the for_loops contain a loop with the given
// CircularBufferLoopStage.
bool containsCircularBufferStage(
    const std::vector<kir::ForLoop*>& for_loops,
    CircularBufferLoopStage stage_type);
} // namespace ir_utils

namespace lower_utils {

bool hasBlockSync(const Expr* expr, const ThreadPredicateMap& pred_map);

// Allocate global buffer for a grid communication calls, i.e. grid reduce, grid
// welford reduce, grid broadcast.
kir::Allocate* allocGlobalBufferForGridComm(
    Val* buffer_size,
    DataType dtype,
    bool zero_init,
    bool resets_to_zero = false);

struct AllocPosInfo {
  // The for loop that the initialization of this allocation must be
  // placed in, nullptr if not within a loop
  kir::ForLoop* init_for_loop = nullptr;

  // Keep track of the actual allocation loop. This can be different
  // from init_for_loop only with unswitched shared memory allocations,
  // which are moved outer loops to avoid duplicated allocations. This means
  // that the alloc position may be outside what's expected. Most applications
  // outside lower_allocation is likely looking for init_for_loop which is
  // more directly related to how large an allocation is and how it's used.
  // (see issue #1133).
  kir::ForLoop* alloc_for_loop = nullptr;

  // The allocation position relative to buffer IDs, it could be outside the
  // compute at position if it's shared memory with a compute at inside an
  // unswitch
  int64_t alloc_pos = 0;
};

// Fill the above allocation struct based on provided information. id_map is
// used if we're looking at a producer tensor but loops on a consumer tensor.
AllocPosInfo getAllocPosInfo(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map = {},
    bool use_id_map = false);

//! Returns true if the expression has a variant that takes a predicate
//!  as an inline argument.
bool supportInlinePredicate(Expr* expr);

//! Test if an expression is a scalar expression.
bool isScalarExpr(Expr* expr);

//! Test if provided IterDomain instance has an extent that matches maximum
//!  extent stored in parallel dimension map for parallel type of provided
//!  IterDomain object. `in_compute_warp` specifies we are checking an
//!  expression in the compute warp, if so, we need to get the parallel type
//!  extent of the compute warp, instead of the global parallel type extent.
bool isExtentEqualToMaxParallelTypeExtent(
    const IterDomain* id,
    bool in_compute_warp = false);

//! Get the uint32_t index of a scalar TensorView. This is usually used for
//! indexing special items in shared memory, like mbarrier.
NVF_API Val* u32IndexScalarSmemTv(TensorView* tv);

//! Get the uint32_t index of a TensorIndex. This is usually used for
//! initializing a pipeline of mbarriers.
NVF_API Val* u32IndexScalarSmemTv(kir::TensorIndex* index);

//! Get the size of a global sync buffer needed to perform a grid reduction for
//! each axis in bitmap.
Val* getGridSyncBufferSize(const ParallelTypeBitmap& bitmap);

//! Returns the fusion outputs that require codegen.
//! The fusion outputs to be computed through expression evaluator are
//! filtered out.
std::vector<Val*> getFusionOutputsRequiringCodegen(Fusion* fusion);

//! Get the number of threads in a tensor view. Note that this function
//! only cares about the given tensor view itself, not the entire fusion.
//! That is, for example, if the tensor view is [TIDx{3}], but the entire
//! fusion has blockDim.x = 128, this function will return 3 instead of 128.
Val* getNumThreadsInTensorView(TensorView* tv);

//! Get the unit dimensions of A and B for the given MmaOp.
std::array<UnitDim, 2> getMmaLayout(const MmaOp* expr);

// Returns true if expr is an expression that initializes a reduction
// buffer.
bool isReductionInitExpr(const Expr* expr);

// Return true if it is sufficient to predicate the end of the loop
// iteration. An aligned vectorized loop is one example where it is
// guaranteed to be valid by the validation checks. More generally,
// the divisible split set is used to find such loops. The divisible
// split set contains splits used in view transformations as well as
// those whose output domains are vectorized. View transformations
// guarantee that any split involved is divisible, whereas
// vectorization only guarantees that the overall root extent is
// divisible by the split factor. Thus, if a loop IterDomain is
// an output of a split included in the divisible view splits, we can
// just predicate the end of the loop iteration. If a loop IterDomain
// is an output of a divisible split due to vectorization, it is only
// valid when the loop IterDomain is mapped with the vectorized inner
// output IterDomain. If it is mapped with an outer IterDomain, since
// the split input IterDomain may be an output IterDomain of a
// non-divisible split, we still need to predicate each loop iteration
// value.
bool predicateAtEnd(kir::ForLoop* loop);

// Given linear_g and domain, prove that linear_g is linear with respect to
// domain and return the stride. linear_g is linear with respect to domain if
// there exists a strided view of domain such that linear_g is one of the
// axes of that strided view. Usually, linear_g is a group in the loop domain of
// some tensor, and domain is the allocation domain of some tensor. In this
// case, if the index of linear_g is i, then this function proves that the index
// is is a linear function of i, with the linear coefficient being the return
// value. Note that this function does the proof and stride calculation in a
// best-effort manner. It can not cover all linear cases. If the return value is
// nullptr, it can be either because linear_g is not linear with respect to
// domain, or because linear_g is actually linear with respect to domain, but it
// is too hard for this function to find a proof.
Val* proveLinearAndGetStride(
    const ValGraph& id_graph,
    const ValGroup& linear_g,
    const ValGroups& domain);

// Get the concrete loop domain of a given loop ID
IterDomain* getConcreteLoopID(IterDomain* loop_id);

// Go through all expressions and compute a local ordering of loops. operator<
// is implemented based on the concrete_id_dependencies analysis done. If
// there's no dependency between two IDs then order doesn't mater, otherwise we
// can tell which is inner most by checking if there's any dependency
// relationships.
//
// Dependency relationships in concrete_id_dependencies has a "global" view in
// the fusion, so it can resolve ordering by only looking at id's and the
// dependency map.
//
// For example two expressions may have domains: [I0], [I1] Yet we
// won't know the ordering unless we see a domain with: [I0, I1]. This happened
// in Indexing9 (also see Indexing17) test when merging T5 with
// the group containing T10 (cache of T5, which is post broadcasted output) and
// T6(pre broadcasted output).
// T5 had the domain [0, 1, 2, 3, 4] produce at 3
// T6 had the domain [0, 3, 4] compute at 3
// Merging [0, 1, 2] and [0, 3, 4] resulted in the domain [0, 3, 4, 1, 2]
//
// If ID's are not in filter, we don't care about their ordering and ignore
// them. This is because we're only focused on loops we will have to merge
// across groups. If the domain is not in a produce at position in the producer
// edges, or a compute at position in the consumer edges, the expressions we
// look at may not have a unique ordering.
//
// The optional kernel_scope_domain parameter is only used in
// expression sorting. It isn't in the CA map, but since we only have
// a single unique IterDomain, the conrete ID is just itself.
struct IterDomainDependencySorter {
  IterDomainDependencySorter(
      const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
          concrete_id_dependencies,
      IterDomain* kernel_scope_domain = nullptr)
      : concrete_id_dependencies_(concrete_id_dependencies),
        kernel_scope_domain_(kernel_scope_domain) {}

  // Return true if id0 should be before id1
  // Orders such that if x maps to {y}, x comes before y in final ordering.
  inline bool operator()(IterDomain* id0, IterDomain* id1) {
    auto concrete_id_0 =
        id0 != kernel_scope_domain_ ? getConcreteLoopID(id0) : id0;
    auto concrete_id_1 =
        id1 != kernel_scope_domain_ ? getConcreteLoopID(id1) : id1;
    if (concrete_id_dependencies_.find(concrete_id_0) !=
        concrete_id_dependencies_.end()) {
      const auto& dependencies_0 = concrete_id_dependencies_.at(concrete_id_0);
      // if id0 depends on id1 it means id1 is inside id0, so id0 < id1
      if (dependencies_0.count(concrete_id_1)) {
        return true;
      }
    }

    return false;
  }

  const std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>&
      concrete_id_dependencies_;
  const IterDomain* kernel_scope_domain_ = nullptr;
};

// Check if all the inputs of the given MmaOp is guarded by mbarrier
bool allMmaInputsGuardedByMBarrier(const MmaOp* mma);

// Check if the given ForLoop is a warp specialized loop by checking
// the circular buffer type of the loop domain.
bool isWarpSpecializedLoop(kir::ForLoop* loop);

// Check if the given Expr is only a data copy, and no math is done on it.
// For example, set, broadcast, squeeze, slice, pad, reshape, etc.
// When an Expr is copy only, regardless of the architecture, it is always
// supported. For example, it is totally OK to broadcast a fp8 tensor of shape
// [1024] to a fp8 tensor of shape [1024, 1] on NVIDIA GeForce 8800 GTX, the
// first device that supports CUDA, because it just involves byte copying, which
// is supported on all architectures.
bool isCopyOnly(Expr* expr);

// Check if the given Val is only copied from/to other Val, and no math is done
// on it.
bool isCopyOnly(Val* val);

} // namespace lower_utils
} // namespace nvfuser
