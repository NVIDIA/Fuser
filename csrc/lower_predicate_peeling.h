#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <ir_all_nodes.h>
#include <kernel_ir.h>

#include <unordered_set>

namespace nvfuser {

//!   Note: [Predicate Peeling]
//!   This is a loop transformation that attempts to eliminate predicate
//!    evaluation in a serial loop.
//!
//!   A simple example showing how this trick works is, say we have
//!   T0 [I(T0.size[0])] -> split(32) -> T0 [Io(ceilDiv(T0.size[0],32)),
//!   Ii(32)], which generates the following code: for i in
//!   0..ceilDiv(T0.size[0],32)
//!     for j in 0..32:
//!        // assume we need to initialize in this kernel
//!       T0[i*32+j]  = 0;
//!       if i*32+j < T0.size[0]
//!         T0[i*32+j] ...
//!   The above code generates 32 predicates as the predicate is inlined in the
//!   inner loop.
//!
//!   The simplification trick is to convert the loop into:
//!
//!   let ceilMod(a, b) = a %b == 0 ? b : a %b;
//!
//!   // peeled residue prolog : (called initial evaluation in cutlass)
//!   //   Very similar to the original loop except the
//!   //  outer loop extent and the predicate extent
//!   //  are modified.
//!
//!   for i in 0..1
//!     for j in 0..32:
//!       T0[i*32+j]  = 0;
//!       if i*32+j < ceilMod(T0.size[0], 32)
//!         T0[i*32+j] ...
//!   // peeled residue main loop
//!   // (called steady-state in cutlass)
//!   for i in 0..ceilDiv(T0.size[0],32)-1
//!     for j in 0..32:
//!         // No need to initialize as we know the predicate
//!        //  is all true.
//!        //  This significantly reduces memory instruction
//!        //  congestion with cp.async kernels.
//!         // No longer need to predicate here as
//!         //  the residue part of the root iterdomain has
//!         //  been peeled away.
//!         T0[i*32+j + ceilMod(T0.size[0],32)] ...
//!
//! Some details on the predicate peeling pass implemented here:
//!  1. The peeled loop is separate into 2 `PredicatePeelingStage`'s:
//!    The first iteration is peeled and marked as
//!    PredicatePeelingStage::Prolog, while
//!  the rest of the iterations are PredicatePeelingStage::Main.
//!
//!  2. The predicate indexing at the (predicate peeling) prolog is modified to
//! make the access within the residue tile
//!
//!  3. The address indexing at the (predicate peeling) main loop is modified
//! by adding the residue tile as offset.
//!
//!  4. The initialization within (predicate peeling) main loop can be lifted
//! out of the main loop if there are no other not-unrolled serial loops.
//!
//! Note: [Supported Case in Predicate Peeling pass]:
//! The predicate peeling transform is a very specialized pattern used in matmul
//!  and some non-trivial overhead would be involved to generalize.
//!
//! The current support for predicate peeling is for a very specific case only
//!  and some consideration is needed regarding whether more complex peeling
//!  pattern along this line should be pursued.
//!
//! The only supported pattern now is:
//!   tile_o, tile_i = split(root_id, inner_factor);
//! where tile_o is required to be on the leaf domain and is where the loop
//!  peeling primitive should be applied.
//!
//! The inner_factor is required to be a compile-time constant.
//!
// Note: [Predicate Peeling Interaction with Circular Buffering]
//!
//! 1. In the case where the original loop is double buffered, the first
//! iteration of the double buffer prolog loop is used as
//! PredicatePeelingStage::Prolog and the rest are labeled as
//! PredicatePeelingStage::Main.
//!
//! 2. If a tv is double buffered or circular buffered, the gmem load stage is
//! (stage_depth-1) iterations ahead, so would need to add an extra (simpler)
//! predicate to avoid out-of-bound access.
//!
//! 3. A circular buffer init prolog is added in the case of a predicate tiled
//! and circular buffered loop, as the circular buffer loop prolog only
//! prefetches up to iteration `stage_depth-1`, and if the initialization were
//! to be lifted out of the main loop stage, would also need to initialize for
//! iteration `stage_depth` to make sure the shared memory buffer is all zero
//! initialized.

//! A data structure used by PredicatePeelingInfo to communicate which
//!  for loop is predicate peeled along with the peeling stage and
//!  original inner tiling factor
//! TODO: some info here is redundant now.
struct PeeledTileEntry {
  //! The peeling stage, see note above.
  PredicatePeelStage peel_stage = PredicatePeelStage::NoApplicable;

  //! The original splitting factor, see [Supported Case in Predicate Peeling
  //! pass].
  Val* inner_factor = nullptr;

  //! The actual for loop that is predicate peeled.
  kir::ForLoop* for_loop = nullptr;
};

//! Keeps track fo predicate peeled loops requested
//!  from scheduler.
class PredicatePeelingInfo {
 public:
  //! Returns true if predicate peeling is requested by scheduler
  //!  for the given loop.
  bool shouldPeelLoop(kir::ForLoop* forloop) const;

  //! Collect predicate peeling information from fusion.
  void build(Fusion* fusion);

  //! Returns the peeled entry info if the given loop is predicate
  //!  peeled and the given root_id matches with the tiled root id.
  //!
  //! see also [Supported Case in Predicate Peeling pass].
  c10::optional<PeeledTileEntry> getMaybePeeledTileEntry(
      const std::vector<kir::ForLoop*>& loops,
      IterDomain* root_id);

  //! Returns true if any iterdomain on the given tv's tensor
  //!  domain corresponds to a predicate peeled loop.
  bool hasPeeledId(const TensorView* tv) const;

 private:
  //! Keeps track of loop concrete iterdomains that were predicate
  //!  peeled.
  std::unordered_set<IterDomain*> concrete_id_of_peeled_loops_;
};

namespace PredicatePeeling {

//! User space check that makes sure the loop can
//!  actually be peeled to remove predicates.
//! See also
//! [Supported Case in Predicate Peeling pass]:
bool supportedPeelingLoop(IterDomain* id);

//! Kernel IR pass that applies the predicate peeling transformation.
std::vector<Expr*> peelPredicatedLoop(const std::vector<Expr*> exprs);

//! Utility to generate the residual extend used in predicate
//!  peeling prolog.
Val* getPrologPredicateOffset(IterDomain* id, Val* tile_factor);

//! Utility to generate the offset applied to tensor indices
//!  in predicate peeling main loop.
Val* getSplitTileMainOffset(IterDomain* id, Val* tile_factor);

} // namespace PredicatePeeling

} // namespace nvfuser
