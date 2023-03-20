#pragma once

#include <disjoint_set.h>
#include <ir_all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

//! Keeps track of loops that will be interleaved, see
//!  [Loop Interleaving].
class InterleaveLoopInfo {
  using ConcreteIdVector = VectorOfUniqueEntries<IterDomain*>;
  using TensorViewVector = VectorOfUniqueEntries<TensorView*>;

 public:
  //! Collect info by traversing fusion expressions.
  void build(Fusion* fusion);

  //! Validate data consistency after interleaving.
  //! see [Supported Interleaving Cases].
  void validate();

  //! See comment on concrete_main_loop_to_subloop_map_
  const auto& concreteMainLoopToSubloopMap() const {
    return concrete_main_loop_to_subloop_map_;
  }

  //! See comment on concrete_main_loop_to_number_of_units_
  const auto& concreteMainLoopToFactorMap() const {
    return concrete_main_loop_to_number_of_units_;
  }

 private:
  //! Build phase 1: check all the tv's for
  //!  main_loops where subloops are interleaved.
  void collectInterleaveMainLoops();

  //! Build phase2: collect all tv's that are
  //!  computed within interleaved loops.
  void collectInterleavedSubLoops();

  //! Register a (main_loop, sub_loop) pair collected from
  //!  tv, see [Loop Interleaving].
  void insertEntry(TensorView* tv, IterDomain* main_loop, IterDomain* sub_loop);

  //! Returns true if the given id is loop mapped with
  //!  an interleaving main loop, see [Loop Interleaving].
  bool isMainLoop(IterDomain* id);

  //! Returns true if the id is loop mapped to a sub loop
  //!  within a main loop mapped to concrete_main_id.
  //!  see also [Loop Interleaving].
  bool isSubLoopOf(IterDomain* id, IterDomain* concrete_main_id);

  //! Validate data consistency after interleaving.
  //! see [Supported Interleaving Cases].
  void validateMainLoop(
      IterDomain* main_loop,
      const TensorViewVector& interleaved_tvs);

  //! Validation utility:
  //!  see [Supported Interleaving Cases].
  bool isExitTv(TensorView* tv, const TensorViewVector& interleaved_tvs);

 private:
  //! Keeps track of interleaving main loops and the
  //!  interleaved subloops within, see  [Loop Interleaving].
  std::unordered_map<IterDomain*, ConcreteIdVector>
      concrete_main_loop_to_subloop_map_;

  //! Keeps track of the interleaving main loops and
  //!  all the tensors that are *produced* within
  //!  the interleaved subloops associated with
  //!  each interleaving main loop.
  std::unordered_map<IterDomain*, TensorViewVector>
      concrete_main_loop_to_interleaved_tv_;

  //! Keeps track of the interleaving factor of each
  //!  interleaving main loop. see [Loop Interleaving].
  std::unordered_map<IterDomain*, int> concrete_main_loop_to_number_of_units_;

  //! Short-cut to the fusion this info keeps track of.
  Fusion* fusion_ = nullptr;

  //! Cached used math vals from fusion_;
  std::vector<TensorView*> used_tvs_;
};

void validateInterleaving(Fusion* fusion);

std::vector<Expr*> interLeaveDoubleBufferUnrolledLoops(
    const std::vector<Expr*>& exprs);

} // namespace nvfuser
