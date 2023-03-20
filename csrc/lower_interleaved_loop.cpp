#include <expr_evaluator.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <lower_interleaved_loop.h>
#include <lower_utils.h>

namespace nvfuser {

// Note: [Loop Interleaving]:
//  This pass is trying to implement a simple yet useful loop structure
//   optimization that tries to interleave sub iterations of unrolled loops.
// With an example:
//
// Before transform:
//    for i0 in 0..4
//      expr1
//    for i1 in 0..8
//      expr2
//    for i2 in 0..4
//      expr3
// After transform:
//    for i0 in {0}
//      expr1
//    for i1 in {0,1}
//      expr2
//    for i2 in {0}
//      expr3
//    for i0 in {1}
//      expr1
//    for i1 in {2,3}
//      expr2
//    ...
//
// To simplify the initial implementation, an outer serial loop is assumed, as
//  an indicator to define at which loop nest level to start interleaving, so
//  the actual transform looks like: (some terminology defined inline)
// Before transform:
//  for i in ...        // This outer serial loop is called "main loop" in this
//  pass
//    for i0 in 0..4    // Each of these unrolled loops is called a "subloop" of
//    the "main loop"
//      expr1
//    for i1 in 0..8
//      expr2
//    for i2 in 0..4
//      expr3
// After transform:
//  for i in ...
//    for i0 in {0}   // Each of these sub-iterations is called an "interleave
//    unit"
//      expr1
//    for i1 in {0,1}
//      expr2
//    for i2 in {0}
//      expr3
//    for i0 in {1}
//      expr1
//    for i1 in {2,3}
//      expr2
//    ...
//
// This optimization is controlled by scheduler through interface:
//   tv->interleave(pos, factor),
// where `pos` is the position of the iterdomain
//  that corresponds to the main loop, and all the subloops are assumed to be at
//  the immediate next position.
//  e.g.
//    tv[Io, Ii] -> interleave(0, pos);
//  means that the "main loop" is selected to be the loop that is loop mapped to
//  Io, and Ii is assumed to be map to one of the "sub loops".
//
// The term `factor` defines the number of "interleave units" to split each "sub
// loop"
//  into, in a best effort manner, with each unit size `ceilDiv(loop_extent,
//  factor)`.
//
// E.g. if the factor is 4
//   subloop `for i in 0..8` becomes:
//  `for i in 0..2`
//  `for i in 2..4`
//  `for i in 4..6`
//  `for i in 6..8`
//   subloop `for i in 0..7` becomes:
//  `for i in 0..2`
//  `for i in 2..4`
//  `for i in 4..6`
//  `for i in 6..7`
//   subloop `for i in 0..6` becomes:
//  `for i in 0..2`
//  `for i in 2..4`
//  `for i in 4..6`
//
// All the subloops are assumed to be constant sized since they need to be
// unrolled
//  for this optimization to be meaningful.
namespace {

int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
};

//! Returns the next level unrolled loop that is within the given
//!  main loop on a tensorview. Returns a c10::nullopt if the unrolled
//!  loop cannot be found.
c10::optional<IterDomain*> getMaybeSubloop(
    TensorView* tv,
    IterDomain* main_loop) {
  bool main_loop_found = false;
  const auto& ca_map = GpuLower::current()->caMap();

  for (auto leaf_id : tv->domain()->domain()) {
    if (main_loop_found && !leaf_id->isParallelized()) {
      return ca_map->getConcreteMappedID(leaf_id, IdMappingMode::LOOP);
    }
    main_loop_found = main_loop_found ||
        ca_map->areMapped(leaf_id, main_loop, IdMappingMode::LOOP);
  }

  return c10::nullopt;
}

} // namespace

void InterleaveLoopInfo::build(Fusion* fusion) {
  fusion_ = fusion;
  auto used_math_vals = fusion->usedMathVals();
  auto filtered_used_math_vals =
      ir_utils::filterByType<TensorView>(used_math_vals);

  // Cache used tvs for multiple visit.
  used_tvs_ = {filtered_used_math_vals.begin(), filtered_used_math_vals.end()};

  // Collect loop information from fusion
  collectInterleaveMainLoops();
  collectInterleavedSubLoops();

  // Validate interleaved expressions for data consistency
  validate();
}

void InterleaveLoopInfo::collectInterleaveMainLoops() {
  for (auto tv : used_tvs_) {
    auto maybe_main_axis = tv->getMaybeInterleavedAxisAndFactor();
    if (maybe_main_axis.has_value()) {
      auto concrete_main_loop_id =
          GpuLower::current()->caMap()->getConcreteMappedID(
              tv->axis(maybe_main_axis.value().first), IdMappingMode::LOOP);

      // Create new record for this loop id if not found
      if (!concrete_main_loop_to_interleaved_tv_.count(concrete_main_loop_id)) {
        // Create record space to later collect the interleaved tensors
        //  and the subloops.
        concrete_main_loop_to_subloop_map_.insert(
            std::make_pair(concrete_main_loop_id, ConcreteIdVector()));
        concrete_main_loop_to_interleaved_tv_.insert(
            std::make_pair(concrete_main_loop_id, TensorViewVector()));

        // Record the interleave factor for this main loop. see [Loop
        // Interleaving].
        concrete_main_loop_to_number_of_units_.insert(std::make_pair(
            concrete_main_loop_id, maybe_main_axis.value().second));
      }
    }
  }
}

bool InterleaveLoopInfo::isMainLoop(IterDomain* id) {
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return concrete_main_loop_to_interleaved_tv_.count(concrete_id);
}

bool InterleaveLoopInfo::isSubLoopOf(
    IterDomain* id,
    IterDomain* concrete_main_id) {
  auto it = concrete_main_loop_to_subloop_map_.find(concrete_main_id);
  TORCH_INTERNAL_ASSERT(
      it != concrete_main_loop_to_subloop_map_.end(), "Invalid main loop");
  auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return it->second.has(concrete_id);
}

void InterleaveLoopInfo::insertEntry(
    TensorView* tv,
    IterDomain* main_loop,
    IterDomain* sub_loop) {
  auto concrete_main_loop = GpuLower::current()->caMap()->getConcreteMappedID(
      main_loop, IdMappingMode::LOOP);
  auto concrete_sub_loop = GpuLower::current()->caMap()->getConcreteMappedID(
      sub_loop, IdMappingMode::LOOP);

  // Insert sub loops from this tv
  auto main_loop_entry_it =
      concrete_main_loop_to_subloop_map_.find(concrete_main_loop);
  TORCH_INTERNAL_ASSERT(
      main_loop_entry_it != concrete_main_loop_to_subloop_map_.end(),
      "unknown main loop: ",
      main_loop->toString(),
      " (",
      concrete_main_loop->toString(),
      ")");
  main_loop_entry_it->second.pushBack(concrete_sub_loop);

  // Insert interleaved tvs.
  auto tv_entry_it =
      concrete_main_loop_to_interleaved_tv_.find(concrete_main_loop);
  TORCH_INTERNAL_ASSERT(
      tv_entry_it != concrete_main_loop_to_interleaved_tv_.end(),
      "unknown main loop: ",
      main_loop->toString(),
      " (",
      concrete_main_loop->toString(),
      ")");
  tv_entry_it->second.pushBack(tv);
}

void InterleaveLoopInfo::collectInterleavedSubLoops() {
  for (auto tv : used_tvs_) {
    IterDomain* main_loop = nullptr;
    for (auto leaf_id : tv->domain()->domain()) {
      if (main_loop == nullptr) {
        if (isMainLoop(leaf_id)) {
          main_loop = leaf_id;
          auto maybe_subloop = getMaybeSubloop(tv, leaf_id);
          TORCH_INTERNAL_ASSERT(
              maybe_subloop.has_value(),
              tv->toString(),
              " cannot be interleaved within ",
              leaf_id);
          insertEntry(tv, main_loop, maybe_subloop.value());
        }
      } else {
        // main loop already found. There should be no more
        // main loop in this tensor
        TORCH_INTERNAL_ASSERT(
            !isMainLoop(leaf_id),
            tv,
            "has nested main loop ",
            main_loop->toString(),
            " and ",
            leaf_id->toString(),
            " which is not yet supported");
      }
    }
  }
}

// Validation of double buffering topology of interleaved expressions:
//  see [Supported Interleaving Cases]
void InterleaveLoopInfo::validate() {
  // Validate expression consistency after interleaving
  for (auto& main_loop_entry : concrete_main_loop_to_interleaved_tv_) {
    validateMainLoop(main_loop_entry.first, main_loop_entry.second);
  }
}

// Returns true if the given tv is an "exit tv",
//  see [Supported Interleaving Cases].
bool InterleaveLoopInfo::isExitTv(
    TensorView* tv,
    const TensorViewVector& interleaved_tvs) {
  // Output is always an exit
  if (tv->isFusionOutput()) {
    return true;
  }

  for (auto use : fusion_->unordered_uses(tv)) {
    // Check if any immediate consumer of tv is interleaved.
    for (auto consumer_tv :
         ir_utils::filterByType<TensorView>(use->outputs())) {
      if (interleaved_tvs.has(consumer_tv)) {
        return false;
      }
    }
  }

  // No immediate consumer of tv is interleaved so the tv is an exit tv.
  return true;
}

void InterleaveLoopInfo::validateMainLoop(
    IterDomain* concrete_main_loop,
    const TensorViewVector& interleaved_tvs) {
  // [Supported Interleaving Cases]
  // All the expressions that are inside the main loop or subloop can
  //  only be 3 cases:
  // 1. It's double/circular buffered across a loop that's either at or on the
  // outer
  //  loop nest than the main loop. E.g.
  //  for i in ... // loop 1
  //   for j in ... // loop 2 (interleave main loop)
  //    for k in ... // loop 3 (interleave sub loop)
  //     tv0 [i%3*buffersize + ... ] = ...;
  //  tv0 is circular buffered around loop1, so interleaving loop 3
  //  with any other serial loops within loop2 will not make any consumer
  //  of tv0 use the wrong value.
  //  No guarantee on the producers of tv0 from this though,
  //   which relies on the same check being run on them as well to ensure
  //   safety.
  //
  // 2. It's inlined into the subloop.
  //  Eg.
  //  for i in ... // loop1 (interleave main loop)
  //   for j in ... // loop2 (interleave sub loop)
  //     for k in ... // loop3
  //      tv0[k] = ...
  //     for w in ... // loop4
  //      ... = t0[w]
  // The inlining semantically means that the consumer of tv0 above is
  //  within loop2, so interleaving loop 2 with other loops within loop1
  //  should not cause the consumer of tv0 to read wrong values, as they
  //  are essentially not changed.
  //
  // 3. It's not a producer of any other interleaved tv's,
  //       i.e. it is an "exit tv".
  //  for i in ... // loop1 (interleave main loop)
  //   for j in ... // loop2 (interleave subloop 1)
  //    tv0[j] = ...
  //   for k in ... // loop3 (interleave subloop 2)
  //    tv1[j] = ...
  //
  //  for m in ...
  //   ... = tv0[m] + tv1[m];
  //
  // In this case tv0 and tv1 are producing values that are used outside
  //  of any of the expressions that are interleaved, so the interleaving
  //  of loop2 and loop3 should have no effect on the semantic.
  for (auto tv : interleaved_tvs.vector()) {
    if (isExitTv(tv, interleaved_tvs)) {
      // Exit tv computation can be interleaved by Point 3 above.
      continue;
    }

    // Double buffered tv doesn't need to be checked, see Point 2 above:
    if (tv->isDoubleBuffered() || tv->isCircularBuffered()) {
      auto db_axis =
          GpuLower::current()->doubleBufferInfo().getDoubleBufferAxis(tv);

      // Check that the double buffer axis is at or on the left of
      //  the main loop.
      bool can_interleave = false;

      // Iterating over the leaf domains from the left
      for (auto id : tv->domain()->domain()) {
        if (id == db_axis) {
          // If we see double buffer axis first then
          //  it's double buffered on the outer loop.
          // So it can be interleaved.
          can_interleave = true;
          break;
        } else if (GpuLower::current()->caMap()->areMapped(
                       id, concrete_main_loop, IdMappingMode::LOOP)) {
          // If we see main loop before seeing the double buffer axis,
          //  it cannot be proven safe to interleave by double buffering
          //  but the other two points might apply.
          can_interleave = false;
        }
      }

      if (can_interleave) {
        continue;
      }
    }

    // If Point3 and Point2 didn't apply at this point,
    //  then Point1 has to apply in order for this interleaving to be valid.
    // TODO:
    //  Maybe in follow ups more supported patterns could be added.

    // Check that the subloop is on the left of CA axis:
    auto& concrete_subloops =
        concrete_main_loop_to_subloop_map_.at(concrete_main_loop);
    bool subloop_found = false;
    for (auto id_it = tv->domain()->domain().begin();
         id_it != tv->domain()->domain().begin() + tv->getComputeAtPosition();
         id_it++) {
      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          *id_it, IdMappingMode::LOOP);
      if (concrete_subloops.has(concrete_id)) {
        subloop_found = true;
        break;
      }
    }
    TORCH_INTERNAL_ASSERT(
        subloop_found,
        "unsupported interleaved tv ",
        tv->toString(),
        " it needs to be either double buffered, or an exit of interleaved region or inlined beyond subloops");
  }
}

namespace {

// A data structure collecting the parameters when realizing the interleaving.
struct InterLeaveConfig {
  // Total number of units, aka. interleave factor,
  //  see [Loop Interleaving].
  int64_t number_of_units = 1;

  // Evaluated loop extent of each sub loop.
  std::unordered_map<IterDomain*, int64_t> concrete_id_to_extent_;
};

//! The loop interleaving pass that implements the interleaving
//!  transform, see [Loop Interleaving].
class LoopInterLeaver : kir::ExprMutator {
 public:
  static std::vector<Expr*> run(std::vector<Expr*> exprs) {
    // Interleave main loops one at a time.
    for (auto& it : GpuLower::current()
                        ->interleavedLoopInfo()
                        .concreteMainLoopToSubloopMap()) {
      LoopInterLeaver interleaver;
      interleaver.concrete_main_loop_ = it.first;

      interleaver.concrete_subloop_set_ = std::unordered_set<IterDomain*>(
          it.second.vector().begin(), it.second.vector().end());
      interleaver.traverseAndInsert(exprs);
      exprs = interleaver.exprs_;
    }
    return exprs;
  }

 private:
  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* fl) final {
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        fl->iter_domain(), IdMappingMode::LOOP);

    // For double buffered loops, only interleave the main stage.
    if (concrete_main_loop_ == concrete_loop_id &&
        fl->doubleBufferLoopStage() == DoubleBufferLoopStage::Main) {
      handleMainLoop(fl);
    } else {
      kir::ExprMutator::handle(fl);
    }
  }

  // Returns true if the expression is a subloop to be interleaved
  //  see [Loop Interleaving].
  bool isInterleavedSubloop(Expr* expr) {
    if (auto loop = dynamic_cast<kir::ForLoop*>(expr)) {
      auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
          loop->iter_domain(), IdMappingMode::LOOP);
      if (concrete_subloop_set_.count(concrete_loop_id) &&
          // Do not interleave double buffer epilogs
          loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Epilog &&

          // Do not interleave any index computation expressions
          !loop->loopTransformInfo().is_base_index_loop &&
          !loop->loopTransformInfo().is_increment_loop) {
        return true;
      }
    }
    return false;
  }

  // Remove the original subloops once the interleaved
  //  versions have been inserted.
  void clearSubLoops(
      std::vector<kir::ForLoop*>& interleaved_subloops,
      kir::ForLoop* main_loop) {
    for (auto fl : interleaved_subloops) {
      registerRemove(fl, &main_loop->body());
    }
    interleaved_subloops.clear();
  }

  // Realize the interleaving with the given for loop
  //  as the main loop, see [Loop Interleaving].
  //
  // [Loop Interleaving Impl]
  // The implementation pass goes as the below example:
  //
  //  for i in ... // main loop
  //   for j in ... // sub loop1
  //    ...
  //   for k in ... // sub loop2
  //    ...
  //   __syncthread();
  //   for m in ... // sub loop3
  //    ...
  //   for n in ... // sub loop4
  //    ...
  //
  // This function loops through the body of the main loop
  //  and puts all the subloops encountered in `interleaved_subloops`
  //  vector.
  // Whenever it sees an expression that is *not* a interleaved subloop, e.g.
  //  the syncthreads in the above example, the currently collected
  //  `interleaved_subloops`, i.e. loop1 and loop2 in this case, are
  //  emitted as interleaved units and the pass continues with an empty
  //  `interleaved_subloops` vector.
  // As a result, in this example, sub loop1 and sub loop2 are interleaved
  //  while sub loop3 and sub loop4 are interleaved.
  void handleMainLoop(kir::ForLoop* fl) {
    // Collect the subloops encountered when looping
    //  over the main loop expressions.
    std::vector<kir::ForLoop*> interleaved_subloops;

    // Loop over the main loop body.
    for (auto expr : fl->body().exprs()) {
      if (auto loop = dynamic_cast<kir::ForLoop*>(expr)) {
        if (
            // Usually not useful to involve double buffer prologs
            //  and epilogs in the interleaving.
            !isProlog(loop->doubleBufferLoopStage()) &&
            loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Epilog &&
            // Check if this expression is a subloop
            isInterleavedSubloop(expr)) {
          // Collect this sub loop to be realized later, see details above.
          interleaved_subloops.push_back(expr->as<kir::ForLoop>());
          continue;
        }
      }

      // Main loop may have allocation expressions that can be safe
      //  to just continue collecting the subloop across as the interleave
      //  units will be realized after this expression, which means the
      //  allocation is still valid.
      if (expr->isA<kir::Allocate>()) {
        continue;
      }

      // This is the point where we see an expression that is *not* an
      //  interleaved subloop that we are collecting, so emit the currently
      //  collected interleaved subloops as interleaved units.
      // And clear the collected vector before proceeding.
      if (!interleaved_subloops.empty()) {
        realizeInterleavedSubloops(expr, interleaved_subloops, true, fl);
        clearSubLoops(interleaved_subloops, fl);
      }
    }

    // It's possible, actually common that all exprs within
    //  the main loop are subloops, so we will need to run
    //  another realization step after visiting the whole main
    //  loop.
    if (!interleaved_subloops.empty()) {
      realizeInterleavedSubloops(
          fl->body().exprs().back(), interleaved_subloops, false, fl);
      clearSubLoops(interleaved_subloops, fl);
    }
  }

  // Performs a deep loopnest clone if the expression
  //  is a loop nest.
  // TODO: use common infra
  Expr* cloneMaybeLoopNest(Expr* expr) {
    auto fl = dynamic_cast<kir::ForLoop*>(expr);
    if (!fl) {
      return expr;
    }

    TORCH_INTERNAL_ASSERT(!expr->isA<kir::IfThenElse>(), "unsupported");
    auto cloned_fl = IrBuilder::create<kir::ForLoop>(fl);

    for (auto loop_expr : fl->body().exprs()) {
      cloned_fl->body().push_back(cloneMaybeLoopNest(loop_expr));
    }

    return cloned_fl;
  }

  void handle(kir::IfThenElse*) final {
    TORCH_INTERNAL_ASSERT(
        false, "LoopInterleaving: no support yet post IfThenElse lowering");
  }

  // Emit the currently collected subloops as interleaved units,
  //  see [Loop Interleaving Impl].
  void realizeInterleavedSubloops(
      // A insertion reference point
      Expr* insert_point,
      // Subloops to interleave.
      std::vector<kir::ForLoop*> sub_loops,
      // Insert interleave units before insertion point
      //  if true, after if false.
      bool insert_before,
      // Main loop to interleave within.
      kir::ForLoop* main_loop) {
    // Container to collect the interleave units in interleaved order.
    std::vector<kir::ForLoop*> interleave_units;

    // Populate parameters on interleaving these sub loops.
    auto config = getInterleaveConfig(main_loop, sub_loops);

    // Repeat for number_of_units times, each time creating
    //  an interleave unit for each subloop.
    for (int idx : c10::irange(config.number_of_units)) {
      // Loop over each sub loop
      for (auto sub_loop : sub_loops) {
        // Collect concrete id and extent
        auto concrete_loop_id =
            GpuLower::current()->caMap()->getConcreteMappedID(
                sub_loop->iter_domain(), IdMappingMode::LOOP);

        auto concrete_extent =
            config.concrete_id_to_extent_.at(concrete_loop_id);

        // Calculate size of this unit
        auto interleave_unit = ceilDiv(concrete_extent, config.number_of_units);

        // Set start and stop of this unit,
        //   stop needs to be the minimum of start+size and original extent
        // to avoid out running the orignal loop.
        int start_idx = idx * interleave_unit;
        auto stop_idx = std::min(start_idx + interleave_unit, concrete_extent);

        // No longer need to generate more of this sub loop if
        //  start is already out of bound.
        if (start_idx < concrete_extent) {
          auto start_val = SimplifyingIrBuilder::create<Int>(start_idx);
          auto stop_val = SimplifyingIrBuilder::create<Int>(stop_idx);
          interleave_units.push_back(
              makeInterleavedUnit(sub_loop, start_val, stop_val));
        }
      }
    }

    if (insert_before) {
      for (auto unit : interleave_units) {
        registerInsertBefore(insert_point, unit, &main_loop->body());
      }
    } else {
      // Need to insert in reverse order when inserting after in order
      //  to maintain the original order defined in interleave_units.
      for (auto it = interleave_units.rbegin(); it != interleave_units.rend();
           it++) {
        registerInsertAfter(insert_point, *it, &main_loop->body());
      }
    }
  }

  // Make an interleaved unit of the given sub loop according to the given
  //  start and stop offset.
  kir::ForLoop* makeInterleavedUnit(kir::ForLoop* fl, Val* start, Val* stop) {
    // Create an outer loop with the same loop expressions but
    //  different start and stop.
    auto outer_loop = IrBuilder::create<kir::ForLoop>(
        fl->iter_domain(),
        fl->index(),
        start,
        stop,
        fl->step(),
        fl->vectorize(),
        fl->vectorize_shift(),
        fl->isUnrolled(),
        fl->loopTransformInfo().interLeaveUnit());

    for (auto expr : fl->body().exprs()) {
      outer_loop->body().push_back(cloneMaybeLoopNest(expr));
    }

    return outer_loop;
  }

  // Collect info needed to realize interleaved loop,
  //  see [Loop Interleaving Impl].
  InterLeaveConfig getInterleaveConfig(
      kir::ForLoop* main_loop,
      const std::vector<kir::ForLoop*> sub_loops_) {
    TORCH_INTERNAL_ASSERT(
        !sub_loops_.empty(), "Cannot generate config for empty subloops");
    InterLeaveConfig interleave_config;
    ExpressionEvaluator const_evaluator;

    for (auto fl : sub_loops_) {
      auto maybe_value = const_evaluator.evaluate(fl->stop());
      TORCH_INTERNAL_ASSERT(
          maybe_value.has_value(), "non constant interleaving not supported");
      auto value = maybe_value.value().as<int64_t>();

      auto concrete_loop_domain =
          GpuLower::current()->caMap()->getConcreteMappedID(
              fl->iter_domain(), IdMappingMode::LOOP);

      // Collect concrete extents of each of the subloops.
      interleave_config.concrete_id_to_extent_[concrete_loop_domain] = value;
    }

    // Calculate interleave factor, simple heuristic as ceilDiv(max, min):
    interleave_config.number_of_units =
        GpuLower::current()
            ->interleavedLoopInfo()
            .concreteMainLoopToFactorMap()
            .at(GpuLower::current()->caMap()->getConcreteMappedID(
                main_loop->iter_domain(), IdMappingMode::LOOP));

    return interleave_config;
  }

 private:
  // Marks the current main loop this pass
  //  is processing.
  IterDomain* concrete_main_loop_ = nullptr;

  // Set of subloop concrete IterDomains that will
  //  be interleaved within main loop.
  std::unordered_set<IterDomain*> concrete_subloop_set_;
};
} // namespace

std::vector<Expr*> interLeaveDoubleBufferUnrolledLoops(
    const std::vector<Expr*>& exprs) {
  return LoopInterLeaver::run(exprs);
}

} // namespace nvfuser
