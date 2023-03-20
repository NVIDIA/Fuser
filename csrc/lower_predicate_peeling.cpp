#include <dispatch.h>
#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <iter_visitor.h>
#include <lower2device.h>
#include <lower_predicate_peeling.h>

#include <unordered_set>

namespace nvfuser {

bool PredicatePeeling::supportedPeelingLoop(IterDomain* id) {
  // Not meaningful to peel a parallel loop
  if (id->isParallelized()) {
    return false;
  }

  auto id_def = id->definition();

  if (id_def == nullptr) {
    // This case is not profitable so skip peeling support.
    return false;
  } else if (auto split = dynamic_cast<Split*>(id_def)) {
    auto split_in = split->in();

    // This is typical case that we want to peel.
    //  where we advance a serial iteration through
    //  constant sized tiles.
    return split_in->definition() == nullptr && id == split->outer() &&
        split->factor()->isConstInt();
  }

  // TODO:
  //  could possibly support more patterns using the separability analysis.
  return false;
}

void PredicatePeelingInfo::build(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    // Only visit tensorviews with peeling serial id info.
    if (tv->peeledSerialId() != nullptr) {
      // Create a set of leaf ids for validation.
      std::unordered_set<IterDomain*> leaf_id_set{
          tv->domain()->domain().begin(), tv->domain()->domain().end()};
      auto peeled_id = tv->peeledSerialId();

      // Quick check that the peeled id at schedule
      //  time is still a leaf domain.
      TORCH_INTERNAL_ASSERT(
          leaf_id_set.count(peeled_id),
          "only exisiting leaf domain supported for peeling\n",
          tv->toString(),
          " does not have\n",
          peeled_id->toString());

      // Insert the peeled concrete id to the recorded map.
      concrete_id_of_peeled_loops_.insert(
          GpuLower::current()->caMap()->getConcreteMappedID(
              peeled_id, IdMappingMode::LOOP));
    }
  }
}

c10::optional<PeeledTileEntry> PredicatePeelingInfo::getMaybePeeledTileEntry(
    const std::vector<kir::ForLoop*>& loops,
    IterDomain* root_id) {
  auto gpu_lower = GpuLower::current();

  std::unordered_map<IterDomain*, kir::ForLoop*> concrete_id_to_loop_map;
  for (auto fl : loops) {
    auto concrete_loop_id = gpu_lower->caMap()->getConcreteMappedID(
        fl->iter_domain(), IdMappingMode::LOOP);
    concrete_id_to_loop_map[concrete_loop_id] = fl;
  }

  for (auto peeled_id : concrete_id_of_peeled_loops_) {
    // Need to locate the peeled loop to validate this check
    auto matching_loop_it = concrete_id_to_loop_map.find(peeled_id);
    if (matching_loop_it != concrete_id_to_loop_map.end()) {
      // This is the only supported case at the initial stage.
      // see also [Supported Case in Predicate Peeling pass]
      auto split = peeled_id->definition()->as<Split>();
      if (gpu_lower->caMap()->areMapped(
              split->in(), root_id, IdMappingMode::EXACT)) {
        // This means the given id has been peeled.
        PeeledTileEntry entry;
        entry.peel_stage =
            matching_loop_it->second->loopTransformInfo().predicate_peel_stage;
        entry.inner_factor = split->factor();
        entry.for_loop = matching_loop_it->second;
        return entry;
      }
    }
  }
  return c10::nullopt;
}

bool PredicatePeelingInfo::hasPeeledId(const TensorView* tv) const {
  for (auto id : concrete_id_of_peeled_loops_) {
    if (std::any_of(
            tv->domain()->domain().begin(),
            tv->domain()->domain().end(),
            [id](IterDomain* tv_id) {
              return GpuLower::current()->caMap()->areMapped(
                  tv_id, id, IdMappingMode::LOOP);
            })) {
      return true;
    }
  }
  return false;
}

bool PredicatePeelingInfo::shouldPeelLoop(kir::ForLoop* forloop) const {
  auto loop_concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      forloop->iter_domain(), IdMappingMode::LOOP);

  return
      // Base index calculation loop should not be involved here.
      !forloop->loopTransformInfo().is_base_index_loop &&
      concrete_id_of_peeled_loops_.count(loop_concrete_id);
}

namespace {

//! A utility class that deep clones the loop body of original_fl
//!  and add the cloned expressions into the loop body of new_fl.
class LoopNestDeepCloner {
 public:
  static void clone(kir::ForLoop* original_fl, kir::ForLoop* new_fl) {
    LoopNestDeepCloner cloner;
    cloner.cloned_scopes_.push_back(&new_fl->body());
    for (auto expr : original_fl->body().exprs()) {
      cloner.handle(expr);
    }
  }

 private:
  void handle(Expr* expr) {
    if (auto fl = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(fl);
    } else {
      cloned_scopes_.back()->push_back(expr);
    }
  }

  void handle(kir::ForLoop* fl) {
    auto new_fl = IrBuilder::create<kir::ForLoop>(fl);
    cloned_scopes_.push_back(&new_fl->body());
    for (auto expr : fl->body().exprs()) {
      handle(expr);
    }
    cloned_scopes_.pop_back();

    cloned_scopes_.back()->push_back(new_fl);
  }

 private:
  std::vector<kir::Scope*> cloned_scopes_;
};

//! The predicate peeling transform pass implementation.
class PredicatePeeledLoops : kir::ExprMutator {
 public:
  static std::vector<Expr*> run(const std::vector<Expr*> exprs) {
    PredicatePeeledLoops peeled_loops;
    peeled_loops.traverseAndInsert(exprs);
    return peeled_loops.exprs_;
  }

 private:
  using kir::ExprMutator::handle;

  // Create the predicate peeled prolog loop
  kir::ForLoop* createPeeledLoop(kir::ForLoop* fl) {
    // Make clone of the outermost loop, but
    //  limit the loop to the first iteration.
    auto peeled_loop = IrBuilder::create<kir::ForLoop>(
        fl->iter_domain(),
        fl->index(),
        fl->kernel()->zeroVal(),
        fl->kernel()->oneVal(),
        fl->kernel()->oneVal(),
        false,
        nullptr,
        fl->isUnrollRequired(),
        fl->loopTransformInfo().predicatePeelStage(PredicatePeelStage::Prolog));

    LoopNestDeepCloner::clone(fl, peeled_loop);
    return peeled_loop;
  }

  // Create the predicate peeled main loop
  kir::ForLoop* createMainLoop(kir::ForLoop* fl) {
    auto start =
        SimplifyingIrBuilder::addExpr(fl->start(), fl->kernel()->oneVal());

    auto main_loop = IrBuilder::create<kir::ForLoop>(
        fl->iter_domain(),
        fl->index(),
        start,
        fl->stop(),
        fl->step(),
        fl->vectorize(),
        fl->vectorize_shift(),
        fl->isUnrollRequired(),
        fl->loopTransformInfo().predicatePeelStage(PredicatePeelStage::Main));

    LoopNestDeepCloner::clone(fl, main_loop);
    return main_loop;
  }

  void handle(kir::ForLoop* fl) final {
    kir::ExprMutator::handle(fl);

    if (GpuLower::current()->predicatePeelingInfo().shouldPeelLoop(fl)) {
      auto loop_concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          fl->iter_domain(), IdMappingMode::LOOP);

      // Peel this loop if shouldPeel is true and it
      //  hasn't been processed by this pass already.
      if (!peeled_iterdomains_.count(loop_concrete_id)) {
        // Create and insert the peeled prolog.
        auto peeled_loop = createPeeledLoop(fl);
        registerInsertBefore(fl, peeled_loop);

        if (fl->stop()->isOneInt()) {
          // This is the case for double buffer prolog,
          //  will just use the peeled loop as the new
          //  double buffer prolog.
          registerRemove(fl);
        } else {
          // Peel off one iteration from the main
          //  component of the original loop.
          auto new_main_loop = createMainLoop(fl);
          registerReplace(fl, new_main_loop);
        }
        // Record peeling of this loop
        peeled_iterdomains_.insert(loop_concrete_id);
      }
    }
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(
        false, "no support for inserted ite before this point");
  }

 private:
  //! Keeps track of loop concrete iterdomains that has already been
  //!  transformed by the loop peeling pass.
  //! This pass runs after double buffering pass, so
  //!  we may encounter forloops corresponding to the
  //!  same loop domain twice. In this case we only need
  //!  to peel the first occurrence (the prolog).
  std::unordered_set<IterDomain*> peeled_iterdomains_;
};

} // namespace

std::vector<Expr*> PredicatePeeling::peelPredicatedLoop(
    const std::vector<Expr*> exprs) {
  return PredicatePeeledLoops::run(exprs);
}

Val* PredicatePeeling::getPrologPredicateOffset(
    IterDomain* id,
    Val* tile_factor) {
  // Assume X = original extent,
  //        L = tile factor
  // Offset needs to satisfy:
  //  X % L == 0 : return L else return X % L

  // X + L - ceildiv(X,L)*L
  auto orig_extent = id->extent();

  // X + L
  auto extent_plus_factor =
      SimplifyingIrBuilder::addExpr(orig_extent, tile_factor);

  // ceildiv(X,L)
  auto extent_ceildiv_factor =
      SimplifyingIrBuilder::ceilDivExpr(orig_extent, tile_factor);

  // ceildiv(X,L)* L
  auto extent_round_up =
      SimplifyingIrBuilder::mulExpr(extent_ceildiv_factor, tile_factor);

  // X + L - ceildiv(X,L)*L
  return SimplifyingIrBuilder::subExpr(extent_plus_factor, extent_round_up);
}

Val* PredicatePeeling::getSplitTileMainOffset(
    IterDomain* id,
    Val* tile_factor) {
  // This offset is to be **subtracted** from the tensor index
  //  on the predicate peeling main loop.
  return SimplifyingIrBuilder::subExpr(
      tile_factor, PredicatePeeling::getPrologPredicateOffset(id, tile_factor));
}

} // namespace nvfuser
