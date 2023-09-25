#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>

namespace nvfuser {

// Returns whether a pointwise expression `expr` expands its input operand
// `in_tv`.
static bool pointwiseExpands(const Expr* expr, const TensorView* in_tv) {
  NVF_CHECK(
      expr->outputs().size() == 1,
      "A pointwise expression is expected to have one output: ",
      expr->toString());
  const Val* out = expr->output(0);

  if (!out->isA<TensorView>()) {
    return false;
  }
  const auto* out_tv = out->as<TensorView>();

  const size_t n_dims = in_tv->getRootDomain().size();
  if (n_dims != out_tv->getRootDomain().size()) {
    return false;
  }

  int num_expands = 0;
  for (size_t i = 0; i < n_dims; i++) {
    const Val* in_extent = in_tv->getRootDomain()[i]->extent();
    const Val* out_extent = out_tv->getRootDomain()[i]->extent();
    if (in_extent->isOneInt() && !out_extent->isOneInt()) {
      num_expands++;
    }
  }
  return num_expands > 0;
}

// Head of the def-use chain.
static const Expr* findExpand(const LoadStoreOp* ldst) {
  std::queue<const Expr*> q;
  std::unordered_set<const Expr*> visited;

  auto enqueueIfNotVisited = [&q, &visited](const Expr* expr) {
    if (visited.insert(expr).second) {
      q.push(expr);
    }
  };

  enqueueIfNotVisited(ldst);
  while (!q.empty()) {
    const Expr* def = q.front();
    q.pop();
    for (const Val* def_out : def->outputs()) {
      if (!def_out->isA<TensorView>()) {
        continue;
      }

      for (const Expr* use : def_out->uses()) {
        if (use->isA<ExpandOp>()) {
          return use;
        }

        if (use->isOneOf<UnaryOp, BinaryOp, TernaryOp, BroadcastOp>() ||
            (use->isA<LoadStoreOp>() &&
             use->as<LoadStoreOp>()->out()->as<TensorView>()->getMemoryType() !=
                 MemoryType::Global)) {
          if (pointwiseExpands(use, def_out->as<TensorView>())) {
            return use;
          }
          enqueueIfNotVisited(use);
        }
      }
    }
  }

  return nullptr;
}

// Returns true if the cache policy is changed.
static bool refineCachePolicy(LoadStoreOp* ldst) {
  scheduler_debug_utils::log("Processing ", ldst->toString());

  if (ldst->opType() != LoadStoreOpType::Set) {
    return false;
  }

  // Currently, we only change cache policy for global->local loads.
  if (ldst->in()->as<TensorView>()->getMemoryType() != MemoryType::Global) {
    return false;
  }
  if (ldst->out()->as<TensorView>()->getMemoryType() != MemoryType::Local) {
    return false;
  }

  const Expr* expand = findExpand(ldst);
  if (expand == nullptr) {
    scheduler_debug_utils::log(
        "Skipped ",
        ldst->toString(),
        " because we cannot find the using expand.");
    return false;
  }

  scheduler_debug_utils::log(
      "Changed the cache op of ",
      ldst->toString(),
      " from ",
      ldst->cacheOp(),
      " to ",
      CacheOp::AllLevels,
      " because it is expanded by ",
      expand->toString());
  ldst->setCacheOp(CacheOp::AllLevels);
  return true;
}

void refineCachePolicy(Fusion* fusion) {
  for (Expr* expr : fusion->exprs()) {
    if (expr->isA<LoadStoreOp>()) {
      refineCachePolicy(expr->as<LoadStoreOp>());
    }
  }
}

} // namespace nvfuser
