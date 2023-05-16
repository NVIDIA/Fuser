// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/loop_rotation.h>
#include <device_lower/utils.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>

#include <device_lower/lower2device.h>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {
namespace {

// Clone an expr, if this expr is a container (ForLoop, IfThenElse), then
// recursively clone all exprs in its scope.
Expr* recursivelyClone(Expr* expr) {
  TORCH_INTERNAL_ASSERT(expr != nullptr);
  if (auto fl = dynamic_cast<kir::ForLoop*>(expr)) {
    auto new_loop = IrBuilder::create<kir::ForLoop>(fl);
    for (auto e : fl->body().exprs()) {
      new_loop->body().push_back(recursivelyClone(e));
    }
    return new_loop;
  } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
    // We are running this pass before UnrollPass, so we actually don't expect
    // to see any IfThenElse here. However, we are still handling IfThenElse
    // just in case in the future we want to move this pass(For example, what if
    // we only want to rotate the loop in the unswitch path?). We should be
    // definitely revisit how to deal with this ite->predicate() if this is the
    // case.
    TORCH_INTERNAL_ASSERT(
        false, "Don't expect to see IfThenElse in loop rotation pass.");
    auto new_ite = IrBuilder::create<kir::IfThenElse>(ite->predicate());
    for (auto e : ite->thenBody().exprs()) {
      new_ite->thenBody().push_back(recursivelyClone(e));
    }
    for (auto e : ite->elseBody().exprs()) {
      new_ite->elseBody().push_back(recursivelyClone(e));
    }
    return new_ite;
  } else {
    auto clone = expr->shallowCopy();
    GpuLower::current()->propagateExprInfo(expr, clone);
    return clone;
  }
}

// Because scheduler does not have access to lowered loop structure, our
// interface with scheduler is to let scheduler select tensors whose allocation
// and computation will be rotated, and we expand the selection to
// kir::Allocate, kir::ForLoop, and kir::IfThenElse. For example, if I have
// kernel:
//   for (int i = 0; i < id1.extent(); i++) {
//     // kir::Allocate 1
//     float T1[5];
//     // kir::ForLoop 1
//     for (int j = 0; j < 5; j++) {
//       if (i < T0.size[0]) {
//         T1[j] = sin(T0[i, j]);
//       }
//     }
//     // kir::Allocate 2
//     float T2[5];
//     // kir::ForLoop 2
//     for (int j = 0; j < 5; j++) {
//       T2[j] = cos(T1[j]);
//     }
//     // kir::Allocate 3
//     float T3[5];
//     // kir::ForLoop 3
//     for (int j = 0; j < 5; j++) {
//       T3[j] = exp(T2[j]);
//     }
//     // kir::ForLoop 4
//     for (int j = 0; j < 5; j++) {
//       if (i < T4.size[0]) {
//         T4[i, j] = log(T3[j]);
//       }
//     }
//   }
// And received a compilation parameter {id1, {T1, T2}}, then the first step
// should expand the selection from {T1, T2} to:
// {kir::Allocate 1, kir::ForLoop 1, kir::Allocate 2, kir::ForLoop 2}
//
// RotateLoop is a bottom-up algorithm that does its work during the returning
// of recursion. By designing like this, when we want to rotate the loop, all
// expressions in the loop body should have already been recursively visited,
// and the selection should already been expanded, so we have the information
// required to determine which expr to rotate.
class RotateLoop : kir::ExprMutator {
 public:
  static std::vector<Expr*> run(
      std::vector<Expr*> exprs,
      const LoopRotationParam& params) {
    // Rotate one loop at a time so that nested loops can be rotated without
    // interacting with each other.
    for (auto item : params) {
      exprs = RotateLoop(
                  std::get<0>(item)->axis((int)std::get<1>(item)),
                  std::get<2>(item))
                  .traverseAndInsert(exprs);
    }
    return exprs;
  }

 private:
  // The concrete id of the loop being rotated
  IterDomain* loop_concrete_id_;
  // The selected tvs/exprs to be rotated
  std::unordered_set<Statement*> selection_;

  RotateLoop(IterDomain* loop_id, std::unordered_set<Statement*> selection)
      : loop_concrete_id_(GpuLower::current()->caMap()->getConcreteMappedID(
            loop_id,
            IdMappingMode::LOOP)),
        selection_(std::move(selection)) {}

  // We use the following strategy on expr selection:
  // - If a Val is selected, then its allocation is automatically selected.
  // - If all the exprs in a container(ForLoop and IfThenElse) are selected,
  //   then the container is automatically selected.
  // This function modifies selection_ to implement this strategy
  void expandSelection(Expr* expr) {
    TORCH_INTERNAL_ASSERT(expr != nullptr);
    for (auto fl : for_loops_) {
      if (fl->iter_domain() != loop_concrete_id_) {
        continue;
      }
      auto entireScopeSelected = [this](const kir::Scope& scope) {
        return std::all_of(
            scope.exprs().begin(), scope.exprs().end(), [this](Expr* expr) {
              return selection_.count(expr) > 0;
            });
      };
      bool should_select_this = false;
      if (auto fl = dynamic_cast<kir::ForLoop*>(expr)) {
        should_select_this = !fl->empty() && entireScopeSelected(fl->body());
      } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
        should_select_this = !ite->empty() &&
            entireScopeSelected(ite->thenBody()) &&
            entireScopeSelected(ite->elseBody());
      } else if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
        should_select_this = selection_.count(alloc->buffer()) > 0;
      } else {
        should_select_this = std::any_of(
            expr->outputs().begin(), expr->outputs().end(), [this](Val* v) {
              return selection_.count(v) > 0;
            });
      }
      if (should_select_this) {
        selection_.insert(expr);
      }
    }
  }

  // Expressions selected in selection_ is not necessarily on the top of the
  // loop, for example, if I have a loop
  //   for (int i = 0; i < n; i++) {
  //     i1 = i + 1;
  //     i2 = i1 + 1;
  //     i3 = i % 3;
  //     i4 = i3 + 1;
  //     i5 = i4 + 1;
  //   }
  // Then a valid selection is {i3, i4}, and the rotated loop will look like
  //   if (0 < n) {
  //     i3 = 0 % 3;
  //     i4 = i3 + 1;
  //   }
  //   for (int i = 0; i < n; i++) {
  //     i1 = i + 1;
  //     i2 = i1 + 1;
  //     i5 = i4 + 1;
  //     if (i + 1 < n) {
  //       i3 = (i + 1) % 3;
  //       i4 = i3 + 1;
  //     }
  //   }
  // This function validates that if the selection comply with data dependency.
  //
  // Note that right now, we don't have a complete data dependency analysis, so
  // we can not generally support all legal selection. Currently, we are doing a
  // simple patter matching for the following patterns:
  // Pattern 1:
  //   for (...) {
  //     selected
  //     selected
  //     unselected
  //     unselected
  //     unselected
  //   }
  // Pattern 2:
  //   for (...) {
  //     unselected double buffer load
  //     selected
  //     selected
  //     unselected
  //     unselected
  //     unselected
  //   }
  bool validateSelection(kir::ForLoop* fl) {
    class IsDoubleBufferLoad : public kir::IrVisitor {
     public:
      bool operator()(Expr* expr) {
        result_ = true;
        handle(expr);
        return result_;
      }

      IsDoubleBufferLoad(kir::ForLoop* loop) : loop_(loop) {}

     private:
      using kir::IrVisitor::handle;

      void handle(Expr* expr) final {
        if (!result_) {
          return;
        }
        for (auto output : expr->outputs()) {
          auto tv = dynamic_cast<TensorView*>(output);
          if (tv == nullptr) {
            result_ = false;
            return;
          }
          if (GpuLower::current()->doubleBufferInfo().getDoubleBufferLoop(
                  tv, {loop_}) != loop_) {
            result_ = false;
            return;
          }
        }
        IrVisitor::handle(expr);
      }

     private:
      bool result_ = true;
      kir::ForLoop* const loop_ = nullptr;
    } is_double_buffer_load(fl);

    bool seen_unselected = false;
    for (auto expr : fl->body().exprs()) {
      if (selection_.count(expr) > 0) {
        if (seen_unselected) {
          return false;
        }
        continue;
      }
      if (is_double_buffer_load(expr)) {
        continue;
      }
      seen_unselected = true;
    }
    return true;
  }

  // Do the rotation. If I have loop:
  //   for (int i = 0; i < n; i++) {
  //     selected1(i);
  //     selected2(i);
  //     unselected3(i);
  //     unselected4(i);
  //   }
  // Then this function should generate
  //   // prologue
  //   if (0 < n) {
  //     // trivial loop
  //     for (int i = 0) {
  //       selected1(i);
  //       selected2(i);
  //     }
  //   }
  //   // main loop
  //   for (int i = 0; i < n; i++) {
  //     unselected3(i);
  //     unselected4(i);
  //     if (i + 1 < n) {
  //       selected1(i + 1);
  //       selected2(i + 1);
  //     }
  //   }
  // Currently, because all out-of-bound access are already covered other
  // predicates, so we are actually generating
  //   // prologue
  //   // trivial loop
  //   for (int i = 0) {
  //     selected1(i);
  //     selected2(i);
  //   }
  //   // main loop
  //   for (int i = 0; i < n; i++) {
  //     unselected3(i);
  //     unselected4(i);
  //     if (true) {
  //       selected1(i + 1);
  //       selected2(i + 1);
  //     }
  //   }
  void rotate(kir::ForLoop* fl) {
    if (isDebugDumpEnabled(DebugDumpOption::LoopRotation)) {
      std::cout << "[Loop rotation] Rotating loop:" << std::endl
                << fl->toString() << std::endl;
    }
    // Insert selected allocations and `prologue` before `fl`, and replace `fl`
    // with `rotated`

    // prologue
    // Currently, all existing predicates should be able to cover the condition
    // of start < end, so no predicate here. In the future, if we decide that
    // we need to predicate this, then we should add an kir::IfThenElse here.
    auto prologue = IrBuilder::create<kir::ForLoop>(
        fl->iter_domain(), fl->start(), fl->doubleBufferLoopStage());
    std::vector<Expr*> lifted_alloc;
    for (auto expr : fl->body().exprs()) {
      if (selection_.count(expr) == 0) {
        continue;
      }
      if (expr->isA<kir::Allocate>()) {
        lifted_alloc.push_back(expr);
        continue;
      }
      prologue->body().push_back(recursivelyClone(expr));
    }
    if (prologue->empty()) {
      if (isDebugDumpEnabled(DebugDumpOption::LoopRotation)) {
        std::cout << "[Loop rotation] Nothing to do." << std::endl;
      }
      return;
    }
    // Insert selected allocations and `prologue` before `fl`. Note that we will
    // not modify `fl` to remove these allocations and other selected
    // expressions. Instead, we will abandon `fl` and create a new loop to
    // replace `fl`.
    for (auto expr : lifted_alloc) {
      registerInsertBefore(fl, expr);
    }
    registerInsertBefore(fl, prologue);
    if (isDebugDumpEnabled(DebugDumpOption::LoopRotation)) {
      std::cout << "[Loop rotation] Prologue:" << std::endl
                << prologue->toString() << std::endl;
    }
    // main
    auto rotated = IrBuilder::create<kir::IfThenElse>(
        IrBuilder::create<kir::Predicate>(PredicateType::LoopRotation));
    auto main = IrBuilder::create<kir::ForLoop>(fl);
    for (auto expr : fl->body().exprs()) {
      if (selection_.count(expr) == 0) {
        main->body().push_back(expr);
      } else if (!expr->isA<kir::Allocate>()) {
        rotated->thenBody().push_back(expr);
      }
    }
    main->body().push_back(rotated);
    if (isDebugDumpEnabled(DebugDumpOption::LoopRotation)) {
      std::cout << "[Loop rotation] Main:" << std::endl
                << main->toString() << std::endl;
    }
    registerReplace(fl, main);
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* fl) final {
    ExprMutator::handle(fl);
    expandSelection(fl);
    auto id = fl->iter_domain();
    if (id == loop_concrete_id_) {
      TORCH_CHECK(
          validateSelection(fl), "Unable to rotate loop ", fl->toString());
      rotate(fl);
    }
  }

  void handle(Expr* expr) final {
    ExprMutator::handle(expr);
    expandSelection(expr);
  }
};

} // namespace

std::vector<Expr*> rotateLoops(
    const std::vector<Expr*>& exprs,
    const LoopRotationParam& params) {
  return RotateLoop::run(exprs, params);
}

} // namespace nvfuser
