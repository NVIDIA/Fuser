// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_simplifier.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <type.h>

#include <dispatch.h>

namespace nvfuser {

template <typename T>
T* ptr(T& obj) {
  return &obj;
}

template <typename T>
T* ptr(T* obj) {
  return obj;
}

/*
 * Generic dispatch for any handler that does not modify the IR directly.
 * For example we may want to walk the graph to construct a topologically sorted
 * set of exprs. This doesn't modify the IR directly. We also use this to print
 * the IR itself.
 * This dispatch is paired with a class that implements the functions:
 * template <typenname node_type>
 * int handler(node_type* node)
 *
 * handler should call:
 * dispatch(this, node_to_dispatch)
 *
 * It could also implement:
 * int handler(Statement* stmt){
 *   dispatch(this, stmt);
 * }
 *
 * And therefore dispatch should never call:
 * ptr(mutator)->mutate(this->as<Statement>());
 */

template <typename T>
void Val::dispatch(T handler, Val* val) {
  switch (*(val->getValType())) {
#define M(e)                            \
  case ValType::e:                      \
    ptr(handler)->handle(val->as<e>()); \
    return;
    DISPATCH_FOR_ALL_VALS(M)
#undef M
#define M(e)                                 \
  case ValType::e:                           \
    ptr(handler)->handle(val->as<kir::e>()); \
    return;
    DISPATCH_FOR_ALL_KIR_VALS(M)
#undef M
#define M(e)                                 \
  case ValType::e:                           \
    ptr(handler)->handle(val->as<hir::e>()); \
    return;
    DISPATCH_FOR_ALL_HIR_VALS(M)
#undef M
    default:
      ptr(handler)->handle(val);
      return;
  }
  NVF_THROW(
      "Unknown valtype in dispatch! val: ",
      val->toString(),
      " = ",
      val->toInlineString());
}

template <typename T>
void Expr::dispatch(T handler, Expr* expr) {
#define M(e)                             \
  if (expr->isStrictlyA<e>()) {          \
    ptr(handler)->handle(expr->as<e>()); \
    return;                              \
  }
  DISPATCH_FOR_ALL_EXPRS(M)
  M(assoc_comm::FlattenedAssocCommOp)
#undef M
#define M(e)                                  \
  if (expr->isStrictlyA<kir::e>()) {          \
    ptr(handler)->handle(expr->as<kir::e>()); \
    return;                                   \
  }
  DISPATCH_FOR_ALL_KIR_EXPRS(M)
#undef M
#define M(e)                                  \
  if (expr->isStrictlyA<hir::e>()) {          \
    ptr(handler)->handle(expr->as<hir::e>()); \
    return;                                   \
  }
  DISPATCH_FOR_ALL_HIR_EXPRS(M)
#undef M
  NVF_THROW("Unknown exprtype in dispatch: ", typeid(*expr).name());
}

template <typename T>
void Statement::dispatch(T handler, Statement* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->dispatch(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->dispatch(stmt->as<Expr>());
  } else {
    NVF_THROW("Unknown stmttype in dispatch!");
  }
}

template <typename T>
void Val::constDispatch(T handler, const Val* val) {
  switch (*(val->getValType())) {
#define M(e)                            \
  case ValType::e:                      \
    ptr(handler)->handle(val->as<e>()); \
    return;
    DISPATCH_FOR_ALL_VALS(M)
#undef M
#define M(e)                                 \
  case ValType::e:                           \
    ptr(handler)->handle(val->as<kir::e>()); \
    return;
    DISPATCH_FOR_ALL_KIR_VALS(M)
#undef M
#define M(e)                                 \
  case ValType::e:                           \
    ptr(handler)->handle(val->as<hir::e>()); \
    return;
    DISPATCH_FOR_ALL_HIR_VALS(M)
#undef M
    default:
      ptr(handler)->handle(val);
      return;
  }
  NVF_THROW(
      "Unknown valtype in dispatch! val: ",
      val->toString(),
      " = ",
      val->toInlineString());
}

template <typename T>
void Expr::constDispatch(T handler, const Expr* expr) {
#define M(e)                             \
  if (expr->isStrictlyA<e>()) {          \
    ptr(handler)->handle(expr->as<e>()); \
    return;                              \
  }
  DISPATCH_FOR_ALL_EXPRS(M)
  M(assoc_comm::FlattenedAssocCommOp)
#undef M
#define M(e)                                  \
  if (expr->isStrictlyA<kir::e>()) {          \
    ptr(handler)->handle(expr->as<kir::e>()); \
    return;                                   \
  }
  DISPATCH_FOR_ALL_KIR_EXPRS(M)
#undef M
#define M(e)                                  \
  if (expr->isStrictlyA<hir::e>()) {          \
    ptr(handler)->handle(expr->as<hir::e>()); \
    return;                                   \
  }
  DISPATCH_FOR_ALL_HIR_EXPRS(M)
#undef M
  NVF_THROW("Unknown exprtype in dispatch: ", typeid(*expr).name());
}

template <typename T>
void Statement::constDispatch(T handler, const Statement* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->dispatch(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->dispatch(stmt->as<Expr>());
  } else {
    NVF_THROW("Unknown stmttype in dispatch!");
  }
}

/*
 * Generic mutatorDispatch for any handler that modifies the IR. This could be
 * a transformation on loop structures, or parallelizing a loop. This
 * mutatorDispatch is paired with a class that implements the functions
 * template <typenname node_type> Statement* mutate(node_type* node) mutate
 * should call (statement* node_to_dispatch)->mutatorDispatch() It could also
 * implement Statement* mutate(Statement* stmt){ stmt->mutatorDispatch(this);
 * }
 * And therefore dispatch should never call:
 *   ptr(mutator)->mutate(this->as<Statement>());
 */
template <typename T>
void Val::mutatorDispatch(T mutator, Val* val) {
  switch (*(val->getValType())) {
#define M(e)                            \
  case ValType::e:                      \
    ptr(mutator)->mutate(val->as<e>()); \
    return;
    DISPATCH_FOR_ALL_VALS(M)
#undef M
#define M(e)                                 \
  case ValType::e:                           \
    ptr(mutator)->mutate(val->as<kir::e>()); \
    return;
    DISPATCH_FOR_ALL_KIR_VALS(M)
#undef M
#define M(e)                                 \
  case ValType::e:                           \
    ptr(mutator)->mutate(val->as<hir::e>()); \
    return;
    DISPATCH_FOR_ALL_HIR_VALS(M)
#undef M
    default:
      ptr(mutator)->mutate(val);
      return;
  }
  NVF_THROW("Unknown valtype in dispatch!");
}

template <typename T>
void Statement::mutatorDispatch(T mutator, Statement* stmt) {
  if (stmt->isVal()) {
    ptr(mutator)->dispatchMutate(stmt->as<Val>());
    return;
  }
  if (stmt->isExpr()) {
    ptr(mutator)->mutate(stmt->as<Expr>());
    return;
  }
  NVF_THROW("Unknown stmttype in dispatch!");
}

/*
 * Handler template instantiations. These should only have to be done on base
 * classes. Actual visitors/mutators should inhereit from these classes and call
 * ->dispatch(this) to avoid needing an explicit instantiation.
 */
template void Statement::dispatch(OptOutDispatch&, Statement*);
template void Statement::dispatch(OptOutDispatch*, Statement*);
template void Val::dispatch(OptOutDispatch&, Val*);
template void Val::dispatch(OptOutDispatch*, Val*);
template void Expr::dispatch(OptOutDispatch&, Expr*);
template void Expr::dispatch(OptOutDispatch*, Expr*);

template void Statement::dispatch(OptInDispatch, Statement*);
template void Statement::dispatch(OptInDispatch*, Statement*);
template void Val::dispatch(OptInDispatch, Val*);
template void Val::dispatch(OptInDispatch*, Val*);
template void Expr::dispatch(OptInDispatch, Expr*);
template void Expr::dispatch(OptInDispatch*, Expr*);

template void Statement::constDispatch(OptOutConstDispatch&, const Statement*);
template void Statement::constDispatch(OptOutConstDispatch*, const Statement*);
template void Val::constDispatch(OptOutConstDispatch&, const Val*);
template void Val::constDispatch(OptOutConstDispatch*, const Val*);
template void Expr::constDispatch(OptOutConstDispatch&, const Expr*);
template void Expr::constDispatch(OptOutConstDispatch*, const Expr*);

template void Statement::constDispatch(OptInConstDispatch&, const Statement*);
template void Statement::constDispatch(OptInConstDispatch*, const Statement*);
template void Val::constDispatch(OptInConstDispatch&, const Val*);
template void Val::constDispatch(OptInConstDispatch*, const Val*);
template void Expr::constDispatch(OptInConstDispatch&, const Expr*);
template void Expr::constDispatch(OptInConstDispatch*, const Expr*);

template void Statement::mutatorDispatch(OptOutMutator&, Statement*);
template void Statement::mutatorDispatch(OptOutMutator*, Statement*);
template void Val::mutatorDispatch(OptOutMutator&, Val*);
template void Val::mutatorDispatch(OptOutMutator*, Val*);

void OptOutDispatch::dispatch(Statement* s) {
  Statement::dispatch(this, s);
}

void OptOutDispatch::dispatch(Expr* e) {
  Expr::dispatch(this, e);
}

void OptOutDispatch::dispatch(Val* v) {
  Val::dispatch(this, v);
}

void OptOutConstDispatch::dispatch(const Statement* s) {
  Statement::constDispatch(this, s);
}

void OptOutConstDispatch::dispatch(const Expr* e) {
  Expr::constDispatch(this, e);
}

void OptOutConstDispatch::dispatch(const Val* v) {
  Val::constDispatch(this, v);
}

void OptInConstDispatch::unhandled(const Statement* stmt) {
  if (stmt->isExpr()) {
    NVF_THROW(
        "Handle not overriden for ", stmt->as<Expr>()->getOpString(), ".");
  } else if (stmt->isVal()) {
    NVF_THROW("Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    NVF_THROW("Unrecognized statement type.");
  }
}

void OptInDispatch::unhandled(Statement* stmt) {
  if (stmt->isExpr()) {
    NVF_THROW(
        "Handle not overriden for ", stmt->as<Expr>()->getOpString(), ".");
  } else if (stmt->isVal()) {
    NVF_THROW("Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    NVF_THROW("Unrecognized statement type.");
  }
}

#define M(e)                                        \
  void OptOutConstDispatch::handle(const e* stmt) { \
    unhandled(stmt);                                \
  }
M(Val)
DISPATCH_FOR_ALL_VALS(M)
DISPATCH_FOR_ALL_EXPRS(M)
M(assoc_comm::FlattenedAssocCommOp)
#undef M
#define M(e)                                             \
  void OptOutConstDispatch::handle(const kir::e* stmt) { \
    unhandled(stmt);                                     \
  }
DISPATCH_FOR_ALL_KIR_EXPRS(M)
DISPATCH_FOR_ALL_KIR_VALS(M)
#undef M
#define M(e)                                             \
  void OptOutConstDispatch::handle(const hir::e* stmt) { \
    unhandled(stmt);                                     \
  }
DISPATCH_FOR_ALL_HIR_VALS(M)
DISPATCH_FOR_ALL_HIR_EXPRS(M)
#undef M

void OptOutDispatch::unhandled(Statement*) {}

// Vals
#define M(e)                             \
  void OptOutDispatch::handle(e* stmt) { \
    unhandled(stmt);                     \
  }
M(Val)
DISPATCH_FOR_ALL_VALS(M)
DISPATCH_FOR_ALL_EXPRS(M)
M(assoc_comm::FlattenedAssocCommOp)
#undef M
#define M(e)                                  \
  void OptOutDispatch::handle(kir::e* stmt) { \
    unhandled(stmt);                          \
  }
DISPATCH_FOR_ALL_KIR_VALS(M)
DISPATCH_FOR_ALL_KIR_EXPRS(M)
#undef M
#define M(e)                                  \
  void OptOutDispatch::handle(hir::e* stmt) { \
    unhandled(stmt);                          \
  }
DISPATCH_FOR_ALL_HIR_VALS(M)
DISPATCH_FOR_ALL_HIR_EXPRS(M)
#undef M

} // namespace nvfuser
