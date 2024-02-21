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
    case ValType::NamedScalar:
      ptr(handler)->handle(val->as<NamedScalar>());
      return;
    case ValType::IterDomain:
      ptr(handler)->handle(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(handler)->handle(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(handler)->handle(val->as<TensorView>());
      return;
    case ValType::Predicate:
      ptr(handler)->handle(val->as<kir::Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<kir::TensorIndex>());
      return;
    case ValType::PipelineVal:
      ptr(handler)->handle(val->as<PipelineVal>());
      return;
    default:
      ptr(handler)->handle(val);
      return;
  }
  NVF_ERROR(
      false,
      "Unknown valtype in dispatch! val: ",
      val->toString(),
      " = ",
      val->toInlineString());
}

template <typename T>
void Expr::dispatch(T handler, Expr* expr) {
  if (expr->isStrictlyA<FullOp>()) {
    ptr(handler)->handle(expr->as<FullOp>());
    return;
  }
  if (expr->isStrictlyA<IotaOp>()) {
    ptr(handler)->handle(expr->as<IotaOp>());
    return;
  }
  if (expr->isStrictlyA<EyeOp>()) {
    ptr(handler)->handle(expr->as<EyeOp>());
    return;
  }
  if (expr->isStrictlyA<UnaryOp>()) {
    ptr(handler)->handle(expr->as<UnaryOp>());
    return;
  }
  if (expr->isStrictlyA<BinaryOp>()) {
    ptr(handler)->handle(expr->as<BinaryOp>());
    return;
  }
  if (expr->isStrictlyA<TernaryOp>()) {
    ptr(handler)->handle(expr->as<TernaryOp>());
    return;
  }
  if (expr->isStrictlyA<ArrayConstruct>()) {
    ptr(handler)->handle(expr->as<ArrayConstruct>());
    return;
  }
  if (expr->isStrictlyA<StructConstruct>()) {
    ptr(handler)->handle(expr->as<StructConstruct>());
    return;
  }
  if (expr->isStrictlyA<GetAttr>()) {
    ptr(handler)->handle(expr->as<GetAttr>());
    return;
  }
  if (expr->isStrictlyA<GetItem>()) {
    ptr(handler)->handle(expr->as<GetItem>());
    return;
  }
  if (expr->isStrictlyA<ReverseArray>()) {
    ptr(handler)->handle(expr->as<ReverseArray>());
    return;
  }
  if (expr->isStrictlyA<GetMetaData>()) {
    ptr(handler)->handle(expr->as<GetMetaData>());
    return;
  }
  if (expr->isStrictlyA<TensorConstruct>()) {
    ptr(handler)->handle(expr->as<TensorConstruct>());
    return;
  }
  if (expr->isStrictlyA<SelectOp>()) {
    ptr(handler)->handle(expr->as<SelectOp>());
    return;
  }
  if (expr->isStrictlyA<IndexSelectOp>()) {
    ptr(handler)->handle(expr->as<IndexSelectOp>());
    return;
  }
  if (expr->isStrictlyA<TorchGatherOp>()) {
    ptr(handler)->handle(expr->as<TorchGatherOp>());
    return;
  }
  if (expr->isStrictlyA<ScatterOp>()) {
    ptr(handler)->handle(expr->as<ScatterOp>());
    return;
  }
  if (expr->isStrictlyA<RNGOp>()) {
    ptr(handler)->handle(expr->as<RNGOp>());
    return;
  }
  if (expr->isStrictlyA<ReductionOp>()) {
    ptr(handler)->handle(expr->as<ReductionOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedReductionOp>()) {
    ptr(handler)->handle(expr->as<GroupedReductionOp>());
    return;
  }
  if (expr->isStrictlyA<WelfordOp>()) {
    ptr(handler)->handle(expr->as<WelfordOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedWelfordOp>()) {
    ptr(handler)->handle(expr->as<GroupedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<LoadStoreOp>()) {
    ptr(handler)->handle(expr->as<LoadStoreOp>());
    return;
  }
  if (expr->isStrictlyA<MmaOp>()) {
    ptr(handler)->handle(expr->as<MmaOp>());
    return;
  }
  if (expr->isStrictlyA<BroadcastOp>()) {
    ptr(handler)->handle(expr->as<BroadcastOp>());
    return;
  }
  if (expr->isStrictlyA<SqueezeOp>()) {
    ptr(handler)->handle(expr->as<SqueezeOp>());
    return;
  }
  if (expr->isStrictlyA<CatOp>()) {
    ptr(handler)->handle(expr->as<CatOp>());
    return;
  }
  if (expr->isStrictlyA<PadOp>()) {
    ptr(handler)->handle(expr->as<PadOp>());
    return;
  }
  if (expr->isStrictlyA<SliceOp>()) {
    ptr(handler)->handle(expr->as<SliceOp>());
    return;
  }
  if (expr->isStrictlyA<Split>()) {
    ptr(handler)->handle(expr->as<Split>());
    return;
  }
  if (expr->isStrictlyA<Merge>()) {
    ptr(handler)->handle(expr->as<Merge>());
    return;
  }
  if (expr->isStrictlyA<Swizzle>()) {
    ptr(handler)->handle(expr->as<Swizzle>());
    return;
  }
  if (expr->isStrictlyA<Swizzle2D>()) {
    ptr(handler)->handle(expr->as<Swizzle2D>());
    return;
  }
  if (expr->isStrictlyA<Resize>()) {
    ptr(handler)->handle(expr->as<Resize>());
    return;
  }
  if (expr->isStrictlyA<ExpandOp>()) {
    ptr(handler)->handle(expr->as<ExpandOp>());
    return;
  }
  if (expr->isStrictlyA<ShiftOp>()) {
    ptr(handler)->handle(expr->as<ShiftOp>());
    return;
  }
  if (expr->isStrictlyA<GatherOp>()) {
    ptr(handler)->handle(expr->as<GatherOp>());
    return;
  }
  if (expr->isStrictlyA<ViewAsScalar>()) {
    ptr(handler)->handle(expr->as<ViewAsScalar>());
    return;
  }
  if (expr->isStrictlyA<ViewOp>()) {
    ptr(handler)->handle(expr->as<ViewOp>());
    return;
  }
  if (expr->isStrictlyA<kir::Allocate>()) {
    ptr(handler)->handle(expr->as<kir::Allocate>());
    return;
  }
  if (expr->isStrictlyA<kir::Asm>()) {
    ptr(handler)->handle(expr->as<kir::Asm>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSync>()) {
    ptr(handler)->handle(expr->as<kir::BlockSync>());
    return;
  }
  if (expr->isStrictlyA<kir::GridSync>()) {
    ptr(handler)->handle(expr->as<kir::GridSync>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierInit>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierInit>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierInvalidate>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierInvalidate>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierArrive>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierArrive>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierArriveExpectTx>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierArriveExpectTx>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierWait>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierWait>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSerializeWait>()) {
    ptr(handler)->handle(expr->as<kir::BlockSerializeWait>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSerializeRelease>()) {
    ptr(handler)->handle(expr->as<kir::BlockSerializeRelease>());
    return;
  }
  if (expr->isStrictlyA<kir::AsyncWait>()) {
    ptr(handler)->handle(expr->as<kir::AsyncWait>());
    return;
  }
  if (expr->isStrictlyA<kir::AsyncCommit>()) {
    ptr(handler)->handle(expr->as<kir::AsyncCommit>());
    return;
  }
  if (expr->isStrictlyA<kir::InitMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::InitMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::UpdateMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::UpdateMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::ForLoop>()) {
    ptr(handler)->handle(expr->as<kir::ForLoop>());
    return;
  }
  if (expr->isStrictlyA<kir::IfThenElse>()) {
    ptr(handler)->handle(expr->as<kir::IfThenElse>());
    return;
  }
  if (expr->isStrictlyA<kir::GridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GridBroadcast>()) {
    ptr(handler)->handle(expr->as<kir::GridBroadcast>());
    return;
  }
  if (expr->isStrictlyA<kir::GridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::VectorizedWelfordOp>()) {
    ptr(handler)->handle(expr->as<kir::VectorizedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<kir::AllocateFusedReduction>()) {
    ptr(handler)->handle(expr->as<kir::AllocateFusedReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GetRNGSeedAndOffsetFromHost>()) {
    ptr(handler)->handle(expr->as<kir::GetRNGSeedAndOffsetFromHost>());
    return;
  }
  if (expr->isStrictlyA<kir::EncodeTensorMapTiled>()) {
    ptr(handler)->handle(expr->as<kir::EncodeTensorMapTiled>());
    return;
  }
  if (expr->isStrictlyA<PipelineStage>()) {
    ptr(handler)->handle(expr->as<PipelineStage>());
    return;
  }
  if (expr->isStrictlyA<PipelineCommunication>()) {
    ptr(handler)->handle(expr->as<PipelineCommunication>());
    return;
  }
  if (expr->isStrictlyA<assoc_comm::FlattenedAssocCommOp>()) {
    ptr(handler)->handle(expr->as<assoc_comm::FlattenedAssocCommOp>());
    return;
  }
  NVF_ERROR(false, "Unknown exprtype in dispatch: ", typeid(*expr).name());
}

template <typename T>
void Statement::dispatch(T handler, Statement* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->dispatch(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->dispatch(stmt->as<Expr>());
  } else {
    NVF_ERROR(false, "Unknown stmttype in dispatch!");
  }
}

template <typename T>
void Val::constDispatch(T handler, const Val* val) {
  switch (*(val->getValType())) {
    case ValType::NamedScalar:
      ptr(handler)->handle(val->as<NamedScalar>());
      return;
    case ValType::IterDomain:
      ptr(handler)->handle(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(handler)->handle(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(handler)->handle(val->as<TensorView>());
      return;
    case ValType::Predicate:
      ptr(handler)->handle(val->as<kir::Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(handler)->handle(val->as<kir::TensorIndex>());
      return;
    case ValType::PipelineVal:
      ptr(handler)->handle(val->as<PipelineVal>());
      return;
    default:
      ptr(handler)->handle(val);
      return;
  }
  NVF_ERROR(
      false,
      "Unknown valtype in dispatch! val: ",
      val->toString(),
      " = ",
      val->toInlineString());
}

template <typename T>
void Expr::constDispatch(T handler, const Expr* expr) {
  if (expr->isStrictlyA<FullOp>()) {
    ptr(handler)->handle(expr->as<FullOp>());
    return;
  }
  if (expr->isStrictlyA<IotaOp>()) {
    ptr(handler)->handle(expr->as<IotaOp>());
    return;
  }
  if (expr->isStrictlyA<EyeOp>()) {
    ptr(handler)->handle(expr->as<EyeOp>());
    return;
  }
  if (expr->isStrictlyA<UnaryOp>()) {
    ptr(handler)->handle(expr->as<UnaryOp>());
    return;
  }
  if (expr->isStrictlyA<BinaryOp>()) {
    ptr(handler)->handle(expr->as<BinaryOp>());
    return;
  }
  if (expr->isStrictlyA<TernaryOp>()) {
    ptr(handler)->handle(expr->as<TernaryOp>());
    return;
  }
  if (expr->isStrictlyA<ArrayConstruct>()) {
    ptr(handler)->handle(expr->as<ArrayConstruct>());
    return;
  }
  if (expr->isStrictlyA<StructConstruct>()) {
    ptr(handler)->handle(expr->as<StructConstruct>());
    return;
  }
  if (expr->isStrictlyA<GetAttr>()) {
    ptr(handler)->handle(expr->as<GetAttr>());
    return;
  }
  if (expr->isStrictlyA<GetItem>()) {
    ptr(handler)->handle(expr->as<GetItem>());
    return;
  }
  if (expr->isStrictlyA<ReverseArray>()) {
    ptr(handler)->handle(expr->as<ReverseArray>());
    return;
  }
  if (expr->isStrictlyA<GetMetaData>()) {
    ptr(handler)->handle(expr->as<GetMetaData>());
    return;
  }
  if (expr->isStrictlyA<TensorConstruct>()) {
    ptr(handler)->handle(expr->as<TensorConstruct>());
    return;
  }
  if (expr->isStrictlyA<SelectOp>()) {
    ptr(handler)->handle(expr->as<SelectOp>());
    return;
  }
  if (expr->isStrictlyA<IndexSelectOp>()) {
    ptr(handler)->handle(expr->as<IndexSelectOp>());
    return;
  }
  if (expr->isStrictlyA<TorchGatherOp>()) {
    ptr(handler)->handle(expr->as<TorchGatherOp>());
    return;
  }
  if (expr->isStrictlyA<ScatterOp>()) {
    ptr(handler)->handle(expr->as<ScatterOp>());
    return;
  }
  if (expr->isStrictlyA<RNGOp>()) {
    ptr(handler)->handle(expr->as<RNGOp>());
    return;
  }
  if (expr->isStrictlyA<ReductionOp>()) {
    ptr(handler)->handle(expr->as<ReductionOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedReductionOp>()) {
    ptr(handler)->handle(expr->as<GroupedReductionOp>());
    return;
  }
  if (expr->isStrictlyA<WelfordOp>()) {
    ptr(handler)->handle(expr->as<WelfordOp>());
    return;
  }
  if (expr->isStrictlyA<GroupedWelfordOp>()) {
    ptr(handler)->handle(expr->as<GroupedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<LoadStoreOp>()) {
    ptr(handler)->handle(expr->as<LoadStoreOp>());
    return;
  }
  if (expr->isStrictlyA<MmaOp>()) {
    ptr(handler)->handle(expr->as<MmaOp>());
    return;
  }
  if (expr->isStrictlyA<BroadcastOp>()) {
    ptr(handler)->handle(expr->as<BroadcastOp>());
    return;
  }
  if (expr->isStrictlyA<SqueezeOp>()) {
    ptr(handler)->handle(expr->as<SqueezeOp>());
    return;
  }
  if (expr->isStrictlyA<CatOp>()) {
    ptr(handler)->handle(expr->as<CatOp>());
    return;
  }
  if (expr->isStrictlyA<PadOp>()) {
    ptr(handler)->handle(expr->as<PadOp>());
    return;
  }
  if (expr->isStrictlyA<SliceOp>()) {
    ptr(handler)->handle(expr->as<SliceOp>());
    return;
  }
  if (expr->isStrictlyA<Split>()) {
    ptr(handler)->handle(expr->as<Split>());
    return;
  }
  if (expr->isStrictlyA<Merge>()) {
    ptr(handler)->handle(expr->as<Merge>());
    return;
  }
  if (expr->isStrictlyA<Swizzle>()) {
    ptr(handler)->handle(expr->as<Swizzle>());
    return;
  }
  if (expr->isStrictlyA<Swizzle2D>()) {
    ptr(handler)->handle(expr->as<Swizzle2D>());
    return;
  }
  if (expr->isStrictlyA<Resize>()) {
    ptr(handler)->handle(expr->as<Resize>());
    return;
  }
  if (expr->isStrictlyA<ExpandOp>()) {
    ptr(handler)->handle(expr->as<ExpandOp>());
    return;
  }
  if (expr->isStrictlyA<ShiftOp>()) {
    ptr(handler)->handle(expr->as<ShiftOp>());
    return;
  }
  if (expr->isStrictlyA<GatherOp>()) {
    ptr(handler)->handle(expr->as<GatherOp>());
    return;
  }
  if (expr->isStrictlyA<ViewAsScalar>()) {
    ptr(handler)->handle(expr->as<ViewAsScalar>());
    return;
  }
  if (expr->isStrictlyA<ViewOp>()) {
    ptr(handler)->handle(expr->as<ViewOp>());
    return;
  }
  if (expr->isStrictlyA<kir::Allocate>()) {
    ptr(handler)->handle(expr->as<kir::Allocate>());
    return;
  }
  if (expr->isStrictlyA<kir::Asm>()) {
    ptr(handler)->handle(expr->as<kir::Asm>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSync>()) {
    ptr(handler)->handle(expr->as<kir::BlockSync>());
    return;
  }
  if (expr->isStrictlyA<kir::GridSync>()) {
    ptr(handler)->handle(expr->as<kir::GridSync>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierInit>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierInit>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierInvalidate>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierInvalidate>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierArrive>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierArrive>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierArriveExpectTx>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierArriveExpectTx>());
    return;
  }
  if (expr->isStrictlyA<kir::MBarrierWait>()) {
    ptr(handler)->handle(expr->as<kir::MBarrierWait>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSerializeWait>()) {
    ptr(handler)->handle(expr->as<kir::BlockSerializeWait>());
    return;
  }
  if (expr->isStrictlyA<kir::BlockSerializeRelease>()) {
    ptr(handler)->handle(expr->as<kir::BlockSerializeRelease>());
    return;
  }
  if (expr->isStrictlyA<kir::AsyncWait>()) {
    ptr(handler)->handle(expr->as<kir::AsyncWait>());
    return;
  }
  if (expr->isStrictlyA<kir::AsyncCommit>()) {
    ptr(handler)->handle(expr->as<kir::AsyncCommit>());
    return;
  }
  if (expr->isStrictlyA<kir::InitMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::InitMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::UpdateMagicZero>()) {
    ptr(handler)->handle(expr->as<kir::UpdateMagicZero>());
    return;
  }
  if (expr->isStrictlyA<kir::ForLoop>()) {
    ptr(handler)->handle(expr->as<kir::ForLoop>());
    return;
  }
  if (expr->isStrictlyA<kir::IfThenElse>()) {
    ptr(handler)->handle(expr->as<kir::IfThenElse>());
    return;
  }
  if (expr->isStrictlyA<kir::GridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridReduction>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GridBroadcast>()) {
    ptr(handler)->handle(expr->as<kir::GridBroadcast>());
    return;
  }
  if (expr->isStrictlyA<kir::GridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::GroupedGridWelford>()) {
    ptr(handler)->handle(expr->as<kir::GroupedGridWelford>());
    return;
  }
  if (expr->isStrictlyA<kir::VectorizedWelfordOp>()) {
    ptr(handler)->handle(expr->as<kir::VectorizedWelfordOp>());
    return;
  }
  if (expr->isStrictlyA<kir::AllocateFusedReduction>()) {
    ptr(handler)->handle(expr->as<kir::AllocateFusedReduction>());
    return;
  }
  if (expr->isStrictlyA<kir::GetRNGSeedAndOffsetFromHost>()) {
    ptr(handler)->handle(expr->as<kir::GetRNGSeedAndOffsetFromHost>());
    return;
  }
  if (expr->isStrictlyA<kir::EncodeTensorMapTiled>()) {
    ptr(handler)->handle(expr->as<kir::EncodeTensorMapTiled>());
    return;
  }
  if (expr->isStrictlyA<PipelineStage>()) {
    ptr(handler)->handle(expr->as<PipelineStage>());
    return;
  }
  if (expr->isStrictlyA<PipelineCommunication>()) {
    ptr(handler)->handle(expr->as<PipelineCommunication>());
    return;
  }
  if (expr->isStrictlyA<assoc_comm::FlattenedAssocCommOp>()) {
    ptr(handler)->handle(expr->as<assoc_comm::FlattenedAssocCommOp>());
    return;
  }
  NVF_ERROR(false, "Unknown exprtype in dispatch: ", typeid(*expr).name());
}

template <typename T>
void Statement::constDispatch(T handler, const Statement* stmt) {
  if (stmt->isVal()) {
    ptr(handler)->dispatch(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    ptr(handler)->dispatch(stmt->as<Expr>());
  } else
    NVF_ERROR(false, "Unknown stmttype in dispatch!");
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
    case ValType::NamedScalar:
      ptr(mutator)->mutate(val->as<NamedScalar>());
      return;
    case ValType::IterDomain:
      ptr(mutator)->mutate(val->as<IterDomain>());
      return;
    case ValType::TensorDomain:
      ptr(mutator)->mutate(val->as<TensorDomain>());
      return;
    case ValType::TensorView:
      ptr(mutator)->mutate(val->as<TensorView>());
      return;
    case ValType::Predicate:
      ptr(mutator)->mutate(val->as<kir::Predicate>());
      return;
    case ValType::TensorIndex:
      ptr(mutator)->mutate(val->as<kir::TensorIndex>());
      return;
    case ValType::PipelineVal:
      ptr(mutator)->mutate(val->as<PipelineVal>());
      return;
    default:
      ptr(mutator)->mutate(val);
      return;
  }
  NVF_ERROR(false, "Unknown valtype in dispatch!");
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
  NVF_ERROR(false, "Unknown stmttype in dispatch!");
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
    NVF_ERROR(
        false,
        "Handle not overriden for ",
        stmt->as<Expr>()->getOpString(),
        ".");
  } else if (stmt->isVal()) {
    NVF_ERROR(
        false, "Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    NVF_ERROR(false, "Unrecognized statement type.");
  }
}

void OptInDispatch::unhandled(Statement* stmt) {
  if (stmt->isExpr()) {
    NVF_ERROR(
        false,
        "Handle not overriden for ",
        stmt->as<Expr>()->getOpString(),
        ".");
  } else if (stmt->isVal()) {
    NVF_ERROR(
        false, "Handle not overriden for ", stmt->getValType().value(), ".");
  } else {
    NVF_ERROR(false, "Unrecognized statement type.");
  }
}

// Vals
void OptOutConstDispatch::handle(const Val* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const NamedScalar* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const IterDomain* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorDomain* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorView* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const kir::Predicate* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::TensorIndex* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const PipelineVal* stmt) {
  unhandled(stmt);
}

// Exprs
void OptOutConstDispatch::handle(const FullOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const IotaOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const EyeOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const UnaryOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const BinaryOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TernaryOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ArrayConstruct* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const StructConstruct* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GetAttr* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GetItem* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ReverseArray* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GetMetaData* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TensorConstruct* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const SelectOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const IndexSelectOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const TorchGatherOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ScatterOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const RNGOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GroupedReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const WelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GroupedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const LoadStoreOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const MmaOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const BroadcastOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const SqueezeOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const CatOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const PadOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const SliceOp* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const Split* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Merge* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Swizzle* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Swizzle2D* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const Resize* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ExpandOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ShiftOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const GatherOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ViewAsScalar* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const ViewOp* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const kir::Allocate* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::Asm* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::BlockSync* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridSync* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::MBarrierInit* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::MBarrierInvalidate* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::MBarrierArrive* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::MBarrierArriveExpectTx* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::MBarrierWait* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::BlockSerializeWait* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::BlockSerializeRelease* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::AsyncWait* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::AsyncCommit* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::InitMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::UpdateMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::ForLoop* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::IfThenElse* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GroupedGridReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridBroadcast* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GridWelford* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GroupedGridWelford* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::VectorizedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::AllocateFusedReduction* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::GetRNGSeedAndOffsetFromHost* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const kir::EncodeTensorMapTiled* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const PipelineStage* stmt) {
  unhandled(stmt);
}
void OptOutConstDispatch::handle(const PipelineCommunication* stmt) {
  unhandled(stmt);
}

void OptOutConstDispatch::handle(const assoc_comm::FlattenedAssocCommOp* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::unhandled(Statement*) {}

// Vals
void OptOutDispatch::handle(Val* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(NamedScalar* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(IterDomain* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorDomain* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorView* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(kir::Predicate* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::TensorIndex* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(PipelineVal* stmt) {
  unhandled(stmt);
}

// Exprs
void OptOutDispatch::handle(FullOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(IotaOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(EyeOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(UnaryOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(BinaryOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TernaryOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ArrayConstruct* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(StructConstruct* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GetAttr* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GetItem* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ReverseArray* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GetMetaData* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TensorConstruct* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(SelectOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(IndexSelectOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(TorchGatherOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ScatterOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(RNGOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GroupedReductionOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(WelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GroupedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(LoadStoreOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(MmaOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(BroadcastOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(SqueezeOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(CatOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(PadOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(SliceOp* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(Split* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Merge* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Swizzle* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Swizzle2D* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(Resize* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ExpandOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ShiftOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(GatherOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ViewAsScalar* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(ViewOp* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(kir::Allocate* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::Asm* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::BlockSync* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridSync* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::MBarrierInit* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::MBarrierInvalidate* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::MBarrierArrive* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::MBarrierArriveExpectTx* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::MBarrierWait* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::BlockSerializeWait* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::BlockSerializeRelease* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::AsyncWait* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::AsyncCommit* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::InitMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::UpdateMagicZero* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::ForLoop* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::IfThenElse* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GroupedGridReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridBroadcast* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GridWelford* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GroupedGridWelford* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::VectorizedWelfordOp* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::AllocateFusedReduction* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::GetRNGSeedAndOffsetFromHost* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(kir::EncodeTensorMapTiled* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(PipelineStage* stmt) {
  unhandled(stmt);
}
void OptOutDispatch::handle(PipelineCommunication* stmt) {
  unhandled(stmt);
}

void OptOutDispatch::handle(assoc_comm::FlattenedAssocCommOp* stmt) {
  unhandled(stmt);
}

} // namespace nvfuser
