// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <exceptions.h>

#include <utils.h>

#include <complex>
#include <unordered_map>

// dispatch.h prevents the need from adding manual dispatch in every class that
// wants to define how to process a series of nodes. dispatch.h provides 4
// classes that can be inherited providing a means to override functions on a
// per-node basis. There are currently 4 provided dispatch mechanisms:
//
// OptOutDispatch:
//
// provides the functions:
// virtual void handle(ValType* irnode){}
//
// This provides a mechanisms to override this handle for particular node
// types. For example if we only wanted to actually run a function on
// BinaryOps, we could inherit OptOutDispatch and simply override: void
// handle(BinaryOp*) { doSomething; } Then we could run through all our
// Statement* and call OptOutDispatch::handle(statement). When a BinaryOp is
// encountered our override function will be called. For every other node,
// nothing will be done.
//
// OptInDispatch:
//
// This class is similar to OptOutDispatch, however if we encounter a node
// that we haven't specified an override for in the derived class, an error
// will be thrown. This is useful if we create a class that is expected to
// handle any type of node it encounters.
//
// OptOutMutator:
//
// This class is similar to OptOutDispatch except the functions provided are of
// type: virtual Statement* mutate(Statement*) this is useful for when we want
// to have an IR node result from our overloaded functions.
//
// OptInMutator:
//
// This class is similar to OptInDispatch except the functions provided are of
// type: virtual Statement* mutate(Statement*) this is useful for when we want
// to have an IR node result from our overloaded functions.

namespace nvfuser {
class IrContainer;
class Fusion;

// Hierarchal dispatch functions for handle
class Statement;
class Expr;
class Val;

// Vals
class IterDomain;
class TensorDomain;
class TensorView;

class NamedScalar;

class PipelineVal;

// Exprs
class FullOp;
class IotaOp;
class EyeOp;
class UnaryOp;
class BinaryOp;
class TernaryOp;
class ArrayConstruct;
class StructConstruct;
class GetAttr;
class GetItem;
class ReverseArray;
class GetMetaData;
class TensorConstruct;
class SelectOp;
class IndexSelectOp;
class TorchGatherOp;
class ScatterOp;
class RNGOp;
class ReductionOp;
class GroupedReductionOp;
class WelfordOp;
class GroupedWelfordOp;
class LoadStoreOp;
class MmaOp;
class BroadcastOp;
class SqueezeOp;
class ExpandOp;
class ShiftOp;
class GatherOp;
class ViewAsScalar;
class ViewOp;
class CatOp;
class PadOp;
class SliceOp;

class PipelineStage;
class PipelineCommunication;

// Exprs
class Split;
class Merge;
class Swizzle2D;
class Resize;

namespace kir {
class Predicate;
class TensorIndex;

class Allocate;
class BlockSync;
class GridSync;
class MBarrierInit;
class MBarrierInvalidate;
class MBarrierArrive;
class MBarrierArriveExpectTx;
class MBarrierWait;
class CpAsyncWait;
class CpAsyncCommit;
class CpAsyncBulkS2GWait;
class CpAsyncBulkS2GCommit;
class ForLoop;
class IfThenElse;
class GridReduction;
class GroupedGridReduction;
class GridBroadcast;
class GridWelford;
class GroupedGridWelford;
class VectorizedWelfordOp;
class AllocateFusedReduction;
class InitMagicZero;
class UpdateMagicZero;
class GetRNGSeedAndOffsetFromHost;
class EncodeTensorMapTiled;

} // namespace kir

// By default, all IR nodes are handled in this dispatch, and will call an empty
// function on all nodes.
class OptOutConstDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(const Statement*) {}

 public:
  // Hierarchal dispatch functions for handle
  virtual void dispatch(const Statement*);
  virtual void dispatch(const Expr*);
  virtual void dispatch(const Val*);

  // Vals
  virtual void handle(const IterDomain* stmt);
  virtual void handle(const TensorDomain* stmt);
  virtual void handle(const TensorView* stmt);
  virtual void handle(const Val* stmt);
  virtual void handle(const NamedScalar* stmt);

  virtual void handle(const kir::Predicate*);
  virtual void handle(const kir::TensorIndex*);

  virtual void handle(const PipelineVal*);

  // Exprs
  virtual void handle(const FullOp* stmt);
  virtual void handle(const IotaOp* stmt);
  virtual void handle(const EyeOp* stmt);
  virtual void handle(const UnaryOp* stmt);
  virtual void handle(const BinaryOp* stmt);
  virtual void handle(const TernaryOp* stmt);
  virtual void handle(const ArrayConstruct* stmt);
  virtual void handle(const StructConstruct* stmt);
  virtual void handle(const GetAttr* stmt);
  virtual void handle(const GetItem* stmt);
  virtual void handle(const ReverseArray* stmt);
  virtual void handle(const GetMetaData* stmt);
  virtual void handle(const TensorConstruct* stmt);
  virtual void handle(const SelectOp* stmt);
  virtual void handle(const IndexSelectOp* stmt);
  virtual void handle(const TorchGatherOp* stmt);
  virtual void handle(const ScatterOp* stmt);
  virtual void handle(const RNGOp* stmt);
  virtual void handle(const ReductionOp* stmt);
  virtual void handle(const GroupedReductionOp* stmt);
  virtual void handle(const WelfordOp* stmt);
  virtual void handle(const GroupedWelfordOp* stmt);
  virtual void handle(const LoadStoreOp* stmt);
  virtual void handle(const MmaOp* stmt);
  virtual void handle(const BroadcastOp* stmt);
  virtual void handle(const SqueezeOp* stmt);
  virtual void handle(const CatOp* stmt);
  virtual void handle(const PadOp* stmt);
  virtual void handle(const SliceOp* stmt);

  virtual void handle(const Split* stmt);
  virtual void handle(const Merge* stmt);
  virtual void handle(const Swizzle2D* stmt);
  virtual void handle(const Resize* stmt);
  virtual void handle(const ExpandOp* stmt);
  virtual void handle(const ShiftOp* stmt);
  virtual void handle(const GatherOp* stmt);
  virtual void handle(const ViewAsScalar* stmt);
  virtual void handle(const ViewOp* stmt);

  virtual void handle(const kir::Allocate*);
  virtual void handle(const kir::BlockSync*);
  virtual void handle(const kir::GridSync*);
  virtual void handle(const kir::MBarrierInit*);
  virtual void handle(const kir::MBarrierInvalidate*);
  virtual void handle(const kir::MBarrierArrive*);
  virtual void handle(const kir::MBarrierArriveExpectTx*);
  virtual void handle(const kir::MBarrierWait*);
  virtual void handle(const kir::CpAsyncWait*);
  virtual void handle(const kir::CpAsyncCommit*);
  virtual void handle(const kir::CpAsyncBulkS2GWait*);
  virtual void handle(const kir::CpAsyncBulkS2GCommit*);
  virtual void handle(const kir::InitMagicZero*);
  virtual void handle(const kir::UpdateMagicZero*);
  virtual void handle(const kir::ForLoop*);
  virtual void handle(const kir::IfThenElse*);
  virtual void handle(const kir::GridReduction*);
  virtual void handle(const kir::GroupedGridReduction*);
  virtual void handle(const kir::GridBroadcast*);
  virtual void handle(const kir::GridWelford*);
  virtual void handle(const kir::GroupedGridWelford*);
  virtual void handle(const kir::VectorizedWelfordOp*);
  virtual void handle(const kir::AllocateFusedReduction*);
  virtual void handle(const kir::GetRNGSeedAndOffsetFromHost*);
  virtual void handle(const kir::EncodeTensorMapTiled*);

  virtual void handle(const PipelineStage*);
  virtual void handle(const PipelineCommunication*);
};

class OptOutDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(Statement*);

 public:
  // Hierarchal dispatch functions for handle
  virtual void dispatch(Statement*);
  virtual void dispatch(Expr*);
  virtual void dispatch(Val*);

  // Vals
  virtual void handle(Val* stmt);
  virtual void handle(NamedScalar* stmt);
  virtual void handle(IterDomain* stmt);
  virtual void handle(TensorDomain* stmt);
  virtual void handle(TensorView* stmt);

  virtual void handle(kir::Predicate*);
  virtual void handle(kir::TensorIndex*);

  virtual void handle(PipelineVal*);

  // Exprs
  virtual void handle(FullOp* stmt);
  virtual void handle(IotaOp* stmt);
  virtual void handle(EyeOp* stmt);
  virtual void handle(UnaryOp* stmt);
  virtual void handle(BinaryOp* stmt);
  virtual void handle(TernaryOp* stmt);
  virtual void handle(ArrayConstruct* stmt);
  virtual void handle(StructConstruct* stmt);
  virtual void handle(GetAttr* stmt);
  virtual void handle(GetItem* stmt);
  virtual void handle(ReverseArray* stmt);
  virtual void handle(GetMetaData* stmt);
  virtual void handle(TensorConstruct* stmt);
  virtual void handle(SelectOp* stmt);
  virtual void handle(IndexSelectOp* stmt);
  virtual void handle(TorchGatherOp* stmt);
  virtual void handle(ScatterOp* stmt);
  virtual void handle(RNGOp* stmt);
  virtual void handle(ReductionOp* stmt);
  virtual void handle(GroupedReductionOp* stmt);
  virtual void handle(WelfordOp* stmt);
  virtual void handle(GroupedWelfordOp* stmt);
  virtual void handle(LoadStoreOp* stmt);
  virtual void handle(MmaOp* stmt);
  virtual void handle(BroadcastOp* stmt);
  virtual void handle(SqueezeOp* stmt);
  virtual void handle(CatOp* stmt);
  virtual void handle(PadOp* stmt);
  virtual void handle(SliceOp* stmt);

  virtual void handle(Split* stmt);
  virtual void handle(Merge* stmt);
  virtual void handle(Swizzle2D* stmt);
  virtual void handle(Resize* stmt);
  virtual void handle(ExpandOp* stmt);
  virtual void handle(ShiftOp* stmt);
  virtual void handle(GatherOp* stmt);
  virtual void handle(ViewAsScalar* stmt);
  virtual void handle(ViewOp* stmt);

  virtual void handle(kir::Allocate* stmt);
  virtual void handle(kir::BlockSync* stmt);
  virtual void handle(kir::GridSync* stmt);
  virtual void handle(kir::MBarrierInit* stmt);
  virtual void handle(kir::MBarrierInvalidate* stmt);
  virtual void handle(kir::MBarrierArrive* stmt);
  virtual void handle(kir::MBarrierArriveExpectTx* stmt);
  virtual void handle(kir::MBarrierWait* stmt);
  virtual void handle(kir::CpAsyncWait* stmt);
  virtual void handle(kir::CpAsyncCommit* stmt);
  virtual void handle(kir::CpAsyncBulkS2GWait* stmt);
  virtual void handle(kir::CpAsyncBulkS2GCommit* stmt);
  virtual void handle(kir::InitMagicZero* stmt);
  virtual void handle(kir::UpdateMagicZero* stmt);
  virtual void handle(kir::ForLoop* stmt);
  virtual void handle(kir::IfThenElse* stmt);
  virtual void handle(kir::GridReduction* stmt);
  virtual void handle(kir::GroupedGridReduction* stmt);
  virtual void handle(kir::GridBroadcast* stmt);
  virtual void handle(kir::GridWelford* stmt);
  virtual void handle(kir::GroupedGridWelford* stmt);
  virtual void handle(kir::VectorizedWelfordOp* stmt);
  virtual void handle(kir::AllocateFusedReduction* stmt);
  virtual void handle(kir::GetRNGSeedAndOffsetFromHost* stmt);
  virtual void handle(kir::EncodeTensorMapTiled* stmt);

  virtual void handle(PipelineStage* stmt);
  virtual void handle(PipelineCommunication* stmt);
};

class OptInConstDispatch : public OptOutConstDispatch {
 public:
  using OptOutConstDispatch::handle;

 protected:
  void unhandled(const Statement* stmt) final;
};

class OptInDispatch : public OptOutDispatch {
 public:
  using OptOutDispatch::handle;

 protected:
  void unhandled(Statement* stmt) final;
};

// Class to perform mutations on Fusion IR. Exprs can simply be redefined, but
// when mutating values they have to be registered through registerMutation so
// that exprs can detect there's been a muatation and know to modify all
// instances of that Val. This means each Val should be mutated "consistently".
// Otherwise behavior may be difficult to understand as it depends on which
// order mutate is called in. This class expects user to topologically call the
// statments of interest so inputs are called and mutated before exprs depending
// on them.
//
// Warning: TensorViews need to be treated carefully. As we don't generally
// register their mutation when their tensor domains only change. If a TV needs
// to be swapped out, it needs to be registered as a "proper" mutation like
// other vals, on top of TensorDomain being updated in the mutated TensorView.
//
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class OptOutMutator : public PolymorphicBase {
 public:
  // Hierarchal dispatch functions for handle
  virtual void dispatchMutate(Statement* s);
  virtual void dispatchMutate(Val* v);

  void registerMutation(Val* val, Val* mutation);

  Val* maybeMutated(Val* val) const;

  std::unordered_map<Val*, Val*> mutations_;

  //****Functions below defined in mutator.cpp*****

  // Vals
  virtual void mutate(Val*);
  virtual void mutate(NamedScalar*);
  virtual void mutate(IterDomain*);
  virtual void mutate(TensorDomain*);
  virtual void mutate(TensorView*);
  virtual void mutate(PipelineVal*);

  virtual void mutate(kir::Predicate*);
  virtual void mutate(kir::TensorIndex*);

  //! This method replaces e if any inputs or attributes are registered for
  //! mutation.
  virtual void mutate(Expr* e) {
    mutateExpr(
        e,
        /*replace_outputs*/ false,
        /*replace_inputs*/ true,
        /*replace_attrs*/ true);
  }

  //! Unlike mutate(Expr*), this method replaces e only if any outputs are
  //! registered for mutation. Inputs and attributes are unchanges. This method
  //! is useful for tranferring the definition of e's current outputs to those
  //! their respective registered mutations.
  Expr* mutateExprOutputsOnly(Expr* e) {
    return mutateExpr(
        e,
        /*replace_outputs*/ true,
        /*replace_inputs*/ false,
        /*replace_attrs*/ false);
  }

 protected:
  virtual void removeExpr(IrContainer*, Expr*) const;
  virtual void registerNewExpr(Expr*) {}

 private:
  //! Replaces Expr if any inputs, attrs, or outputs are registered for
  //! mutation. See comment on mutateExprOutputsOnly for more information.
  Expr* mutateExpr(
      Expr*,
      bool replace_outputs = false,
      bool replace_inputs = true,
      bool replace_attrs = true);
};

} // namespace nvfuser
