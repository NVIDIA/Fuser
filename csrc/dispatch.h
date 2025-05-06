// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <utils.h>
#include <visibility.h>

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

#define DISPATCH_FOR_ALL_VALS(f) \
  f(IterDomain);                 \
  f(TensorDomain);               \
  f(TensorView);                 \
  f(NamedScalar);
#define DISPATCH_FOR_ALL_KIR_VALS(f) f(Predicate) f(TensorIndex)
#define DISPATCH_FOR_ALL_HIR_VALS(f) f(Stream)

#define DISPATCH_FOR_ALL_EXPRS(f) \
  f(FullOp);                      \
  f(IotaOp);                      \
  f(EyeOp);                       \
  f(UnaryOp);                     \
  f(BinaryOp);                    \
  f(TernaryOp);                   \
  f(ArrayConstruct);              \
  f(StructConstruct);             \
  f(GetAttr);                     \
  f(GetItem);                     \
  f(ReverseArray);                \
  f(GetMetaData);                 \
  f(TensorConstruct);             \
  f(SelectOp);                    \
  f(IndexSelectOp);               \
  f(IndexPutAccumulateOp);        \
  f(GatherOp);                    \
  f(ScatterOp);                   \
  f(RNGOp);                       \
  f(ReductionOp);                 \
  f(GroupedReductionOp);          \
  f(WelfordOp);                   \
  f(GroupedWelfordOp);            \
  f(LoadStoreOp);                 \
  f(MmaOp);                       \
  f(BroadcastOp);                 \
  f(SqueezeOp);                   \
  f(ExpandOp);                    \
  f(RepeatOp);                    \
  f(ViewAsScalar);                \
  f(ViewOp);                      \
  f(CatOp);                       \
  f(PadOp);                       \
  f(SliceOp);                     \
  f(Split);                       \
  f(Merge);                       \
  f(Swizzle);                     \
  f(Swizzle2D);                   \
  f(Resize);                      \
  f(MatmulOp);                    \
  f(LinearOp);                    \
  f(SdpaFwdOp);                   \
  f(SdpaBwdOp);                   \
  f(EmbeddingFwdOp);              \
  f(Communication);               \
  f(ForLoop);                     \
  f(P2PCommunication);
#define DISPATCH_FOR_ALL_KIR_EXPRS(f) \
  f(Allocate);                        \
  f(AllocTMem);                       \
  f(Asm);                             \
  f(BlockSync);                       \
  f(GridSync);                        \
  f(FenceAsyncProxy);                 \
  f(WgMmaFence);                      \
  f(SetMaxNReg);                      \
  f(Continue);                        \
  f(Return);                          \
  f(MBarrierInit);                    \
  f(MBarrierInvalidate);              \
  f(MBarrierArrive);                  \
  f(MBarrierArriveExpectTx);          \
  f(MBarrierWait);                    \
  f(MBarrierWaitParity);              \
  f(BlockSerializeWait);              \
  f(BlockSerializeRelease);           \
  f(AsyncWait);                       \
  f(AsyncCommit);                     \
  f(IfThenElse);                      \
  f(GridReduction);                   \
  f(GroupedGridReduction);            \
  f(GridBroadcast);                   \
  f(GridWelford);                     \
  f(GroupedGridWelford);              \
  f(VectorizedWelfordOp);             \
  f(AllocateFusedReduction);          \
  f(InitMagicZero);                   \
  f(UpdateMagicZero);                 \
  f(GetRNGSeedAndOffsetFromHost);     \
  f(EncodeTensorMapTiled);            \
  f(RNGOp);
#define DISPATCH_FOR_ALL_HIR_EXPRS(f) \
  f(HostUnit);                        \
  f(PostOnStream);                    \
  f(LaunchKernel);                    \
  f(SetCurrentStream);                \
  f(GetCurrentStream);                \
  f(Wait);                            \
  f(Synchronize);                     \
  f(StartCoalescing);                 \
  f(EndCoalescing);                   \
  f(ShareMemHandles);                 \
  f(HirAliasSelect);                  \
  f(Deallocate);

// Forward declarations for all Val and Expr types

#define M(e) class e;
DISPATCH_FOR_ALL_VALS(M);
DISPATCH_FOR_ALL_EXPRS(M);
#undef M

namespace kir {

#define M(e) class e;
DISPATCH_FOR_ALL_KIR_VALS(M)
DISPATCH_FOR_ALL_KIR_EXPRS(M)
#undef M

} // namespace kir

namespace hir {

#define M(e) class e;
DISPATCH_FOR_ALL_HIR_VALS(M)
DISPATCH_FOR_ALL_HIR_EXPRS(M)
#undef M

} // namespace hir

namespace assoc_comm {
class FlattenedAssocCommOp;
} // namespace assoc_comm

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

#define M(e) virtual void handle(const e* stmt);
  M(Val);
  DISPATCH_FOR_ALL_VALS(M)
  DISPATCH_FOR_ALL_EXPRS(M)
  M(assoc_comm::FlattenedAssocCommOp);
#undef M
#define M(e) virtual void handle(const kir::e* stmt);
  DISPATCH_FOR_ALL_KIR_VALS(M)
  DISPATCH_FOR_ALL_KIR_EXPRS(M)
#undef M
#define M(e) virtual void handle(const hir::e* stmt);
  DISPATCH_FOR_ALL_HIR_VALS(M)
  DISPATCH_FOR_ALL_HIR_EXPRS(M)
#undef M
};

class NVF_API OptOutDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(Statement*);

 public:
  // Hierarchal dispatch functions for handle
  virtual void dispatch(Statement*);
  virtual void dispatch(Expr*);
  virtual void dispatch(Val*);

#define M(e) virtual void handle(e* stmt);
  M(Val);
  DISPATCH_FOR_ALL_VALS(M)
  DISPATCH_FOR_ALL_EXPRS(M)
  M(assoc_comm::FlattenedAssocCommOp);
#undef M
#define M(e) virtual void handle(kir::e* stmt);
  DISPATCH_FOR_ALL_KIR_VALS(M)
  DISPATCH_FOR_ALL_KIR_EXPRS(M)
#undef M
#define M(e) virtual void handle(hir::e* stmt);
  DISPATCH_FOR_ALL_HIR_VALS(M)
  DISPATCH_FOR_ALL_HIR_EXPRS(M)
#undef M
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
class NVF_API OptOutMutator : public PolymorphicBase {
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

#define M(e) virtual void mutate(e* stmt);
  DISPATCH_FOR_ALL_VALS(M)
#undef M
#define M(e) virtual void mutate(kir::e* stmt);
  DISPATCH_FOR_ALL_KIR_VALS(M)
#undef M

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
