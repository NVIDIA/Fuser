// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>

#include <instrumentation.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <logical_domain_map.h>

#include <unordered_set>
#include <vector>

namespace nvfuser {

// TODO: Replace with mutator as IndexLowering is replacing expr's with
// versions that are doing indexing
class IndexLowering : private OptOutConstDispatch {
 public:
  static std::vector<Expr*> getIndexedExprs(std::vector<Expr*> incoming_exprs);

 private:
  IndexLowering() = default;

  void pushBack(Expr*);

  // Return the most recently inserted
  //  expression in the current active
  //  scope or global scope.
  Expr* back() const;

  // Insert an expression before the current top-level expression.
  void insertAtTopLevel(Expr* expr);

  void handle(const FullOp*) final;
  void handle(const IotaOp*) final;
  void handle(const EyeOp*) final;
  void handle(const ViewAsScalar*) final;
  void handle(const UnaryOp*) final;
  void handle(const BinaryOp*) final;
  void handle(const TernaryOp*) final;
  void handle(const ArrayConstruct*) final;
  void handle(const StructConstruct*) final;
  void handle(const GetAttr*) final;
  void handle(const GetItem*) final;
  void handle(const GetMetaData*) final;
  void handle(const TensorConstruct*) final;
  void handle(const SelectOp*) final;
  void handle(const IndexSelectOp*) final;
  void handle(const GatherOp*) final;
  void handle(const ScatterOp*) final;
  void handle(const RNGOp*) final;
  void handle(const ReductionOp*) final;
  void handle(const GroupedReductionOp*) final;
  void handle(const WelfordOp*) final;
  void handle(const GroupedWelfordOp*) final;
  void handle(const LoadStoreOp*) final;
  void handle(const MmaOp*) final;
  void handle(const BroadcastOp*) final;
  void handle(const PadOp*) final;
  void handle(const SliceOp*) final;
  void handle(const CatOp*) final;

  void handle(const kir::Asm*) final;
  void handle(const ForLoop*) final;
  void handle(const kir::IfThenElse*) final;
  void handle(const kir::Allocate*) final;
  void handle(const kir::AllocTMem*) final;
  void handle(const kir::BlockSync*) final;
  void handle(const kir::GridSync*) final;
  void handle(const kir::FenceAsyncProxy*) final;
  void handle(const kir::WgMmaFence*) final;
  void handle(const kir::SetMaxNReg*) final;
  void handle(const kir::Continue*) final;
  void handle(const kir::Return*) final;
  void handle(const kir::MBarrierInit*) final;
  void handle(const kir::MBarrierInvalidate*) final;
  void handle(const kir::MBarrierArrive*) final;
  void handle(const kir::MBarrierArriveExpectTx*) final;
  void handle(const kir::MBarrierWait*) final;
  void handle(const kir::MBarrierWaitParity*) final;
  void handle(const kir::AsyncWait*) final;
  void handle(const kir::AsyncCommit*) final;
  void handle(const kir::BlockSerializeWait*) final;
  void handle(const kir::BlockSerializeRelease*) final;

  void generate(const std::vector<Expr*>& exprs);

  // Get the loop in which the currently visiting expr is a rotated expr.
  const std::unordered_set<ForLoop*>& getRotatedLoop() const {
    return rotated_loop_;
  }

  // lower index for producer. The `override_index` is a mapping `id->index`,
  // where `id` must be an IterDomain in the rFactor domain of the producer.
  // This is can used to manually set the index for the given rFactor ID.
  // Currently, this `override_index` is only used by indexing ops like
  // select/index_select.
  // The argument `generate_pointer` specifies whether to generate pointer for
  // the tensor. If global tensor, then generate T1.data. If shared memory
  // tensor, then use `cvta` ptx to convert shared memory address to unsigned
  // int for indexing. Search `toSmem` in the codebase for additional
  // information. This argument is effective only if the indexed tensor is a
  // shared memory or global tensor. On other memory type, this argument will
  // cause an error.
  Val* lowerSrcIndex(
      Val* val,
      Val* dst,
      const std::unordered_map<IterDomain*, Val*>& override_index = {},
      bool generate_pointer = false,
      DataType as_type = DataType::Null) const;

  Val* lowerDstIndex(
      Val* dst,
      const std::unordered_map<int, Val*>& override_index = {},
      bool generate_pointer = false,
      DataType as_type = DataType::Null) const;

  void handleCpAsyncBulkLoad(const LoadStoreOp* ldst);
  void handleCpAsyncBulkStore(const LoadStoreOp* ldst);

  void handleBlockReduction(const ReductionOp* rop, Val* out, Val* in);
  void handleGridReduction(const ReductionOp* rop, Val* out, Val* in);
  //! Called by handleGridReduction, this returns true if rop is lowered as a
  //! serial grid reduction.
  void handleSerialGridReduction(const ReductionOp* rop, Val* out, Val* in);

  void handleBlockReduction(
      const GroupedReductionOp* rop,
      const std::vector<Val*>& outputs,
      const std::vector<Val*>& inputs);
  void handleGridReduction(
      const GroupedReductionOp* rop,
      const std::vector<Val*>& outputs,
      const std::vector<Val*>& inputs);

  void handleGridWelford(WelfordOp* new_wop);

  void handleGroupedBlockWelford(
      const GroupedWelfordOp* wop,
      const std::vector<WelfordTriplet>& output_vals,
      const std::vector<WelfordTriplet>& input_vals,
      const std::vector<WelfordTriplet>& init_vals);
  void handleGroupedGridWelford(
      const GroupedWelfordOp* wop,
      const std::vector<WelfordTriplet>& output_vals,
      const std::vector<WelfordTriplet>& input_vals,
      const std::vector<WelfordTriplet>& init_vals);

  // Allocate a unique buffer for grid reductions and broadcast. A
  // buffer is uniquely allocated for each output tensor of an
  // expression.
  kir::Allocate* allocateUniqueBuffer(
      Val* buffer_size,
      DataType dtype,
      bool zero_init,
      TensorView* out_tv,
      std::unordered_map<TensorView*, kir::Allocate*>& alloc_map);

  std::vector<kir::Allocate*> allocateWelfordWorkBuffer(
      const std::vector<WelfordTriplet>& triplets,
      WelfordTriplet::ValName name,
      Val* buffer_size);

  // Allocate a fused reduction object uniquely for a given
  // TensorView. Parameter expr is the expression corresponding to the
  // fused reduction.
  void allocateUniqueFusedReduction(Expr* expr, TensorView* out_tv);

 private:
  std::vector<Expr*> lowered_exprs_;

  // This is a slight work around as scope has a couple definitions, we have the
  // Scope that's in ForLoop/IfThenElse which is really just a wrapper around
  // std::vector<Expr*> and then we have the actual ForLoop/IfThenElse. We want
  // to be able to carry both around because when we push back to a scope it
  // could be either the body or else body of the IfThenElse. However, we want
  // to understand the nesting of IfThenElse/ForLoop nodes.
  Scope* active_scope_ = nullptr;

  // Track for loops to send to indexing. Similar to what's done in
  // kir::IrVisitor
  std::vector<ForLoop*> for_loops_;

  // Keep track of the loop in which the currently visiting expr is a rotated.
  std::unordered_set<ForLoop*> rotated_loop_;

  // Maps to keep track of allocated buffers and objects that must be
  // allocated only once
  std::unordered_map<TensorView*, kir::Allocate*> sync_buffer_map_;
  std::unordered_map<TensorView*, kir::Allocate*> work_buffer_map_;
  std::unordered_map<TensorView*, kir::AllocateFusedReduction*>
      fused_reduction_map_;
};

} // namespace nvfuser
