// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/base_nodes.h>
#include <parallel_type_bitmap.h>
#include <tma.h>
#include <type.h>
#include <utils.h>

#include <c10/macros/Export.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

class IrBuilderPasskey;

namespace kir {
class Kernel;

// Values
class Predicate;
class TensorIndex;

// Expressions
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
class InitMagicZero;
class UpdateMagicZero;
class ForLoop;
class IfThenElse;
class GridReduction;
class GroupedGridReduction;
class GridBroadcast;
class GridWelford;
class GroupedGridWelford;
class AllocateFusedReduction;

// Expr container
class Scope;

class Predicate final : public Val {
 public:
  explicit Predicate(
      IrBuilderPasskey passkey,
      PredicateType ptype,
      const Expr* expr = nullptr,
      Val* thread_pred = nullptr);

  explicit Predicate(IrBuilderPasskey passkey, ForLoop* unrolled_loop);

  explicit Predicate(IrBuilderPasskey passkey, Val* value);

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  PredicateType predicate_type() const {
    return ptype_;
  }

  const Expr* expr() const {
    NVF_ERROR(
        ptype_ != PredicateType::Unswitch &&
        ptype_ != PredicateType::Vectorize && ptype_ != PredicateType::Manual);
    return expr_;
  }

  Val* thread_pred() const {
    NVF_ERROR(
        ptype_ == PredicateType::Inline ||
        ptype_ == PredicateType::Misaligned || ptype_ == PredicateType::Shift ||
        ptype_ == PredicateType::Padding ||
        ptype_ == PredicateType::ReductionWrite);
    return thread_pred_;
  }

  ForLoop* unrolled_loop() const {
    NVF_ERROR(ptype_ == PredicateType::Unswitch);
    return unrolled_loop_;
  }

  bool hasValue() const {
    return value_ != nullptr;
  }

  Val* value() const {
    NVF_ERROR(
        value_ != nullptr,
        "The conditional expression for this Predicate is invalid.");
    return value_;
  }

  void setValue(Val* value) {
    NVF_ERROR(value != nullptr, "The Bool expression is invalid.");
    value_ = value;
  }

  bool isConst() const final {
    return hasValue() && value_->isConst();
  }

  bool isTrivial() const {
    return isConst() && value_->value().is<bool>() &&
        value_->value().as<bool>();
  }

 private:
  PredicateType ptype_ = PredicateType::Manual;

  // For PredicateCompute::getInlinePredicate,
  // ShiftPredicateInserter::getShiftPredicate and getPaddingPredicate
  const Expr* expr_ = nullptr;

  // For PredicateCompute::getInlinePredicate
  Val* thread_pred_ = nullptr;

  // For ParallelType::Unswitch - UnswitchPredicate::get
  ForLoop* unrolled_loop_ = nullptr;

  // The Bool conditional value
  // The value is nullptr until lower_predicate pass
  Val* value_ = nullptr;
};

class TensorIndex final : public Val {
 public:
  TensorIndex(IrBuilderPasskey, const TensorView* view, Val* index);

  Val* index() const {
    return index_;
  }

  TensorView* view() const {
    NVF_ERROR(view_ != nullptr);
    return const_cast<TensorView*>(view_); // NOLINT
  }

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

 private:
  const TensorView* view_ = nullptr;
  Val* index_ = nullptr;
};

//! Allocate is a lower level Node that describes a buffer of memory that
//! is required as an intermediate within a kernel. The extent is the expression
//! of the size of the buffer that is generated from the TensorView that
//! describes the output of an operation.
class Allocate final : public Expr {
 public:
  using Expr::Expr;

  //! Allocation of a multi-dimensional buffer
  //!
  //! param shape Size of each dimension
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      std::vector<Val*> shape = {},
      bool zero_init = false,
      Allocate* alias = nullptr);

  //! Allocation of a non-dimensional buffer
  //!
  //! param size Size of allocation
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      Val* size,
      bool zero_init = false);

  const char* getOpString() const override {
    return "Allocate";
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* buffer() const {
    return attributeVal(0);
  }

  MemoryType memoryType() const {
    return attribute<MemoryType>(1);
  }

  //! Total size
  Val* size() const {
    return input(0);
  }

  //! Size of each dimension
  std::vector<Val*> shape() const {
    std::vector<Val*> result;
    result.reserve(attributes().size() - 5);
    for (auto i = attributes().begin() + 5; i != attributes().end(); ++i) {
      result.emplace_back((*i)->as<Val>());
    }
    return result;
  }

  bool zeroInit() const {
    return attribute<bool>(2);
  }

  // This alias tracks the next Allocate node in a linked chain of aliases
  // If the alias is nullptr, then the Allocate node uses memory in the kernel
  const Allocate* alias() const {
    return dynamic_cast<const Allocate*>(attribute(3));
  }

  // Set the address of a shared memory allocation within the dynamic shared
  // memory array. The addr argument should be a scalar expression describing an
  // aligned address in bytes.
  void setAddress(Val* addr) {
    NVF_CHECK(
        memoryType() == MemoryType::Shared,
        "Allocation address may only be set for shared memory allocations. Memory type is ",
        memoryType());
    NVF_CHECK(
        address() == nullptr,
        "Attempted to set address twice for allocation ",
        toString());
    attributes_[4] = addr;
  }

  // This is an integer scalar describing the byte address within the dynamic
  // shared memory array for a shared memory allocation. For memory types other
  // than Shared, or before allocation, this function might return nullptr.
  Val* address() const {
    return attributeVal(4);
  }
};

// Sync represents __syncthreads barrier for block level coordination.
//
// TODO(kir): change name to SyncThreads as we could have other barriers.
//
class BlockSync final : public Expr {
 public:
  using Expr::Expr;

  explicit BlockSync(IrBuilderPasskey passkey, bool war_sync = false);

  const char* getOpString() const override {
    return "BlockSync";
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  // TODO: war_sync_ is only used for testing/validation purposes.
  bool isWarHazardSync() const {
    return attribute<bool>(0);
  }
};

// Synchronize all blocks in device, implies cooperative group launch is
// required.
class GridSync final : public Expr {
 public:
  using Expr::Expr;

  explicit GridSync(
      IrBuilderPasskey passkey,
      ParallelTypeBitmap sync_dims,
      Val* sync_buffer);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GridSync";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  ParallelTypeBitmap syncDims() const {
    return attribute<ParallelTypeBitmap>(0);
  }

  Val* syncBuffer() const {
    return attributeVal(1);
  }
};

class MBarrierInit final : public Expr {
 public:
  using Expr::Expr;
  explicit MBarrierInit(
      IrBuilderPasskey passkey,
      Val* mbarrier,
      Val* thread_count);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MBarrierInit";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* mbarrier() const {
    return input(0);
  }

  Val* threadCount() const {
    return input(1);
  }
};

class MBarrierInvalidate final : public Expr {
 public:
  using Expr::Expr;
  explicit MBarrierInvalidate(IrBuilderPasskey passkey, Val* mbarrier);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MBarrierInvalidate";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* mbarrier() const {
    return input(0);
  }
};

class MBarrierArrive final : public Expr {
 public:
  using Expr::Expr;
  explicit MBarrierArrive(IrBuilderPasskey passkey, Val* state, Val* mbarrier);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MBarrierArrive";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* state() const {
    return output(0);
  }

  Val* mbarrier() const {
    return input(0);
  }
};

// IR node for: mbarrier.arrive.expect_tx
// This is usually used to specify the number of bytes that will be
// transferred for cp.async and cp.async.bulk, so that future mbarrier.wait
// can wait for the completion of the transfer.
class MBarrierArriveExpectTx final : public Expr {
 public:
  using Expr::Expr;
  explicit MBarrierArriveExpectTx(
      IrBuilderPasskey passkey,
      Val* state,
      Val* mbarrier,
      Val* tx_count);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MBarrierArriveExpectTx";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* state() const {
    return output(0);
  }

  Val* mbarrier() const {
    return input(0);
  }

  Val* txCount() const {
    return input(1);
  }
};

class MBarrierWait final : public Expr {
 public:
  using Expr::Expr;
  explicit MBarrierWait(IrBuilderPasskey passkey, Val* mbarrier, Val* state);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MBarrierWait";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* mbarrier() const {
    return input(0);
  }

  Val* state() const {
    return input(1);
  }
};

// CpAsyncWait represents wait intrinsics for cp.async
class CpAsyncWait final : public Expr {
 public:
  using Expr::Expr;

  explicit CpAsyncWait(IrBuilderPasskey passkey, int64_t keep_stages = 0);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "CpAsyncWait";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! Returns the remaining number of stages that are not synchronized
  //!  after this op.
  int64_t keepStages() const {
    return attribute<int64_t>(0);
  }
};

// CpAsyncCommit represents commit intrinsics for cp.async
//  A commit intrinsic communicates delimiter of transaction groups
// to the async load hardware. Example usage see [Cicular buffer].
class CpAsyncCommit final : public Expr {
 public:
  using Expr::Expr;

  explicit CpAsyncCommit(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "CpAsyncCommit";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

class CpAsyncBulkS2GWait final : public Expr {
 public:
  using Expr::Expr;

  explicit CpAsyncBulkS2GWait(
      IrBuilderPasskey passkey,
      int64_t keep_stages = 0);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "CpAsyncBulkS2GWait";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  int64_t keepStages() const {
    return attribute<int64_t>(0);
  }
};

class CpAsyncBulkS2GCommit final : public Expr {
 public:
  using Expr::Expr;

  explicit CpAsyncBulkS2GCommit(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "CpAsyncBulkS2GCommit";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// Simply prints "DEFINE_MAGIC_ZERO" in the code in accordance with magic_zero
// in helpers.cu
class InitMagicZero final : public Expr {
 public:
  using Expr::Expr;

  explicit InitMagicZero(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "InitMagicZero";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// Simply prints "UPDATE_MAGIC_ZERO" in the code in accordance with magic_zero
// in helpers.cu
class UpdateMagicZero final : public Expr {
 public:
  using Expr::Expr;

  explicit UpdateMagicZero(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "UpdateMagicZero";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// TODO(kir): promote to IR node
class Scope {
 public:
  explicit Scope(Expr* owner) : owner_(owner) {}

  std::string toString(int indent_size = 0) const;

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

  bool empty() const {
    return exprs_.empty();
  }

  auto size() const {
    return exprs_.size();
  }

  auto& at(size_t i) {
    return exprs_.at(i);
  }

  auto& at(size_t i) const {
    return exprs_.at(i);
  }

  auto& operator[](size_t i) {
    return at(i);
  }

  auto& operator[](size_t i) const {
    return at(i);
  }

  // Insert expr before expression at pos
  std::vector<Expr*>::iterator insert(size_t pos, Expr* expr);

  // Insert expr before ref
  std::vector<Expr*>::iterator insert_before(Expr* ref, Expr* expr);

  // Insert expr after ref
  std::vector<Expr*>::iterator insert_after(Expr* ref, Expr* expr);

  void push_back(Expr* e) {
    exprs_.push_back(e);
  }

  // Erase expr at pos
  void erase(size_t pos);

  // Erase expr ref
  void erase(Expr* ref);

  bool contains(Expr* expr) const;

  void clear();

  Expr* owner() const {
    return owner_;
  }

  bool operator==(const Scope&) const {
    NVF_ERROR(false, "Should not reach here");
  }

  // Insert expr before pos
  std::vector<Expr*>::iterator insert(
      std::vector<Expr*>::const_iterator pos,
      Expr* expr);

 private:
  // Erase expr at pos
  void erase(std::vector<Expr*>::const_iterator pos);

 private:
  std::vector<Expr*> exprs_;

  //! Owner exprssion of this scope, e.g., IfThenElse
  Expr* owner_ = nullptr;
};

//! ForLoop provides scoping around an int iterator from 0 to range. Exprs
//! placed in its body are considered inside the scope of the for loop. In the
//! future the implementation should look quite different so that we can do
//! proper dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
//! ForLoop may represent a part of an iteration domain representend
//! by iter_domain_. In that case, the loop extent field, extent_, may
//! be smaller than the extent of iter_domain_.
class ForLoop final : public Expr {
 public:
  using Expr::Expr;

  //! By default, start and stop are the same as those of iter_domain.
  //! Step is one by default.
  //!
  //! TODO: cleaner way to set options?
  ForLoop(
      IrBuilderPasskey passkey,
      IterDomain* iter_domain,
      Val* index,
      Val* start,
      Val* stop,
      Val* step,
      bool vectorize,
      Val* vectorize_shift,
      bool unroll_required,
      DoubleBufferLoopStage double_buffer_loop_stage);

  ForLoop(
      IrBuilderPasskey passkey,
      IterDomain* iter_domain,
      Val* index,
      DoubleBufferLoopStage double_buffer_loop_stage);

  ForLoop(IrBuilderPasskey passkey, IterDomain* iter_domain);

  ForLoop(IrBuilderPasskey passkey, const ForLoop* other);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ForLoop";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* index() const {
    return input(0);
  }

  Val* indexOrStartIfTrivial() const {
    return isTrivial() ? start() : index();
  }

  Val* start() const;

  Val* stop() const;

  Val* step() const;

  Val* simplifiedStop() const;

  // [pre | vectorize | post] <= inner-most, merged root domain
  // shift_ is applied to vectorize and post sections.
  Val* vectorize_shift() const {
    return attributeVal(4);
  }

  IterDomain* iter_domain() const {
    return input(1)->as<IterDomain>();
  }

  // TODO: Return pointer instead of reference to be more consistent
  Scope& body() {
    return attribute<Scope>(7);
  }

  const Scope& body() const {
    return attribute<Scope>(7);
  }

  bool empty() const {
    return body().empty();
  }

  // vectorize is true when the for-loop contains a vectorize set
  // the flag is used to omit the for-loop from the kernel
  bool vectorize() const {
    return attribute<bool>(3);
  }

  //! True if unrolled (i.e., "#pragma unroll" is attached)
  bool isUnrolled() const;

  //! True if unroll is required for avoiding stack allocation
  bool isUnrollRequired() const {
    return attribute<bool>(5);
  }

  //! Set unrolling required
  void requireUnroll() {
    attribute<bool>(5) = true;
  }

  //! True if no actual for-loop is materialized
  bool isTrivial() const;

  //! True if loop is grouped reduction/welford
  bool isGroup() const;

  //! Returns the stage of a double buffered iterdomain
  //!  that this for loop materializes.
  auto doubleBufferLoopStage() const {
    return attribute<DoubleBufferLoopStage>(6);
  }

 private:
  //! Returns if a loop could be unrolled.
  bool isUnrollable() const;

  //! Not storing this as an attribute because this is only a cache for
  //! simplifiedStop. We are not interested in keeping this across clone/serde,
  //! etc.
  mutable Val* simplified_stop_ = nullptr;
};

//! IfThenElse provides scoping for an boolean operator. Exprs placed in its
//! body are considered inside the scope of the if statement. In the future the
//! implementation should look quite different so that we can do proper
//! dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
class IfThenElse final : public Expr {
 public:
  using Expr::Expr;

  explicit IfThenElse(IrBuilderPasskey passkey, Predicate* cond);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "IfThenElse";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Scope& thenBody() {
    return attribute<Scope>(0);
  }
  const Scope& thenBody() const {
    return attribute<Scope>(0);
  }

  Scope& elseBody() {
    return attribute<Scope>(1);
  }

  const Scope& elseBody() const {
    return attribute<Scope>(1);
  }

  bool hasElse() const {
    return !elseBody().empty();
  }

  bool empty() const {
    return thenBody().empty() && elseBody().empty();
  }
};

//! Grid reduction operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! reduction and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! reduction and sync buffers.
class GridReduction final : public ReductionOp {
  static constexpr int num_reduction_op_attr = 3;

 public:
  using ReductionOp::ReductionOp;

  GridReduction(
      IrBuilderPasskey passkey,
      BinaryOpType reduction_op_type,
      Val* init,
      Val* out,
      Val* in,
      Allocate* reduction_buffer,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GridReduction";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Allocate* reduction_buffer() const {
    return attribute(num_reduction_op_attr)->as<Allocate>();
  }

  Allocate* sync_buffer() const {
    return attribute(num_reduction_op_attr + 1)->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(num_reduction_op_attr + 2);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(num_reduction_op_attr + 3);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute<ParallelTypeBitmap>(num_reduction_op_attr + 4);
  }

  ParallelTypeBitmap& threadPredicate() {
    return attribute<ParallelTypeBitmap>(num_reduction_op_attr + 4);
  }

  GridReduction* withThreadPredicate(
      const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GridReduction>();
    result->threadPredicate() = thread_predicate;
    return result;
  }
};

class GroupedGridReduction final : public GroupedReductionOp {
 public:
  using GroupedReductionOp::GroupedReductionOp;

  GroupedGridReduction(
      IrBuilderPasskey passkey,
      std::vector<BinaryOpType> reduction_op_type,
      std::vector<Val*> init,
      std::vector<Val*> out,
      std::vector<Val*> in,
      std::vector<Allocate*> reduction_buffers,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances,
      Val* buffer_stride,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  // number of attributes in the parent class
  size_t numGroupedReductionOpAttr() const {
    return 2 + outputs().size();
  }

  const char* getOpString() const override {
    return "GroupedGridReduction";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::vector<Allocate*> reduction_buffers() const {
    auto offset = numGroupedReductionOpAttr() + 5;
    auto size = outputs().size();
    std::vector<Allocate*> result;
    result.reserve(size);
    for (auto i : c10::irange(offset, offset + size)) {
      result.emplace_back(attribute(i)->as<Allocate>());
    }
    return result;
  }

  Allocate* reduction_buffer(size_t i) const {
    return reduction_buffers().at(i);
  }

  Allocate* sync_buffer() const {
    return attribute(numGroupedReductionOpAttr())->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(numGroupedReductionOpAttr() + 1);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(numGroupedReductionOpAttr() + 2);
  }

  // Stride of reduction buffers
  Val* buffer_stride() const {
    return attributeVal(numGroupedReductionOpAttr() + 3);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute<ParallelTypeBitmap>(numGroupedReductionOpAttr() + 4);
  }

  ParallelTypeBitmap& threadPredicate() {
    return attribute<ParallelTypeBitmap>(numGroupedReductionOpAttr() + 4);
  }

  GroupedGridReduction* withThreadPredicate(
      const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GroupedGridReduction>();
    result->threadPredicate() = thread_predicate;
    return result;
  }
};

//! Grid broadcast operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! broadcast and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! broadcast and sync buffers.
class GridBroadcast final : public Expr {
 public:
  using Expr::Expr;

  GridBroadcast(
      IrBuilderPasskey passkey,
      BroadcastOp* broadcast_op,
      Allocate* broadcast_buffer,
      Allocate* sync_buffer);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GridBroadcast";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  BroadcastOp* broadcast_op() const {
    return attribute(0)->as<BroadcastOp>();
  }

  Allocate* broadcast_buffer() const {
    return attribute(1)->as<Allocate>();
  }

  Allocate* sync_buffer() const {
    return attribute(2)->as<Allocate>();
  }
};

//! Grid welford operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! reduction and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! reduction and sync buffers.
//!
//! TODO: Make this a subclass of WelfordOp
class GridWelford final : public Expr {
 public:
  using Expr::Expr;

  GridWelford(
      IrBuilderPasskey passkey,
      WelfordOp* welford_op,
      Allocate* var_buffer,
      Allocate* avg_buffer,
      Allocate* n_buffer,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GridWelford";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  WelfordOp* welford_op() const {
    return attribute(0)->as<WelfordOp>();
  }

  Allocate* var_buffer() const {
    return attribute(1)->as<Allocate>();
  }

  Allocate* avg_buffer() const {
    return attribute(2)->as<Allocate>();
  }

  Allocate* N_buffer() const {
    return attribute(3)->as<Allocate>();
  }

  Allocate* sync_buffer() const {
    return attribute(4)->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(5);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(6);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute<ParallelTypeBitmap>(7);
  }
  ParallelTypeBitmap& threadPredicate() {
    return attribute<ParallelTypeBitmap>(7);
  }

  GridWelford* withThreadPredicate(const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GridWelford>();
    result->threadPredicate() = thread_predicate;
    return result;
  }
};

class GroupedGridWelford final : public GroupedWelfordOp {
 public:
  using GroupedWelfordOp::GroupedWelfordOp;

  // input, output and init vals are vectors of triplets
  GroupedGridWelford(
      IrBuilderPasskey passkey,
      std::vector<WelfordTriplet> output_vals,
      std::vector<WelfordTriplet> input_vals,
      std::vector<WelfordTriplet> init_vals,
      std::array<std::vector<Allocate*>, 3> reduction_buffers,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances,
      Val* buffer_stride,
      bool is_allreduce = false,
      bool use_outer_opt = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  size_t numGroupedWelfordOpAttr() const {
    return 1 + outputs().size();
  }

  const char* getOpString() const override {
    return "GroupedGridWelford";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::array<std::vector<Allocate*>, 3> reduction_buffers() const {
    auto offset = numGroupedWelfordOpAttr() + 5;
    auto size = outputs().size() / 3;
    std::array<std::vector<Allocate*>, 3> result;
    result[0].reserve(size);
    result[1].reserve(size);
    result[2].reserve(size);
    for (auto i : c10::irange(size)) {
      result[0].emplace_back(attribute(offset + i * 3)->as<Allocate>());
      result[1].emplace_back(attribute(offset + i * 3 + 1)->as<Allocate>());
      result[2].emplace_back(attribute(offset + i * 3 + 2)->as<Allocate>());
    }
    return result;
  }

  Allocate* sync_buffer() const {
    return attribute(numGroupedWelfordOpAttr())->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(numGroupedWelfordOpAttr() + 1);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(numGroupedWelfordOpAttr() + 2);
  }

  // Stride of reduction buffers
  Val* buffer_stride() const {
    return attributeVal(numGroupedWelfordOpAttr() + 3);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute<ParallelTypeBitmap>(numGroupedWelfordOpAttr() + 4);
  }
  ParallelTypeBitmap& threadPredicate() {
    return attribute<ParallelTypeBitmap>(numGroupedWelfordOpAttr() + 4);
  }

  GroupedGridWelford* withThreadPredicate(
      const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GroupedGridWelford>();
    result->threadPredicate() = thread_predicate;
    return result;
  }

  // True if the outer-optimized kernel should be used
  bool useOuterOpt() const {
    auto offset = numGroupedWelfordOpAttr() + 5 + outputs().size();
    return attribute<bool>(offset);
  }

  //! Return the required smem buffer size
  int getSmemBufferSize(int bdimx, int bdimy, int bdimz) const;
};

//! Represents a WelfordOp with the division by count is hoisted out
//! of an innermost loop
class VectorizedWelfordOp final : public WelfordOp {
 public:
  using WelfordOp::WelfordOp;

  VectorizedWelfordOp(
      IrBuilderPasskey,
      const WelfordTriplet& output,
      const WelfordTriplet& input,
      const WelfordTriplet& init,
      Val* count,
      Val* reciprocal_of_count,
      Val* hoisted_predicate);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "VectorizedWelfordOp";
  }

  //! New count that should be set to outN
  Val* count() const {
    return attributeVal(WelfordOp::kNumAttrs);
  }

  //! Reciprocal of count
  Val* reciprocalOfCount() const {
    return attributeVal(WelfordOp::kNumAttrs + 1);
  }

  //! Predicate of this expression hoisted out of an innermost loop
  Val* hoistedPredicate() const {
    return attributeVal(WelfordOp::kNumAttrs + 2);
  }
};

// Allocate an instance of the fused reduction class.
class AllocateFusedReduction final : public Expr {
  explicit AllocateFusedReduction(IrBuilderPasskey passkey, Expr* grid_expr);

 public:
  using Expr::Expr;

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GridReduction* grid_reduction)
      : AllocateFusedReduction(passkey, dynamic_cast<Expr*>(grid_reduction)) {}

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GridWelford* grid_welford)
      : AllocateFusedReduction(passkey, dynamic_cast<Expr*>(grid_welford)) {}

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GroupedGridReduction* grouped_grid_reduction)
      : AllocateFusedReduction(
            passkey,
            dynamic_cast<Expr*>(grouped_grid_reduction)) {}

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GroupedGridWelford* grouped_grid_welford)
      : AllocateFusedReduction(
            passkey,
            dynamic_cast<Expr*>(grouped_grid_welford)) {}

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "AllocateFusedReduction";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! GridReduction, GridWelford, GroupedGridReduction or GroupedGridWelford
  Expr* gridExpr() const {
    return attribute(0)->asExpr();
  }

  TensorIndex* out() const;

  const ParallelTypeBitmap& threadPredicate() const;
};

class GetRNGSeedAndOffsetFromHost : public Expr {
 public:
  using Expr::Expr;

  GetRNGSeedAndOffsetFromHost(
      IrBuilderPasskey,
      Val* seed_ptr,
      Val* seed_val,
      Val* first_offset_ptr,
      Val* first_offset_val,
      int64_t offsets = -1);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GetRNGSeedAndOffsetFromHost";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  const int64_t& offsets() const {
    return attribute<int64_t>(0);
  }

  int64_t& offsets() {
    return attribute<int64_t>(0);
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

// Expr for driver API cuTensorMapEncodeTiled
class EncodeTensorMapTiled : public Expr {
 public:
  using Expr::Expr;

  EncodeTensorMapTiled(
      IrBuilderPasskey,
      Val* output,
      DataType data_type,
      Val* global_address,
      Val* global_dim,
      Val* global_strides,
      Val* box_dim,
      Val* element_strides,
      tma::TensorMapInterleave interleave,
      tma::TensorMapSwizzle swizzle,
      tma::TensorMapL2Promotion l2_promotion,
      tma::TensorMapFloatOOBFill oob_fill);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "EncodeTensorMapTiled";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* globalAddress() const {
    return input(0);
  }

  Val* globalDim() const {
    return input(1);
  }

  Val* globalStrides() const {
    return input(2);
  }

  Val* boxDim() const {
    return input(3);
  }

  Val* elementStrides() const {
    return input(4);
  }

  const DataType& dataType() const {
    return attribute<DataType>(0);
  }

  const int64_t& tensorRank() const {
    return attribute<int64_t>(1);
  }

  const tma::TensorMapInterleave& interleave() const {
    return attribute<tma::TensorMapInterleave>(2);
  }

  const tma::TensorMapSwizzle& swizzle() const {
    return attribute<tma::TensorMapSwizzle>(3);
  }

  const tma::TensorMapL2Promotion& l2Promotion() const {
    return attribute<tma::TensorMapL2Promotion>(4);
  }

  const tma::TensorMapFloatOOBFill& oobFill() const {
    return attribute<tma::TensorMapFloatOOBFill>(5);
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

} // namespace kir
} // namespace nvfuser
