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
#include <mma_type.h>
#include <parallel_type_bitmap.h>
#include <tma.h>
#include <type.h>
#include <utils.h>
#include <visibility.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
class Asm;
class BlockSync;
class GridSync;
class FenceAsyncProxy;
class WgMmaFence;
class SetMaxNReg;
class Continue;
class Return;
class MBarrierInit;
class MBarrierInvalidate;
class MBarrierArrive;
class MBarrierArriveExpectTx;
class MBarrierWait;
class MBarrierWaitParity;
class BlockSerializeWait;
class BlockSerializeRelease;
class AsyncWait;
class AsyncCommit;
class InitMagicZero;
class UpdateMagicZero;
class IfThenElse;
class GridReduction;
class GroupedGridReduction;
class GridBroadcast;
class GridWelford;
class GroupedGridWelford;
class AllocateFusedReduction;
class RNGOp;

// Expr container

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
        ptype_ == PredicateType::Misaligned ||
        ptype_ == PredicateType::ReductionWrite ||
        ptype_ == PredicateType::ElectSync);
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
  TensorIndex(
      IrBuilderPasskey,
      const TensorView* view,
      Val* index,
      DataType dtype = DataType::Null);

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

// In theory, we should just put this struct into class Asm, but unfortunately,
// due to compiler bug, we can not do that:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165
struct AsmOptions {
  bool volatile_ = false;
  bool memory = false;
  std::unordered_set<int64_t> readable_outputs = {};
  std::unordered_set<int64_t> immediate_inputs = {};
};

class Asm final : public Expr {
 public:
  using Options = AsmOptions;

  using Expr::Expr;

  explicit Asm(
      IrBuilderPasskey passkey,
      const std::string& code,
      const std::vector<Val*>& outputs,
      const std::vector<Val*>& inputs,
      const Options& options = Options());

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Asm";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  const std::string& code() const {
    return attribute<std::string>(0);
  }

  // The name of the utility function that we want to wrap the inline PTX code
  // in. If this is empty, then the inline PTX code will be emitted directly
  // into the kernel.
  std::string utility() const;

  // The signature of the utility function that we want to wrap the inline PTX
  // code in. Something like "void my_utility(int*, int*, int*)". This is
  // used to determine if the utility function has already been generated when
  // we convert Kernel IR to CUDA C++ code.
  std::string signature() const;

  const Options& options() const {
    return attribute<Options>(1);
  }

  Options& options() {
    return attribute<Options>(1);
  }

  bool volatile_() const {
    return options().volatile_;
  }

  bool& volatile_() {
    return options().volatile_;
  }

  bool memory() const {
    return options().memory;
  }

  bool& memory() {
    return options().memory;
  }

  bool hasBooleanInput() const {
    for (auto input : inputs()) {
      if (input->dtype() == DataType::Bool) {
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<std::string, Val*>> constraintsAndOutputs() const;
  std::vector<std::pair<std::string, Val*>> constraintsAndInputs() const;

  std::string parameters() const;
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
  //! param zero_init Should this memory be zero-initialized?
  //! param resets_to_zero Will this memory be set to zero upon completion of
  //!   this kernel?
  //! param alias Is this an alias of previously-allocated memory
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      std::vector<Val*> shape = {},
      bool zero_init = false,
      bool resets_to_zero = false,
      Allocate* alias = nullptr);

  //! Allocation of a non-dimensional buffer
  //!
  //! param size Size of allocation
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      Val* size,
      bool zero_init = false,
      bool resets_to_zero = false);

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
    result.reserve(attributes().size() - 8);
    for (auto i = attributes().begin() + 8; i != attributes().end(); ++i) {
      result.emplace_back((*i)->as<Val>());
    }
    return result;
  }

  //! Does this allocation require its memory to be initialized to zero before
  //! this kernel is launched? If this is true, then an additional memset
  //! kernel might be launched before the current Fusion kernel is launched in
  //! order to guarantee that this buffer is filled with zeroes (see
  //! resetsToZero() below).
  bool zeroInit() const {
    return attribute<bool>(2);
  }

  //! Is this buffer guaranteed to be reset to all zero values at the end of
  //! this kernel? This is used to avoid an additional memset kernel launch for
  //! buffers that require zeroed memory (see zeroInit() above).
  //!
  //! A common use case for zeroInit() allocations is semaphore buffers that
  //! hold counters starting at zero. Typically, each participating thread would
  //! increment the counter and the last thread would leave the counter in a
  //! non-zeroed state. The next time that kernel is run, it can no longer
  //! re-use the non-zero semaphore buffer, so KernelExecutor will launch
  //! at::zeroes to allocate a new buffer, resulting in a memset kernel launch.
  //!
  //! Instead, if the last thread resets the counter to zero, then the buffer
  //! can be re-used, and at::zeroes need only be run at the first kernel
  //! launch. If resetsToZero() is true, then KernelExecutor will use
  //! contigZeroedTensor() and releaseZeroedMemory() from global_allocator.h to
  //! reuse zeroed memory avoiding the additional kernel launch.
  //!
  //! Whenever possible, we should try to guarantee that resetsToZero() is true
  //! if zeroInit() is true by modifying our code to clean up global counters,
  //! because the latency penalty of an additional kernel launch should be
  //! greater than that required to reset this memory at the end of the fusion.
  //! The exception is when a kernel is launched only a single time, in which
  //! case resetting the memory is unnecessary, but we expect that kernels will
  //! instead be launched many times.
  bool resetsToZero() const {
    return attribute<bool>(3);
  }

  // This alias tracks the next Allocate node in a linked chain of aliases
  // If the alias is nullptr, then the Allocate node uses memory in the kernel
  const Allocate* alias() const {
    return dynamic_cast<const Allocate*>(attribute(4));
  }

  // This function can only be used for shared memory or tensor memory.
  //
  // For shared memory, this function sets the address of a shared memory
  // allocation within the dynamic shared memory array. The addr argument should
  // be a scalar expression describing an aligned address in bytes.
  //
  // For tensor memory, this function sets the address of a tensor memory
  // "region" in the tensor memory. Each tensor memory "region" is a piece of
  // tensor memory allocated by a single tcgen05.alloc, see note [Tensor Memory
  // Allocation] for detailed description. Note that this address may not be the
  // address of a TensorView, because each region may contain multiple
  // TensorViews. This address must be a uint32 scalar, as described in the PTX
  // documentation:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-addressing
  void setAddress(Val* addr) {
    NVF_CHECK(
        memoryType() == MemoryType::Shared ||
            memoryType() == MemoryType::Tensor,
        "Allocation address may only be set for shared/tensor memory allocations. Memory type is ",
        memoryType());
    NVF_CHECK(
        address() == nullptr,
        "Attempted to set address twice for allocation ",
        toString());
    attributes_[5] = addr;
  }

  // Set the lane offset of a TensorView in a tensor memory "region". See note
  // [Tensor Memory Allocation] for more detail.
  void setLaneOffset(Val* lane_offset) {
    NVF_CHECK(
        memoryType() == MemoryType::Tensor,
        "Lane offset may only be set for tensor memory allocations. Memory type is ",
        memoryType());
    NVF_CHECK(
        laneOffset() == nullptr,
        "Attempted to set lane offset twice for allocation ",
        toString());
    attributes_[6] = lane_offset;
  }

  // Set the column offset of a TensorView in a tensor memory "region". See note
  // [Tensor Memory Allocation] for more detail.
  void setColOffset(Val* col_offset) {
    NVF_CHECK(
        memoryType() == MemoryType::Tensor,
        "Column offset may only be set for tensor memory allocations. Memory type is ",
        memoryType());
    NVF_CHECK(
        colOffset() == nullptr,
        "Attempted to set column offset twice for allocation ",
        toString());
    attributes_[7] = col_offset;
  }

  // This is an integer scalar describing the byte address within the dynamic
  // shared memory array for a shared memory allocation. For memory types other
  // than Shared, or before allocation, this function might return nullptr.
  Val* address() const {
    NVF_CHECK(
        memoryType() == MemoryType::Shared ||
            memoryType() == MemoryType::Tensor,
        "Allocation address may only be set for shared memory allocations. Memory type is ",
        memoryType());
    return attributeVal(5);
  }

  Val* laneOffset() const {
    NVF_CHECK(
        memoryType() == MemoryType::Tensor,
        "Lane offset may only be set for tensor memory allocations. Memory type is ",
        memoryType());
    return attributeVal(6);
  }

  Val* colOffset() const {
    NVF_CHECK(
        memoryType() == MemoryType::Tensor,
        "Column offset may only be set for tensor memory allocations. Memory type is ",
        memoryType());
    return attributeVal(7);
  }
};

// Allocate tensor memory tcgen05.alloc
class AllocTMem final : public Expr {
 public:
  using Expr::Expr;
  AllocTMem(IrBuilderPasskey passkey, Val* address, Val* num_columns);

  const char* getOpString() const override {
    return "AllocTMem";
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* address() const {
    return output(0);
  }

  Val* numColumns() const {
    return input(0);
  }
};

// Sync represents __syncthreads barrier for block level coordination.
//
// TODO(kir): change name to SyncThreads as we could have other barriers.
//
class BlockSync final : public Expr {
 public:
  using Expr::Expr;

  explicit BlockSync(
      IrBuilderPasskey passkey,
      bool war_sync = false,
      std::optional<bool> optional_compute_or_load_sync = std::nullopt);

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

  std::optional<bool> warpSpecializedState() const {
    return attribute<std::optional<bool>>(1);
  }

  bool isComputeWarpSync() const {
    return attribute<std::optional<bool>>(1).value_or(false);
  }

  bool isAsyncWarpSync() const {
    auto optional_compute_or_load_sync = attribute<std::optional<bool>>(1);
    return optional_compute_or_load_sync.has_value() &&
        !optional_compute_or_load_sync.value();
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

// PTX: fence.proxy.async
class FenceAsyncProxy final : public Expr {
 public:
  using Expr::Expr;

  explicit FenceAsyncProxy(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "FenceAsyncProxy";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// PTX: wgmma.fence.sync.aligned
class WgMmaFence final : public Expr {
 public:
  using Expr::Expr;

  explicit WgMmaFence(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "WgMmaFence";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// PTX: setmaxnreg.inc.sync.aligned.u32 and setmaxnreg.dec.sync.aligned.u32
class SetMaxNReg final : public Expr {
 public:
  using Expr::Expr;

  explicit SetMaxNReg(
      IrBuilderPasskey passkey,
      Val* number_of_registers,
      bool increase_registers);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return (increaseRegisters()) ? "IncSetMaxNReg" : "DecSetMaxNReg";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  bool increaseRegisters() const {
    return attribute<bool>(0);
  }

  Val* numberOfRegisters() const {
    return input(0);
  }
};

class Continue final : public Expr {
 public:
  using Expr::Expr;

  explicit Continue(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Continue";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

class Return final : public Expr {
 public:
  using Expr::Expr;

  explicit Return(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Return";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
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
    if (!outputs().empty()) {
      return output(0);
    }
    return nullptr;
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
    if (!outputs().empty()) {
      return output(0);
    }
    return nullptr;
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

class MBarrierWaitParity final : public Expr {
 public:
  using Expr::Expr;
  explicit MBarrierWaitParity(
      IrBuilderPasskey passkey,
      Val* mbarrier,
      Val* parity);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MBarrierWaitParity";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* mbarrier() const {
    return input(0);
  }

  Val* parity() const {
    return input(1);
  }
};

// For all but first block in each reduction segment, first thread waits for
// sync flag to indicate it is our turn to proceed (sync flag is incremented by
// BlockSerializeRelease). Then block sync. This has the effect of
// serializing blocks in each reduction segment. This is a block syncing
// operation.
class BlockSerializeWait final : public Expr {
 public:
  using Expr::Expr;

  explicit BlockSerializeWait(
      IrBuilderPasskey passkey,
      ParallelTypeBitmap sync_dims,
      Val* sync_buffer);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "BlockSerializeWait";
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

// This first performs a block sync. For all but last block in the reduction
// segment, first thread then writes the next segment ID to the sync flag. When
// used with BlockSerializeWait, this has the effect of serializing blocks in
// order each reduction segment.
class BlockSerializeRelease final : public Expr {
 public:
  using Expr::Expr;

  explicit BlockSerializeRelease(
      IrBuilderPasskey passkey,
      ParallelTypeBitmap sync_dims,
      Val* sync_buffer);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "BlockSerializeRelease";
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

// AsyncWait represents wait intrinsics for cp.async, cp.async.bulk and
// wgmma.mma_async
class AsyncWait final : public Expr {
 public:
  using Expr::Expr;

  explicit AsyncWait(
      IrBuilderPasskey passkey,
      AsyncOpType async_op_type,
      int64_t keep_stages = 0);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "AsyncWait";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  const char* ptx() const;
  bool memory() const;

  AsyncOpType asyncOpType() const {
    return attribute<AsyncOpType>(0);
  }

  //! Returns the remaining number of stages that are not synchronized
  //!  after this op.
  int64_t keepStages() const {
    return attribute<int64_t>(1);
  }
};

// AsyncCommit represents commit intrinsics for cp.async
//  A commit intrinsic communicates delimiter of transaction groups
// to the async load hardware. Example usage see [Cicular buffer].
class AsyncCommit final : public Expr {
 public:
  using Expr::Expr;

  explicit AsyncCommit(IrBuilderPasskey passkey, AsyncOpType async_op_type);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "AsyncCommit";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  const char* ptx() const;

  //! Returns if the corresponding PTX needs a `:memory` in the end, this value
  //! will be used to set AsmOptions::memory when lowering to inline PTX.
  bool memory() const;

  AsyncOpType asyncOpType() const {
    return attribute<AsyncOpType>(0);
  }
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
//! This node provides KernelExecutor the information it needs to allocate the
//! reduction and sync buffers.
class GridReduction final : public ReductionOp {
  static constexpr int num_reduction_op_attr = 4;

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
      bool is_allreduce = false,
      TensorIndex* serial_reduction_tensor = nullptr);

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

  TensorIndex* serialReductionTensor() const {
    return dynamic_cast<TensorIndex*>(attributeVal(num_reduction_op_attr + 5));
  }

  bool isSerial() const {
    return serialReductionTensor() != nullptr;
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
    for (auto i : arange(offset, offset + size)) {
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
//! This node provides KernelExecutor the information it needs to allocate the
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
//! This node provides KernelExecutor the information it needs to allocate the
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
    for (auto i : arange(size)) {
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
  int64_t getSmemBufferSize(int64_t bdimx, int64_t bdimy, int64_t bdimz) const;
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
      MmaInputSmemSwizzle swizzle,
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

  const MmaInputSmemSwizzle& swizzle() const {
    return attribute<MmaInputSmemSwizzle>(3);
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

class RNGOp : public Expr {
 public:
  using Expr::Expr;

  RNGOp(
      IrBuilderPasskey,
      Val* out,
      Val* rng_result,
      Val* rng_component,
      DataType dtype,
      RNGOpType rng_type,
      // range high and low, or avg and std dev
      std::vector<Val*> parameters = {});

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  const char* getOpString() const override {
    return "RNGOp";
  }

  RNGOpType getRNGOpType() const {
    return attribute<RNGOpType>(0);
  }

  DataType dtype() const {
    return attribute<DataType>(1);
  }
};

} // namespace kir
} // namespace nvfuser
