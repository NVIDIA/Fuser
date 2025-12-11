// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Host Irs are used to represent a host program. They need to be registered in
// a HostIrContainer. Each Ir represents a Host data or instruction.

#pragma once

#include "fusion.h"
#include "ir/base_nodes.h"
#include "ir/builder.h"
#include "multidevice/communication.h"
#include "scheduler/heuristic.h"

namespace nvfuser {
// This works around a circular dependency: compiled_kernel.h ==>
// expr_evaluator.h ==> ir/all_nodes.h ==> host_ir/ir.h ==> compiled_kernel.h
//
// ir/all_nodes.h probably shouldn't include host_ir/ir.h. The former is for
// fusion IR and the latter is for host IR.
class CompiledKernel;
} // namespace nvfuser

namespace nvfuser::hir {

// HostUnit represents a Fusion in the Host Program. In other words, it
// represents a compute graph (or a segment of a larger compute graph)
// represented by a Fusion that should be compiled and executed as a bulked
// item from the host perspective.
//
// This IR can be thought as a thin layer around the class `Fusion`, which
// furthermore inherits from `Expr` so that it is an "IR" in nvFuser IR
// semantics.
//
// This IRs fundamentally allows nested IR structures. It could potentially be
// useful in other instances than HostIrs.
//
// Its implementation is minimal, the only specifity being the moethod
// `fusion_to_execute()` that returns the fusion that the IR represents.
//
// Note: HostUnit has no I/O itself -- however the Fusion it embbeds has I/O of
// course, which are not registered in the surrounding HostIrContainer.
//
// Note: Whether HostUnit should inherit from Expr or Val is debatable. Both
// are possible, I define it as an Expr for now here but am open to change it.
class HostUnit : public Expr {
 public:
  using Expr::Expr;
  HostUnit(IrBuilderPasskey passkey, std::unique_ptr<Fusion> fusion);
  HostUnit(const HostUnit* src, IrCloner* ir_cloner);

  HostUnit(const HostUnit& other) = delete;
  HostUnit& operator=(const HostUnit& other) = delete;
  HostUnit(HostUnit&& other) = delete;
  HostUnit& operator=(HostUnit&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE
  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::HostUnit";
  }

  bool sameAs(const Statement* other) const override;

  Fusion* fusion_to_execute() const {
    return fusion_.get();
  }

 private:
  std::unique_ptr<Fusion> fusion_;
};

// PostOnStream represents the host instruction of executing a HostUnit. Its I/O
// represents in the host program the concrete I/O that will be bound at runtime
// to the Fusion's I/O for compilation and execution. At runtime, PostOnStream
// will compile and launch the kernel lowered from the HostUnit's embedded
// Fusion.
//
// Note: later PostOnStream will take a "Stream" argument
//
// Note: later PostOnStream will also be able to launch network Communications
//
// Note: later compilation and kernel launch will be separated and represented
// by distinct Host IRs
class PostOnStream : public Expr {
 public:
  using Expr::Expr;
  PostOnStream(
      IrBuilderPasskey passkey,
      Expr* host_op,
      std::vector<Val*> inputs,
      std::vector<Val*> outputs);

  PostOnStream(const PostOnStream& other) = delete;
  PostOnStream& operator=(const PostOnStream& other) = delete;
  PostOnStream(PostOnStream&& other) = delete;
  PostOnStream& operator=(PostOnStream&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::PostOnStream";
  }

  bool sameAs(const Statement* other) const override;

  Expr* hostOpToPost() const {
    return attributes_.at(0)->as<Expr>();
  }
};

class LaunchKernel : public Expr {
 public:
  using Expr::Expr;
  LaunchKernel(
      IrBuilderPasskey passkey,
      int64_t group_id,
      const LaunchParams& launch_constraints,
      CompiledKernel* compile_kernel,
      const std::vector<Val*>& inputs,
      const std::vector<Val*>& outputs,
      Val* cache_id);
  LaunchKernel(const LaunchKernel* src, IrCloner* ir_cloner);

  LaunchKernel(const LaunchKernel& other) = delete;
  LaunchKernel& operator=(const LaunchKernel& other) = delete;
  LaunchKernel(LaunchKernel&& other) = delete;
  LaunchKernel& operator=(LaunchKernel&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::LaunchKernel";
  }

  int64_t groupId() const {
    return attribute<int64_t>(0);
  }

  const auto& launchParams() const {
    return attribute<LaunchParams>(1);
  }

  const auto& compileParams() const {
    return attribute<CompileParams>(2);
  }

  // A NamedScalar that holds the input cache ID. This NamedScalar is expected
  // to be bound by HostIrEvaluate::runWithInputs. If it's not bound,
  // KernelExecutor::runFusion will create a KernelArgumentHolder without cache
  // ID and initializeExecutorEntry every time, slow yet functional.
  Val* cacheId() const {
    return attributeVal(3);
  }

  CompiledKernel* compiledKernel() const {
    return compiled_kernel_;
  }

 private:
  CompiledKernel* compiled_kernel_ = nullptr;
};

class Deallocate : public Expr {
 public:
  using Expr::Expr;
  Deallocate(IrBuilderPasskey passkey, TensorView* tv);

  Deallocate(const Deallocate& other) = delete;
  Deallocate& operator=(const Deallocate& other) = delete;
  Deallocate(Deallocate&& other) = delete;
  Deallocate& operator=(Deallocate&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::Deallocate";
  }

  TensorView* buffer() const;
};

class Stream : public Val {
 public:
  // if index is provided, the IR represents the streams whose index is the
  // dynamic value of that index. Otherwise, it statically represents a new
  // Stream.
  Stream(IrBuilderPasskey passkey, Val* index = nullptr);
  Stream(const Stream* src, IrCloner* ir_cloner);
  bool sameAs(const Statement* other) const override;

  NVFUSER_DECLARE_CLONE
  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* index() const {
    return index_;
  }

 private:
  Val* index_ = nullptr;
};

class SetCurrentStream : public Expr {
 public:
  using Expr::Expr;
  SetCurrentStream(IrBuilderPasskey passkey, Stream* stream);

  SetCurrentStream(const SetCurrentStream& other) = delete;
  SetCurrentStream& operator=(const SetCurrentStream& other) = delete;
  SetCurrentStream(SetCurrentStream&& other) = delete;
  SetCurrentStream& operator=(SetCurrentStream&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::SetCurrentStream";
  }

  bool sameAs(const Statement* other) const override;

  Stream* stream() const {
    return attributes_.at(0)->as<Stream>();
  }
};

class GetCurrentStream : public Expr {
 public:
  using Expr::Expr;
  GetCurrentStream(IrBuilderPasskey passkey);

  GetCurrentStream(const GetCurrentStream& other) = delete;
  GetCurrentStream& operator=(const GetCurrentStream& other) = delete;
  GetCurrentStream(GetCurrentStream&& other) = delete;
  GetCurrentStream& operator=(GetCurrentStream&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::GetCurrentStream";
  }

  Stream* stream() const {
    return attributes_.at(0)->as<Stream>();
  }
};

class Wait : public Expr {
 public:
  using Expr::Expr;
  Wait(IrBuilderPasskey passkey, Expr* expr);

  Wait(const Wait& other) = delete;
  Wait& operator=(const Wait& other) = delete;
  Wait(Wait&& other) = delete;
  Wait& operator=(Wait&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::Wait";
  }

  bool sameAs(const Statement* other) const override;

  Expr* communication() const {
    return attributes_.at(0)->as<Expr>();
  }
};

// Makes the current stream wait on the given stream. Non-blocking from the host
// point of view.
class Synchronize : public Expr {
 public:
  using Expr::Expr;
  Synchronize(IrBuilderPasskey passkey, Stream* stream);

  Synchronize(const Synchronize& other) = delete;
  Synchronize& operator=(const Synchronize& other) = delete;
  Synchronize(Synchronize&& other) = delete;
  Synchronize& operator=(Synchronize&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::Synchronize";
  }

  bool sameAs(const Statement* other) const override;

  Stream* stream() const {
    return attributes_.at(0)->as<Stream>();
  }
};

// For ProcessGroupNCCL, startCoalescing and endCoalescing correspond to
// ncclGroupStart and ncclGroupEnd respectively. Those calls group p2p calls
// that need to be progressed together -- one global work handle returned by
// endCoalescing needs to be progressed. This has the following main advantages:
// 1) calls are progressed concurrently
// 2) since NICs are two-sided, a send and a recv calls need to be coalesced to
//    achieve full BW.
// 3) If not coalesced, we can easily reach a deadlock if the
//    send/recv pairs are not ordered correctly.
// It is in general preferable to coalesce send/recv calls. The only drawback is
// that we don't have a fine-grain control on synchronicity, in other words, we
// can only synchronize with the grouped communication at once.
// Remark: ProcessGroupUCC does not implement coalesced groups for now
class StartCoalescing : public Expr {
 public:
  using Expr::Expr;
  StartCoalescing(IrBuilderPasskey passkey);

  StartCoalescing(const StartCoalescing& other) = delete;
  StartCoalescing& operator=(const StartCoalescing& other) = delete;
  StartCoalescing(StartCoalescing&& other) = delete;
  StartCoalescing& operator=(StartCoalescing&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::StartCoalescing";
  }
};

class EndCoalescing : public Expr {
 public:
  using Expr::Expr;
  EndCoalescing(IrBuilderPasskey passkey);

  EndCoalescing(const EndCoalescing& other) = delete;
  EndCoalescing& operator=(const EndCoalescing& other) = delete;
  EndCoalescing(EndCoalescing&& other) = delete;
  EndCoalescing& operator=(EndCoalescing&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::EndCoalescing";
  }
};

class ShareMemHandles : public Expr {
 public:
  using Expr::Expr;
  ShareMemHandles(
      IrBuilderPasskey passkey,
      std::vector<P2PCommunication*> communications);

  ShareMemHandles(const ShareMemHandles& other) = delete;
  ShareMemHandles& operator=(const ShareMemHandles& other) = delete;
  ShareMemHandles(ShareMemHandles&& other) = delete;
  ShareMemHandles& operator=(ShareMemHandles&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::ShareMemHandles";
  }

  const std::vector<P2PCommunication*>& communications() const {
    return attribute<std::vector<P2PCommunication*>>(0);
  }
};

// This op mimicks the semantics of SelectOp but is used in HIR non-SSA context
// to index into a TensorView, returning an alias "slice" of the original
// TensorView.
class HirAliasSelect : public Expr {
 public:
  using Expr::Expr;
  HirAliasSelect(
      IrBuilderPasskey passkey,
      TensorView* in,
      TensorView* out,
      int64_t axis,
      Val* index);

  HirAliasSelect(const HirAliasSelect& other) = delete;
  HirAliasSelect& operator=(const HirAliasSelect& other) = delete;
  HirAliasSelect(HirAliasSelect&& other) = delete;
  HirAliasSelect& operator=(HirAliasSelect&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::HirAliasSelect";
  }

  TensorView* in() const {
    return inputs().at(0)->as<TensorView>();
  }

  TensorView* out() const {
    return attributeVal(0)->as<TensorView>();
  }

  int64_t axis() const {
    return attribute<int64_t>(1);
  }

  Val* index() const {
    return inputs().at(1);
  }
};

// This is essentially a LoadStoreOp whose output allocation is stream
// parallelized. The input and output TensorViews will have the same logical
// domain except that the input may have extra reduction dimensions. Upon
// evaluation, the output tensor will be an aliasing slice of the input tensor.
//
// I only plan to use ShardByStream around evaluated Exprs (e.g. MatmulOp)
// because otherwise I would have to change every such Expr to support stream
// parallelization.
//
// I don't plan to use ShardByStream around `LaunchKernel`s.  nvFuser codegen
// should be able to generate the right indexing by analyzing a
// stream-parallelized allocation/loop domain. Having the kernel do the indexing
// has two benefits:
// 1. Preserve maximum allocation information. With ShardByStream, the input
// allocation domain will be `[i{5}, i{2}], contiguity=[false, true]`. Without,
// it will be `[i{5}, i{6}], contiguity=[true, true]`. With the latter, the
// scheduler knows at compile time that the stride of i{5} is 6. This extra
// information could help scheduling and codegen.
// 2. Unify different parallel types. For TIDs or BIDs, we've already been
// indexing into a global tensor, not a per-thread or per-block "local" tensor.
// For DIDs, we currently assume the index is always 0 but that'll probably
// change when nvFuser generates kernel-initiated communication.
//
// I considered keeping this a LoadStoreOp but I couldn't figure out a good way
// to pass in the stream index, which is needed for slicing.
//
// This op is similar to HirAliasSelect, but the semantics are slightly
// different. For example, `out` is for some reason an attribute there. I could
// merge them into one but I prefer keeping them separated to not slow down
// MultiDeviceExecutor development.
class ShardByStream : public Expr {
 public:
  using Expr::Expr;
  ShardByStream(
      IrBuilderPasskey passkey,
      TensorView* out,
      TensorView* in,
      Val* stream_index);

  ShardByStream(const ShardByStream& other) = delete;
  ShardByStream& operator=(const ShardByStream& other) = delete;
  ShardByStream(ShardByStream&& other) = delete;
  ShardByStream& operator=(ShardByStream&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::ShardByStream";
  }

  TensorView* in() const {
    return inputs().at(0)->as<TensorView>();
  }

  TensorView* out() const {
    return outputs().at(0)->as<TensorView>();
  }

  Val* stream_index() const {
    return inputs().at(1);
  }
};

// SymmetricContiguousView takes a sharded TensorView with contiguous symmetric
// memory type (where the outermost dimension is parallelized with DIDx) and
// produces an unsharded TensorView. At runtime, it performs IPC handle exchange
// and creates a contiguous virtual address mapping across all ranks. This
// effectively "unshards" the tensor by making all ranks' data visible in a
// contiguous address space.
class SymmetricContiguousView : public Expr {
 public:
  using Expr::Expr;
  SymmetricContiguousView(
      IrBuilderPasskey passkey,
      TensorView* out,
      TensorView* in);

  SymmetricContiguousView(const SymmetricContiguousView& other) = delete;
  SymmetricContiguousView& operator=(const SymmetricContiguousView& other) =
      delete;
  SymmetricContiguousView(SymmetricContiguousView&& other) = delete;
  SymmetricContiguousView& operator=(SymmetricContiguousView&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::SymmetricContiguousView";
  }

  TensorView* in() const {
    return inputs().at(0)->as<TensorView>();
  }

  TensorView* out() const {
    return outputs().at(0)->as<TensorView>();
  }
};

class ForLoop : public Expr {
 public:
  using Expr::Expr;

  ForLoop(IrBuilderPasskey passkey, Val* index, Val* start, Val* stop);

  ForLoop(const ForLoop& other) = delete;
  ForLoop& operator=(const ForLoop& other) = delete;
  ForLoop(ForLoop&& other) = delete;
  ForLoop& operator=(ForLoop&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  static ForLoop* createFromIterDomain(IterDomain* iter_domain);

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "hir::ForLoop";
  }

  Val* index() const {
    return inputs().at(0);
  }

  Val* start() const {
    return inputs().at(1);
  }

  Val* stop() const {
    return inputs().at(2);
  }

  const Scope& body() const {
    return attribute<Scope>(0);
  }

  Scope& body() {
    return attribute<Scope>(0);
  }
};

} // namespace nvfuser::hir
