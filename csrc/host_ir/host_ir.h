// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/builder.h>
#include <multidevice/communication.h>
#include <atomic>

namespace nvfuser {

namespace hir {

/*
  Host Irs are used to represent a host program. They need to be registered in a
  HostIrContainer. Each Ir represents a Host data or instruction.
*/

/*
  HostUnit represents a Fusion in the Host Program. In other words, it
  represents a compute graph (or a segment of a larger compute graph)
  represented by a Fusion that should be compiled and executed as a bulked item
  from the host perspective.

  This IR can be thought as a thin layer around the class `Fusion`, which
  furthermore inherits from `Expr` so that it is an "IR" in nvFuser IR
  semantics.

  This IRs fundamentally allows nested IR structures. It could potentially be
  useful in other instances than HostIrs.

  Its implementation is minimal, the only specifity being the moethod
  `fusion_to_execute()` that returns the fusion that the IR represents.

  Note: HostUnit has no I/O itself -- however the Fusion it embbeds has I/O of
  course, which are not registered in the surrounding HostIrContainer.

  Note: Whether HostUnit should inherit from Expr or Val is debatable. Both are
  possible, I define it as an Expr for now here but am open to change it.
*/
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

/*
  PostOnStream represents the host instruction of executing a HostUnit. Its I/O
  represents in the host program the concrete I/O that will be bound at runtime
  to the Fusion's I/O for compilation and execution. At runtime, PostOnStream
  will compile and launch the kernel lowered from the HostUnit's embedded
  Fusion.

  Note: later PostOnStream will take a "Stream" argument

  Note: later PostOnStream will also be able to launch network Communications

  Note: later compilation and kernel launch will be separated and represented by
  distinct Host IRs
*/
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

} // namespace hir

} // namespace nvfuser
