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
class NVF_API HostUnit : public Expr {
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
  represents in the host program the concrete I/O that will be binded at runtime
  to the Fusion's I/O for compilation and execution. At runtime, PostOnStream
  will compile and launch the kernel lowered from the HostUnit's embedded
  Fusion.

  Note: later PostOnStream will take a "Stream" argument

  Note: later PostOnStream will also be able to launch network Communications

  Note: later compilation and kernel launch will be separated and represented by
  distinct Host IRs
*/
class NVF_API PostOnStream : public Expr {
 public:
  using Expr::Expr;
  PostOnStream(
      IrBuilderPasskey passkey,
      HostUnit* hu,
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

  HostUnit* hostUnit() const {
    return attributes_.at(0)->as<HostUnit>();
  }
};

} // namespace hir

} // namespace nvfuser
