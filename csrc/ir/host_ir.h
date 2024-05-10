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

namespace nvfuser {

namespace hir {

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
  virtual const char* getOpString() const override {
    return "hir::HostUnit";
  }

  bool sameAs(const Statement* other) const override;

  Fusion* fusion_to_execute() const {
    return fusion_.get();
  }

 private:
  std::unique_ptr<Fusion> fusion_;
};

class NVF_API StreamIr : public Val {
 public:
  StreamIr(IrBuilderPasskey passkey);
  StreamIr(const StreamIr* src, IrCloner* ir_cloner);
  bool sameAs(const Statement* other) const override;

  NVFUSER_DECLARE_CLONE
  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

 private:
  const int idx_;
  static int running_counter_;
};

class NVF_API PostOnStream : public Expr {
 public:
  using Expr::Expr;
  PostOnStream(IrBuilderPasskey passkey,
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
  virtual const char* getOpString() const override {
    return "hir::PostOnStream";
  }

  bool sameAs(const Statement* other) const override;

  Expr* hostOpToPost() {
    return attributes_.at(0)->as<Expr>();
  }
};

// class NVF_API ExecuteComm : public Expr {
//  public:
//   using Expr::Expr;
//   ExecuteComm(IrBuilderPasskey passkey, std::unique_ptr<Fusion> fusion, std::vector<Val*> inputs, std::vector<Val*> outputs);

//   ExecuteComm(const ExecuteComm& other) = delete;
//   ExecuteComm& operator=(const ExecuteComm& other) = delete;
//   ExecuteComm(ExecuteComm&& other) = delete;
//   ExecuteComm& operator=(ExecuteComm&& other) = delete;

//   NVFUSER_DECLARE_CLONE_AND_CREATE

//   std::string toString(int indent_size = 0) const override;
//   std::string toInlineString(int indent_size = 0) const override;
//   virtual const char* getOpString() const override {
//     return "ExecuteComm";
//   }

//   bool sameAs(const Statement* other) const override;

//  private:
//   std::unique_ptr<Fusion> fusion_;
// };

} // namespace hir

} // namespace nvfuser
