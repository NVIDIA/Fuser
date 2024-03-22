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

class NVF_API ExecutableUnit : public Expr {
  public:
  using Expr::Expr;
  ExecutableUnit(IrBuilderPasskey passkey, std::unique_ptr<Fusion> fusion);
  ExecutableUnit(const ExecutableUnit* src, IrCloner* ir_cloner);

  ExecutableUnit(const ExecutableUnit& other) = delete;
  ExecutableUnit& operator=(const ExecutableUnit& other) = delete;
  ExecutableUnit(ExecutableUnit&& other) = delete;
  ExecutableUnit& operator=(ExecutableUnit&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE
  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  virtual const char* getOpString() const override {
    return "hir::ExecutableUnit";
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
               ExecutableUnit* eu,
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

  ExecutableUnit* executableUnit() {
    return attributes_.at(0)->as<ExecutableUnit>();
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
