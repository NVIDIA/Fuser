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

class NVF_API FusionIr : public Val {
  FusionIr(IrBuilderPasskey passkey, std::unique_ptr<Fusion> fusion);
  FusionIr(const FusionIr* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE
  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  bool sameAs(const Statement* other) const override;

  // returns the Val from which this PipelineVal has been created
  Fusion* fusion() const {
    return fusion_.get();
  }

 private:
  std::unique_ptr<Fusion> fusion_;
};


class NVF_API ExecuteFusion : public Expr {
 public:
  using Expr::Expr;
  ExecuteFusion(IrBuilderPasskey passkey, Fusion* fusion, std::vector<Val*> inputs, std::vector<Val*> outputs);

  ExecuteFusion(const ExecuteFusion& other) = delete;
  ExecuteFusion& operator=(const ExecuteFusion& other) = delete;
  ExecuteFusion(ExecuteFusion&& other) = delete;
  ExecuteFusion& operator=(ExecuteFusion&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  virtual const char* getOpString() const override {
    return "ExecuteFusion";
  }

  bool sameAs(const Statement* other) const override;

 private:
  std::unique_ptr<Fusion> fusion_;
};

class NVF_API ExecuteComm : public Expr {
 public:
  using Expr::Expr;
  ExecuteComm(IrBuilderPasskey passkey, Fusion* fusion, std::vector<Val*> inputs, std::vector<Val*> outputs);

  ExecuteComm(const ExecuteComm& other) = delete;
  ExecuteComm& operator=(const ExecuteComm& other) = delete;
  ExecuteComm(ExecuteComm&& other) = delete;
  ExecuteComm& operator=(ExecuteComm&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  virtual const char* getOpString() const override {
    return "ExecuteComm";
  }

  bool sameAs(const Statement* other) const override;

 private:
  std::unique_ptr<Fusion> fusion_;
};

} // namespace hir

} // namespace nvfuser
