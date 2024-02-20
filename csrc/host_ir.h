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

class ExecuteFusion : public Expr {
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

class SaveSlicedOutput : public Expr {
 public:
  using Expr::Expr;
  SaveSlicedOutput(IrBuilderPasskey passkey, Val* src, Val* dst, Val* index);

  SaveSlicedOutput(const SaveSlicedOutput& other) = delete;
  SaveSlicedOutput& operator=(const SaveSlicedOutput& other) = delete;
  SaveSlicedOutput(SaveSlicedOutput&& other) = delete;
  SaveSlicedOutput& operator=(SaveSlicedOutput&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  virtual const char* getOpString() const override {
    return "SaveSlicedOutput";
  }

  bool sameAs(const Statement* other) const override;

 private:
  Val* src_;
  Val* dst_;
  Val* index_;
};

class HostFusion final : public Fusion {
 public:
  HostFusion() = default;
  HostFusion(const HostFusion&) = delete;
  HostFusion& operator=(const HostFusion&) = delete;

  Fusion* gpu_fusion;
};

std::unique_ptr<HostFusion> makeHostFusionFromFusion(Fusion* fusion);

} // namespace hir

} // namespace nvfuser
