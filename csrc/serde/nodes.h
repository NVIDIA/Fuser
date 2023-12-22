
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/all_nodes.h>
#include <serde/factory.h>

namespace nvfuser::serde {

// Forward definition for RecordFunctor
class Expr;
class Val;

class ValueFactory : public Factory<serde::Value, nvfuser::Val*> {
 public:
  ValueFactory() : Factory((nvfuser::toUnderlying(ValData::MAX) + 1)) {
    registerAllParsers();
  }

 private:
  void registerAllParsers();
};

class ExpressionFactory : public Factory<serde::Expression, nvfuser::Expr*> {
 public:
  ExpressionFactory() : Factory((nvfuser::toUnderlying(ExprType::MAX) + 2)) {
    registerAllParsers();
  }

 private:
  void registerAllParsers();
};

} // namespace nvfuser::serde
