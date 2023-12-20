
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

class ValueFactory : public NodeFactory<serde::Value, nvfuser::Val*> {
 private:
  void registerAllParsers() override;
};

class ExpressionFactory
    : public NodeFactory<serde::Expression, nvfuser::Expr*> {
 private:
  void registerAllParsers() override;
};

} // namespace nvfuser::serde
