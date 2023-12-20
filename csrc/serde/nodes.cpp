// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/builder.h>
#include <serde/nodes.h>
#include <serde/utils.h>

namespace fb = flatbuffers;
namespace nvf = nvfuser;

namespace {

std::vector<nvf::Val*> getValues(
    nvf::IrContainer& container,
    const fb::Vector<int64_t>* buffer) {
  NVF_CHECK(buffer != nullptr, "Values buffer is nullptr");
  std::vector<nvf::Val*> result;
  result.reserve(buffer->size());
  std::transform(
      buffer->begin(),
      buffer->end(),
      std::back_inserter(result),
      [&](int64_t index) { return container.getVal(index); });
  return result;
}

std::vector<nvf::Expr*> getExpressions(
    nvf::IrContainer& container,
    const fb::Vector<int64_t>* buffer) {
  NVF_CHECK(buffer != nullptr, "Expressions buffer is nullptr");
  std::vector<nvf::Expr*> result;
  result.reserve(buffer->size());
  std::transform(
      buffer->begin(),
      buffer->end(),
      std::back_inserter(result),
      [&](int64_t index) { return container.getExpr(index); });
  return result;
}

std::vector<nvf::Statement*> getStatements(
    nvf::IrContainer& container,
    const fb::Vector<fb::Offset<nvf::serde::Statement>>* buffer) {
  NVF_CHECK(buffer != nullptr, "Statements buffer is nullptr");
  std::vector<nvf::Statement*> result;
  result.reserve(buffer->size());
  std::transform(
      buffer->begin(),
      buffer->end(),
      std::back_inserter(result),
      [&](auto stmt) -> nvf::Statement* {
        if (stmt->is_val()) {
          return container.getVal(stmt->index());
        } else {
          return container.getExpr(stmt->index());
        }
      });
  return result;
}

template <typename ExprType>
nvf::Expr* deserialize(
    nvf::IrContainer& container,
    const nvf::serde::Expression* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::Expression is nullptr");
  std::vector<nvf::Val*> inputs =
      getValues(container, buffer->input_vals());
  std::vector<nvf::Val*> outputs =
      getValues(container, buffer->output_vals());
  std::vector<nvf::Statement*> attributes =
      getStatements(container, buffer->attributes_stmts());
  return nvf::IrBuilder::create<ExprType>(
      buffer->type(), inputs, outputs, attributes);
}

} // namespace

namespace nvfuser::serde {

void ValueFactory::registerAllParsers() {
  auto deserializeVal = [](const nvf::IrContainer& container, const Value* buffer) {
    return IrBuilder::create<nvf::Val>(
        nvf::ValType::Others, mapToDtypeStruct(buffer->dtype_enum()));
  };
  registerParser(serde::ValData::NONE, deserializeVal);
}

void ExpressionFactory::registerAllParsers() {
  registerParser(serde::ExprType::Allocate, deserialize<nvf::kir::Allocate>);
}

} // namespace nvfuser::serde
