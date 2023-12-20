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
  std::vector<nvf::Val*> inputs = getValues(container, buffer->input_vals());
  std::vector<nvf::Val*> outputs = getValues(container, buffer->output_vals());
  std::vector<nvf::Statement*> attributes =
      getStatements(container, buffer->attributes_stmts());
  return nvf::IrBuilder::create<ExprType>(
      buffer->type(), inputs, outputs, attributes);
}

} // namespace

namespace nvfuser::serde {

void ValueFactory::registerAllParsers() {
  auto deserializeVal = [](const nvf::IrContainer& container,
                           const serde::Value* buffer) {
    return IrBuilder::create<nvf::Val>(
        nvf::ValType::Others, mapToDtypeStruct(buffer->dtype_enum()));
  };
  registerParser(serde::ValData::NONE, deserializeVal);

  auto deserializeNamedScalar = [](const nvf::IrContainer& container,
                                   const serde::Value* buffer) {
    auto data = buffer->data_as_NamedScalar();
    return IrBuilder::create<nvf::NamedScalar>(
        data->name()->str(), mapToDtypeStruct(buffer->dtype_enum()));
  };
  registerParser(serde::ValData::NamedScalar, deserializeNamedScalar);
}

void ExpressionFactory::registerAllParsers() {
  // fusion
  registerParser(
      serde::ExprType::ArrayConstruct, deserialize<nvf::ArrayConstruct>);
  registerParser(serde::ExprType::Binary, deserialize<nvf::BinaryOp>);
  registerParser(serde::ExprType::Broadcast, deserialize<nvf::BroadcastOp>);
  registerParser(serde::ExprType::Cat, deserialize<nvf::CatOp>);
  registerParser(serde::ExprType::Expand, deserialize<nvf::ExpandOp>);
  registerParser(serde::ExprType::Eye, deserialize<nvf::EyeOp>);
  registerParser(serde::ExprType::Full, deserialize<nvf::FullOp>);
  registerParser(serde::ExprType::Gather, deserialize<nvf::GatherOp>);
  registerParser(serde::ExprType::GetAttr, deserialize<nvf::GetAttr>);
  registerParser(serde::ExprType::GetItem, deserialize<nvf::GetItem>);
  registerParser(serde::ExprType::GetMetaData, deserialize<nvf::GetMetaData>);
  registerParser(serde::ExprType::IndexSelect, deserialize<nvf::IndexSelectOp>);
  registerParser(serde::ExprType::Iota, deserialize<nvf::IotaOp>);
  registerParser(serde::ExprType::LoadStore, deserialize<nvf::LoadStoreOp>);
  registerParser(serde::ExprType::Merge, deserialize<nvf::Merge>);
  registerParser(serde::ExprType::Mma, deserialize<nvf::MmaOp>);
  registerParser(serde::ExprType::Pad, deserialize<nvf::PadOp>);
  registerParser(serde::ExprType::Reduction, deserialize<nvf::ReductionOp>);
  registerParser(serde::ExprType::Resize, deserialize<nvf::Resize>);
  registerParser(serde::ExprType::ReverseArray, deserialize<nvf::ReverseArray>);
  registerParser(serde::ExprType::RNG, deserialize<nvf::RNGOp>);
  registerParser(serde::ExprType::Scatter, deserialize<nvf::ScatterOp>);
  registerParser(serde::ExprType::Select, deserialize<nvf::SelectOp>);
  registerParser(serde::ExprType::Shift, deserialize<nvf::ShiftOp>);
  registerParser(serde::ExprType::Slice, deserialize<nvf::SliceOp>);
  registerParser(serde::ExprType::Split, deserialize<nvf::Split>);
  registerParser(serde::ExprType::Squeeze, deserialize<nvf::SqueezeOp>);
  registerParser(
      serde::ExprType::StructConstruct, deserialize<nvf::StructConstruct>);
  registerParser(serde::ExprType::Swizzle, deserialize<nvf::Swizzle>);
  registerParser(serde::ExprType::Swizzle2D, deserialize<nvf::Swizzle2D>);
  registerParser(
      serde::ExprType::TensorConstruct, deserialize<nvf::TensorConstruct>);
  registerParser(serde::ExprType::Ternary, deserialize<nvf::TernaryOp>);
  registerParser(serde::ExprType::TorchGather, deserialize<nvf::TorchGatherOp>);
  registerParser(serde::ExprType::Unary, deserialize<nvf::UnaryOp>);
  registerParser(serde::ExprType::View, deserialize<nvf::ViewAsScalar>);
  registerParser(serde::ExprType::ViewAsScalar, deserialize<nvf::ViewAsScalar>);
  registerParser(serde::ExprType::Welford, deserialize<nvf::WelfordOp>);

  // kir
  registerParser(serde::ExprType::Allocate, deserialize<nvf::kir::Allocate>);
  registerParser(
      serde::ExprType::AllocateFusedReduction,
      deserialize<nvf::kir::AllocateFusedReduction>);
  registerParser(serde::ExprType::Asm, deserialize<nvf::kir::Asm>);
  registerParser(
      serde::ExprType::AsyncCommit, deserialize<nvf::kir::AsyncCommit>);
  registerParser(serde::ExprType::AsyncWait, deserialize<nvf::kir::AsyncWait>);
  registerParser(
      serde::ExprType::BlockSerializeRelease,
      deserialize<nvf::kir::BlockSerializeRelease>);
  registerParser(
      serde::ExprType::BlockSerializeWait,
      deserialize<nvf::kir::BlockSerializeWait>);
  registerParser(serde::ExprType::BlockSync, deserialize<nvf::kir::BlockSync>);
  registerParser(
      serde::ExprType::EncodeTensorMapTiled,
      deserialize<nvf::kir::EncodeTensorMapTiled>);
  registerParser(serde::ExprType::ForLoop, deserialize<nvf::kir::ForLoop>);
  registerParser(
      serde::ExprType::GetRNGSeedAndOffsetFromHost,
      deserialize<nvf::kir::GetRNGSeedAndOffsetFromHost>);
  registerParser(
      serde::ExprType::GridBroadcast, deserialize<nvf::kir::GridBroadcast>);
  registerParser(serde::ExprType::GridSync, deserialize<nvf::kir::GridSync>);
  registerParser(
      serde::ExprType::GridWelford, deserialize<nvf::kir::GridWelford>);
  registerParser(
      serde::ExprType::IfThenElse, deserialize<nvf::kir::IfThenElse>);
  registerParser(
      serde::ExprType::InitMagicZero, deserialize<nvf::kir::InitMagicZero>);
  registerParser(
      serde::ExprType::MBarrierArrive, deserialize<nvf::kir::MBarrierArrive>);
  registerParser(
      serde::ExprType::MBarrierInit, deserialize<nvf::kir::MBarrierInit>);
  registerParser(
      serde::ExprType::MBarrierWait, deserialize<nvf::kir::MBarrierWait>);
  registerParser(
      serde::ExprType::UpdateMagicZero, deserialize<nvf::kir::UpdateMagicZero>);

  // multidevice
  registerParser(
      serde::ExprType::PipelineCommunication,
      deserialize<nvf::PipelineCommunication>);
  registerParser(
      serde::ExprType::PipelineStage, deserialize<nvf::PipelineStage>);

  // Expression Simplifier
  auto deserialize_unsupported =
      [](nvf::IrContainer& container,
         const nvf::serde::Expression* buffer) -> nvf::Expr* {
    NVF_ERROR(buffer != nullptr, "serde::Expression is nullptr.");
    NVF_ERROR(false, "FlattenedAssocComm is not supported.");
  };
  registerParser(serde::ExprType::FlattenedAssocComm, deserialize_unsupported);
}

} // namespace nvfuser::serde
