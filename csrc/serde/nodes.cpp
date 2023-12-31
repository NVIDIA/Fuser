// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_simplifier.h>
#include <ir/builder.h>
#include <serde/nodes.h>
#include <serde/utils.h>
#include <optional>

namespace nvf = nvfuser;

namespace nvfuser::serde {

void ValueFactory::registerAllParsers() {
  auto deserialize_unsupported =
      [](const nvf::serde::Value* buffer) -> nvf::Val* {
    NVF_ERROR(buffer != nullptr, "serde::Value is nullptr.");
    NVF_ERROR(false, "serde::ValueData::NONE is not supported.");
  };
  registerParser(serde::ValueData::NONE, deserialize_unsupported);

  registerParser(
      serde::ValueData::DataType,
      nvf::IrBuilder::deserializeVal<
          nvf::Val,
          serde::DataType,
          serde::ValueData::DataType>);

  registerParser(
      serde::ValueData::IterDomain,
      nvf::IrBuilder::deserializeVal<
          nvf::IterDomain,
          serde::IterDomain,
          serde::ValueData::IterDomain>);

  registerParser(
      serde::ValueData::NamedScalar,
      nvf::IrBuilder::deserializeVal<
          nvf::NamedScalar,
          serde::NamedScalar,
          serde::ValueData::NamedScalar>);

  registerParser(
      serde::ValueData::PipelineVal,
      nvf::IrBuilder::deserializeVal<
          nvf::PipelineVal,
          serde::PipelineVal,
          serde::ValueData::PipelineVal>);

  registerParser(
      serde::ValueData::PolymorphicValue,
      nvf::IrBuilder::deserializeVal<
          nvf::Val,
          serde::PolymorphicValue,
          serde::ValueData::PolymorphicValue>);

  registerParser(
      serde::ValueData::PolymorphicValueDtype,
      nvf::IrBuilder::deserializeVal<
          nvf::Val,
          serde::PolymorphicValueDtype,
          serde::ValueData::PolymorphicValueDtype>);

  registerParser(
      serde::ValueData::Predicate,
      nvf::IrBuilder::deserializeVal<
          nvf::kir::Predicate,
          serde::Predicate,
          serde::ValueData::Predicate>);

  registerParser(
      serde::ValueData::PrimDataType,
      nvf::IrBuilder::deserializeVal<
          nvf::Val,
          serde::PrimDataType,
          serde::ValueData::PrimDataType>);

  registerParser(
      serde::ValueData::TensorDomain,
      nvf::IrBuilder::deserializeVal<
          nvf::TensorDomain,
          serde::TensorDomain,
          serde::ValueData::TensorDomain>);

  registerParser(
      serde::ValueData::TensorIndex,
      nvf::IrBuilder::deserializeVal<
          nvf::kir::TensorIndex,
          serde::TensorIndex,
          serde::ValueData::TensorIndex>);

  registerParser(
      serde::ValueData::TensorView,
      nvf::IrBuilder::deserializeVal<
          nvf::TensorView,
          serde::TensorView,
          serde::ValueData::TensorView>);
}

void ExpressionFactory::registerAllParsers() {
  // fusion
  registerParser(
      serde::ExprType::ArrayConstruct,
      nvf::IrBuilder::deserializeExpr<nvf::ArrayConstruct>);
  registerParser(
      serde::ExprType::Binary, nvf::IrBuilder::deserializeExpr<nvf::BinaryOp>);
  registerParser(
      serde::ExprType::Broadcast,
      nvf::IrBuilder::deserializeExpr<nvf::BroadcastOp>);
  registerParser(
      serde::ExprType::Cat, nvf::IrBuilder::deserializeExpr<nvf::CatOp>);
  registerParser(
      serde::ExprType::Expand, nvf::IrBuilder::deserializeExpr<nvf::ExpandOp>);
  registerParser(
      serde::ExprType::Eye, nvf::IrBuilder::deserializeExpr<nvf::EyeOp>);
  registerParser(
      serde::ExprType::Full, nvf::IrBuilder::deserializeExpr<nvf::FullOp>);
  registerParser(
      serde::ExprType::Gather, nvf::IrBuilder::deserializeExpr<nvf::GatherOp>);
  registerParser(
      serde::ExprType::GetAttr, nvf::IrBuilder::deserializeExpr<nvf::GetAttr>);
  registerParser(
      serde::ExprType::GetItem, nvf::IrBuilder::deserializeExpr<nvf::GetItem>);
  registerParser(
      serde::ExprType::GetMetaData,
      nvf::IrBuilder::deserializeExpr<nvf::GetMetaData>);
  registerParser(
      serde::ExprType::IndexSelect,
      nvf::IrBuilder::deserializeExpr<nvf::IndexSelectOp>);
  registerParser(
      serde::ExprType::Iota, nvf::IrBuilder::deserializeExpr<nvf::IotaOp>);
  registerParser(
      serde::ExprType::LoadStore,
      nvf::IrBuilder::deserializeExpr<nvf::LoadStoreOp>);
  registerParser(
      serde::ExprType::Merge, nvf::IrBuilder::deserializeExpr<nvf::Merge>);
  registerParser(
      serde::ExprType::Mma, nvf::IrBuilder::deserializeExpr<nvf::MmaOp>);
  registerParser(
      serde::ExprType::Pad, nvf::IrBuilder::deserializeExpr<nvf::PadOp>);
  registerParser(
      serde::ExprType::Reduction,
      nvf::IrBuilder::deserializeExpr<nvf::ReductionOp>);
  registerParser(
      serde::ExprType::Resize, nvf::IrBuilder::deserializeExpr<nvf::Resize>);
  registerParser(
      serde::ExprType::ReverseArray,
      nvf::IrBuilder::deserializeExpr<nvf::ReverseArray>);
  registerParser(
      serde::ExprType::RNG, nvf::IrBuilder::deserializeExpr<nvf::RNGOp>);
  registerParser(
      serde::ExprType::Scatter,
      nvf::IrBuilder::deserializeExpr<nvf::ScatterOp>);
  registerParser(
      serde::ExprType::Select, nvf::IrBuilder::deserializeExpr<nvf::SelectOp>);
  registerParser(
      serde::ExprType::Shift, nvf::IrBuilder::deserializeExpr<nvf::ShiftOp>);
  registerParser(
      serde::ExprType::Slice, nvf::IrBuilder::deserializeExpr<nvf::SliceOp>);
  registerParser(
      serde::ExprType::Split, nvf::IrBuilder::deserializeExpr<nvf::Split>);
  registerParser(
      serde::ExprType::Squeeze,
      nvf::IrBuilder::deserializeExpr<nvf::SqueezeOp>);
  registerParser(
      serde::ExprType::StructConstruct,
      nvf::IrBuilder::deserializeExpr<nvf::StructConstruct>);
  registerParser(
      serde::ExprType::Swizzle, nvf::IrBuilder::deserializeExpr<nvf::Swizzle>);
  registerParser(
      serde::ExprType::Swizzle2D,
      nvf::IrBuilder::deserializeExpr<nvf::Swizzle2D>);
  registerParser(
      serde::ExprType::TensorConstruct,
      nvf::IrBuilder::deserializeExpr<nvf::TensorConstruct>);
  registerParser(
      serde::ExprType::Ternary,
      nvf::IrBuilder::deserializeExpr<nvf::TernaryOp>);
  registerParser(
      serde::ExprType::TorchGather,
      nvf::IrBuilder::deserializeExpr<nvf::TorchGatherOp>);
  registerParser(
      serde::ExprType::Unary, nvf::IrBuilder::deserializeExpr<nvf::UnaryOp>);
  registerParser(
      serde::ExprType::View, nvf::IrBuilder::deserializeExpr<nvf::ViewOp>);
  registerParser(
      serde::ExprType::ViewAsScalar,
      nvf::IrBuilder::deserializeExpr<nvf::ViewAsScalar>);
  registerParser(
      serde::ExprType::Welford,
      nvf::IrBuilder::deserializeExpr<nvf::WelfordOp>);

  // kir
  registerParser(
      serde::ExprType::Allocate,
      nvf::IrBuilder::deserializeExpr<nvf::kir::Allocate>);
  registerParser(
      serde::ExprType::AllocateFusedReduction,
      nvf::IrBuilder::deserializeExpr<nvf::kir::AllocateFusedReduction>);
  registerParser(
      serde::ExprType::Asm, nvf::IrBuilder::deserializeExpr<nvf::kir::Asm>);
  registerParser(
      serde::ExprType::AsyncCommit,
      nvf::IrBuilder::deserializeExpr<nvf::kir::AsyncCommit>);
  registerParser(
      serde::ExprType::AsyncWait,
      nvf::IrBuilder::deserializeExpr<nvf::kir::AsyncWait>);
  registerParser(
      serde::ExprType::BlockSerializeRelease,
      nvf::IrBuilder::deserializeExpr<nvf::kir::BlockSerializeRelease>);
  registerParser(
      serde::ExprType::BlockSerializeWait,
      nvf::IrBuilder::deserializeExpr<nvf::kir::BlockSerializeWait>);
  registerParser(
      serde::ExprType::BlockSync,
      nvf::IrBuilder::deserializeExpr<nvf::kir::BlockSync>);
  registerParser(
      serde::ExprType::EncodeTensorMapTiled,
      nvf::IrBuilder::deserializeExpr<nvf::kir::EncodeTensorMapTiled>);
  registerParser(
      serde::ExprType::ForLoop,
      nvf::IrBuilder::deserializeExpr<nvf::kir::ForLoop>);
  registerParser(
      serde::ExprType::GetRNGSeedAndOffsetFromHost,
      nvf::IrBuilder::deserializeExpr<nvf::kir::GetRNGSeedAndOffsetFromHost>);
  registerParser(
      serde::ExprType::GridBroadcast,
      nvf::IrBuilder::deserializeExpr<nvf::kir::GridBroadcast>);
  registerParser(
      serde::ExprType::GridSync,
      nvf::IrBuilder::deserializeExpr<nvf::kir::GridSync>);
  registerParser(
      serde::ExprType::GridWelford,
      nvf::IrBuilder::deserializeExpr<nvf::kir::GridWelford>);
  registerParser(
      serde::ExprType::IfThenElse,
      nvf::IrBuilder::deserializeExpr<nvf::kir::IfThenElse>);
  registerParser(
      serde::ExprType::InitMagicZero,
      nvf::IrBuilder::deserializeExpr<nvf::kir::InitMagicZero>);
  registerParser(
      serde::ExprType::MBarrierArrive,
      nvf::IrBuilder::deserializeExpr<nvf::kir::MBarrierArrive>);
  registerParser(
      serde::ExprType::MBarrierInit,
      nvf::IrBuilder::deserializeExpr<nvf::kir::MBarrierInit>);
  registerParser(
      serde::ExprType::MBarrierWait,
      nvf::IrBuilder::deserializeExpr<nvf::kir::MBarrierWait>);
  registerParser(
      serde::ExprType::UpdateMagicZero,
      nvf::IrBuilder::deserializeExpr<nvf::kir::UpdateMagicZero>);

  // multidevice
  registerParser(
      serde::ExprType::PipelineCommunication,
      nvf::IrBuilder::deserializeExpr<nvf::PipelineCommunication>);
  registerParser(
      serde::ExprType::PipelineStage,
      nvf::IrBuilder::deserializeExpr<nvf::PipelineStage>);

  // Expression Simplifier
  registerParser(
      serde::ExprType::FlattenedAssocComm,
      nvf::IrBuilder::deserializeExpr<nvf::assoc_comm::FlattenedAssocCommOp>);
}

} // namespace nvfuser::serde
