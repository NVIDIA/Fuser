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
#include <optional>

namespace nvf = nvfuser;

namespace nvfuser::serde {

void ValueFactory::registerAllParsers() {
  registerParser(
      serde::ValData::NONE,
      nvf::IrBuilder::deserializeVal<nvf::Val, void, serde::ValData::NONE>);

  registerParser(
      serde::ValData::NamedScalar,
      nvf::IrBuilder::deserializeVal<
          nvf::NamedScalar,
          serde::NamedScalar,
          serde::ValData::NamedScalar>);

  registerParser(
      serde::ValData::IterDomain,
      nvf::IrBuilder::deserializeVal<
          nvf::IterDomain,
          serde::IterDomain,
          serde::ValData::IterDomain>);

  registerParser(
      serde::ValData::PipelineVal,
      nvf::IrBuilder::deserializeVal<
          nvf::PipelineVal,
          serde::PipelineVal,
          serde::ValData::PipelineVal>);

  registerParser(
      serde::ValData::Predicate,
      nvf::IrBuilder::deserializeVal<
          nvf::kir::Predicate,
          serde::Predicate,
          serde::ValData::Predicate>);

  registerParser(
      serde::ValData::TensorIndex,
      nvf::IrBuilder::deserializeVal<
          nvf::kir::TensorIndex,
          serde::TensorIndex,
          serde::ValData::TensorIndex>);

  registerParser(
      serde::ValData::TensorDomain,
      nvf::IrBuilder::deserializeVal<
          nvf::TensorDomain,
          serde::TensorDomain,
          serde::ValData::TensorDomain>);

  registerParser(
      serde::ValData::TensorView,
      nvf::IrBuilder::deserializeVal<
          nvf::TensorView,
          serde::TensorView,
          serde::ValData::TensorView>);
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
      serde::ExprType::View,
      nvf::IrBuilder::deserializeExpr<nvf::ViewAsScalar>);
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
  auto deserialize_unsupported =
      [](const nvf::serde::Expression* buffer) -> nvf::Expr* {
    NVF_ERROR(buffer != nullptr, "serde::Expression is nullptr.");
    NVF_ERROR(false, "FlattenedAssocComm is not supported.");
  };
  registerParser(serde::ExprType::FlattenedAssocComm, deserialize_unsupported);
}

} // namespace nvfuser::serde
