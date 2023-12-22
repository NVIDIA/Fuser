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
  auto deserializeVal = [](nvf::IrContainer& container,
                           const serde::Value* buffer) {
    return IrBuilder::create<nvf::Val>(
        nvf::ValType::Others, mapToDtypeStruct(buffer->dtype_enum()));
  };
  registerParser(serde::ValData::NONE, deserializeVal);

  auto deserializeNamedScalar = [](nvf::IrContainer& container,
                                   const serde::Value* buffer) {
    auto data = buffer->data_as_NamedScalar();
    NVF_ERROR(data != nullptr, "Expected NamedScalar data.");
    return IrBuilder::create<nvf::NamedScalar>(
        data->name()->str(), mapToDtypeStruct(buffer->dtype_enum()));
  };
  registerParser(serde::ValData::NamedScalar, deserializeNamedScalar);

  auto deserializePredicate = [](nvf::IrContainer& container,
                                 const serde::Value* buffer) {
    auto data = buffer->data_as_Predicate();
    NVF_ERROR(data != nullptr, "Expected Predicate data.");
    return IrBuilder::create<nvf::kir::Predicate>(
        static_cast<PredicateType>(data->predicate_type_enum()),
        container.getExpr(data->expr()),
        container.getVal<nvf::Val>(data->thread_pred()));
  };
  registerParser(serde::ValData::Predicate, deserializePredicate);

  auto deserializeTensorIndex = [](nvf::IrContainer& container,
                                   const serde::Value* buffer) {
    auto data = buffer->data_as_TensorIndex();
    NVF_ERROR(data != nullptr, "Expected TensorIndex data.");

    const nvf::Val* view = container.getVal<nvf::Val>(data->view());
    NVF_ERROR(
        view->isA<nvf::TensorView>(),
        "Expected nvfuser::Val to be a TensorView.");
    return IrBuilder::create<nvf::kir::TensorIndex>(
        view->as<nvf::TensorView>(),
        container.getVal<nvf::Val>(data->index()),
        mapToDtypeStruct(buffer->dtype_enum()));
  };
  registerParser(serde::ValData::TensorIndex, deserializeTensorIndex);

  auto deserializePipelineVal = [](nvf::IrContainer& container,
                                   const serde::Value* buffer) {
    auto data = buffer->data_as_PipelineVal();
    NVF_ERROR(data != nullptr, "Expected PipelineVal data.");
    return IrBuilder::create<nvf::PipelineVal>(
        container.getVal<nvf::Val>(data->original_val()));
  };
  registerParser(serde::ValData::PipelineVal, deserializePipelineVal);

  auto deserializeIterDomain = [](nvf::IrContainer& container,
                                  const serde::Value* buffer) {
    auto data = buffer->data_as_IterDomain();
    NVF_ERROR(data != nullptr, "Expected IterDomain data.");
    return IrBuilder::create<nvf::IterDomain>(
        container.getVal<nvf::Val>(data->start_val()),
        container.getVal<nvf::Val>(data->extent_val()),
        container.getVal<nvf::Val>(data->expanded_extent_val()),
        container.getVal<nvf::Val>(data->stop_offset_val()),
        static_cast<ParallelType>(data->parallel_type_enum()),
        static_cast<IterType>(data->iter_type_enum()),
        data->is_rfactor_domain(),
        data->is_padded_dimension(),
        (data->padded_to_size() != -1)
            ? std::optional<int64_t>(data->padded_to_size())
            : std::nullopt,
        data->is_mma_swizzled());
  };
  registerParser(serde::ValData::IterDomain, deserializeIterDomain);

  auto deserializeTensorDomain = [](nvf::IrContainer& container,
                                    const serde::Value* buffer) {
    auto data = buffer->data_as_TensorDomain();
    NVF_ERROR(data != nullptr, "Expected TensorDomain data.");
    return IrBuilder::create<nvf::TensorDomain>(
        container.getValues<nvf::IterDomain>(data->root_domain()),
        container.getValues<nvf::IterDomain>(data->rfactor_domain()),
        container.getValues<nvf::IterDomain>(data->allocation_domain()),
        container.getValues<nvf::IterDomain>(data->leaf_domain()),
        mapSerdeContiguityEnum(data->contiguity()));
  };
  registerParser(serde::ValData::TensorDomain, deserializeTensorDomain);

  auto deserializeTensorView = [](nvf::IrContainer& container,
                                  const serde::Value* buffer) {
    auto data = buffer->data_as_TensorView();
    NVF_ERROR(data != nullptr, "Expected TensorView data.");

    return IrBuilder::create<nvf::TensorView>(
        container.getVal<nvf::TensorDomain>(data->domain()),
        mapToDtypeStruct(buffer->dtype_enum()),
        static_cast<MemoryType>(data->memory_type_enum()));
  };
  registerParser(serde::ValData::TensorView, deserializeTensorView);
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
