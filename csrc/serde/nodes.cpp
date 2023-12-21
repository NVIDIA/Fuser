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

namespace {

template <typename ExprType>
nvf::Expr* deserialize(
    nvf::IrContainer& container,
    const nvf::serde::Expression* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::Expression is nullptr");
  std::vector<nvf::Val*> inputs =
      container.getValues<nvf::Val>(buffer->input_vals());
  std::vector<nvf::Val*> outputs =
      container.getValues<nvf::Val>(buffer->output_vals());
  std::vector<nvf::Statement*> attributes =
      container.getStatements(buffer->attributes_stmts());
  return nvf::IrBuilder::create<ExprType>(
      buffer->type(), inputs, outputs, attributes);
}

} // namespace

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
        container.getVal(data->thread_pred()));
  };
  registerParser(serde::ValData::Predicate, deserializePredicate);

  auto deserializeTensorIndex = [](nvf::IrContainer& container,
                                   const serde::Value* buffer) {
    auto data = buffer->data_as_TensorIndex();
    NVF_ERROR(data != nullptr, "Expected TensorIndex data.");

    const nvf::Val* view = container.getVal(data->view());
    NVF_ERROR(
        view->isA<nvf::TensorView>(),
        "Expected nvfuser::Val to be a TensorView.");
    return IrBuilder::create<nvf::kir::TensorIndex>(
        view->as<nvf::TensorView>(),
        container.getVal(data->index()),
        mapToDtypeStruct(buffer->dtype_enum()));
  };
  registerParser(serde::ValData::TensorIndex, deserializeTensorIndex);

  auto deserializePipelineVal = [](nvf::IrContainer& container,
                                   const serde::Value* buffer) {
    auto data = buffer->data_as_PipelineVal();
    NVF_ERROR(data != nullptr, "Expected PipelineVal data.");
    return IrBuilder::create<nvf::PipelineVal>(
        container.getVal(data->original_val()));
  };
  registerParser(serde::ValData::PipelineVal, deserializePipelineVal);

  auto deserializeIterDomain = [](nvf::IrContainer& container,
                                  const serde::Value* buffer) {
    auto data = buffer->data_as_IterDomain();
    NVF_ERROR(data != nullptr, "Expected IterDomain data.");
    return IrBuilder::create<nvf::IterDomain>(
        container.getVal(data->start_val()),
        container.getVal(data->extent_val()),
        container.getVal(data->expanded_extent_val()),
        container.getVal(data->stop_offset_val()),
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

    nvf::Val* domain = container.getVal(data->domain());
    NVF_ERROR(
        domain->isA<nvf::TensorDomain>(),
        "Expected nvfuser::Val to be a TensorDomain.");
    return IrBuilder::create<nvf::TensorView>(
        domain->as<nvf::TensorDomain>(),
        mapToDtypeStruct(buffer->dtype_enum()),
        static_cast<MemoryType>(data->memory_type_enum()));
  };
  registerParser(serde::ValData::TensorView, deserializeTensorView);
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
