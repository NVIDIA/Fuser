// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/builder.h>
#include <ir/cloner.h>
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>

namespace nvfuser {

PipelineStage::PipelineStage(
    IrBuilderPasskey passkey,
    const PipelineStageDescriptor* descriptor,
    ValSet input_vals,
    ValSet output_vals)
    : Expr(passkey, serde::ExprType::PipelineStage) {
  NVF_ERROR(
      passkey.ir_container_ ? passkey.ir_container_->isA<Pipeline>() : false,
      "IR type only valid for Pipeline container.");

  for (auto v : output_vals) {
    NVF_ERROR(v->isA<PipelineVal>());
    addOutput(v);
    v->as<PipelineVal>()->setStage(this);
  }
  for (auto v : input_vals) {
    NVF_ERROR(v->isA<PipelineVal>());
    addInput(v);
    v->as<PipelineVal>()->setStage(this);
  }

  addDataAttribute(descriptor);
}

std::string PipelineStage::toString(int indent_size) const {
  std::stringstream ss;
  ss << "PipelineStage representing Stage " << descriptor()->unique_id << ".";
  ss << "Inputs={";
  for (auto input : inputs()) {
    ss << input->as<PipelineVal>()->getOriginalVal()->toString(indent_size);
    ss << ", ";
  }
  ss << "}. Outputs={";
  for (auto output : outputs()) {
    ss << output->as<PipelineVal>()->getOriginalVal()->toString(indent_size);
    ss << ", ";
  }
  ss << "}.";
  return ss.str();
}

std::string PipelineStage::toInlineString(int indent_size) const {
  return toString(indent_size);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PipelineStage)

bool PipelineStage::sameAs(const Statement* other) const {
  if (!Expr::sameAs(other)) {
    return false;
  }
  return descriptor() == other->as<PipelineStage>()->descriptor();
}

PipelineCommunication::PipelineCommunication(
    IrBuilderPasskey passkey,
    Val* in,
    Val* out)
    : Expr(passkey, serde::ExprType::PipelineCommunication) {
  NVF_ERROR(
      passkey.ir_container_ ? passkey.ir_container_->isA<Pipeline>() : false,
      "IR type only valid for Pipeline container.");
  NVF_ERROR(
      in->isA<PipelineVal>() && out->isA<PipelineVal>(),
      "I/O must be PipelineVal IRs");
  addOutput(out);
  addInput(in);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PipelineCommunication)

std::string PipelineCommunication::toString(int indent_size) const {
  std::stringstream ss;
  ss << "PipelineCommunication that transfers " << in() << " to " << out();
  return ss.str();
}

std::string PipelineCommunication::toInlineString(int indent_size) const {
  return toString(indent_size);
}

PipelineVal::PipelineVal(IrBuilderPasskey passkey, Val* val)
    : Val(passkey, ValType::PipelineVal, val->dtype()), original_val_(val) {
  NVF_ERROR(
      passkey.ir_container_->isA<Pipeline>(),
      "IR type only valid for Pipeline container.");
}

PipelineVal::PipelineVal(const PipelineVal* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      original_val_(src->original_val_),
      stage_(src->stage_) {}

PipelineVal::PipelineVal(
    IrContainer* container,
    IrBuilderPasskey passkey,
    const serde::Value* buffer,
    const serde::PipelineVal* data)
    : PipelineVal(passkey, container->getVal<Val>(data->value())) {}

bool PipelineVal::sameAs(const Statement* other) const {
  if (!Val::sameAs(other)) {
    return false;
  }
  const auto other_aggregate_val = other->as<PipelineVal>();
  return original_val_->sameAs(other_aggregate_val->original_val_) &&
      stage_->sameAs(other_aggregate_val->stage_);
}

NVFUSER_DEFINE_CLONE(PipelineVal)

std::string PipelineVal::toString(int indent_size) const {
  std::stringstream ss;
  ss << "PipelineVal representing Val "
     << getOriginalVal()->toString(indent_size);
  if (getStage() != nullptr) {
    ss << " on stage " << getStage()->descriptor()->unique_id;
  }
  return ss.str();
}

std::string PipelineVal::toInlineString(int indent_size) const {
  return toString(indent_size);
}

std::pair<serde::ValueData, flatbuffers::Offset<void>> PipelineVal::
    serializeData(
        const IrSerde& container,
        flatbuffers::FlatBufferBuilder& builder) const {
  flatbuffers::Offset<serde::PipelineVal> data = serde::CreatePipelineVal(
      builder, container.map(original_val_), container.map(stage_));
  return {serde::ValueData::PipelineVal, data.Union()};
}

void PipelineVal::deserializeExpr(
    IrContainer* container,
    const serde::Value* buffer) {
  NVF_ERROR(container != nullptr, "IrContainer is nullptr.");
  NVF_ERROR(buffer != nullptr, "serde::Value is nullptr.");
  const serde::PipelineVal* data = buffer->data_as_PipelineVal();
  NVF_ERROR(data != nullptr);
  Val::deserializeExpr(container, buffer);
  stage_ = container->getExpr<PipelineStage>(data->stage_expr());
}

} // namespace nvfuser
