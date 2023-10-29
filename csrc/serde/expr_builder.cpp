// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <serde/expr_builder.h>
#include <serde/expr_utils.h>
#include <serde/polymorphic_value.h>
#include <type.h>

namespace nvfuser::serde {

void ExpressionBuilder::registerAllParsers() {
  auto deserializeBinaryOp = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_BinaryOp();
    NVF_ERROR(data != nullptr, "serde::BinaryOp is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto bop = buildBinaryOp(data);
    operation_stack_.push_back(bop);
  };
  registerParser(InstructionData_BinaryOp, deserializeBinaryOp);

  auto deserializeGetAttr = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_GetAttr();
    NVF_ERROR(data != nullptr, "serde::GetAttr is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto expr =
        IrBuilder::getAttrExpr(retrieve(data->struct_()), data->attr()->str());
    operation_stack_.push_back(expr);
  };
  registerParser(InstructionData_GetAttr, deserializeGetAttr);

  auto deserializeGetItem = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_GetItem();
    NVF_ERROR(data != nullptr, "serde::GetItem is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto item = IrBuilder::getItemExpr(
        retrieve(data->array()), retrieve(data->index()));
    operation_stack_.push_back(item);
  };
  registerParser(InstructionData_GetItem, deserializeGetItem);

  auto deserializeGetMetaData = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_GetMetaData();
    NVF_ERROR(data != nullptr, "serde::GetMetaData is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto metadata = kernel_->metadataOf(retrieve(data->in()));
    operation_stack_.push_back(metadata);
  };
  registerParser(InstructionData_GetMetaData, deserializeGetMetaData);

  auto deserializeIterDomain = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_IterDomain();
    NVF_ERROR(data != nullptr, "serde::IterDomain is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto id = buildIterDomain(data);
    operation_stack_.push_back(id);
  };
  registerParser(InstructionData_IterDomain, deserializeIterDomain);

  auto deserializeMerge = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_Merge();
    NVF_ERROR(data != nullptr, "serde::Merge is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto inner = retrieve(data->inner());
    auto outer = retrieve(data->outer());
    NVF_ERROR(inner->isA<nvfuser::IterDomain>());
    NVF_ERROR(outer->isA<nvfuser::IterDomain>());

    auto merged_id = nvfuser::IterDomain::merge(
        inner->as<nvfuser::IterDomain>(), outer->as<nvfuser::IterDomain>());
    operation_stack_.push_back(merged_id);
  };
  registerParser(InstructionData_Merge, deserializeMerge);

  auto deserializeNamedScalar = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_NamedScalar();
    NVF_ERROR(data != nullptr, "serde::Scalar is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto named_scalar = IrBuilder::create<nvfuser::NamedScalar>(
        data->name()->str(), nvfuser::DataType::Index);
    operation_stack_.push_back(named_scalar);
  };
  registerParser(InstructionData_NamedScalar, deserializeNamedScalar);

  auto deserializeResize = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_Resize();
    NVF_ERROR(data != nullptr, "serde::Resize is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto in = retrieve(data->in());
    NVF_ERROR(in->isA<nvfuser::IterDomain>());

    auto left_expansion = retrieve(data->left_expansion());
    auto right_expansion = retrieve(data->right_expansion());

    // TODO add mark_as_rfactor attribute
    // TODO add optional itertype attribute
    auto resize = nvfuser::IterDomain::resize(
        in->as<nvfuser::IterDomain>(),
        left_expansion,
        right_expansion,
        false /* mark_as_rfactor */);
    operation_stack_.push_back(resize);
  };
  registerParser(InstructionData_Resize, deserializeResize);

  auto deserializeScalar = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_Scalar();
    NVF_ERROR(data != nullptr, "serde::Scalar is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto int_val = IrBuilder::create<nvfuser::Val>(
        data->long_value(), nvfuser::DataType::Index);
    operation_stack_.push_back(int_val);
  };
  registerParser(InstructionData_Scalar, deserializeScalar);

  auto deserializeSplit = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_Split();
    NVF_ERROR(data != nullptr, "serde::Split is nullptr.")
    if (exists(data->inner()) && exists(data->outer())) {
      return;
    }
    auto in = retrieve(data->in());
    NVF_ERROR(in->isA<nvfuser::IterDomain>());

    auto split_ids = nvfuser::IterDomain::split(
        in->as<nvfuser::IterDomain>(),
        retrieve(data->factor()),
        data->inner_split(),
        data->trim_out_of_bounds());
    operation_stack_.push_back(split_ids.first);
    operation_stack_.push_back(split_ids.second);
  };
  registerParser(InstructionData_Split, deserializeSplit);

  auto deserializeSwizzle2D = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_Swizzle2D();
    NVF_ERROR(data != nullptr, "serde::Swizzle2D is nullptr.")
    if (exists(data->out_x()) && exists(data->out_y())) {
      return;
    }
    auto in_x = retrieve(data->in_x());
    auto in_y = retrieve(data->in_y());
    NVF_ERROR(in_x->isA<nvfuser::IterDomain>());
    NVF_ERROR(in_y->isA<nvfuser::IterDomain>());

    auto swizzle_ids = nvfuser::IterDomain::swizzle(
        static_cast<nvfuser::Swizzle2DType>(data->swizzle_type()),
        in_x->as<nvfuser::IterDomain>(),
        in_y->as<nvfuser::IterDomain>(),
        static_cast<nvfuser::SwizzleMode>(data->swizzle_mode()));
    operation_stack_.push_back(swizzle_ids.first);
    operation_stack_.push_back(swizzle_ids.second);
  };
  registerParser(InstructionData_Swizzle2D, deserializeSwizzle2D);

  auto deserializeSymbolic = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_Symbolic();
    NVF_ERROR(data != nullptr, "serde::Symbolic is nullptr.")
    NVF_ERROR(data->out() < (int64_t)operation_stack_.size());
  };
  registerParser(InstructionData_Symbolic, deserializeSymbolic);

  auto deserializeUnaryOp = [&](const serde::Instruction* buffer) {
    auto data = buffer->data_as_UnaryOp();
    NVF_ERROR(data != nullptr, "serde::UnaryOp is nullptr.")
    if (exists(data->out())) {
      return;
    }
    auto uop = buildUnaryOp(data);
    operation_stack_.push_back(uop);
  };
  registerParser(InstructionData_UnaryOp, deserializeUnaryOp);
}

void ExpressionBuilder::deserialize(const NaiveValueGenerator* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::NaiveValueGenerator is nullptr.");
  FusionGuard fg(kernel_);
  for (auto inst : *buffer->instructions()) {
    parse(inst->data_type(), inst);
  }
}

Val* ExpressionBuilder::buildUnaryOp(const UnaryOp* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::UnaryOp is nullptr.")
  switch (buffer->unary_type()) {
    case serde::UnaryOpType_Cast:
      return castOp(
          mapToDtypeStruct(buffer->data_type()), retrieve(buffer->src0()));
    case serde::UnaryOpType_Neg:
      return neg(retrieve(buffer->src0()));
    default:
      NVF_ERROR(false, "Unsupported binary operation.\t");
      return nullptr;
  }
}

Val* ExpressionBuilder::buildBinaryOp(const BinaryOp* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::BinaryOp is nullptr.")
  switch (buffer->binary_type()) {
    case serde::BinaryOpType_Add:
      return add(retrieve(buffer->src0()), retrieve(buffer->src1()));
    case serde::BinaryOpType_CeilDiv:
      return ceilDiv(retrieve(buffer->src0()), retrieve(buffer->src1()));
    case serde::BinaryOpType_Div:
      return div(retrieve(buffer->src0()), retrieve(buffer->src1()));
    case serde::BinaryOpType_Mod:
      return mod(retrieve(buffer->src0()), retrieve(buffer->src1()));
    case serde::BinaryOpType_Mul:
      return mul(retrieve(buffer->src0()), retrieve(buffer->src1()));
    case serde::BinaryOpType_Sub:
      return sub(retrieve(buffer->src0()), retrieve(buffer->src1()));
    default:
      NVF_ERROR(false, "Unsupported binary operation.\t");
      return nullptr;
  }
}

nvfuser::IterDomain* ExpressionBuilder::buildIterDomain(
    const IterDomain* buffer) {
  nvfuser::IterDomainBuilder builder(
      retrieve(buffer->start()), retrieve(buffer->extent()));
  builder.stop_offset(retrieve(buffer->stop_offset()));
  builder.iter_type(static_cast<nvfuser::IterType>(buffer->iter_type()));

  if (buffer->expanded_extent() > 0) {
    builder.expanded_extent(retrieve(buffer->expanded_extent()));
  }

  // Scheduler parameters
  builder.parallel_type(
      static_cast<nvfuser::ParallelType>(buffer->parallel_type()));
  builder.is_rfactor_domain(buffer->is_rfactor_domain());
  builder.is_padded_dimension(buffer->is_padded_dimension());
  builder.is_mma_swizzled(buffer->is_mma_swizzled());

  if (buffer->padded_to_size() != 0) {
    builder.padded_to_size(buffer->padded_to_size());
  }

  return builder.build();
}

std::vector<const kir::Allocate*> ExpressionBuilder::deserialize(
    const ExpressionBuilder::Allocations* buffers) {
  FusionGuard fg(kernel_);

  std::vector<const kir::Allocate*> results;
  for (auto buffer : *buffers) {
    std::vector<nvfuser::IterDomain*> new_root;
    if (buffer->tv()->root() != nullptr) {
      for (auto fb_id : *buffer->tv()->root()) {
        auto val = retrieve(fb_id);
        NVF_ERROR(val->isA<nvfuser::IterDomain>());
        new_root.push_back(val->as<nvfuser::IterDomain>());
      }
    }

    std::vector<nvfuser::IterDomain*> new_rfactor;
    if (buffer->tv()->rfactor() != nullptr) {
      for (auto fb_id : *buffer->tv()->rfactor()) {
        auto val = retrieve(fb_id);
        NVF_ERROR(val->isA<nvfuser::IterDomain>());
        new_rfactor.push_back(val->as<nvfuser::IterDomain>());
      }
    }

    std::vector<nvfuser::IterDomain*> new_allocation;
    if (buffer->tv()->allocate() != nullptr) {
      for (auto fb_id : *buffer->tv()->allocate()) {
        auto val = retrieve(fb_id);
        NVF_ERROR(val->isA<nvfuser::IterDomain>());
        new_allocation.push_back(val->as<nvfuser::IterDomain>());
      }
    }

    std::vector<nvfuser::IterDomain*> new_leaf;
    if (buffer->tv()->leaf() != nullptr) {
      for (auto fb_id : *buffer->tv()->leaf()) {
        auto val = retrieve(fb_id);
        NVF_ERROR(val->isA<nvfuser::IterDomain>());
        new_leaf.push_back(val->as<nvfuser::IterDomain>());
      }
    }

    const auto buffer_domain = IrBuilder::create<nvfuser::TensorDomain>(
        new_root, new_rfactor, new_allocation, new_leaf);

    const auto buffer_tv = IrBuilder::create<nvfuser::TensorView>(
        buffer_domain,
        mapToNvfuserDtype(buffer->tv()->dtype()),
        MemoryType::Global);

    std::vector<nvfuser::Val*> shape;
    for (auto fb_id : *buffer->shape()) {
      shape.push_back(retrieve(fb_id));
    }

    auto node = IrBuilder::create<kir::Allocate>(
        buffer_tv, buffer_tv->getMemoryType(), shape, buffer->zero_init());

    results.push_back(node);
  }
  return results;
}

} // namespace nvfuser::serde
