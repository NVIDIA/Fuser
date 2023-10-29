// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <serde/expr_builder.h>
#include <serde/polymorphic_value.h>
#include <serde/utils.h>
#include <type.h>

namespace nvfuser::serde {

ExpressionBuilder::ExpressionBuilder(kir::Kernel* kernel) : kernel_(kernel) {
  operation_stack_ = gatherSymbolicValues(kernel_);
}

void ExpressionBuilder::deserialize(const NaiveValueGenerator* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::NaiveValueGenerator is nullptr.");
  for (auto inst : *buffer->instructions()) {
    deserialize(inst);
  }
}

void ExpressionBuilder::deserialize(const Instruction* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::Instruction is nullptr.");
  auto exists = [&](size_t idx) { return idx < operation_stack_.size(); };

  FusionGuard fg(kernel_);
  switch (buffer->data_type()) {
    case serde::InstructionData_Symbolic: {
      auto data = buffer->data_as_Symbolic();
      NVF_ERROR(data != nullptr, "serde::Symbolic is nullptr.")
      NVF_ERROR((size_t)data->out() < operation_stack_.size());
      break;
    }
    case serde::InstructionData_NamedScalar: {
      auto data = buffer->data_as_NamedScalar();
      NVF_ERROR(data != nullptr, "serde::Scalar is nullptr.")
      if (!exists(data->out())) {
        auto ns = IrBuilder::create<nvfuser::NamedScalar>(
            data->name()->str(), nvfuser::DataType::Index);
        operation_stack_.push_back(ns);
      }
      break;
    }
    case serde::InstructionData_Scalar: {
      auto data = buffer->data_as_Scalar();
      NVF_ERROR(data != nullptr, "serde::Scalar is nullptr.")
      if (!exists(data->out())) {
        auto int_val = IrBuilder::create<nvfuser::Val>(
            data->long_value(), nvfuser::DataType::Index);
        operation_stack_.push_back(int_val);
      }
      break;
    }
    case serde::InstructionData_UnaryOp: {
      auto data = buffer->data_as_UnaryOp();
      NVF_ERROR(data != nullptr, "serde::UnaryOp is nullptr.")
      if (!exists(data->out())) {
        auto uop = buildUnaryOp(data);
        operation_stack_.push_back(uop);
      }
      break;
    }
    case serde::InstructionData_BinaryOp: {
      auto data = buffer->data_as_BinaryOp();
      NVF_ERROR(data != nullptr, "serde::BinaryOp is nullptr.")
      if (!exists(data->out())) {
        auto bop = buildBinaryOp(data);
        operation_stack_.push_back(bop);
      }
      break;
    }
    case serde::InstructionData_IterDomain: {
      auto data = buffer->data_as_IterDomain();
      NVF_ERROR(data != nullptr, "serde::IterDomain is nullptr.")
      if (!exists(data->out())) {
        operation_stack_.push_back(buildIterDomain(data));
      }
      break;
    }
    case serde::InstructionData_GetAttr: {
      auto data = buffer->data_as_GetAttr();
      NVF_ERROR(data != nullptr, "serde::GetAttr is nullptr.")
      if (!exists(data->out())) {
        auto aop = IrBuilder::getAttrExpr(
            operation_stack_.at(data->struct_()), data->attr()->str());
        operation_stack_.push_back(aop);
      }
      break;
    }
    case serde::InstructionData_GetItem: {
      auto data = buffer->data_as_GetItem();
      NVF_ERROR(data != nullptr, "serde::GetItem is nullptr.")
      if (!exists(data->out())) {
        auto iop = IrBuilder::getItemExpr(
            operation_stack_.at(data->array()),
            operation_stack_.at(data->index()));
        operation_stack_.push_back(iop);
      }
      break;
    }
    case serde::InstructionData_GetMetaData: {
      auto data = buffer->data_as_GetMetaData();
      NVF_ERROR(data != nullptr, "serde::GetMetaData is nullptr.")
      if (!exists(data->out())) {
        auto val = operation_stack_.at(data->in());
        auto mop = kernel_->metadataOf(val);
        operation_stack_.push_back(mop);
      }
      break;
    }
    case serde::InstructionData_Merge: {
      auto data = buffer->data_as_Merge();
      NVF_ERROR(data != nullptr, "serde::Merge is nullptr.")
      if (!exists(data->out())) {
        auto inner = operation_stack_.at(data->inner());
        auto outer = operation_stack_.at(data->outer());
        NVF_ERROR(inner->isA<nvfuser::IterDomain>());
        NVF_ERROR(outer->isA<nvfuser::IterDomain>());

        auto merged_id = nvfuser::IterDomain::merge(
            inner->as<nvfuser::IterDomain>(), outer->as<nvfuser::IterDomain>());
        operation_stack_.push_back(merged_id);
      }
      break;
    }
    case serde::InstructionData_Split: {
      auto data = buffer->data_as_Split();
      NVF_ERROR(data != nullptr, "serde::Split is nullptr.")
      if (!exists(data->inner()) || !exists(data->outer())) {
        auto in = operation_stack_.at(data->in());
        NVF_ERROR(in->isA<nvfuser::IterDomain>());

        auto factor = operation_stack_.at(data->factor());
        auto split_ids = nvfuser::IterDomain::split(
            in->as<nvfuser::IterDomain>(),
            factor,
            data->inner_split(),
            data->trim_out_of_bounds());
        operation_stack_.push_back(split_ids.first);
        operation_stack_.push_back(split_ids.second);
      }
      break;
    }
    case serde::InstructionData_Resize: {
      auto data = buffer->data_as_Resize();
      NVF_ERROR(data != nullptr, "serde::Resize is nullptr.")
      if (!exists(data->out())) {
        auto in = operation_stack_.at(data->in());
        NVF_ERROR(in->isA<nvfuser::IterDomain>());

        auto left_expansion = operation_stack_.at(data->left_expansion());
        auto right_expansion = operation_stack_.at(data->right_expansion());

        // TODO add mark_as_rfactor attribute
        // TODO add optional itertype attribute
        auto resized_id = nvfuser::IterDomain::resize(
            in->as<nvfuser::IterDomain>(),
            left_expansion,
            right_expansion,
            false /* mark_as_rfactor */);
        operation_stack_.push_back(resized_id);
      }
      break;
    }
    case serde::InstructionData_Swizzle2D: {
      auto data = buffer->data_as_Swizzle2D();
      NVF_ERROR(data != nullptr, "serde::Swizzle2D is nullptr.")
      if (!exists(data->out_x()) || !exists(data->out_y())) {
        auto in_x = operation_stack_.at(data->in_x());
        auto in_y = operation_stack_.at(data->in_y());
        NVF_ERROR(in_x->isA<nvfuser::IterDomain>());
        NVF_ERROR(in_y->isA<nvfuser::IterDomain>());

        auto swizzle_ids = nvfuser::IterDomain::swizzle(
            static_cast<nvfuser::Swizzle2DType>(data->swizzle_type()),
            in_x->as<nvfuser::IterDomain>(),
            in_y->as<nvfuser::IterDomain>(),
            static_cast<nvfuser::SwizzleMode>(data->swizzle_mode()));
        operation_stack_.push_back(swizzle_ids.first);
        operation_stack_.push_back(swizzle_ids.second);
      }
      break;
    }
    default: {
      NVF_ERROR(false, "Unsupported instruction during deserialization.");
    }
  }
}

Val* ExpressionBuilder::buildUnaryOp(const UnaryOp* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::UnaryOp is nullptr.")
  switch (buffer->unary_type()) {
    case serde::UnaryOpType_Cast:
      return castOp(
          mapToDtypeStruct(buffer->data_type()),
          operation_stack_.at(buffer->src0()));
    case serde::UnaryOpType_Neg:
      return neg(operation_stack_.at(buffer->src0()));
    default:
      NVF_ERROR(false, "Unsupported binary operation.\t");
      return nullptr;
  }
}

Val* ExpressionBuilder::buildBinaryOp(const BinaryOp* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::BinaryOp is nullptr.")
  switch (buffer->binary_type()) {
    case serde::BinaryOpType_Add:
      return add(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_CeilDiv:
      return ceilDiv(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Div:
      return div(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Mod:
      return mod(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Mul:
      return mul(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    case serde::BinaryOpType_Sub:
      return sub(
          operation_stack_.at(buffer->src0()),
          operation_stack_.at(buffer->src1()));
    default:
      NVF_ERROR(false, "Unsupported binary operation.\t");
      return nullptr;
  }
}

nvfuser::IterDomain* ExpressionBuilder::buildIterDomain(
    const IterDomain* buffer) {
  nvfuser::IterDomainBuilder builder(
      operation_stack_.at(buffer->start()),
      operation_stack_.at(buffer->extent()));
  builder.stop_offset(operation_stack_.at(buffer->stop_offset()));
  builder.iter_type(static_cast<nvfuser::IterType>(buffer->iter_type()));

  if (buffer->expanded_extent() > 0) {
    builder.expanded_extent(operation_stack_.at(buffer->expanded_extent()));
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
        auto val = operation_stack_.at(fb_id);
        NVF_ERROR(val->isA<nvfuser::IterDomain>());
        new_root.push_back(val->as<nvfuser::IterDomain>());
      }
    }

    std::vector<nvfuser::IterDomain*> new_rfactor;
    if (buffer->tv()->rfactor() != nullptr) {
      for (auto fb_id : *buffer->tv()->rfactor()) {
        auto val = operation_stack_.at(fb_id);
        NVF_ERROR(val->isA<nvfuser::IterDomain>());
        new_rfactor.push_back(val->as<nvfuser::IterDomain>());
      }
    }

    std::vector<nvfuser::IterDomain*> new_allocation;
    if (buffer->tv()->allocate() != nullptr) {
      for (auto fb_id : *buffer->tv()->allocate()) {
        auto val = operation_stack_.at(fb_id);
        NVF_ERROR(val->isA<nvfuser::IterDomain>());
        new_allocation.push_back(val->as<nvfuser::IterDomain>());
      }
    }

    std::vector<nvfuser::IterDomain*> new_leaf;
    if (buffer->tv()->leaf() != nullptr) {
      for (auto fb_id : *buffer->tv()->leaf()) {
        auto val = operation_stack_.at(fb_id);
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
      shape.push_back(operation_stack_.at(fb_id));
    }

    auto node = IrBuilder::create<kir::Allocate>(
        buffer_tv, buffer_tv->getMemoryType(), shape, buffer->zero_init());

    results.push_back(node);
  }
  return results;
}

} // namespace nvfuser::serde
