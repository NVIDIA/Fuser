// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <serde/expr_serializer.h>
#include <serde/polymorphic_value.h>
#include <serde/utils.h>
#include <type.h>

namespace nvfuser::serde {

namespace {

// Bind the value to all_values_ container
void bind(std::unordered_set<nvfuser::Val*>& container, nvfuser::Val* v) {
  container.insert(v);
}

void bind(
    std::unordered_set<nvfuser::Val*>& container,
    nvfuser::IterDomain* id) {
  bind(container, id->start());
  bind(container, id->extent());
  if (id->hasExpandedExtent()) {
    bind(container, id->expandedExtent());
  }
  bind(container, id->stopOffset());
}

// Bind the iterDomain's extent for the given domain
void bindDomain(
    std::unordered_set<nvfuser::Val*>& container,
    const std::vector<nvfuser::IterDomain*>& domain) {
  for (auto d : domain) {
    bind(container, d->as<Val>());
    bind(container, d);
  }
}

// 1. Generate extents for IterDomains that compose root domain
// 2. Create new extents using split, merge, reorder operations for rfactor,
// allocation, and leaf domains
void bind(
    std::unordered_set<nvfuser::Val*>& container,
    nvfuser::TensorView* tv) {
  bindDomain(container, tv->getRootDomain());
  bindDomain(container, tv->getRFactorDomain());
  bindDomain(container, tv->getAllocationDomain());
  bindDomain(container, tv->getLeafDomain());
}

// Gather all values contained in kir::Allocate nodes
std::vector<nvfuser::Val*> gatherAllValues(
    const std::vector<const kir::Allocate*>& allocations) {
  std::unordered_set<nvfuser::Val*> all_values;
  for (auto allocate : allocations) {
    if (auto tv = dynamic_cast<nvfuser::TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
      for (auto v : allocate->shape()) {
        bind(all_values, v);
      }
    }
  }
  return std::vector(all_values.begin(), all_values.end());
}

std::vector<nvfuser::Val*> getAttributes(nvfuser::Val* val) {
  if (!val->definition()) {
    return std::vector<nvfuser::Val*>();
  }
  std::vector<nvfuser::Val*> data_attributes;
  for (auto a : val->definition()->attributes()) {
    if (a->isVal()) {
      data_attributes.push_back(a->asVal());
    }
  }
  return data_attributes;
}

std::vector<nvfuser::Val*> getImmediateProducers(nvfuser::Val* val) {
  if (val->definition() != nullptr) {
    return val->definition()->inputs();
  } else if (auto id = dynamic_cast<nvfuser::IterDomain*>(val)) {
    std::vector<nvfuser::Val*> inputs;
    inputs.push_back(id->start());
    inputs.push_back(id->extent());
    if (id->hasExpandedExtent()) {
      inputs.push_back(id->expandedExtent());
    }
    inputs.push_back(id->stopOffset());
    return inputs;
  } else {
    return std::vector<nvfuser::Val*>();
  }
}

std::vector<nvfuser::Val*> getConsumers(nvfuser::Val* val) {
  return (val->definition()) ? val->definition()->outputs()
                             : std::vector<nvfuser::Val*>({val});
}

//! IR-Generic utility, collects all the producers required for the
//!  given list of IR values and returns them along with the original
//!  list in topological order.
std::vector<nvfuser::Val*> makeSortedEvaluationList(
    const std::vector<const kir::Allocate*>& allocations) {
  std::unordered_set<nvfuser::Val*> visited;
  std::vector<nvfuser::Val*> sorted;

  // Topological Sort
  auto to_sort = gatherAllValues(allocations);
  while (!to_sort.empty()) {
    auto top_val = to_sort.back();
    if (visited.count(top_val)) {
      to_sort.pop_back();
    } else {
      bool ready_to_pop = true;
      for (auto producer : getImmediateProducers(top_val)) {
        if (!visited.count(producer)) {
          ready_to_pop = false;
          to_sort.push_back(producer);
        }
      }

      if (ready_to_pop) {
        // Some definition operations generate multiple outputs. e.g., split and
        // resize. We add sibling outputs together in the sorted list.
        for (auto consumer : getConsumers(top_val)) {
          visited.insert(consumer);
          sorted.push_back(consumer);
        }
      }
    }
  }
  return sorted;
}

} // namespace

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeAttribute(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::Val* val) {
  operation_stack_.emplace(val, operation_stack_.size());
  auto sv_fb =
      CreateSymbolicDirect(builder, val->name(), val->toString().c_str());
  return CreateInstruction(
      builder, serde::InstructionData_Symbolic, sv_fb.Union());
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeUnaryOp(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::UnaryOp* uop) {
  DataType dtype = (uop->getUnaryOpType() == nvfuser::UnaryOpType::Cast)
      ? mapToSerdeDtype(uop->out()->getDataType().value())
      : serde::DataType_None;
  auto uop_fb = CreateUnaryOpDirect(
      builder,
      mapToSerdeUnaryOp(uop->getUnaryOpType()),
      dtype,
      operation_stack_.at(uop->inputs().front()),
      (int64_t)operation_stack_.size(),
      uop->toString().c_str());
  return CreateInstruction(
      builder, serde::InstructionData_UnaryOp, uop_fb.Union());
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeBinaryOp(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::BinaryOp* bop) {
  auto bop_fb = CreateBinaryOpDirect(
      builder,
      mapToSerdeBinaryOp(bop->getBinaryOpType()),
      operation_stack_.at(bop->inputs().front()),
      operation_stack_.at(bop->inputs().back()),
      (int64_t)operation_stack_.size(),
      bop->toString().c_str());
  return CreateInstruction(
      builder, serde::InstructionData_BinaryOp, bop_fb.Union());
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeMerge(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::Merge* merge) {
  auto merge_fb = CreateMerge(
      builder,
      operation_stack_.at(merge->inner()),
      operation_stack_.at(merge->outer()),
      (int64_t)operation_stack_.size());
  return CreateInstruction(
      builder, serde::InstructionData_Merge, merge_fb.Union());
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeGetAttr(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::GetAttr* attr) {
  auto attr_fb = CreateGetAttr(
      builder,
      operation_stack_.at(attr->struct_()),
      builder.CreateString(attr->attr()),
      (int64_t)operation_stack_.size());
  return CreateInstruction(
      builder, serde::InstructionData_GetAttr, attr_fb.Union());
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeGetItem(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::GetItem* item) {
  auto item_fb = CreateGetItem(
      builder,
      operation_stack_.at(item->array()),
      operation_stack_.at(item->index()),
      (int64_t)operation_stack_.size());
  return CreateInstruction(
      builder, serde::InstructionData_GetItem, item_fb.Union());
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeGetMetaData(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::GetMetaData* metadata) {
  auto metadata_fb = CreateGetMetaData(
      builder,
      operation_stack_.at(metadata->in()),
      (int64_t)operation_stack_.size());
  return CreateInstruction(
      builder, serde::InstructionData_GetMetaData, metadata_fb.Union());
}

std::array<flatbuffers::Offset<Instruction>, 3> ExpressionSerializer::
    serializeResize(
        flatbuffers::FlatBufferBuilder& builder,
        nvfuser::Resize* resize) {
  auto left_expand_inst = serializeAttribute(builder, resize->leftExpand());
  auto right_expand_inst = serializeAttribute(builder, resize->leftExpand());
  auto resize_fb = CreateResize(
      builder,
      operation_stack_.at(resize->in()),
      operation_stack_.at(resize->leftExpand()),
      operation_stack_.at(resize->rightExpand()),
      (int64_t)operation_stack_.size());
  auto resize_inst = CreateInstruction(
      builder, serde::InstructionData_Resize, resize_fb.Union());
  return {left_expand_inst, right_expand_inst, resize_inst};
}

std::array<flatbuffers::Offset<Instruction>, 2> ExpressionSerializer::
    serializeSplit(
        flatbuffers::FlatBufferBuilder& builder,
        nvfuser::Split* split) {
  auto factor_inst = serializeAttribute(builder, split->factor());
  auto split_fb = CreateSplit(
      builder,
      operation_stack_.at(split->in()),
      operation_stack_.at(split->factor()),
      (int64_t)operation_stack_.size(),
      (int64_t)operation_stack_.size() + 1);
  auto split_inst = CreateInstruction(
      builder, serde::InstructionData_Split, split_fb.Union());
  return {factor_inst, split_inst};
}

flatbuffers::Offset<Instruction> ExpressionSerializer::serializeSwizzle2D(
    flatbuffers::FlatBufferBuilder& builder,
    nvfuser::Swizzle2D* swizzle) {
  auto swizzle_fb = CreateSwizzle2D(
      builder,
      operation_stack_.at(swizzle->inX()),
      operation_stack_.at(swizzle->inY()),
      castEnumToUnderlyingType(swizzle->swizzleType()),
      castEnumToUnderlyingType(swizzle->swizzleMode()),
      (int64_t)operation_stack_.size(),
      (int64_t)operation_stack_.size() + 1);
  return CreateInstruction(
      builder, serde::InstructionData_Swizzle2D, swizzle_fb.Union());
}

void ExpressionSerializer::processAllocations(
    const std::vector<const kir::Allocate*>& allocations) {
  // Built in deterministic order from kernel inputs
  symbolic_values_ = gatherSymbolicValues(kernel_);

  // 1) All kir::Allocate nodes are contained in FusionExecutor::KernelSummary
  // 2) Sort all values by dependency order
  // 3) Divide values into NamedScalar, Int, Symbolic, and Derived values
  for (auto v : makeSortedEvaluationList(allocations)) {
    if (v->definition() == nullptr) {
      if (auto ns = dynamic_cast<nvfuser::NamedScalar*>(v)) {
        insertUniqueItem(named_scalar_values_, ns);
      } else if (v->isConstInt()) {
        insertUniqueItem(const_int_values_, v);
      } else if (auto id = dynamic_cast<nvfuser::IterDomain*>(v)) {
        insertUniqueItem(derived_values_, id);
      } else {
        NVF_ERROR(
            !insertUniqueItem(symbolic_values_, v),
            "Expect all symbolic values to come from kernel inputs.");
      }
    } else {
      insertUniqueItem(derived_values_, v);
    }
  }
}

flatbuffers::Offset<NaiveValueGenerator> ExpressionSerializer::
    serializeNaiveValueGenerator(
        flatbuffers::FlatBufferBuilder& builder,
        const std::vector<const kir::Allocate*>& allocations) {
  // Short Circuit: Return empty offset if there aren't any kir::Allocate nodes
  if (allocations.empty()) {
    return 0;
  }

  processAllocations(allocations);

  // 4) Serialize NaiveValueGenerator by converting each NvFuser value of into
  // an instruction.

  using fb_instruction = flatbuffers::Offset<Instruction>;
  std::vector<fb_instruction> instructions_fb;

  for (auto& val : symbolic_values_) {
    auto sv_fb = CreateSymbolicDirect(
        builder,
        val->name(),
        val->toString().c_str(),
        (int64_t)operation_stack_.size());
    auto inst = CreateInstruction(
        builder, serde::InstructionData_Symbolic, sv_fb.Union());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(val, operation_stack_.size());
  }

  for (const auto& ns : named_scalar_values_) {
    auto ns_fb = CreateNamedScalarDirect(
        builder, ns->name().c_str(), (int64_t)operation_stack_.size());
    auto inst = CreateInstruction(
        builder, serde::InstructionData_NamedScalar, ns_fb.Union());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(ns, operation_stack_.size());
  }

  for (const auto& int_val : const_int_values_) {
    auto val_fb = serializeScalar(
        builder,
        int_val->evaluateInt(),
        nvfuser::DataType::Int,
        (int64_t)operation_stack_.size());
    auto inst = CreateInstruction(
        builder, serde::InstructionData_Scalar, val_fb.Union());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(int_val, operation_stack_.size());
  }

  while (!derived_values_.empty()) {
    auto& val = derived_values_.front();
    auto def = val->definition();
    derived_values_.pop_front();

    if (operation_stack_.count(val)) {
      continue;
    }

    if (def == nullptr && val->isA<nvfuser::IterDomain>()) {
      auto id = dynamic_cast<nvfuser::IterDomain*>(val);
      auto fb_id = serialize(builder, id);
      auto fb_inst = CreateInstruction(
          builder, serde::InstructionData_IterDomain, fb_id.Union());
      instructions_fb.push_back(fb_inst);
      operation_stack_.emplace(val, operation_stack_.size());
      continue;
    }

    NVF_ERROR(def != nullptr, "Expected definition with derived value.");
    if (auto uop = dynamic_cast<nvfuser::UnaryOp*>(def)) {
      instructions_fb.push_back(serializeUnaryOp(builder, uop));
      operation_stack_.emplace(val, operation_stack_.size());

    } else if (auto bop = dynamic_cast<nvfuser::BinaryOp*>(def)) {
      instructions_fb.push_back(serializeBinaryOp(builder, bop));
      operation_stack_.emplace(val, operation_stack_.size());

    } else if (auto mop = dynamic_cast<nvfuser::Merge*>(def)) {
      instructions_fb.push_back(serializeMerge(builder, mop));
      operation_stack_.emplace(val, operation_stack_.size());

    } else if (auto sop = dynamic_cast<nvfuser::Split*>(def)) {
      auto inst = serializeSplit(builder, sop);
      for (auto i : inst) {
        instructions_fb.push_back(i);
      }
      operation_stack_.emplace(val, operation_stack_.size());

      auto next_val = derived_values_.front();
      NVF_ERROR(next_val->definition() == def);
      operation_stack_.emplace(next_val, operation_stack_.size());
      derived_values_.pop_front();

    } else if (auto swop = dynamic_cast<nvfuser::Swizzle2D*>(def)) {
      instructions_fb.push_back(serializeSwizzle2D(builder, swop));
      operation_stack_.emplace(val, operation_stack_.size());

      auto next_val = derived_values_.front();
      NVF_ERROR(next_val->definition() == def);
      operation_stack_.emplace(next_val, operation_stack_.size());
      derived_values_.pop_front();

    } else if (auto rop = dynamic_cast<nvfuser::Resize*>(def)) {
      auto inst = serializeResize(builder, rop);
      for (auto i : inst) {
        instructions_fb.push_back(i);
      }
      operation_stack_.emplace(val, operation_stack_.size());

    } else if (auto mop = dynamic_cast<nvfuser::GetMetaData*>(def)) {
      instructions_fb.push_back(serializeGetMetaData(builder, mop));
      operation_stack_.emplace(val, operation_stack_.size());

    } else if (auto iop = dynamic_cast<nvfuser::GetItem*>(def)) {
      instructions_fb.push_back(serializeGetItem(builder, iop));
      operation_stack_.emplace(val, operation_stack_.size());

    } else if (auto aop = dynamic_cast<nvfuser::GetAttr*>(def)) {
      instructions_fb.push_back(serializeGetAttr(builder, aop));
      operation_stack_.emplace(val, operation_stack_.size());

    } else {
      NVF_ERROR(false, "Serialization unknown expression.\t", def->toString());
    }
  }
  return CreateNaiveValueGeneratorDirect(builder, &instructions_fb);
}

std::vector<flatbuffers::Offset<AllocateBuffer>> ExpressionSerializer::
    serializeAllocations(
        flatbuffers::FlatBufferBuilder& builder,
        const std::vector<const kir::Allocate*>& allocations) {
  using fb_allocate = flatbuffers::Offset<AllocateBuffer>;
  std::vector<fb_allocate> fb_allocations;

  for (auto alloc : allocations) {
    auto alloc_buffer_tv = alloc->buffer()->as<nvfuser::TensorView>();
    NVF_ERROR(alloc_buffer_tv != nullptr);
    auto fb_alloc = CreateAllocateBuffer(
        builder,
        serialize(builder, alloc_buffer_tv),
        serialize(builder, alloc->shape()),
        alloc->zeroInit());
    fb_allocations.push_back(fb_alloc);
  }
  return fb_allocations;
}

// TODO create separate functions for TensorDomain and IterDomain
flatbuffers::Offset<flatbuffers::Vector<int64_t>> ExpressionSerializer::
    serialize(
        flatbuffers::FlatBufferBuilder& builder,
        std::vector<nvfuser::Val*> domain) {
  std::vector<long> fb_domain;
  for (auto val : domain) {
    NVF_ERROR(
        operation_stack_.count(val),
        "Missing value in NaiveValueGenerator stack.\t",
        val->toString());
    fb_domain.push_back(operation_stack_.at(val));
  }
  return builder.CreateVector(fb_domain);
}

flatbuffers::Offset<IterDomain> ExpressionSerializer::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::IterDomain* id) {
  NVF_ERROR(
      operation_stack_.count(id->start()),
      "Missing iterDomain extent in NaiveValueGenerator stack.\t",
      id->start()->toString());

  NVF_ERROR(
      operation_stack_.count(id->extent()),
      "Missing iterDomain extent in NaiveValueGenerator stack.\t",
      id->extent()->toString());

  NVF_ERROR(
      !id->hasExpandedExtent() || operation_stack_.count(id->expandedExtent()),
      "Missing iterDomain expandedExtent in NaiveValueGenerator stack.\t",
      id->expandedExtent()->toString());

  NVF_ERROR(
      operation_stack_.count(id->stopOffset()),
      "Missing iterDomain stopOffset in NaiveValueGenerator stack.\t",
      id->stopOffset()->toString());

  return CreateIterDomain(
      builder,
      operation_stack_.at(id->start()),
      operation_stack_.at(id->extent()),
      id->hasExpandedExtent() ? operation_stack_.at(id->expandedExtent()) : -1,
      operation_stack_.at(id->stopOffset()),
      (int64_t)operation_stack_.size(),
      castEnumToUnderlyingType(id->getParallelType()),
      castEnumToUnderlyingType(id->getIterType()),
      id->isRFactorProduct(),
      id->hasPaddingToMultipleOfWarp(),
      id->getMaybeSizeAfterPadding().value_or(0),
      id->isMmaSwizzled());
}

// TODO create separate functions for TensorDomain and IterDomain
flatbuffers::Offset<SymbolicTensor> ExpressionSerializer::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::TensorView* tv) {
  std::vector<int64_t> fb_root_domain;
  for (auto id : tv->getRootDomain()) {
    fb_root_domain.push_back(operation_stack_.at(id));
  }
  auto root_domain_fb = CreateDomainDirect(builder, &fb_root_domain);

  flatbuffers::Offset<Domain> rfactor_domain_fb = 0;
  if (tv->hasRFactor()) {
    std::vector<int64_t> fb_rfactor_domain;
    for (auto id : tv->getRFactorDomain()) {
      fb_rfactor_domain.push_back(operation_stack_.at(id));
    }
    rfactor_domain_fb = CreateDomainDirect(builder, &fb_rfactor_domain);
  }

  flatbuffers::Offset<Domain> allocation_domain_fb = 0;
  if (tv->hasAllocation()) {
    std::vector<int64_t> fb_allocation_domain;
    for (auto id : tv->getAllocationDomain()) {
      fb_allocation_domain.push_back(operation_stack_.at(id));
    }
    allocation_domain_fb = CreateDomainDirect(builder, &fb_allocation_domain);
  }

  std::vector<int64_t> fb_leaf_domain;
  for (auto id : tv->getLeafDomain()) {
    fb_leaf_domain.push_back(operation_stack_.at(id));
  }
  auto leaf_domain_fb = CreateDomainDirect(builder, &fb_leaf_domain);

  SymbolicTensorBuilder tensor_builder(builder);
  tensor_builder.add_dtype(mapToSerdeDtype(tv->getDataType().value()));
  tensor_builder.add_root(root_domain_fb);
  tensor_builder.add_rfactor(rfactor_domain_fb);
  tensor_builder.add_allocate(allocation_domain_fb);
  tensor_builder.add_leaf(leaf_domain_fb);
  return tensor_builder.Finish();
}

} // namespace nvfuser::serde
