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

template <typename VALTYPE>
std::vector<VALTYPE*> getAttributes(VALTYPE* val) {
  if (!val->definition()) {
    return std::vector<VALTYPE*>();
  }
  std::vector<VALTYPE*> data_attributes;
  for (auto a : val->definition()->attributes()) {
    if (a->isVal()) {
      data_attributes.push_back(a->asVal());
    }
  }
  return data_attributes;
}

template <typename VALTYPE>
std::vector<VALTYPE*> getImmediateProducers(VALTYPE* val) {
  if (val->definition() != nullptr) {
    return val->definition()->inputs();
  } else if (auto id = dynamic_cast<nvfuser::IterDomain*>(val)) {
    std::vector<VALTYPE*> inputs;
    inputs.push_back(id->start());
    inputs.push_back(id->extent());
    return inputs;
  } else {
    return std::vector<VALTYPE*>();
  }
}

template <typename VALTYPE>
std::vector<VALTYPE*> getConsumers(VALTYPE* val) {
  return (val->definition()) ? val->definition()->outputs()
                             : std::vector<VALTYPE*>({val});
}

//! IR-Generic utility, collects all the producers required for the
//!  given list of IR values and returns them along with the original
//!  list in topological order.
template <typename VALTYPE>
std::vector<VALTYPE*> makeSortedEvaluationList(std::vector<VALTYPE*> input) {
  // Deduplicate
  std::vector<VALTYPE*> to_sort;
  std::unordered_set<VALTYPE*> visited;
  for (auto val : input) {
    if (!visited.count(val)) {
      to_sort.push_back(val);
      visited.insert(val);
    }
  }

  std::vector<VALTYPE*> sorted;
  visited.clear();

  // Topological Sort
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

//! Kernel IR utility, collects all the symbolic values used in allocation
//! nodes.
std::vector<kir::Allocate*> collectBufferSizes(
    const std::vector<nvfuser::Expr*>& exprs) {
  std::vector<kir::Allocate*> buffers;
  std::vector<nvfuser::Expr*> to_visit(exprs);
  while (!to_visit.empty()) {
    auto expr = to_visit.back();
    to_visit.pop_back();
    if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      buffers.push_back(allocate);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      auto for_loop_exprs = for_loop->body().exprs();
      to_visit.insert(
          to_visit.end(), for_loop_exprs.begin(), for_loop_exprs.end());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      auto ite_then_exprs = ite->thenBody().exprs();
      auto ite_else_exprs = ite->elseBody().exprs();
      to_visit.insert(
          to_visit.end(), ite_then_exprs.begin(), ite_then_exprs.end());
      to_visit.insert(
          to_visit.end(), ite_else_exprs.begin(), ite_else_exprs.end());
    }
  }
  return buffers;
}

void bind(std::vector<nvfuser::Val*>& all_values, nvfuser::Val* v) {
  all_values.push_back(v);
}

// Bind the iterDomain's extent for the given root domain
void bindRootDomain(
    std::vector<nvfuser::Val*>& all_values,
    std::vector<nvfuser::IterDomain*> domain) {
  for (auto d : domain) {
    NVF_ERROR(d->definition() == nullptr);
    bind(all_values, d->start());
    bind(all_values, d->extent());
  }
}

// Bind the iterDomain's extent for the given domain
void bindDomain(
    std::vector<nvfuser::Val*>& all_values,
    std::vector<nvfuser::IterDomain*> domain) {
  for (auto d : domain) {
    bind(all_values, d->as<Val>());
  }
}

// 1. Generate extents for IterDomains that compose root domain
// 2. Create new extents using split, merge, reorder operations for rfactor,
// allocation, and leaf domains
void bind(std::vector<nvfuser::Val*>& all_values, nvfuser::TensorView* tv) {
  bindRootDomain(all_values, tv->getRootDomain());
  bindDomain(all_values, tv->getRFactorDomain());
  bindDomain(all_values, tv->getAllocationDomain());
  bindDomain(all_values, tv->getLeafDomain());
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

flatbuffers::Offset<NaiveValueGenerator> ExpressionSerializer::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    kir::Kernel* kernel,
    const std::vector<const kir::Allocate*>& allocations) {
  // 1) Collect allocation sizes
  std::vector<nvfuser::Val*> all_values;
  for (auto allocate : collectBufferSizes(kernel->topLevelExprs())) {
    if (nvfuser::TensorView* tv =
            dynamic_cast<nvfuser::TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
      for (auto v : allocate->shape()) {
        bind(all_values, v);
      }
    }
  }

  // A deserialized fusion may not contain all its allocations in its
  // kir::Kernel. Add allocations directly to handle this case.
  for (auto allocate : allocations) {
    if (nvfuser::TensorView* tv =
            dynamic_cast<nvfuser::TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
      for (auto v : allocate->shape()) {
        bind(all_values, v);
      }
    }
  }

  std::vector<nvfuser::NamedScalar*> named_scalar_values;
  std::vector<nvfuser::Val*> const_int_values;
  std::vector<nvfuser::Val*> symbolic_values;
  std::deque<nvfuser::Val*> derived_values;

  auto insert_item = [](auto& container, auto v) {
    if (std::find(container.begin(), container.end(), v) == container.end()) {
      container.push_back(v);
    }
  };

  // Add TensorView RootDomain IterDomain Extents for all kernel inputs
  // TODO Get deterministic order
  for (auto input : kernel->inputs()) {
    if (nvfuser::TensorView* tv = dynamic_cast<nvfuser::TensorView*>(input)) {
      insert_item(symbolic_values, tv);
      for (auto id : tv->getRootDomain()) {
        auto extent = id->extent();
        if (!extent->isA<nvfuser::NamedScalar>() && !extent->isConstInt()) {
          insert_item(symbolic_values, extent);
        }
      }
    }
  }

  std::vector<nvfuser::Val*> iterdomains;
  std::copy_if(
      all_values.begin(),
      all_values.end(),
      std::back_inserter(iterdomains),
      [](nvfuser::Val* v) { return v->isA<nvfuser::IterDomain>(); });
  for (auto v : iterdomains) {
    auto id = dynamic_cast<nvfuser::IterDomain*>(v);
    NVF_CHECK(id != nullptr);
    all_values.push_back(id->start());
    all_values.push_back(id->extent());
    if (id->hasExpandedExtent()) {
      all_values.push_back(id->expandedExtent());
    }
    all_values.push_back(id->stopOffset());
  }

  // 2) Sort values by dependency order
  // 3) Divide values into NamedScalar, Int, Symbolic, and Derived values
  for (auto v : makeSortedEvaluationList(all_values)) {
    if (v->definition() == nullptr) {
      if (auto ns = dynamic_cast<nvfuser::NamedScalar*>(v)) {
        insert_item(named_scalar_values, ns);
      } else if (v->isConstInt()) {
        insert_item(const_int_values, v);
      } else if (auto id = dynamic_cast<nvfuser::IterDomain*>(v)) {
        insert_item(derived_values, id);
      } else {
        insert_item(symbolic_values, v);
      }
    } else {
      insert_item(derived_values, v);
    }
  }

  // 4) Serialize NaiveValueGenerator by converting each NvFuser value of into
  // an instruction.
  //
  // table NaiveValueGenerator {
  //   instructions : [Instruction];
  // }

  using fb_instruction = flatbuffers::Offset<Instruction>;
  std::vector<fb_instruction> instructions_fb;

  for (auto& val : symbolic_values) {
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

  for (const auto& ns : named_scalar_values) {
    auto ns_fb = CreateNamedScalarDirect(
        builder, ns->name().c_str(), (int64_t)operation_stack_.size());
    auto inst = CreateInstruction(
        builder, serde::InstructionData_NamedScalar, ns_fb.Union());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(ns, operation_stack_.size());
  }

  for (const auto& int_val : const_int_values) {
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

  while (!derived_values.empty()) {
    auto& val = derived_values.front();
    auto def = val->definition();
    derived_values.pop_front();

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

      auto next_val = derived_values.front();
      NVF_ERROR(next_val->definition() == def);
      operation_stack_.emplace(next_val, operation_stack_.size());
      derived_values.pop_front();

    } else if (auto swop = dynamic_cast<nvfuser::Swizzle2D*>(def)) {
      instructions_fb.push_back(serializeSwizzle2D(builder, swop));
      operation_stack_.emplace(val, operation_stack_.size());

      auto next_val = derived_values.front();
      NVF_ERROR(next_val->definition() == def);
      operation_stack_.emplace(next_val, operation_stack_.size());
      derived_values.pop_front();

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
    serialize(
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
