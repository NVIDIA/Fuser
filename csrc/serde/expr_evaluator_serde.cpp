// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <serde/expr_evaluator_serde.h>
#include <serde/polymorphic_value_serde.h>
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
      serde::Swizzle2DType_ZShape,
      serde::SwizzleMode_Data,
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
    }
  }

  // A deserialized fusion may not contain all its allocations in its
  // kir::Kernel. Add allocations directly to handle this case.
  for (auto allocate : allocations) {
    if (nvfuser::TensorView* tv =
            dynamic_cast<nvfuser::TensorView*>(allocate->buffer())) {
      bind(all_values, tv);
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
    if (TensorView* tv = dynamic_cast<TensorView*>(input)) {
      insert_item(symbolic_values, tv);
      for (auto id : tv->getRootDomain()) {
        auto extent = id->extent();
        if (!extent->isA<NamedScalar>() && !extent->isConstInt()) {
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
    auto sv_fb =
        CreateSymbolicDirect(builder, val->name(), val->toString().c_str());
    auto inst = CreateInstruction(
        builder, serde::InstructionData_Symbolic, sv_fb.Union());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(val, operation_stack_.size());
  }

  for (const auto& ns : named_scalar_values) {
    auto ns_fb = CreateNamedScalarDirect(builder, ns->name().c_str());
    auto inst = CreateInstruction(
        builder, serde::InstructionData_NamedScalar, ns_fb.Union());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(ns, operation_stack_.size());
  }

  for (const auto& int_val : const_int_values) {
    auto val_fb = serializeScalar(
        builder, int_val->evaluateInt(), nvfuser::DataType::Int);
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
    NVF_ERROR(alloc_buffer_tv);

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

  return CreateIterDomain(
      builder,
      operation_stack_.at(id->start()),
      operation_stack_.at(id->extent()),
      castEnumToUnderlyingType(id->getParallelType()),
      castEnumToUnderlyingType(id->getIterType()),
      id->isRFactorProduct(),
      id->hasPaddingToMultipleOfWarp(),
      id->isMmaSwizzled());
}

// TODO create separate functions for TensorDomain and IterDomain
flatbuffers::Offset<SymbolicTensor> ExpressionSerializer::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::TensorView* tv) {
  std::vector<flatbuffers::Offset<IterDomain>> fb_root_domain;
  for (auto id : tv->getRootDomain()) {
    fb_root_domain.push_back(serialize(builder, id));
  }
  auto root_domain_fb = CreateDomainDirect(builder, &fb_root_domain);

  flatbuffers::Offset<Domain> rfactor_domain_fb = 0;
  if (tv->hasRFactor()) {
    std::vector<flatbuffers::Offset<IterDomain>> fb_rfactor_domain;
    for (auto id : tv->getRFactorDomain()) {
      fb_rfactor_domain.push_back(serialize(builder, id));
    }
    rfactor_domain_fb = CreateDomainDirect(builder, &fb_rfactor_domain);
  }

  flatbuffers::Offset<Domain> allocation_domain_fb = 0;
  if (tv->hasAllocation()) {
    std::vector<flatbuffers::Offset<IterDomain>> fb_allocation_domain;
    for (auto id : tv->getAllocationDomain()) {
      fb_allocation_domain.push_back(serialize(builder, id));
    }
    allocation_domain_fb = CreateDomainDirect(builder, &fb_allocation_domain);
  }

  std::vector<flatbuffers::Offset<IterDomain>> fb_leaf_domain;
  for (auto id : tv->getLeafDomain()) {
    fb_leaf_domain.push_back(serialize(builder, id));
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

ExpressionBuilder::ExpressionBuilder(kir::Kernel* kernel) : kernel_(kernel) {
  auto insert_item = [](auto& container, auto v) {
    if (std::find(container.begin(), container.end(), v) == container.end()) {
      container.push_back(v);
    }
  };

  // Add TensorView RootDomain IterDomain Extents for all kernel inputs
  // TODO Get deterministic order
  std::vector<nvfuser::Val*> symbolic_values;
  for (auto input : kernel->inputs()) {
    if (TensorView* tv = dynamic_cast<TensorView*>(input)) {
      insert_item(symbolic_values, tv);
      for (auto id : tv->getRootDomain()) {
        auto extent = id->extent();
        if (!extent->isA<NamedScalar>() && !extent->isConstInt()) {
          insert_item(symbolic_values, extent);
        }
      }
    }
  }
  operation_stack_.insert(
      operation_stack_.end(), symbolic_values.begin(), symbolic_values.end());
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
    case serde::InstructionData_Symbolic:
      // TODO Add check for symbolic extent
      break;
    case serde::InstructionData_NamedScalar: {
      auto data = buffer->data_as_NamedScalar();
      auto ns = IrBuilder::create<nvfuser::NamedScalar>(
          data->name()->str(), nvfuser::DataType::Index);
      operation_stack_.push_back(ns);
      break;
    }
    case serde::InstructionData_Scalar: {
      auto data = buffer->data_as_Scalar();
      auto int_val = IrBuilder::create<nvfuser::Val>(
          data->long_value(), nvfuser::DataType::Index);
      operation_stack_.push_back(int_val);
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
      operation_stack_.push_back(buildIterDomain(data));
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

        // TODO support all enum types - Swizzle2DType and SwizzleMode
        auto swizzle_ids = nvfuser::IterDomain::swizzle(
            nvfuser::Swizzle2DType::ZShape,
            in_x->as<nvfuser::IterDomain>(),
            in_y->as<nvfuser::IterDomain>(),
            nvfuser::SwizzleMode::Data);
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
  return nvfuser::IterDomainBuilder(
             operation_stack_.at(buffer->start()),
             operation_stack_.at(buffer->extent()))
      .parallel_type(
          static_cast<nvfuser::ParallelType>(buffer->parallel_type()))
      .iter_type(static_cast<nvfuser::IterType>(buffer->iter_type()))
      .is_rfactor_domain(buffer->is_rfactor_domain())
      .is_padded_dimension(buffer->is_padded_dimension())
      .is_mma_swizzled(buffer->is_mma_swizzled())
      .build();
}

std::vector<const kir::Allocate*> ExpressionBuilder::deserialize(
    const ExpressionBuilder::Allocations* buffers) {
  FusionGuard fg(kernel_);

  std::vector<const kir::Allocate*> results;
  for (auto buffer : *buffers) {
    std::vector<nvfuser::IterDomain*> new_root;
    for (auto fb_id : *buffer->tv()->root()->dims()) {
      new_root.push_back(buildIterDomain(fb_id));
    }

    // TODO Deserialize rfactor, allocation, and root domains

    const auto buffer_domain =
        IrBuilder::create<nvfuser::TensorDomain>(new_root);

    const auto buffer_tv = IrBuilder::create<nvfuser::TensorView>(
        buffer_domain,
        mapToNvfuserDtype(buffer->tv()->dtype()),
        MemoryType::Global);

    // TODO use stl map
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
