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

struct SortedValues {
  std::vector<nvfuser::Val*> symbolic_values;
  std::vector<nvfuser::NamedScalar*> named_scalar_values;
  std::vector<nvfuser::Val*> const_int_values;
  std::deque<nvfuser::Val*> derived_values;
};

SortedValues processAllocations(
    kir::Kernel* kernel,
    const std::vector<const kir::Allocate*>& allocations) {
  SortedValues result;

  // Built in deterministic order from kernel inputs
  result.symbolic_values = gatherSymbolicValues(kernel);

  // 1) All kir::Allocate nodes are contained in FusionExecutor::KernelSummary
  // 2) Sort all values by dependency order
  // 3) Divide values into Symbolic, NamedScalar, Constant Int, and Derived
  // values
  for (auto v : makeSortedEvaluationList(allocations)) {
    if (v->definition() == nullptr) {
      if (auto ns = dynamic_cast<nvfuser::NamedScalar*>(v)) {
        insertUniqueItem(result.named_scalar_values, ns);
      } else if (v->isConstInt()) {
        insertUniqueItem(result.const_int_values, v);
      } else if (auto id = dynamic_cast<nvfuser::IterDomain*>(v)) {
        insertUniqueItem(result.derived_values, id);
      } else {
        NVF_ERROR(
            !insertUniqueItem(result.symbolic_values, v),
            "Expect all symbolic values to come from kernel inputs.");
      }
    } else {
      insertUniqueItem(result.derived_values, v);
    }
  }
  return result;
}

class DerivedExpressionSerializer final : private OptInConstDispatch {
  using fb_instruction = flatbuffers::Offset<Instruction>;

 public:
  static void serialize(
      flatbuffers::FlatBufferBuilder& builder,
      std::deque<nvfuser::Val*> derived_values,
      std::unordered_map<const Val*, long>& operation_stack,
      std::vector<fb_instruction>& instructions_fb) {
    DerivedExpressionSerializer serializer(
        builder, derived_values, operation_stack, instructions_fb);

    while (!derived_values.empty()) {
      auto& val = derived_values.front();
      auto def = val->definition();

      if (operation_stack.count(val)) {
        derived_values.pop_front();
        continue;
      }

      if (def == nullptr && val->isA<nvfuser::IterDomain>()) {
        serializer.OptInConstDispatch::dispatch(val);
      } else {
        NVF_ERROR(def != nullptr, "Expected definition with derived value.");
        serializer.OptInConstDispatch::dispatch(def);
      }
    }
  }

  // CLANGTIDY - Virtual Destructor is public
  ~DerivedExpressionSerializer() final = default;

 private:
  explicit DerivedExpressionSerializer(
      flatbuffers::FlatBufferBuilder& builder,
      std::deque<nvfuser::Val*>& derived_values,
      std::unordered_map<const Val*, long>& operation_stack,
      std::vector<fb_instruction>& instructions_fb)
      : builder_(builder),
        derived_values_(derived_values),
        operation_stack_(operation_stack),
        instructions_fb_(instructions_fb) {}

  void handle(const nvfuser::BinaryOp* bop) override {
    instructions_fb_.push_back(serializeBinaryOp(bop));
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::GetAttr* attr) override {
    instructions_fb_.push_back(serializeGetAttr(attr));
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::GetMetaData* metadata) override {
    instructions_fb_.push_back(serializeGetMetaData(metadata));
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::GetItem* item) override {
    instructions_fb_.push_back(serializeGetItem(item));
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::IterDomain* id) override {
    NVF_ERROR(id->definition() == nullptr);
    auto fb_id = serializeIterDomain(id);
    auto fb_inst = CreateInstruction(
        builder_, serde::InstructionData_IterDomain, fb_id.Union());
    instructions_fb_.push_back(fb_inst);
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::Merge* merge) override {
    instructions_fb_.push_back(serializeMerge(merge));
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::Resize* resize) override {
    auto inst = serializeResize(resize);
    for (auto i : inst) {
      instructions_fb_.push_back(i);
    }
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::Split* split) override {
    auto inst = serializeSplit(split);
    for (auto i : inst) {
      instructions_fb_.push_back(i);
    }
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();

    auto next_val = derived_values_.front();
    NVF_ERROR(next_val->definition() == split);
    operation_stack_.emplace(next_val, operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::Swizzle2D* swizzle) override {
    instructions_fb_.push_back(serializeSwizzle2D(swizzle));
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();

    auto next_val = derived_values_.front();
    NVF_ERROR(next_val->definition() == swizzle);
    operation_stack_.emplace(next_val, operation_stack_.size());
    derived_values_.pop_front();
  }

  void handle(const nvfuser::UnaryOp* uop) override {
    instructions_fb_.push_back(serializeUnaryOp(uop));
    operation_stack_.emplace(derived_values_.front(), operation_stack_.size());
    derived_values_.pop_front();
  }

  flatbuffers::Offset<Instruction> serializeAttribute(const nvfuser::Val* val) {
    operation_stack_.emplace(val, operation_stack_.size());
    auto sv_fb =
        CreateSymbolicDirect(builder_, val->name(), val->toString().c_str());
    return CreateInstruction(
        builder_, serde::InstructionData_Symbolic, sv_fb.Union());
  }

  flatbuffers::Offset<Instruction> serializeBinaryOp(
      const nvfuser::BinaryOp* bop) {
    auto bop_fb = CreateBinaryOpDirect(
        builder_,
        mapToSerdeBinaryOp(bop->getBinaryOpType()),
        operation_stack_.at(bop->inputs().front()),
        operation_stack_.at(bop->inputs().back()),
        (int64_t)operation_stack_.size(),
        bop->toString().c_str());
    return CreateInstruction(
        builder_, serde::InstructionData_BinaryOp, bop_fb.Union());
  }

  flatbuffers::Offset<Instruction> serializeGetAttr(
      const nvfuser::GetAttr* attr) {
    auto attr_fb = CreateGetAttr(
        builder_,
        operation_stack_.at(attr->struct_()),
        builder_.CreateString(attr->attr()),
        (int64_t)operation_stack_.size());
    return CreateInstruction(
        builder_, serde::InstructionData_GetAttr, attr_fb.Union());
  }

  flatbuffers::Offset<Instruction> serializeGetItem(
      const nvfuser::GetItem* item) {
    auto item_fb = CreateGetItem(
        builder_,
        operation_stack_.at(item->array()),
        operation_stack_.at(item->index()),
        (int64_t)operation_stack_.size());
    return CreateInstruction(
        builder_, serde::InstructionData_GetItem, item_fb.Union());
  }

  flatbuffers::Offset<Instruction> serializeGetMetaData(
      const nvfuser::GetMetaData* metadata) {
    auto metadata_fb = CreateGetMetaData(
        builder_,
        operation_stack_.at(metadata->in()),
        (int64_t)operation_stack_.size());
    return CreateInstruction(
        builder_, serde::InstructionData_GetMetaData, metadata_fb.Union());
  }

  flatbuffers::Offset<IterDomain> serializeIterDomain(
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
        !id->hasExpandedExtent() ||
            operation_stack_.count(id->expandedExtent()),
        "Missing iterDomain expandedExtent in NaiveValueGenerator stack.\t",
        id->expandedExtent()->toString());

    NVF_ERROR(
        operation_stack_.count(id->stopOffset()),
        "Missing iterDomain stopOffset in NaiveValueGenerator stack.\t",
        id->stopOffset()->toString());

    return CreateIterDomain(
        builder_,
        operation_stack_.at(id->start()),
        operation_stack_.at(id->extent()),
        id->hasExpandedExtent() ? operation_stack_.at(id->expandedExtent())
                                : -1,
        operation_stack_.at(id->stopOffset()),
        (int64_t)operation_stack_.size(),
        castEnumToUnderlyingType(id->getParallelType()),
        castEnumToUnderlyingType(id->getIterType()),
        id->isRFactorProduct(),
        id->hasPaddingToMultipleOfWarp(),
        id->getMaybeSizeAfterPadding().value_or(0),
        id->isMmaSwizzled());
  }

  flatbuffers::Offset<Instruction> serializeMerge(const nvfuser::Merge* merge) {
    auto merge_fb = CreateMerge(
        builder_,
        operation_stack_.at(merge->inner()),
        operation_stack_.at(merge->outer()),
        (int64_t)operation_stack_.size());
    return CreateInstruction(
        builder_, serde::InstructionData_Merge, merge_fb.Union());
  }

  std::array<flatbuffers::Offset<Instruction>, 3> serializeResize(
      const nvfuser::Resize* resize) {
    auto left_expand_inst = serializeAttribute(resize->leftExpand());
    auto right_expand_inst = serializeAttribute(resize->leftExpand());
    auto resize_fb = CreateResize(
        builder_,
        operation_stack_.at(resize->in()),
        operation_stack_.at(resize->leftExpand()),
        operation_stack_.at(resize->rightExpand()),
        (int64_t)operation_stack_.size());
    auto resize_inst = CreateInstruction(
        builder_, serde::InstructionData_Resize, resize_fb.Union());
    return {left_expand_inst, right_expand_inst, resize_inst};
  }

  std::array<flatbuffers::Offset<Instruction>, 2> serializeSplit(
      const nvfuser::Split* split) {
    auto factor_inst = serializeAttribute(split->factor());
    auto split_fb = CreateSplit(
        builder_,
        operation_stack_.at(split->in()),
        operation_stack_.at(split->factor()),
        (int64_t)operation_stack_.size(),
        (int64_t)operation_stack_.size() + 1);
    auto split_inst = CreateInstruction(
        builder_, serde::InstructionData_Split, split_fb.Union());
    return {factor_inst, split_inst};
  }

  flatbuffers::Offset<Instruction> serializeSwizzle2D(
      const nvfuser::Swizzle2D* swizzle) {
    auto swizzle_fb = CreateSwizzle2D(
        builder_,
        operation_stack_.at(swizzle->inX()),
        operation_stack_.at(swizzle->inY()),
        castEnumToUnderlyingType(swizzle->swizzleType()),
        castEnumToUnderlyingType(swizzle->swizzleMode()),
        (int64_t)operation_stack_.size(),
        (int64_t)operation_stack_.size() + 1);
    return CreateInstruction(
        builder_, serde::InstructionData_Swizzle2D, swizzle_fb.Union());
  }

  flatbuffers::Offset<Instruction> serializeUnaryOp(
      const nvfuser::UnaryOp* uop) {
    DataType dtype = (uop->getUnaryOpType() == nvfuser::UnaryOpType::Cast)
        ? mapToSerdeDtype(uop->out()->getDataType().value())
        : serde::DataType_None;
    auto uop_fb = CreateUnaryOpDirect(
        builder_,
        mapToSerdeUnaryOp(uop->getUnaryOpType()),
        dtype,
        operation_stack_.at(uop->inputs().front()),
        (int64_t)operation_stack_.size(),
        uop->toString().c_str());
    return CreateInstruction(
        builder_, serde::InstructionData_UnaryOp, uop_fb.Union());
  }

 private:
  flatbuffers::FlatBufferBuilder& builder_;
  std::deque<nvfuser::Val*>& derived_values_;
  std::unordered_map<const Val*, long>& operation_stack_;
  std::vector<fb_instruction>& instructions_fb_;
};

} // namespace

flatbuffers::Offset<NaiveValueGenerator> ExpressionSerializer::
    serializeNaiveValueGenerator(
        flatbuffers::FlatBufferBuilder& builder,
        const std::vector<const kir::Allocate*>& allocations) {
  // Short Circuit: Return empty offset if there aren't any kir::Allocate nodes
  if (allocations.empty()) {
    return 0;
  }

  // 1) All kir::Allocate nodes are contained in FusionExecutor::KernelSummary
  // 2) Sort all values in topological order
  // 3) Divide values into Symbolic, NamedScalar, Constant Integer, and Derived
  // values 4) Serialize NaiveValueGenerator by converting each NvFuser value of
  // into an instruction.
  auto sorted_values = processAllocations(kernel_, allocations);

  using fb_instruction = flatbuffers::Offset<Instruction>;
  std::vector<fb_instruction> instructions_fb;

  for (auto& val : sorted_values.symbolic_values) {
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

  for (const auto& ns : sorted_values.named_scalar_values) {
    auto ns_fb = CreateNamedScalarDirect(
        builder, ns->name().c_str(), (int64_t)operation_stack_.size());
    auto inst = CreateInstruction(
        builder, serde::InstructionData_NamedScalar, ns_fb.Union());
    instructions_fb.push_back(inst);
    operation_stack_.emplace(ns, operation_stack_.size());
  }

  for (const auto& int_val : sorted_values.const_int_values) {
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

  DerivedExpressionSerializer::serialize(
      builder, sorted_values.derived_values, operation_stack_, instructions_fb);

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

template <typename T>
flatbuffers::Offset<flatbuffers::Vector<int64_t>> ExpressionSerializer::
    serialize(
        flatbuffers::FlatBufferBuilder& builder,
        const std::vector<T*>& values) {
  if (values.empty()) {
    return 0;
  }

  std::vector<int64_t> fb_values;
  for (auto v : values) {
    NVF_ERROR(
        operation_stack_.count(v),
        "Missing value in NaiveValueGenerator stack.\t",
        v->toString());
    fb_values.push_back(operation_stack_.at(v));
  }
  return builder.CreateVector(fb_values);
}

flatbuffers::Offset<SymbolicTensor> ExpressionSerializer::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::TensorView* tv) {
  auto root_domain_fb = serialize(builder, tv->getRootDomain());
  auto rfactor_domain_fb = serialize(builder, tv->getRFactorDomain());
  auto allocation_domain_fb = serialize(builder, tv->getAllocationDomain());
  auto leaf_domain_fb = serialize(builder, tv->getLeafDomain());

  SymbolicTensorBuilder tensor_builder(builder);
  tensor_builder.add_dtype(mapToSerdeDtype(tv->getDataType().value()));
  tensor_builder.add_root(root_domain_fb);
  tensor_builder.add_rfactor(rfactor_domain_fb);
  tensor_builder.add_allocate(allocation_domain_fb);
  tensor_builder.add_leaf(leaf_domain_fb);
  return tensor_builder.Finish();
}

} // namespace nvfuser::serde
