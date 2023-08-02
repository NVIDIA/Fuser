// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <evaluator_common.h>

#include <debug.h>
#include <device_lower/lower2device.h>
#include <executor_kernel_arg.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/utils.h>

#include <optional>

namespace nvfuser {

namespace {

template <typename VALTYPE>
std::vector<VALTYPE*> getImmediateProducers(VALTYPE* val) {
  if (val->definition()) {
    auto expr = val->definition();
    return expr->inputs();
  } else {
    return {};
  }
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
  //  Note: didn't explicitly exclude producers that are not in the original
  //   list. This should be acceptable for the intended use.
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
        visited.insert(top_val);
        sorted.push_back(top_val);
        to_sort.pop_back();
      }
    }
  }

  return sorted;
}

//! Kernel IR utility, collects all the symbolic values
//!  used in allocation nodes.
void collectBufferSizes(
    std::vector<Val*>& into,
    const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      into.push_back(allocate->size());
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      collectBufferSizes(into, for_loop->body().exprs());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      collectBufferSizes(into, ite->thenBody().exprs());
      collectBufferSizes(into, ite->elseBody().exprs());
    }
  }
}

std::vector<Val*> collectRuntimeUsedValues(Fusion* fusion) {
  std::vector<Val*> ret;
  auto all_tvs = ir_utils::allTvs(fusion);
  // Collect extent and inputs
  for (auto tv : all_tvs) {
    for (auto id : tv->getLeafDomain()) {
      ret.push_back(id->extent());
    }
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->hasExpandedExtent()) {
        ret.push_back(id->expandedExtent());
      }
    }
  }
  for (auto inp : fusion->inputs()) {
    if (!inp->isA<TensorView>()) {
      ret.push_back(inp);
    }
  }
  // Collect allocation sizes:
  if (fusion->isA<kir::Kernel>()) {
    collectBufferSizes(ret, fusion->as<kir::Kernel>()->topLevelExprs());
  }
  return makeSortedEvaluationList(ret);
}

} // namespace

PrecomputedValues::PrecomputedValues(Fusion* fusion) : fusion_(fusion) {
  loadSymbols(collectRuntimeUsedValues(fusion));
  initializeValueList(symbols());
  initializeNamedScalars();
  initializeIntegerMachine();
}

void PrecomputedValues::bindParallelExtents(
    const ParallelExtentMap& parallel_extents,
    const LaunchParams& launch_constraint) {
  // Bind values of extents of parallelized
  //  iterdomains from launch_constraint when applicable.
  // Consistency will be checked at validate().
  for (const auto& it : parallel_extents) {
    auto raw_val = launch_constraint.getRawVal(it.first);
    if (raw_val > 0) {
      for (auto extent : it.second) {
        bindValue(extent->evaluatorIndex(), raw_val);
      }
    }
  }
}

void PrecomputedValues::bindConcreteParallelTypeValue(
    ParallelType pt,
    PolymorphicValue value) {
  auto index_list_it = thread_dim_value_indices_.find(pt);
  if (index_list_it != thread_dim_value_indices_.end()) {
    for (auto index : *(index_list_it->second)) {
      bindValue(index, value);
    }
  }
}

void PrecomputedValues::bindInputs(const KernelArgumentHolder& args) {
  if (hasValidValues()) {
    invalidate();
  }

  const auto& inputs = fusion_->inputs();
  TORCH_INTERNAL_ASSERT(
      args.size() == inputs.size(), "kernel inputs size does not match args");

  for (const auto i : c10::irange((int64_t)inputs.size())) {
    const auto input = inputs[i];
    TORCH_INTERNAL_ASSERT(input != nullptr);
    bindValue(input->evaluatorIndex(), *args[i]);
    if (auto tensor_input = dynamic_cast<TensorView*>(input)) {
      const auto& tensor = args[i]->as<at::Tensor>();
      if (!tensor.is_cpu()) {
        bindTensorMetaData(tensor_input, tensor);
      }
    }
  }
}

void PrecomputedValues::initializeValueList(
    const std::vector<Val*>& sorted_value_list) {
  // Initialize workspace
  num_of_values_ = (int)sorted_value_list.size();
  defined_ = std::vector<bool>(num_of_values_, false);
  is_constant_ = std::vector<bool>(num_of_values_, false);
  values_ = std::vector<PolymorphicValue>(num_of_values_, PolymorphicValue());

  // Fill in constants and assign evaluator indices
  for (const auto i : c10::irange(num_of_values_)) {
    // Use an expression evaluator to test if value is const
    if (sorted_value_list[i]->isConstScalar()) {
      is_constant_[i] = true;
      if (sorted_value_list[i]->isIntegralScalar()) {
        values_[i] = PolymorphicValue(sorted_value_list[i]->evaluateInt());
      }
      if (sorted_value_list[i]->isFloatingPointScalar()) {
        values_[i] = PolymorphicValue(sorted_value_list[i]->evaluateDouble());
      }
      if (sorted_value_list[i]->isABool()) {
        values_[i] = PolymorphicValue(sorted_value_list[i]->evaluateBool());
      }
    }
    sorted_value_list[i]->setEvaluatorIndex(i);
  }
}

PolymorphicValue PrecomputedValues::getMaybeValueFor(const Val* val) const {
  auto index = val->evaluatorIndex();
  if (index < 0) {
    return std::monostate{};
  }
  if (!defined_[index] && !is_constant_[index]) {
    return std::monostate{};
  }
  return values_[index];
}

void PrecomputedValues::print() const {
  debug() << "Precomputed Values:\n";
  for (auto i : c10::irange(symbols_.size())) {
    if (defined_[i]) {
      debug() << symbols_[i]->toInlineString() << " = " << values_[i]
              << std::endl;
    }
  }
}

void PrecomputedValues::evaluate() {
  FUSER_PERF_SCOPE("PrecomputedValues::Evaluate");
  value_machine_->run();
  validate();
}

void PrecomputedValues::invalidate() {
  // clear binding values
  binding_log_.clear();

  // invalidate value entries
  std::fill(defined_.begin(), defined_.end(), false);

  // invalidate flag
  has_valid_values_ = false;
}

PrecomputedValues PrecomputedValues::clone(IrCloner& ir_cloner) const {
  PrecomputedValues pv(static_cast<Fusion*>(ir_cloner.container()));

  // this is a map to unique pointers to vectors, so we need to copy the
  // vectors and create new unique pointers
  for (const auto& kv : thread_dim_value_indices_) {
    std::vector<int> new_vec(kv.second->begin(), kv.second->end());
    pv.thread_dim_value_indices_[kv.first] =
        std::make_unique<std::vector<int>>(new_vec);
  }

  pv.has_valid_values_ = has_valid_values_;
  pv.num_of_values_ = num_of_values_;
  pv.defined_.insert(pv.defined_.end(), defined_.begin(), defined_.end());
  pv.is_constant_.insert(
      pv.is_constant_.end(), is_constant_.begin(), is_constant_.end());
  pv.values_.insert(pv.values_.end(), values_.begin(), values_.end());
  pv.binding_log_.insert(
      pv.binding_log_.end(), binding_log_.begin(), binding_log_.end());

  pv.symbols_.resize(symbols_.size());
  for (const auto i : c10::irange(symbols_.size())) {
    pv.symbols_[i] = ir_cloner.clone(symbols_[i]);
  }

  pv.value_machine_->copyFrom(*value_machine_.get());

  return pv;
}

namespace {

//! Compares the name of given scalar with thread size strings
//!  and returns the corresponding parallel type if a match
//!  is found.
std::optional<ParallelType> getMaybeThreadSizeParallelType(
    NamedScalar* named_scalar) {
  auto& var_name = named_scalar->name();
  for (auto ptype : kParallelTypeThreads) {
    if (var_name == stringifyThreadSize(ptype)) {
      return ptype;
    }
  }
  return std::nullopt;
}

} // namespace

void PrecomputedValues::initializeNamedScalars() {
  for (auto val : symbols()) {
    if (auto named_scalar = dynamic_cast<NamedScalar*>(val)) {
      auto maybe_parallel_type = getMaybeThreadSizeParallelType(named_scalar);
      if (maybe_parallel_type.has_value()) {
        auto& index_list =
            thread_dim_value_indices_[maybe_parallel_type.value()];
        if (!index_list) {
          index_list = std::make_unique<std::vector<int>>();
        }
        index_list->push_back(val->evaluatorIndex());
      }
    }
  }
}

void PrecomputedValues::validate() {
  FUSER_PERF_SCOPE("PrecomputedValuess::Validate");
  for (const auto& it : binding_log_) {
    TORCH_INTERNAL_ASSERT(
        values_[it.first] == it.second,
        "Precomputed values failed to validate.",
        "\nSomething unexpected changed between the compilation and execution.\n",
        values_[it.first],
        " != ",
        it.second);
  }
  has_valid_values_ = true;
}

void PrecomputedValues::bindTensorMetaData(
    TensorView* tv,
    const at::Tensor& tensor) {
  const auto root_domain =
      TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  TORCH_INTERNAL_ASSERT(
      tensor.dim() == static_cast<int64_t>(root_domain.size()),
      "Something went wrong configuring launch. Inputs do not match.");

  for (const auto dim : c10::irange(root_domain.size())) {
    auto value = tensor.size((int64_t)dim);
    if (root_domain[dim]->hasExpandedExtent()) {
      auto extent = root_domain[dim]->extent();
      auto expanded_extent = root_domain[dim]->expandedExtent();
      bindValue(extent->evaluatorIndex(), 1L);
      bindValue(expanded_extent->evaluatorIndex(), value);
    } else {
      auto extent = root_domain[dim]->extent();
      bindValue(extent->evaluatorIndex(), value);
    }
  }
}

NaiveValueMachine::NaiveValueMachine(PrecomputedValues& precomputed_values)
    : precomputed_values_(precomputed_values), num_of_instructions_{0} {
  for (auto val : precomputed_values_.symbols_) {
    auto def = val->definition();
    if (def) {
      if (auto uop = dynamic_cast<UnaryOp*>(def)) {
        makeUnaryOp(uop);
      } else if (auto bop = dynamic_cast<BinaryOp*>(def)) {
        makeBinaryOp(bop);
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unsupported expr");
      }
    }
  }
}

void NaiveValueMachine::copyFrom(const NaiveValueMachine& other) {
  num_of_instructions_ = other.num_of_instructions_;

  inst_type_.clear();
  inst_type_.insert(
      inst_type_.end(), other.inst_type_.begin(), other.inst_type_.end());

  uop_type_.clear();
  uop_type_.insert(
      uop_type_.end(), other.uop_type_.begin(), other.uop_type_.end());

  data_type_.clear();
  data_type_.insert(
      data_type_.end(), other.data_type_.begin(), other.data_type_.end());

  bop_type_.clear();
  bop_type_.insert(
      bop_type_.end(), other.bop_type_.begin(), other.bop_type_.end());

  src0_.clear();
  src0_.insert(src0_.end(), other.src0_.begin(), other.src0_.end());

  src1_.clear();
  src1_.insert(src1_.end(), other.src1_.begin(), other.src1_.end());

  dest_.clear();
  dest_.insert(dest_.end(), other.dest_.begin(), other.dest_.end());
}

void NaiveValueMachine::run() {
  for (const auto i : c10::irange(num_of_instructions_)) {
    // Skip this instruction if the dest location
    //  has already been computed or is constant.
    if (precomputed_values_.defined_[dest_[i]] ||
        precomputed_values_.is_constant_[dest_[i]]) {
      continue;
    }
    runInstruction(i);
  }
}

void NaiveValueMachine::makeUnaryOp(UnaryOp* uop) {
  int in = uop->inputs()[0]->evaluatorIndex();
  int out = uop->outputs()[0]->evaluatorIndex();
  TORCH_INTERNAL_ASSERT(in >= 0, "Integer Machine: unknown input: ", uop);
  TORCH_INTERNAL_ASSERT(out >= 0, "Integer Machine: unknown out: ", uop);

  int index = makeInstructionEntry();
  inst_type_[index] = InstructionType::UNARY_OP;
  uop_type_[index] = uop->getUnaryOpType();
  if (uop_type_[index] == UnaryOpType::Cast) {
    data_type_[index] = uop->out()->getDataType().value();
  }
  src0_[index] = in;
  dest_[index] = out;
}

void NaiveValueMachine::makeBinaryOp(BinaryOp* bop) {
  int in0 = bop->inputs()[0]->evaluatorIndex();
  int in1 = bop->inputs()[1]->evaluatorIndex();
  int out = bop->outputs()[0]->evaluatorIndex();

  TORCH_INTERNAL_ASSERT(in0 >= 0, "Integer Machine: unknown lhs: ", bop);
  TORCH_INTERNAL_ASSERT(in1 >= 0, "Integer Machine: unknown rhs: ", bop);
  TORCH_INTERNAL_ASSERT(out >= 0, "Integer Machine: unknown out: ", bop);

  int index = makeInstructionEntry();
  inst_type_[index] = InstructionType::BINARY_OP;
  bop_type_[index] = bop->getBinaryOpType();
  src0_[index] = in0;
  src1_[index] = in1;
  dest_[index] = out;
}

int NaiveValueMachine::makeInstructionEntry() {
  int index = num_of_instructions_++;
  inst_type_.emplace_back(InstructionType::UNARY_OP);
  uop_type_.emplace_back(UnaryOpType::Abs);
  bop_type_.emplace_back(BinaryOpType::Add);
  data_type_.emplace_back(DataType::Null);
  src0_.emplace_back(-1);
  src1_.emplace_back(-1);
  dest_.emplace_back(-1);
  return index;
}

void NaiveValueMachine::runInstruction(int index) {
  switch (inst_type_[index]) {
    case InstructionType::SET_OP:
      precomputed_values_.values_[dest_[index]] =
          precomputed_values_.values_[src0_[index]];
      break;
    case InstructionType::UNARY_OP:
      runUnaryOp(index);
      break;
    case InstructionType::BINARY_OP:
      runBinaryOp(index);
      break;
  }
}

void NaiveValueMachine::runUnaryOp(int index) {
  using namespace PolymorphicValue_functions;
  int src_index = src0_[index];
  bool src_defined = precomputed_values_.defined_[src_index];
  bool src_is_const = precomputed_values_.is_constant_[src_index];
  if (!src_defined && !src_is_const) {
    return;
  }

  int dest_index = dest_[index];

  auto& src = precomputed_values_.values_[src_index];
  auto& dest = precomputed_values_.values_[dest_index];

  switch (uop_type_[index]) {
    case UnaryOpType::Neg:
      dest = -src;
      break;
    case UnaryOpType::Cast:
      if (data_type_[index] == DataType::Double) {
        dest = PolymorphicValue((double)src);
      } else if (data_type_[index] == DataType::Int) {
        dest = PolymorphicValue((int64_t)src);
      } else if (data_type_[index] == DataType::Bool) {
        dest = PolymorphicValue((bool)src);
      } else {
        TORCH_INTERNAL_ASSERT(false, "dtype not supported in evaluator");
      }
      break;
    case UnaryOpType::Abs:
      dest = abs(src);
      break;
    default:
      TORCH_CHECK(!"Unexpected operator type ", uop_type_[index]);
  }

  precomputed_values_.defined_[dest_index] = true;
}

void NaiveValueMachine::runBinaryOp(int index) {
  using namespace PolymorphicValue_functions;
  int src0_index = src0_[index];
  int src1_index = src1_[index];
  bool src0_is_const = precomputed_values_.is_constant_[src0_index];
  bool src1_is_const = precomputed_values_.is_constant_[src1_index];

  bool src_defined =
      (precomputed_values_.defined_[src0_index] || src0_is_const) &&
      (precomputed_values_.defined_[src1_index] || src1_is_const);

  if (!src_defined) {
    return;
  }
  int dest_index = dest_[index];

  auto& lhs = precomputed_values_.values_[src0_index];
  auto& rhs = precomputed_values_.values_[src1_index];
  auto& dest = precomputed_values_.values_[dest_index];

  switch (bop_type_[index]) {
    case BinaryOpType::Add:
      dest = lhs + rhs;
      break;
    case BinaryOpType::Sub:
      dest = lhs - rhs;
      break;
    case BinaryOpType::Mul:
      dest = lhs * rhs;
      break;
    case BinaryOpType::Div:
      TORCH_CHECK(rhs != 0);
      dest = lhs / rhs;
      break;
    case BinaryOpType::Mod:
      TORCH_CHECK(rhs != 0);
      dest = lhs % rhs;
      break;
    case BinaryOpType::CeilDiv:
      TORCH_CHECK(rhs != 0);
      dest = ceildiv(lhs, rhs);
      break;
    case BinaryOpType::And:
      dest = lhs && rhs;
      break;
    case BinaryOpType::Max:
      dest = lhs > rhs ? lhs : rhs;
      break;
    case BinaryOpType::Min:
      dest = lhs < rhs ? lhs : rhs;
      break;
    case BinaryOpType::Gcd:
      dest = gcd(lhs, rhs);
      break;
    default:
      TORCH_CHECK(!"Unexpected operator type");
  }

  precomputed_values_.defined_[dest_index] = true;
}

} // namespace nvfuser
