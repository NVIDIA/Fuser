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
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <runtime/executor_kernel_arg.h>
#include <tensor_metadata.h>

#include <optional>

namespace nvfuser {

namespace {

std::vector<Val*> getImmediateProducers(Val* val) {
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
std::vector<Val*> makeSortedEvaluationList(std::vector<Val*> input) {
  // Deduplicate
  std::vector<Val*> to_sort;
  std::unordered_set<Val*> visited;
  for (auto val : input) {
    if (!visited.count(val)) {
      to_sort.push_back(val);
      visited.insert(val);
    }
  }

  std::vector<Val*> sorted;
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
      // Struct types must be bound directly. This is because it would
      // otherwise require us to compute T0 just to compute GetMetaData(T0),
      // for example. We skip computing producers of Structs here in order to
      // avoid computing the TensorViews themselves.
      if (!isStructType(top_val->dtype())) {
        for (auto producer : getImmediateProducers(top_val)) {
          if (!visited.count(producer)) {
            ready_to_pop = false;
            to_sort.push_back(producer);
          }
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
    } else if (auto for_loop = dynamic_cast<ForLoop*>(expr)) {
      collectBufferSizes(into, for_loop->body().exprs());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      collectBufferSizes(into, ite->thenBody().exprs());
      collectBufferSizes(into, ite->elseBody().exprs());
    }
  }
}

std::vector<Val*> collectRuntimeUsedValues(Fusion* fusion) {
  std::vector<Val*> ret;
  auto all_tvs = fusion->allTvs();
  // Collect extent and inputs
  for (auto tv : all_tvs) {
    for (auto id : tv->getLoopDomain()) {
      ret.push_back(id->extent());
    }
    for (auto id : tv->getLogicalDomain()) {
      if (id->hasExpandedExtent()) {
        ret.push_back(id->expandedExtent());
      }
    }
  }
  for (auto inp : fusion->inputs()) {
    if (auto* tv = dynamic_cast<TensorView*>(inp)) {
      // For TensorView inputs, do not bind the TV itself. Only bind its
      // TensorMetaData
      Val* metadata = fusion->metadataOf(tv);
      ret.push_back(metadata);
    } else {
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
  FUSER_PERF_SCOPE("PrecomputedValues::PrecomputedValues");
  loadSymbols(collectRuntimeUsedValues(fusion));
  initializeValueList(symbols());
  initializeNamedScalars();
  initializeIntegerMachine();
}

PrecomputedValues::~PrecomputedValues() {
  // Reset evaluator index to -1
  // so we can create other PrecomputedValues objects.
  for (Val* v : symbols()) {
    v->setEvaluatorIndex(-1);
  }
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
  FUSER_PERF_SCOPE("PrecomputedValues::bindInputs");
  if (hasValidValues()) {
    invalidate();
  }
  bindValues(fusion_->inputs(), args);
}

void PrecomputedValues::bindValues(
    const std::vector<Val*>& inputs,
    const KernelArgumentHolder& args) {
  NVF_ERROR_EQ(
      args.size(),
      std::ssize(inputs),
      "kernel inputs size does not match args");

  for (const auto i : arange((int64_t)inputs.size())) {
    const auto input = inputs[i];
    NVF_ERROR(input != nullptr);
    if (auto* tv = dynamic_cast<TensorView*>(input)) {
      const auto& tensor = args[i].as<at::Tensor>();
      if (!tensor.is_cpu()) {
        bindTensorMetaData(tv, tensor);
      }
    } else {
      bindValue(input->evaluatorIndex(), args[i]);
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
  for (const auto i : arange(num_of_values_)) {
    // Use an expression evaluator to test if value is const
    // Structs must be bound directly
    if (!isStructType(sorted_value_list[i]->dtype()) &&
        sorted_value_list[i]->isConstScalar()) {
      is_constant_[i] = true;
      values_[i] = sorted_value_list[i]->evaluate();
    }
    sorted_value_list[i]->setEvaluatorIndex(i);
  }
}

const PolymorphicValue& PrecomputedValues::getMaybeValueFor(
    const Val* val) const {
  auto index = val->evaluatorIndex();
  if (index < 0) {
    return null_;
  }
  if (!defined_[index] && !is_constant_[index]) {
    return null_;
  }
  return values_[index];
}

void PrecomputedValues::print() const {
  debug() << "Precomputed Values:\n";
  for (auto i : arange(symbols_.size())) {
    if (defined_[i]) {
      debug() << symbols_[i]->toInlineString() << " = "
              << PolymorphicValue_functions::toString(values_[i]) << std::endl;
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
  for (const auto i : arange(symbols_.size())) {
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
  using namespace PolymorphicValue_functions;
  for (const auto& it : binding_log_) {
    NVF_ERROR(
        isSame(values_[it.first], it.second),
        "Precomputed values failed to validate.",
        "\nSomething unexpected changed between the compilation and "
        "execution.\n",
        values_[it.first],
        " != ",
        it.second);
  }
  has_valid_values_ = true;
}

void PrecomputedValues::bindTensorMetaData(
    TensorView* tv,
    const at::Tensor& tensor) {
  const auto logical_domain =
      TensorDomain::noReductions(tv->getLogicalDomain());
  NVF_ERROR(
      tensor.dim() == static_cast<int64_t>(logical_domain.size()),
      "Something went wrong configuring launch. Inputs do not match.");

  std::vector<int64_t> logical_sizes = unshardedSizes(tv, tensor.sizes());

  // Adjust the last dimension of the logical domain to support DataType
  // that is not supported by PyTorch. See the comment of getLastDimAdjustment
  // in type.h for more details.
  const auto adjust_last_dim = getLastDimAdjustment(tv->dtype());
  if (!logical_sizes.empty()) {
    auto& last_dim = logical_sizes.back();
    last_dim = adjust_last_dim.fromATenToNVF(last_dim);
  } else {
    NVF_ERROR(
        adjust_last_dim.denominator == 1 && adjust_last_dim.numerator == 1,
        "DataType not supported");
  }

  for (const auto dim : arange(static_cast<int64_t>(logical_domain.size()))) {
    IterDomain* id = logical_domain[dim];
    const auto dim_size = logical_sizes.at(dim);
    if (id->isBroadcast()) {
      // DIDs are ignored for broadcast. See MultideviceShardingTest.Broadcast
      // and .ExpandedBroadcast.
      bindValue(id->extent()->evaluatorIndex(), 1L);
      if (id->hasExpandedExtent()) {
        bindValue(id->expandedExtent()->evaluatorIndex(), dim_size);
      }
    } else {
      bindValue(id->extent()->evaluatorIndex(), dim_size);
    }
  }

  // Here we bind TensorMetaData so that GetMetaData expressions can be
  // evaluated. Note that we do not bind the at::Tensor itself here since that
  // would mean PrecomputedValues will own the tensor. Unlike
  // ExpressionEvaluator, PrecomputedValues objects are typically long-lived, so
  // we do not want them to own large objects.
  // To do this we create a temporary ExpressionEvaluator so that we can compute
  // the metadata once, then save it
  ExpressionEvaluator ee;
  ee.bindPrecomputedValues(this);
  ee.bind(tv, tensor);
  auto metadata_val = IrBuilder::metadataExpr(tv);
  auto metadata = ee.evaluate(metadata_val);
  // NOTE: In some cases we may not be able to evaluate metadata. For example,
  // if there exists a split expression between the root and logical domains
  // of tv whose split factor is not able to be evaluated. For that reason,
  // calling code should ensure that all inputs required to propagate strides
  // from root to allocation domains are already bound to "this" before binding
  // a TensorView's metadata.
  NVF_ERROR(
      metadata.hasValue(),
      "Could not evaluate metadata expression for ",
      tv->toString(),
      " with input tensor ",
      tensor);
  bindValue(metadata_val->evaluatorIndex(), metadata);
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
      } else if (auto top = dynamic_cast<TernaryOp*>(def)) {
        makeTernaryOp(top);
      } else {
        // There could be some ops not supported yet. For these ops, we will
        // bind their outputs. So ignoring them here.
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
  for (const auto i : arange(num_of_instructions_)) {
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
  NVF_ERROR(in >= 0, "Integer Machine: unknown input: ", uop);
  NVF_ERROR(out >= 0, "Integer Machine: unknown out: ", uop);

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

  NVF_ERROR(in0 >= 0, "Integer Machine: unknown lhs: ", bop);
  NVF_ERROR(in1 >= 0, "Integer Machine: unknown rhs: ", bop);
  NVF_ERROR(out >= 0, "Integer Machine: unknown out: ", bop);

  int index = makeInstructionEntry();
  inst_type_[index] = InstructionType::BINARY_OP;
  bop_type_[index] = bop->getBinaryOpType();
  src0_[index] = in0;
  src1_[index] = in1;
  dest_[index] = out;
}

void NaiveValueMachine::makeTernaryOp(TernaryOp* top) {
  int in0 = top->inputs()[0]->evaluatorIndex();
  int in1 = top->inputs()[1]->evaluatorIndex();
  int in2 = top->inputs()[2]->evaluatorIndex();
  int out = top->outputs()[0]->evaluatorIndex();

  NVF_ERROR(in0 >= 0, "Integer Machine: unknown first input: ", top);
  NVF_ERROR(in1 >= 0, "Integer Machine: unknown second input: ", top);
  NVF_ERROR(in2 >= 0, "Integer Machine: unknown third input: ", top);
  NVF_ERROR(out >= 0, "Integer Machine: unknown out: ", top);

  int index = makeInstructionEntry();
  inst_type_[index] = InstructionType::TERNARY_OP;
  top_type_[index] = top->getTernaryOpType();
  src0_[index] = in0;
  src1_[index] = in1;
  src2_[index] = in2;
  dest_[index] = out;
}

int NaiveValueMachine::makeInstructionEntry() {
  int index = num_of_instructions_++;
  inst_type_.emplace_back(InstructionType::UNARY_OP);
  uop_type_.emplace_back(UnaryOpType::Abs);
  bop_type_.emplace_back(BinaryOpType::Add);
  top_type_.emplace_back(TernaryOpType::Where);
  data_type_.emplace_back(DataType::Null);
  src0_.emplace_back(-1);
  src1_.emplace_back(-1);
  src2_.emplace_back(-1);
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
    case InstructionType::TERNARY_OP:
      runTernaryOp(index);
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
      if (isFloatingPointType(data_type_[index])) {
        dest = PolymorphicValue((double)src);
      } else if (isIntegralType(data_type_[index])) {
        dest = PolymorphicValue((int64_t)src);
      } else if (data_type_[index] == DataType::Bool) {
        dest = PolymorphicValue((bool)src);
      } else {
        NVF_THROW("dtype not supported in evaluator: ", data_type_[index]);
      }
      break;
    case UnaryOpType::Abs:
      dest = abs(src);
      break;
    case UnaryOpType::LogicalNot:
      dest = !src;
      break;
    case UnaryOpType::BitwiseNot:
      dest = ~src;
      break;
    case UnaryOpType::Reciprocal:
      dest = 1.0 / src;
      break;
    case UnaryOpType::Signbit:
      dest = signbit(src);
      break;
    default:
      NVF_CHECK(!"Unexpected operator type ", uop_type_[index]);
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
      NVF_CHECK(rhs != 0);
      dest = lhs / rhs;
      break;
    case BinaryOpType::Mod:
      NVF_CHECK(rhs != 0);
      dest = lhs % rhs;
      break;
    case BinaryOpType::CeilDiv:
      NVF_CHECK(rhs != 0);
      dest = ceildiv(lhs, rhs);
      break;
    case BinaryOpType::LogicalAnd:
      dest = lhs && rhs;
      break;
    case BinaryOpType::BitwiseAnd:
      dest = lhs & rhs;
      break;
    case BinaryOpType::LogicalOr:
      dest = lhs || rhs;
      break;
    case BinaryOpType::BitwiseOr:
      dest = lhs | rhs;
      break;
    case BinaryOpType::BitwiseXor:
      dest = lhs ^ rhs;
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
    case BinaryOpType::LT:
      dest = lhs < rhs;
      break;
    case BinaryOpType::LE:
      dest = lhs <= rhs;
      break;
    case BinaryOpType::Eq:
      dest = lhs == rhs;
      break;
    case BinaryOpType::NE:
      dest = lhs != rhs;
      break;
    case BinaryOpType::GE:
      dest = lhs >= rhs;
      break;
    case BinaryOpType::GT:
      dest = lhs > rhs;
      break;
    case BinaryOpType::Fmod:
      dest = fmod(lhs, rhs);
      break;
    case BinaryOpType::Pow:
      dest = pow(lhs, rhs);
      break;
    default:
      NVF_CHECK(false, "Unexpected operator type ", bop_type_[index]);
  }

  precomputed_values_.defined_[dest_index] = true;
}

void NaiveValueMachine::runTernaryOp(int index) {
  using namespace PolymorphicValue_functions;
  int src0_index = src0_[index];
  int src1_index = src1_[index];
  int src2_index = src2_[index];
  bool src0_is_const = precomputed_values_.is_constant_[src0_index];
  bool src1_is_const = precomputed_values_.is_constant_[src1_index];
  bool src2_is_const = precomputed_values_.is_constant_[src2_index];

  bool src_defined =
      (precomputed_values_.defined_[src0_index] || src0_is_const) &&
      (precomputed_values_.defined_[src1_index] || src1_is_const) &&
      (precomputed_values_.defined_[src2_index] || src2_is_const);

  if (!src_defined) {
    return;
  }
  int dest_index = dest_[index];

  auto& a = precomputed_values_.values_[src0_index];
  auto& b = precomputed_values_.values_[src1_index];
  auto& c = precomputed_values_.values_[src2_index];
  auto& dest = precomputed_values_.values_[dest_index];

  switch (top_type_[index]) {
    case TernaryOpType::Clamp:
      dest = std::min(std::max(a, b), c);
      break;
    case TernaryOpType::Lerp:
      // This is the same lerp computed in helpers.cu
      // https://math.stackexchange.com/a/1798323
      dest = (c < 0.5) ? a + c * (b - a) : b - (b - a) * (1.0 - c);
      break;
    case TernaryOpType::Threshold:
      dest = a <= b ? c : a;
      break;
    case TernaryOpType::Where:
      dest = a ? b : c;
      break;
    default:
      NVF_CHECK(!"Unexpected operator type");
  }

  precomputed_values_.defined_[dest_index] = true;
}

} // namespace nvfuser
