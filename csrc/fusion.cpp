// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <codegen.h>
#include <debug.h>
#include <device_lower/analysis/bank_conflict.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor_params.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel.h>
#include <ops/arith.h>

#include <iterator>
#include <queue>

namespace nvfuser {

static thread_local Fusion* ACTIVE_FUSION = nullptr; // NOLINT

FusionGuard::FusionGuard(Fusion* fusion) : prev_fusion{ACTIVE_FUSION} {
  ACTIVE_FUSION = fusion;
}

FusionGuard::~FusionGuard() {
  ACTIVE_FUSION = prev_fusion;
}

Fusion* FusionGuard::getCurFusion() {
  return ACTIVE_FUSION;
}
void FusionGuard::setCurFusion(Fusion* fusion) {
  ACTIVE_FUSION = fusion;
}

void swap(Fusion& a, Fusion& b) noexcept {
  FUSER_PERF_SCOPE("Fusion swap");

  using std::swap;

  swap(static_cast<IrContainer&>(a), static_cast<IrContainer&>(b));

  swap(a.inputs_, b.inputs_);
  swap(a.outputs_, b.outputs_);

  swap(a.io_alias_, b.io_alias_);
  swap(a.permuted_input_map_, b.permuted_input_map_);
  swap(a.permuted_output_map_, b.permuted_output_map_);
}

std::unique_ptr<SegmentedFusion> Fusion::segment(
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("Segment Fusion");
  return SegmentCandidateFinder::segment(this, args);
}

IrCloner Fusion::copy(const Fusion* from, Fusion* to) {
  to->clear();
  auto ir_cloner = IrContainer::copy(from, to);

  for (auto val : from->vals_) {
    ir_cloner.clone(val)->setDefinition(ir_cloner.clone(val->definition_));
    ir_cloner.clone(val)->setUses(ir_cloner.clone(val->uses_));
  }

  to->inputs_ = ir_cloner.clone(from->inputs_);
  to->outputs_ = ir_cloner.clone(from->outputs_);
  for (auto inp : to->inputs_) {
    inp->setIsFusionInput(true);
  }
  for (auto out : to->outputs_) {
    out->setIsFusionOutput(true);
  }

  // TODO: put this into ir_cloner instead
  for (const auto& entry : from->io_alias_) {
    Val* copied_output = ir_cloner.clone(entry.first);
    Val* copied_input = ir_cloner.clone(entry.second);
    to->io_alias_[copied_output] = copied_input;
  }

  to->permuted_input_map_ = from->permuted_input_map_;
  to->permuted_output_map_ = from->permuted_output_map_;

  to->all_tv_uses_valid_ = from->all_tv_uses_valid_;
  // This should never be true on copy, but copying for completeness.
  to->is_during_update_uses_ = from->is_during_update_uses_;

  for (const auto& i : from->managed_data_) {
    if (i.first.has_value()) {
      to->managed_data_.emplace_back(i.second(ir_cloner, i.first), i.second);
    } else {
      // Don't clone managed data if it has been reset
      to->managed_data_.emplace_back(i.first, i.second);
    }
  }

  for (auto [k, v] : from->managed_named_data_) {
    if (v.first.has_value()) {
      to->managed_named_data_.insert(std::make_pair(
          k, std::make_pair(v.second(ir_cloner, v.first), v.second)));
    }
  }

  return ir_cloner;
}

// Clang tidy complains when using default constructor for IrContainer instead
// of copy constructor. Fusion::copy has a call to IrContainer::copy, so it's
// redundant to use the IrContainer copy constructor, but it is harmless since
// Fusion::copy starts by calling clear().
Fusion::Fusion(const Fusion& other) : IrContainer(other) {
  FUSER_PERF_SCOPE("Fusion copy");
  Fusion::copy(&other, this);
}

Fusion::Fusion(Fusion&& other) noexcept {
  FUSER_PERF_SCOPE("Fusion move");
  swap(*this, other);
}

Fusion& Fusion::operator=(const Fusion& other) {
  FUSER_PERF_SCOPE("Fusion copy assign");
  Fusion copy(other);
  clear();
  swap(*this, copy);
  return *this;
}

Fusion& Fusion::operator=(Fusion&& other) noexcept {
  FUSER_PERF_SCOPE("Fusion move assign");
  clear();
  swap(*this, other);
  return *this;
}

Fusion::~Fusion() {
  clear();
}

void Fusion::clear() noexcept {
  // Perf scope isn't safe here as this function could be called by
  // the Fusion destructor and the scope initializer could call the
  // constructor of Trace, which could throw an exception.
  // FUSER_PERF_SCOPE("Fusion clear");

  IrContainer::clear();

  inputs_.clear();
  outputs_.clear();

  io_alias_.clear();

  permuted_input_map_.clear();
  permuted_output_map_.clear();
  managed_data_.clear();
  managed_named_data_.clear();

  all_tv_uses_valid_ = false;
  is_during_update_uses_ = false;
}

void Fusion::removeExpr(Expr* expr) {
  assertInContainer(expr, "Cannot remove expr ");
  // If we hit this error too frequently, we could lighten the restrictions so
  // that removing something that doesn't exist simply does nothing. For now,
  // we're going with the strictest model which errors.

  for (auto out : expr->outputs()) {
    out->setDefinition(nullptr);
  }

  // Remove uses in inputs
  for (auto inp : expr->inputs()) {
    // Note that if inp is a TensorView, this may call invalidateTvUses
    inp->removeUse(expr);
  }

  IrContainer::removeExpr(expr);
}

void Fusion::removeVal(Val* val) {
  assertInContainer(val, "Cannot remove val ");

  TORCH_CHECK(
      !val->isFusionInput(),
      "Cannot remove val as it is an input of the fusion.");
  TORCH_CHECK(
      !val->isFusionOutput(),
      "Cannot remove val as it is an output of the fusion.");

  Expr* orig = val->definition();
  if (orig != nullptr)
    removeExpr(val->definition());

  for (Expr* use : unordered_uses(val)) {
    removeExpr(use);
  }
  IrContainer::removeVal(val);
}

void Fusion::addInput(Val* input) {
  assertInContainer(input, "Cannot register input ");

  if (input->getValType().value() == ValType::TensorView) {
    auto tv = input->as<TensorView>();
    tv->setMemoryType(MemoryType::Global);
  } else if (input->getValType().value() == ValType::Others) {
    TORCH_CHECK(
        !input->isConst(),
        "Immediate scalar value cannot be added as an input. It is not necessary to pass it as an input.");
  }

  inputs_.push_back(input);
  input->setIsFusionInput(true);

  all_tv_uses_valid_ = false;
}

void Fusion::addOutput(Val* output) {
  // We currently don't support explicitly outputing aliased inputs. This is
  // because they are already marked as output for in-place update. It's tricky
  // to allow marking them explicitly as real output, since that requires us to
  // register/identify output not only by `Val*` pointer, but also by indices;
  // it also requires us to magically arrange `outputs_` entries in proper order
  // ^^^ this doesn't look intuitive on `outputs_` in fusion.
  // I think we can solve this by marking addOutput on io_alias_ keys after
  // fusion is fully defined. Tracking this in #1488
  // Apparently we can't do this neither at the time. I think segmentation
  // unfortunately would call addOutput after we marked io_alias_ map.
  // TORCH_CHECK(io_alias_.count(output) == 0,
  //     "can't register aliased output as real output");
  assertInContainer(output, "Cannot register output ");
  if (output->getValType().value() == ValType::TensorView) {
    auto tv = output->as<TensorView>();
    tv->setMemoryType(MemoryType::Global);
  }
  outputs_.push_back(output);
  output->setIsFusionOutput(true);

  all_tv_uses_valid_ = false;
}

void Fusion::removeInput(Val* input) {
  auto find_input = std::find(inputs_.begin(), inputs_.end(), input);
  if (find_input != inputs_.end()) {
    inputs_.erase(find_input);
  }
  input->setIsFusionInput(false);
  all_tv_uses_valid_ = false;
}

void Fusion::removeOutput(Val* output) {
  auto find_output = std::find(outputs_.begin(), outputs_.end(), output);
  if (find_output != outputs_.end()) {
    outputs_.erase(find_output);
  }
  output->setIsFusionOutput(false);
  all_tv_uses_valid_ = false;
}

void Fusion::replaceOutput(Val* output, Val* replacement) {
  auto find_output = std::find(outputs_.begin(), outputs_.end(), output);
  TORCH_CHECK(find_output != outputs_.end(), "Unable to find output in Fusion");

  if (find_output != outputs_.end()) {
    std::replace_if(
        outputs_.begin(),
        outputs_.end(),
        [&output](Val* v) { return v == output; },
        replacement);

    if (replacement->getValType().value() == ValType::TensorView) {
      replacement->setIsFusionOutput(true);
      replacement->as<TensorView>()->setMemoryType(MemoryType::Global);
    }
    if (output->getValType().value() == ValType::TensorView) {
      output->setIsFusionOutput(false);
      output->as<TensorView>()->setMemoryType(MemoryType::Local);
    }
    // Mark uses invalid so that they will be reset next time uses() is called
    invalidateTvUses();
  }

  // Temporary WAR for issue #1112
  // (https://github.com/csarofeen/pytorch/issues/1112)
  if (io_alias_.count(output) != 0) {
    auto input = io_alias_[output];
    io_alias_.erase(output);
    io_alias_[replacement] = input;
  }
}

std::vector<Expr*> Fusion::exprs() {
  return StmtSort::getExprs(this);
}

bool Fusion::isNoOp() {
  if (exprs().empty()) {
    return true;
  }
  for (auto out_tv : ir_utils::filterByType<TensorView>(outputs())) {
    auto root_dom = TensorDomain::noReductions(out_tv->getMaybeRFactorDomain());
    bool size_zero = false;
    for (auto id : root_dom) {
      if (id->extent()->isConstScalar() && id->extent()->evaluateInt() == 0) {
        size_zero = true;
        break;
      }
    }
    if (!size_zero) {
      return false;
    }
  }
  return true;
}

std::vector<Val*> Fusion::inputsOf(Val* val) {
  return InputsOf::output(this, val);
}

void Fusion::validateInputs() {
  std::unordered_set<Val*> all_inputs;
  for (Val* out : outputs()) {
    for (Val* input : inputsOf(out)) {
      all_inputs.insert(input);
    }
  }

  std::unordered_set<Val*> input_dims;
  auto inp_tvs = ir_utils::filterByType<TensorView>(inputs());
  for (auto tv : inp_tvs) {
    for (auto id : tv->getMaybeRFactorDomain()) {
      input_dims.emplace(id->extent());
    }
  }
  for (Val* input : all_inputs) {
    if (!input->isConstScalar()) {
      TORCH_CHECK(
          input->isFusionInput() ||
              // TODO: Switch:
              inContainer(input),
          // to: input_dims.find(input) != input_dims.end(),
          // https://github.com/csarofeen/pytorch/issues/1365
          "Could not figure out how ",
          input->toString(),
          " is generated, however it was not specified as an input.");
    }
  }
}

std::ostream& Fusion::print(std::ostream& os, bool include_tensor_transforms) {
  FUSER_PERF_SCOPE("Fusion::print");
  FusionGuard fg(this);
  os << "\n%kernel {\n";
  IrMathPrinter op_exprs(os);
  op_exprs.handle(this);
  if (include_tensor_transforms) {
    os << "\nTransformPrinter : \n";
    IrTransformPrinter t_exprs(os);
    t_exprs.handle(this);
  }
  os << "}\n";

  return os;
}

void Fusion::printKernel(const CompileParams& compile_params) {
  FUSER_PERF_SCOPE("Fusion::printKernel");
  TORCH_INTERNAL_ASSERT(
      !this->isA<kir::Kernel>(),
      "Cannot \"print kernel\" of a kernel container. ",
      "This would require lowering during lowering.");
  debug() << codegen::generateCudaKernel(
      GpuLower(this, compile_params).kernel());
}

std::unordered_map<TensorView*, std::pair<std::vector<int>, std::vector<int>>>
Fusion::bankConflictInfo(const CompileParams& compile_params) {
  std::vector<TensorView*> smem_tvs;
  for (auto v : usedMathVals()) {
    auto tv = dynamic_cast<TensorView*>(v);
    if (tv == nullptr) {
      continue;
    }
    if (tv->getMemoryType() == MemoryType::Shared) {
      smem_tvs.push_back(tv);
    }
  }
  if (smem_tvs.empty()) {
    return {};
  }
  manage("smem_tvs", smem_tvs);

  GpuLower lower(this, compile_params);
  auto kernel = lower.kernel();
  auto info = getBankConflictInfo(kernel);

  // Convert TVs in kernel to TVs in fusion
  auto smem_tvs_in_kernel =
      kernel->getManaged<std::vector<TensorView*>>("smem_tvs");
  TORCH_INTERNAL_ASSERT(smem_tvs_in_kernel.size() == smem_tvs.size());
  auto getSmemTvInFusion = [&](Val* v) -> TensorView* {
    auto ti = dynamic_cast<kir::TensorIndex*>(v);
    if (ti == nullptr) {
      return nullptr;
    }
    auto tv = ti->view();
    auto it =
        std::find(smem_tvs_in_kernel.begin(), smem_tvs_in_kernel.end(), tv);
    if (it == smem_tvs_in_kernel.end()) {
      return nullptr;
    }
    auto index = std::distance(smem_tvs_in_kernel.begin(), it);
    return smem_tvs.at(index);
  };

  std::unordered_map<TensorView*, std::pair<std::vector<int>, std::vector<int>>>
      result;
  result.reserve(info.size());
  for (auto i : info) {
    auto expr = i.first;

    // Currently only set and load store op are supported
    TORCH_INTERNAL_ASSERT(expr->inputs().size() == 1);
    TORCH_INTERNAL_ASSERT(expr->outputs().size() == 1);

    auto input = getSmemTvInFusion(expr->input(0));
    auto output = getSmemTvInFusion(expr->output(0));
    if (input == nullptr) {
      TORCH_INTERNAL_ASSERT(i.second.first == 0);
    } else {
      TORCH_INTERNAL_ASSERT(i.second.first != 0);
      result[input].first.push_back(i.second.first);
    }
    if (output == nullptr) {
      TORCH_INTERNAL_ASSERT(i.second.second == 0);
    } else {
      TORCH_INTERNAL_ASSERT(i.second.second != 0);
      result[output].second.push_back(i.second.second);
    }
  }
  return result;
}

void Fusion::printMath(bool from_outputs_only) {
  FUSER_PERF_SCOPE("Fusion::printMath");

  FusionGuard fg(this);
  auto exprs_for_print = exprs();
  debug() << "Inputs:" << std::endl;
  for (auto inp : inputs()) {
    debug() << "  " << inp << ", " << inp->getDataType().value() << std::endl;
  }

  debug() << "Outputs:" << std::endl;
  for (auto out : outputs()) {
    debug() << "  " << out << ", " << out->getDataType().value() << std::endl;
  }

  // If we want everything in the fusion, grab all values without uses to
  // traverse from.
  if (!from_outputs_only) {
    std::vector<Val*> leaf_vals;
    for (auto val : deterministic_vals()) {
      if (val->uses().empty()) {
        leaf_vals.push_back(val);
      }
    }
    exprs_for_print = StmtSort::getExprsTo(this, leaf_vals);
  }

  debug() << "\n%kernel_math {\n";
  for (auto expr : exprs_for_print) {
    debug() << expr;
  }
  debug() << "}\n\n";
}

std::vector<Val*> Fusion::inputsAndCreated() {
  auto result = inputs_;
  for (auto expr : exprs()) {
    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    if (tv_inputs.empty()) {
      for (auto v : expr->outputs()) {
        result.emplace_back(v);
      }
    }
  }
  return result;
}

void Fusion::printTransforms() {
  FUSER_PERF_SCOPE("Fusion::printTransforms");

  FusionGuard fg(this);
  IrTransformPrinter t_exprs(debug());
  t_exprs.handle(this);
}

void Fusion::registerVal(Val* val) {
  if (inContainer(val)) {
    return;
  }

  if (val->fusion()) {
    TORCH_CHECK(
        val->fusion() == this, val, " was not found in the active fusion.");
  }

  IrContainer::registerVal(val);
}

void Fusion::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }

  if (expr->fusion()) {
    TORCH_CHECK(
        expr->fusion() == this, expr, " was not found in the active fusion.");
  }

  IrContainer::registerExpr(expr);

  for (Val* input : expr->inputs()) {
    assertInContainer(input, "Input to expr is invalid, ");
    // Don't just add this expr as a use of the input if it's a tensor as the
    // whole fusion needs to be traversed to rebuild the usage lists
    if (input->isA<TensorView>()) {
      invalidateTvUses();
    } else {
      input->addUse(expr);
    }
  }

  // Kernel is the only container type that is non-ssa. This is mainly (maybe
  // only) because of initialization expressions which would overwrite tensor
  // view definitions.
  bool is_ssa = !this->isA<kir::Kernel>();

  for (Val* output : expr->outputs()) {
    assertInContainer(output, "Output to expr is invalid, ");
    if (output->definition() != nullptr && is_ssa) {
      removeExpr(output->definition());
    }
    if (is_ssa || (!is_ssa && output->definition() == nullptr)) {
      output->setDefinition(expr);
      if (output->isA<TensorView>()) {
        // Updating the definition might change the path to output TVs.
        // If that happens, our definition-based traversal can change and
        // introduce whole new branches, so we need to recompute the uses_
        // vector after setDefinition.
        invalidateTvUses();
      }
    }
  }
}

void Fusion::resetTvUses() {
  FUSER_PERF_SCOPE("Fusion::resetTvUses");
  is_during_update_uses_ = true;

  // getExprs only uses definition, so even if we've modified uses already to
  // remove dead exprs, this could reinsert them. getExprs is also boundeds by
  // inputs as registered inputs will return nullptr as their definition.
  const auto all_tvs = ir_utils::filterByType<TensorView>(vals_);
  const auto used_exprs = StmtSort::getExprs(this);

  for (auto tv : all_tvs) {
    tv->setUses({});
  }

  // Same as in register expr
  for (auto expr : used_exprs) {
    for (Val* input : expr->inputs()) {
      input->addUse(expr);
    }
  }

  all_tv_uses_valid_ = true;
  is_during_update_uses_ = false;
}

std::vector<Val*> Fusion::usedMathVals() {
  // Note that using fusion->inputs() as the argument for the first
  // parameter of getAllValsBetween does not grab all used vals as
  // there can be vals that are created inside a fusion without using
  // anything from inputs. See, for example, tv0 in the
  // FusionOuterSplit test.
  const auto inputs = InputsOf::outputs(this, outputs());
  auto used_math_vals = DependencyCheck::getAllValsBetween(
      {inputs.begin(), inputs.end()}, outputs());
  // When an expre has multiple outputs and only some of them are
  // used, the rest aren't included in used_math_vals as they are not
  // used. However, we want them to be included as they must show up
  // in the fusion.
  std::vector<Val*> vals_to_add;
  std::unordered_set<Val*> added_vals;

  for (auto val : used_math_vals) {
    auto def = val->definition();
    if (def == nullptr || def->outputs().size() < 2) {
      continue;
    }
    for (auto out : def->outputs()) {
      if (std::find(used_math_vals.begin(), used_math_vals.end(), out) ==
          used_math_vals.end()) {
        if (!added_vals.count(out)) {
          vals_to_add.push_back(out);
          added_vals.insert(out);
        }
      }
    }
  }

  used_math_vals.insert(
      used_math_vals.end(), vals_to_add.begin(), vals_to_add.end());

  return used_math_vals;
}

std::vector<Val*> Fusion::terminatingMathVals() {
  VectorOfUniqueEntries<Val*> result;
  auto used_vals = usedMathVals();
  for (auto v : used_vals) {
    // Locate the vals that are not expr outputs but have valid definitions.
    if (unordered_uses(v).empty() && v->definition() != nullptr) {
      result.pushBack(v);
    }
  }
  return result.vector();
}

std::unordered_set<Expr*> Fusion::unordered_uses(const Val* val) const {
  return std::unordered_set<Expr*>(val->uses().begin(), val->uses().end());
}

Expr* Fusion::definition(const Val* val) const {
  assertInContainer(val, "Cannot detect the definition of val, ");
  return val->definition();
}

// Indicate to kernel to set itself up to generate random numbers
bool Fusion::isStochastic() {
  for (auto expr : exprs()) {
    if (expr->isA<RNGOp>()) {
      // Note that RNGOps without seed is not stochastic since the random seed
      // and offset are given as Vals.
      return !expr->as<RNGOp>()->isDeterministic();
    }
  }
  return false;
}

std::vector<Val*> Fusion::getTerminatingOutputs() const {
  FUSER_PERF_SCOPE("getTerminatingOutputs");

  auto is_reachable_to_output = [](Val* val) {
    // traverse to consumers of val and see if there is an output
    std::deque<Val*> consumers;
    std::unordered_set<Val*> visited;
    for (auto use : val->uses()) {
      for (auto consumer : use->outputs()) {
        consumers.push_back(consumer);
      }
    }
    while (!consumers.empty()) {
      auto consumer = consumers.back();
      consumers.pop_back();
      if (consumer->isFusionOutput()) {
        return true;
      }
      // short-cut to break infinite loop with cycles
      if (visited.count(consumer) > 0) {
        continue;
      }
      // consumer is not an output; proceed to its consumers
      for (auto use : consumer->uses()) {
        for (auto consumer_of_consumer : use->outputs()) {
          consumers.push_back(consumer_of_consumer);
        }
      }
      visited.insert(consumer);
    }
    return false;
  };

  std::vector<Val*> terminating_outputs;

  for (auto out : outputs()) {
    // If there is another output reachable from this output, it's not
    // terminating.
    if (is_reachable_to_output(out)) {
      continue;
    }
    terminating_outputs.push_back(out);
  }

  return terminating_outputs;
}

bool Fusion::isAliasCompatible(Val* left, Val* right) {
  // Nullptr check
  if (left == nullptr || right == nullptr) {
    return false;
  }

  // DataType check
  if (!left->getDataType().has_value() || !right->getDataType().has_value() ||
      left->getDataType().value() != right->getDataType().value()) {
    return false;
  }

  // ValType check
  if (!left->getValType().has_value() || !right->getValType().has_value() ||
      left->getValType().value() != right->getValType().value()) {
    return false;
  }

  // Check same number of dimensions if both values are TensorViews
  if (ir_utils::isTV(left) && ir_utils::isTV(right)) {
    return left->as<TensorView>()->nDims() == right->as<TensorView>()->nDims();
  }
  return false;
}

void Fusion::aliasOutputToInput(Val* output, Val* input) {
  // Because we could cast output when input is cast.
  TORCH_INTERNAL_ASSERT(
      !output->isFusionOutput(),
      "Do NOT add aliased output to fusion output outside of `aliasOutputToInput");

  if (!input->isFusionInput()) {
    auto input_expr = input->definition();
    // TORCH_INTERNAL_ASSERT(input_def->isA<UnaryOp>(),
    //     "expected unary op for aliased input");
    TORCH_INTERNAL_ASSERT(
        input_expr->isA<UnaryOp>(), "expected unary op for aliased input");
    auto input_uop = input_expr->as<UnaryOp>();
    TORCH_INTERNAL_ASSERT(
        input_uop->getUnaryOpType() == UnaryOpType::Cast,
        "expected aliased input to be output of cast op");
    input = input_uop->in();
  }
  TORCH_INTERNAL_ASSERT(
      input->getDataType().has_value() && output->getDataType().has_value(),
      "requires DataType to be available for aliased output to input");

  if (input->getDataType().value() != output->getDataType().value()) {
    output = castOp(input->getDataType().value(), output);
  }

  TORCH_INTERNAL_ASSERT(
      isAliasCompatible(input, output),
      "The input and output values are not alias-compatible.");
  io_alias_[output] = input;

  // TODO: output should be marked at the end of fusion definition #1488
  addOutput(output);
}

Val* Fusion::getOutputAlias(Val* output) {
  auto search = io_alias_.find(output);
  if (search != io_alias_.end()) {
    return search->second;
  }
  return nullptr;
}

std::unordered_set<int> Fusion::getIndicesOfAliasedOutputs() const {
  if (io_alias_.empty()) {
    return {};
  }

  std::unordered_set<int> alias_indices;

  for (const auto i : c10::irange(outputs_.size())) {
    if (io_alias_.count(outputs_[i]) != 0) {
      alias_indices.insert((int)i);
    }
  }
  return alias_indices;
}

std::vector<std::pair<int, int>> Fusion::getOutputToInputAliasIndices() const {
  if (io_alias_.empty()) {
    return {};
  }

  std::vector<std::pair<int, int>> alias_indices;
  for (const auto output_idx : c10::irange(outputs_.size())) {
    if (io_alias_.count(outputs_[output_idx]) != 0) {
      bool found = false;
      for (const auto input_idx : c10::irange(inputs_.size())) {
        if (io_alias_.at(outputs_[output_idx]) == inputs_[input_idx]) {
          alias_indices.emplace_back(output_idx, input_idx);
          found = true;
          break;
        }
      }
      TORCH_INTERNAL_ASSERT(
          found,
          "io_alias_ mapping failure, alias output is not present in inputs");
    }
  }
  // can't assert here, we could have segmented fusion where not all alias
  // outputs are present

  return alias_indices;
}

bool Fusion::hasDynamicTransform() {
  return !ir_utils::getTVsWithDynamicTransform(this).empty();
}

// There are three notions of computability used in this function:
//
//   - immediately computable Vals are either immediate constants or Fusion
//     inputs
//   - computable Vals are ones which can be computed by an
//     ExpressionEvaluator if it has the Fusion inputs bound
//   - eventually-computable Vals are ones which can be converted to
//     computable Vals by recursively replacing some producers with other
//     equivalent Vals.
//
// We pass over the scalar_equality_ UnionFind multiple times in this function,
// in order to detect computable, then eventually computable Vals, and finally
// to replace eventually computable Vals with new scalars which are computable.
//
// By the end of this function, the scalar_equality_ UnionFind is updated so
// that any Val that is initially equivalent to an eventually-computable Val has
// a computable Val as its equivalence class representative (root). We then
// replace all scalars with their root.
void Fusion::replaceUncomputableScalars() {
  // NOTE: the UnionFind scalar_equality_ defaults to zero size. At any point,
  // it might have fewer elements than there are scalars in the Fusion. Those
  // scalars which are not represented in the UnionFind have not yet been set
  // equivalent to any others.
  //
  // Note that these unrepresented scalars may need to be replaced since they
  // may have been produced by uncomputable scalars. They may also themselves be
  // producers of represented scalars, since replaceValInExpr exists (i.e. we
  // cannot assume name() corresponds to a topological ordering).
  //
  // Because of this, the first thing we do is resize the UnionFind such that it
  // represents all scalars.
  scalar_equality_.enlarge(
      val_type_name_to_index_[(size_t)ValType::Others].size());

  // whether particular Val (not just some equivalent Val) is computable
  std::vector<bool> computable(scalar_equality_.size(), false);
  // whether this Val is computable as-is or if inputs could be replaced with
  // equivalent Vals (recursively) to form a new computable Val
  std::vector<bool> can_be_made_computable(scalar_equality_.size(), false);

  // Mark all input scalars and immediate constants computable, and for each one
  // proceed downstream as far as possible to propagate computability.
  std::vector<const Val*> immediately_computable_vals;
  for (const UnionFindIndexType s_name : c10::irange(scalar_equality_.size())) {
    const auto s = getValFromName(ValType::Others, s_name);
    if (s->isConst()) {
      immediately_computable_vals.push_back(s);
    }
  }
  for (const auto inp : inputs()) {
    if (const auto inptv = dynamic_cast<TensorView*>(inp)) {
      for (const auto id : inptv->getRootDomain()) {
        immediately_computable_vals.push_back(id->getMaybeExpandedExtent());
      }
    } else if (inp->vtype() == ValType::Others) {
      immediately_computable_vals.push_back(inp);
    }
  }
  // Process comp_queue to implement recursion
  // Since we only recurse once all inputs to a use are marked computable, this
  // is a breadth-first traversal from inputs.
  std::queue<const Val*> comp_queue;
  for (const auto v : immediately_computable_vals) {
    comp_queue.push(v);
  }
  while (!comp_queue.empty()) {
    const auto s = comp_queue.front();
    comp_queue.pop();
    bool s_computable = computable.at(s->name());
    if (s_computable) {
      continue;
    }
    // if there is no definition, mark this as computable and proceed, since
    // that means it was added as a constant or a Fusion input
    const auto def = s->definition();
    if (!def) {
      s_computable = true;
    } else {
      // s is computable iff all of its producers are computable
      s_computable = std::all_of(
          def->inputs().begin(), def->inputs().end(), [&computable](Val* v) {
            return (v->vtype() == ValType::Others) ? computable.at(v->name())
                                                   : false;
          });
    }
    if (s_computable) {
      // Mark as computable and push to queue to recurse into uses
      computable.at(s->name()) = true;
      for (const auto use : s->uses()) {
        if (!ir_utils::isScalarOp(use)) {
          continue;
        }
        for (const auto outp : use->outputs()) {
          comp_queue.push(outp);
        }
      }
    }
  }
  // Now we traverse again to propagate can_be_made_computable. This loosens the
  // recursion criterion to not just whether the Val itself is computable, but
  // also whether it _could_ be made compatible by replacement of some producer
  // Vals.
  //
  // This traversal modifies the UnionFind. We do not change the equivalence
  // classes, but we do change the representatives, so that whenever a Val can
  // be made computable it can be done by recursively replacing all producers
  // with their roots. To do this, whenever a Val is marked
  // can_be_made_computable, if the previous root of its class could not be made
  // computable, that new Val becomes the new root.
  for (const auto v : immediately_computable_vals) {
    comp_queue.push(v);
  }
  while (!comp_queue.empty()) {
    const auto s = comp_queue.front();
    comp_queue.pop();
    bool s_can_be_made_computable = can_be_made_computable.at(s->name());
    if (s_can_be_made_computable) {
      continue; // already marked computable
    }
    // if there is no definition, mark this as computable and proceed, since
    // that means it was added as a constant or a Fusion input
    const auto def = s->definition();
    if (!def) {
      s_can_be_made_computable = true;
    } else {
      // s can be made computable iff all of its producers can be made
      // computable
      s_can_be_made_computable = std::all_of(
          def->inputs().begin(),
          def->inputs().end(),
          [&can_be_made_computable](Val* v) -> bool {
            if (v->vtype() == ValType::Others) {
              return can_be_made_computable.at(v->name());
            }
            return false;
          });
    }
    if (s_can_be_made_computable) {
      can_be_made_computable.at(s->name()) = true;

      // Now we check the equiv class root to see if it was marked computable.
      // If not and this is computable, or if root is not marked
      // "can_be_made_computable" switch s to become new root.
      auto root_name = scalar_equality_.find(s->name());
      if (root_name != s->name() &&
          ((computable.at(s->name()) && !computable.at(root_name)) ||
           !can_be_made_computable.at(root_name))) {
        scalar_equality_.setAsRoot(s->name());
      }

      // Recurse
      for (const auto use : s->uses()) {
        if (!ir_utils::isScalarOp(use)) {
          continue;
        }
        for (const auto outp : use->outputs()) {
          comp_queue.push(outp);
        }
      }
    }
  }

  // Insert new scalars to use as replacements
  //
  // TODO: we only really need to create new scalars for those that are
  // producers of the outputs. Replacing unused scalars pollutes the Fusion, but
  // should not break anything.
  // After this pass, any scalars that can be made computable will have a
  // computable scalar as their equivalence class root.
  for (const auto s_name : c10::irange(scalar_equality_.size())) {
    if (scalar_equality_.find(s_name) == s_name) {
      // Need to process all current roots that can be made computable
      if (!computable.at(s_name) && can_be_made_computable.at(s_name)) {
        const auto s = getValFromName(ValType::Others, s_name);
        comp_queue.push(s);
      }
    }
  }
  while (!comp_queue.empty()) {
    const auto s = comp_queue.front();
    comp_queue.pop();
    if (computable.at(s->name()) || !can_be_made_computable.at(s->name())) {
      continue;
    }

    const auto def = s->definition();
    TORCH_INTERNAL_ASSERT(
        def != nullptr,
        "Uncomputable scalar ",
        s->toString(),
        " cannot be made computible since it has no definition.");

    std::vector<const Val*> producers_to_process;
    for (const auto p : def->inputs()) {
      if (p->vtype() == ValType::Others) {
        if (computable.at(p->name())) {
          continue;
        }
        const auto p_root = scalar_equality_.find(p->name());
        if (computable.at(p_root)) {
          continue;
        }
        // neither p itself nor its representative is computable
        producers_to_process.push_back(getValFromName(ValType::Others, p_root));
      }
    }
    if (producers_to_process.empty()) {
      // we are ready to replace the root of s
      auto newObjectFunc = def->newObjectFunc();
      std::vector<Val*> computable_inputs, computable_outputs;
      std::vector<Statement*> computable_attrs;
      computable_inputs.reserve(def->inputs().size());
      computable_outputs.reserve(def->outputs().size());
      computable_attrs.reserve(def->attributes().size());
      for (auto inp : def->inputs()) {
        TORCH_INTERNAL_ASSERT(
            inp->vtype() == ValType::Others,
            "Found unexpected non-scalar expression ",
            def->toString());
        if (computable.at(inp->name())) {
          computable_inputs.push_back(inp);
        } else {
          const auto root_name = scalar_equality_.find(inp->name());
          auto root = getValFromName(ValType::Others, root_name);
          TORCH_INTERNAL_ASSERT(
              computable.at(root_name),
              "Uncomputable input ",
              s->toString(),
              " has unexpectedly uncomputable root ",
              root->toString());
          computable_inputs.push_back(root);
        }
      }
      for (auto outp : def->outputs()) {
        // Create new Vals for each output
        auto new_outp = IrBuilder::create<Val>(outp->dtype());
        // Mark these outputs equivalent to the originals
        assumeEqual(outp, new_outp);
        if (outp == s) {
          scalar_equality_.setAsRoot(new_outp->name());
        }
        computable_outputs.push_back(new_outp);
      }
      for (auto attr : def->attributes()) {
        if (auto attr_val = dynamic_cast<Val*>(attr)) {
          TORCH_INTERNAL_ASSERT(
              attr_val->vtype() == ValType::Others,
              "Found unexpected non-scalar expression ",
              def->toString());
          if (computable.at(attr_val->name())) {
            computable_attrs.push_back(attr_val);
          } else {
            const auto root_name = scalar_equality_.find(attr_val->name());
            auto root = getValFromName(ValType::Others, root_name);
            TORCH_INTERNAL_ASSERT(
                computable.at(root_name),
                "Uncomputable attribute Val ",
                s->toString(),
                " has unexpectedly uncomputable root ",
                root->toString());
            computable_attrs.push_back(root);
          }
        } else {
          computable_attrs.emplace_back(attr);
        }
      }
      newObjectFunc(
          this, computable_inputs, computable_outputs, computable_attrs);
    } else {
      // recurse to producers if any are uncomputable, then return to s
      comp_queue.push(s);
      for (const auto p : producers_to_process) {
        comp_queue.push(p);
      }
    }
  }

  // Replacement map should be the mapping from val to root in the UnionFind
  std::unordered_map<Val*, Val*> replacement_map;
  for (const auto s_name : c10::irange(scalar_equality_.size())) {
    if (computable.at(s_name)) {
      continue;
    }
    const auto s = getValFromName(ValType::Others, s_name);
    if (s == nullptr) {
      continue; // s is possibly a deleted Val
    }
    const auto root_name = scalar_equality_.find(s_name);
    const auto root = getValFromName(ValType::Others, root_name);
    TORCH_INTERNAL_ASSERT(
        root != nullptr, "Replacement scalar should not be null");
    if (computable.at(root_name)) {
      replacement_map.emplace(s, root);
    } else {
      TORCH_INTERNAL_ASSERT(
          !can_be_made_computable.at(root_name),
          "Scalar ",
          s->toString(),
          "can be made computable but replacement has not yet been set");
    }
  }

  if (replacement_map.empty()) {
    // return early since replaceValue() does not
    return;
  }
}

} // namespace nvfuser
