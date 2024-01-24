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

namespace nvfuser {

/*static*/ thread_local Fusion* FusionGuard::active_fusion_ = nullptr;

FusionGuard::FusionGuard(Fusion* fusion) : prev_fusion_(active_fusion_) {
  active_fusion_ = fusion;
}

FusionGuard::~FusionGuard() {
  active_fusion_ = prev_fusion_;
}

// Cast to non-cast because many users need it.
/*static*/ Fusion* FusionGuard::getCurFusion() {
  return active_fusion_;
}

/*static*/ void FusionGuard::setCurFusion(Fusion* fusion) {
  active_fusion_ = fusion;
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
    Val* copied_input = ir_cloner.clone(entry.second.first);
    to->io_alias_[copied_output] = {copied_input, entry.second.second};
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

flatbuffers::Offset<serde::Fusion> Fusion::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  NVF_ERROR(
      managed_data_.empty(), "Serialization does not support managed_data");
  NVF_ERROR(
      managed_named_data_.empty(),
      "Serialization does not support managed_named_data");

  IrSerde container(this);

  std::vector<int64_t> fb_inputs_vals;
  fb_inputs_vals.reserve(inputs_.size());
  for (auto inp : inputs_) {
    fb_inputs_vals.push_back(container.map(inp));
  }

  std::vector<int64_t> fb_outputs_vals;
  fb_inputs_vals.reserve(outputs_.size());
  for (auto out : outputs_) {
    fb_outputs_vals.push_back(container.map(out));
  }

  std::vector<int64_t> fb_io_alias_keys;
  std::vector<flatbuffers::Offset<serde::AliasInfo>> fb_io_alias_values;
  fb_io_alias_keys.reserve(io_alias_.size());
  fb_io_alias_values.reserve(io_alias_.size());
  for (const auto& entry : io_alias_) {
    fb_io_alias_keys.push_back(container.map(entry.first));
    auto&& [val, info] = entry.second;
    fb_io_alias_values.push_back(serde::CreateAliasInfo(
        builder,
        container.map(val),
        toUnderlying(info.type),
        info.hide_output));
  }

  std::vector<flatbuffers::Offset<serde::Permutation>> fb_permuted_input_map;
  fb_permuted_input_map.reserve(permuted_input_map_.size());
  for (const auto& entry : permuted_input_map_) {
    fb_permuted_input_map.push_back(
        serde::CreatePermutationDirect(builder, entry.first, &entry.second));
  }

  std::vector<flatbuffers::Offset<serde::Permutation>> fb_permuted_output_map;
  fb_permuted_output_map.reserve(permuted_output_map_.size());
  for (const auto& entry : permuted_output_map_) {
    fb_permuted_output_map.push_back(
        serde::CreatePermutationDirect(builder, entry.first, &entry.second));
  }

  return serde::CreateFusionDirect(
      builder,
      IrContainer::serialize(container, builder),
      &fb_inputs_vals,
      &fb_outputs_vals,
      &fb_io_alias_keys,
      &fb_io_alias_values,
      &fb_permuted_input_map,
      &fb_permuted_output_map,
      all_tv_uses_valid_,
      is_during_update_uses_);
}

void Fusion::deserialize(const serde::Fusion* buffer) {
  FUSER_PERF_SCOPE("Fusion constructor deserialize");
  NVF_ERROR(buffer != nullptr, "serde::Fusion is nullptr.");

  FusionGuard fg(this);
  IrContainer::deserialize(buffer->container());

  for (int64_t index :
       c10::irange((int64_t)buffer->container()->vals()->size())) {
    getVal<Val>(index)->deserializeExpr(
        this, buffer->container()->vals()->Get(index));
  }

  inputs_.reserve(buffer->input_vals()->size());
  std::transform(
      buffer->input_vals()->begin(),
      buffer->input_vals()->end(),
      std::back_inserter(inputs_),
      [&](int64_t index) {
        auto inp = getVal<Val>(index);
        inp->setIsFusionInput(true);
        return inp;
      });

  outputs_.reserve(buffer->output_vals()->size());
  std::transform(
      buffer->output_vals()->begin(),
      buffer->output_vals()->end(),
      std::back_inserter(outputs_),
      [&](int64_t index) {
        auto out = getVal<Val>(index);
        out->setIsFusionOutput(true);
        return out;
      });

  NVF_CHECK(
      buffer->io_alias_keys()->size() == buffer->io_alias_values()->size());
  for (size_t index : c10::irange(buffer->io_alias_keys()->size())) {
    Val* key_val = getVal<Val>(buffer->io_alias_keys()->Get(index));
    const serde::AliasInfo* fb_alias_info =
        buffer->io_alias_values()->Get(index);
    NVF_ERROR(fb_alias_info != nullptr, "serde::AliasInfo is nullptr.");

    Val* val_lhs = getVal<Val>(fb_alias_info->value());
    AliasType at_type_enum =
        static_cast<AliasType>(fb_alias_info->alias_type_enum());
    io_alias_.emplace(
        key_val,
        std::make_pair(
            val_lhs, AliasInfo{at_type_enum, fb_alias_info->hide_output()}));
  }

  for (const serde::Permutation* fb_permutation :
       *buffer->permuted_input_map()) {
    NVF_ERROR(fb_permutation != nullptr, "serde::Permutation is nullptr.");
    permuted_input_map_.emplace(
        fb_permutation->key(),
        std::vector<int64_t>(
            fb_permutation->value()->begin(), fb_permutation->value()->end()));
  }

  for (const serde::Permutation* fb_permutation :
       *buffer->permuted_output_map()) {
    NVF_ERROR(fb_permutation != nullptr, "serde::Permutation is nullptr.");
    permuted_output_map_.emplace(
        fb_permutation->key(),
        std::vector<int64_t>(
            fb_permutation->value()->begin(), fb_permutation->value()->end()));
  }

  all_tv_uses_valid_ = buffer->all_tv_uses_valid();
  is_during_update_uses_ = buffer->is_during_update_uses();
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

  NVF_CHECK(
      !val->isFusionInput(),
      "Cannot remove val as it is an input of the fusion.");
  NVF_CHECK(
      !val->isFusionOutput(),
      "Cannot remove val as it is an output of the fusion.");

  if (Expr* orig = val->definition()) {
    removeExpr(orig);
  }

  // We previously first looped over val->uses() and removed them all from the
  // Fusion. This seems correct at first glance, but it is incomplete since
  // `val->uses()` actually only gives all live uses. When there is dead code in
  // the Fusion that includes some uses of a val that is to be removed, we can
  // wind up with an expression that holds an invalid pointer to the removed
  // value in its inputs(). In https://github.com/NVIDIA/Fuser/issues/1270 this
  // caused a segfault when the fusion was cloned since that will clone not only
  // live objects but also these dangerous dangling dead ones.
  std::vector<Expr*> exprs_to_remove;
  for (Expr* e : exprs_) {
    if (!inContainer(e)) {
      continue;
    }
    if (std::find(e->inputs().begin(), e->inputs().end(), val) !=
        e->inputs().end()) {
      // Avoid removing until after we've looped through exprs_
      exprs_to_remove.push_back(e);
    }
  }
  for (auto e : exprs_to_remove) {
    removeExpr(e);
  }
  IrContainer::removeVal(val);
}

void Fusion::addInput(Val* input) {
  assertInContainer(input, "Cannot register input ");

  if (input->getValType().value() == ValType::TensorView) {
    auto tv = input->as<TensorView>();
    tv->setMemoryType(MemoryType::Global);
  } else if (input->getValType().value() == ValType::Others) {
    NVF_CHECK(
        !input->isConst(),
        "Immediate scalar value cannot be added as an input. It is not necessary to pass it as an input.");
  }

  NVF_CHECK(
      !input->isFusionInput(),
      "Val: ",
      input->toString(),
      " is already registered as input, duplicated inputs is not allowed");
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
  // NVF_CHECK(io_alias_.count(output) == 0,
  //     "can't register aliased output as real output");
  assertInContainer(output, "Cannot register output ");
  if (output->isA<TensorView>()) {
    output->as<TensorView>()->setMemoryType(MemoryType::Global);
  } else {
    NVF_CHECK(
        output->isA<PipelineVal>() &&
            output->as<PipelineVal>()->getOriginalVal()->isA<TensorView>(),
        "Non-TensorView outputs are not supported at this point: ",
        output->toString());
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
  NVF_CHECK(find_output != outputs_.end(), "Unable to find output in Fusion");

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
    const std::vector<IterDomain*>& root_dom =
        TensorDomain::noReductions(out_tv->getMaybeRFactorDomain());
    const bool size_zero =
        std::any_of(root_dom.begin(), root_dom.end(), [](IterDomain* id) {
          return id->extent()->isConstScalar() && id->extent()->evaluate() == 0;
        });
    if (!size_zero) {
      return false;
    }
  }

  return true;
}

std::vector<Val*> Fusion::inputsOf(Val* val) {
  return InputsOf::output(val);
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
      NVF_CHECK(
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

std::ostream& Fusion::print(std::ostream& os, bool include_tensor_transforms)
    const {
  FUSER_PERF_SCOPE("Fusion::print");
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
  NVF_ERROR(
      !this->isA<kir::Kernel>(),
      "Cannot \"print kernel\" of a kernel container. ",
      "This would require lowering during lowering.");
  GpuLower lower(this, compile_params);
  lower.run();
  debug() << codegen::generateCudaKernel(lower.kernel());
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
  lower.run();
  auto kernel = lower.kernel();
  auto info = getBankConflictInfo(kernel);

  // Convert TVs in kernel to TVs in fusion
  auto smem_tvs_in_kernel =
      kernel->getManaged<std::vector<TensorView*>>("smem_tvs");
  NVF_ERROR(smem_tvs_in_kernel.size() == smem_tvs.size());
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
    NVF_ERROR(expr->inputs().size() == 1);
    NVF_ERROR(expr->outputs().size() == 1);

    auto input = getSmemTvInFusion(expr->input(0));
    auto output = getSmemTvInFusion(expr->output(0));
    if (input == nullptr) {
      NVF_ERROR(i.second.first == 0);
    } else {
      NVF_ERROR(i.second.first != 0);
      result[input].first.push_back(i.second.first);
    }
    if (output == nullptr) {
      NVF_ERROR(i.second.second == 0);
    } else {
      NVF_ERROR(i.second.second != 0);
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
    exprs_for_print = StmtSort::getExprsTo(leaf_vals);
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
    NVF_CHECK(
        val->fusion() == this, val, " was not found in the active fusion.");
  }

  IrContainer::registerVal(val);
}

void Fusion::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }

  if (expr->fusion()) {
    NVF_CHECK(
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
  const auto inputs = InputsOf::outputs(outputs());
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

  return true;
}

void Fusion::aliasOutputToInput(Val* output, Val* input, const AliasType type) {
  if (type == AliasType::InplaceUpdate) {
    // `input` can be a cast of a fusion input.
    if (!input->isFusionInput()) {
      auto input_expr = input->definition();
      NVF_ERROR(
          input_expr->isA<UnaryOp>(), "expected unary op for aliased input");
      auto input_uop = input_expr->as<UnaryOp>();
      NVF_ERROR(
          input_uop->getUnaryOpType() == UnaryOpType::Cast,
          "expected aliased input to be output of cast op");
      input = input_uop->in();
    }
    NVF_ERROR(
        input->getDataType().has_value() && output->getDataType().has_value(),
        "requires DataType to be available for aliased output to input");

    if (input->getDataType().value() != output->getDataType().value()) {
      output = castOp(input->getDataType().value(), output);
    }
  }

  NVF_ERROR(
      isAliasCompatible(input, output),
      "The input and output values are not alias-compatible.");
  // Let integration hide any output that wasn't a fusion output when
  // `aliasOutputToInput` was called. For example, running mean and var for
  // batch norm.
  io_alias_[output] = {input, AliasInfo{type, !output->isFusionOutput()}};

  // TODO: output should be marked at the end of fusion definition #1488
  if (!output->isFusionOutput()) {
    addOutput(output);
  }
}

std::pair<Val*, const AliasInfo*> Fusion::getOutputAlias(Val* output) const {
  if (auto search = io_alias_.find(output); search != io_alias_.end()) {
    const std::pair<Val*, AliasInfo>& in_val_and_info = search->second;
    return {in_val_and_info.first, &in_val_and_info.second};
  }
  return {nullptr, nullptr};
}

bool Fusion::hasDynamicTransform() {
  return !ir_utils::getTVsWithDynamicTransform(this).empty();
}

} // namespace nvfuser
