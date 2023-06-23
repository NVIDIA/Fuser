// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <dynamic_transform.h>
#include <executor_kernel_arg.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <transform_iter.h>
#include <transform_view.h>
#include <utils.h>

#include <optional>

namespace nvfuser {

DynamicTransformInitialInfo DynamicTransformInitialInfo::clone(
    IrCloner& ir_cloner) const {
  DynamicTransformInitialInfo cloned_info(
      static_cast<Fusion*>(ir_cloner.container()));
  cloned_info.dynamic_reshaped_tvs_.reserve(dynamic_reshaped_tvs_.size());
  for (const auto op : dynamic_reshaped_tvs_) {
    if (op) {
      cloned_info.dynamic_reshaped_tvs_.push_back(ir_cloner.clone(op));
    }
  }
  cloned_info.dynamic_resized_ids_.reserve(dynamic_resized_ids_.size());
  for (const auto op : dynamic_resized_ids_) {
    if (op) {
      cloned_info.dynamic_resized_ids_.push_back(ir_cloner.clone(op));
    }
  }
  cloned_info.maybe_zero_extents_.reserve(maybe_zero_extents_.size());
  for (const auto v : maybe_zero_extents_) {
    if (v) {
      cloned_info.maybe_zero_extents_.insert(ir_cloner.clone(v));
    }
  }
  cloned_info.name_to_tensorview_.reserve(name_to_tensorview_.size());
  for (const auto kv : name_to_tensorview_) {
    if (kv.second) {
      cloned_info.name_to_tensorview_[kv.first] = ir_cloner.clone(kv.second);
    }
  }
  cloned_info.root_dynamic_vals_.reserve(root_dynamic_vals_.size());
  for (const auto v : root_dynamic_vals_) {
    if (v) {
      cloned_info.root_dynamic_vals_.insert(ir_cloner.clone(v));
    }
  }
  return cloned_info;
}

std::string DynamicTransformInitialInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformInitialInfo\n";
  std::string indent = "  ";
  ss << indent << "Dynamic reshaped TensorViews:\n";
  for (const auto& op : dynamic_reshaped_tvs_) {
    ss << indent << indent << op->toString() << "\n";
  }
  ss << indent << "Dynamic resized IterDomains:\n";
  for (const auto& op : dynamic_resized_ids_) {
    ss << indent << indent << op->toString() << "\n";
  }
  ss << indent << "Dynamic extent Vals:\n";
  for (const auto& v : maybe_zero_extents_) {
    ss << indent << indent << v->toString() << "\n";
  }
  ss << indent << "Name to TensorView mapping:\n";
  for (const auto& kv : name_to_tensorview_) {
    ss << indent << indent << kv.first << " => " << kv.second->toString()
       << "\n";
  }
  ss << indent << "Root dynamic Vals:\n";
  for (const auto& v : root_dynamic_vals_) {
    ss << indent << indent << v->toString() << "\n";
  }
  return ss.str();
}

//! Gather information about concretizing transformations without
//! concrete input sizes.
class DynamicTransformInitialInfoBuilder : public IterVisitor {
 public:
  DynamicTransformInitialInfoBuilder(Fusion* fusion) : info_(fusion) {
    TORCH_INTERNAL_ASSERT(
        !fusion->isA<kir::Kernel>(),
        "Invalid container. Kernel container not allowed.\n");

    traverseTo(fusion, fusion->getTerminatingOutputs(), false, false);

    finalizeDynamicVals();
  }

  const auto& getInfo() const {
    return info_;
  }

 private:
  using IterVisitor::handle;

  //! Find views that have symbolic outputs
  void handle(ViewOp* op) override {
    auto inp_tv = op->in()->as<TensorView>();
    auto out_tv = op->out()->as<TensorView>();
    // If there's no symbolic axis, this is a static reshape op
    if (out_tv->domain()->hasSymbolicAxis()) {
      info_.dynamic_reshaped_tvs_.push_back(out_tv);

      // Input and output extent expressions both affect concretization
      for (const auto& id :
           TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain())) {
        leaf_dynamic_vals_.push_back(id->extent());
      }
      for (const auto& id : out_tv->getMaybeRFactorDomain()) {
        leaf_dynamic_vals_.push_back(id->extent());
      }
    }
  }

  //! Detect possibly empty TensorViews and dynamic IterDomain transforms
  void handle(TensorView* tv) override {
    info_.name_to_tensorview_[tv->name()] = tv;
    const auto& rfd = tv->getMaybeRFactorDomain();
    for (auto id : rfd) {
      if (!id->extent()->isConstScalar() || id->extent()->evaluateInt() == 0) {
        info_.maybe_zero_extents_.insert(id->extent());
        leaf_dynamic_vals_.push_back(id->extent());
      }
      if (!id->definition() || id->getIterType() != IterType::Symbolic) {
        continue;
      }
      if (id->definition()->isA<Resize>()) {
        info_.dynamic_resized_ids_.push_back(id);
        // extent of output determines its IterType
        leaf_dynamic_vals_.push_back(id->extent());
      }
    }
  }

  //! Process vector of leaf dynamic values by finding inputs and recording the
  //! result into info_
  void finalizeDynamicVals() {
    const auto inputs = InputsOf::outputs(info_.fusion(), leaf_dynamic_vals_);
    info_.root_dynamic_vals_.insert(inputs.begin(), inputs.end());

    // initial_info_ provides a set of Vals that are used for concretization.
    // Here we check which scalar inputs, if any, correspond to any of those
    // Vals. These will be the inputs that are explicitly used in the cache ID
    // for KernelArgumentHolder.
    auto dyn_vals = info_.getRootDynamicVals();
    for (const auto i : c10::irange(info_.fusion()->inputs().size())) {
      auto input = info_.fusion()->inputs().at(i);
      if (dyn_vals.find(input) != dyn_vals.end()) {
        info_.scalar_inputs_affecting_concretization_.insert(i);
      }
    }
  }

 private:
  DynamicTransformInitialInfo info_;

  //! This is a collection of scalars that are explicitly checked during
  //! concretization of dynamic ops, meaning they influence the structure of the
  //! resulting concretized Fusion. We track these while traversing the graph
  //! and when we are finished traversing we extract all of the corresponding
  //! non-constant root Vals, which provides us with a minimal list of input
  //! scalars that influence concretization. That list of scalars is then used
  //! to compute a minimal cache key in InputsIdLookup::lookupId().
  std::vector<Val*> leaf_dynamic_vals_;
};

namespace { // Anonymous namespace for local function findEmptyTensors

//! This performs a depth-first search from outputs toward inputs for empty
//! tensors. It does not traverse past any zero tensors it finds; this is why
//! this is implemented as a single function instead of with BackwardVisitor.
//! Additionally, we check inputs since they might actually be disconnected from
//! outputs.
std::vector<EmptyTensorDescriptor> findEmptyTensors(
    ExpressionEvaluator* expr_eval) {
  auto fusion = FusionGuard::getCurFusion();
  std::vector<EmptyTensorDescriptor> empty_tensors;
  std::vector<Val*> vals(fusion->inputs());
  vals.insert(vals.end(), fusion->outputs().begin(), fusion->outputs().end());
  std::unordered_set<TensorView*> visited;

  while (!vals.empty()) {
    auto val = vals.back();
    vals.pop_back();
    if (!val->isA<TensorView>()) {
      continue;
    }
    auto tv = val->as<TensorView>();
    if (visited.find(tv) != visited.end()) {
      continue;
    }
    visited.insert(tv);

    std::vector<size_t> empty_axes;
    auto rfactor = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    bool empty = false;
    for (size_t i : c10::irange(rfactor.size())) {
      auto id = rfactor.at(i);
      auto extent_eval = expr_eval->evaluate(id->extent());
      TORCH_INTERNAL_ASSERT(
          extent_eval.has_value(),
          "When finding empty tensors: could not evaluate extent of ",
          id->toString());
      if (extent_eval.value().as<int64_t>() == 0) {
        empty_axes.push_back(i);
        empty = true;
      }
    }
    if (empty) {
      // Replace with full. Note that even if the definition was a FullOp, we
      // still mark this tensor for replacement, so that we can ensure the
      // empty axes are marked with constant zeroes
      empty_tensors.push_back(EmptyTensorDescriptor{tv->name(), empty_axes});
      continue;
    }
    if (tv->definition()) {
      for (auto inp : tv->definition()->inputs()) {
        vals.push_back(inp);
      }
    }
  }
  return empty_tensors;
}

} // namespace

DynamicTransformConcretizationInfo::DynamicTransformConcretizationInfo(
    const DynamicTransformInitialInfo* initial_info,
    ExpressionEvaluator* expr_eval)
    : initial_info_(initial_info) {
  TORCH_INTERNAL_ASSERT(
      !fusion()->isA<kir::Kernel>(),
      "Invalid container. Kernel container not allowed.\n");

  // Make sure all exactly mapped IDs have the same value in the
  // evaluator when any one of the IDs has a known value
  expr_eval->propagateBoundValuesThroughExactMaps(initial_info_->fusion());

  analyzeReshapes(expr_eval);

  analyzeResizes(expr_eval);

  bool has_empty_tensor = false;
  for (auto ext : initial_info_->getDynamicExtentVals()) {
    auto ext_opt = expr_eval->evaluate(ext);
    TORCH_INTERNAL_ASSERT(
        ext_opt.has_value(),
        "Could not evaluate dynamic extent: ",
        ext->toString());
    if (ext_opt.value().as<int64_t>() == 0) {
      has_empty_tensor = true;
      break;
    }
  }
  // Find a minimal set of empty tensors to replace with full() calls
  // NOTE: this does a backward traversal from outputs.
  if (has_empty_tensor) {
    empty_tensors_ = findEmptyTensors(expr_eval);
  }
}

void DynamicTransformConcretizationInfo::analyzeReshapes(
    ExpressionEvaluator* expr_eval) {
  const auto& reshape_tvs = initial_info_->getDynamicReshapedTensorViews();
  for (const auto tv_index : c10::irange(reshape_tvs.size())) {
    auto out_tv = reshape_tvs.at(tv_index);
    auto op = out_tv->definition()->as<ViewOp>();
    auto inp_tv = op->in()->as<TensorView>();

    // If there's no symblic axis, this is a static reshape op
    if (!out_tv->domain()->hasSymbolicAxis()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        out_tv->hasRFactor(),
        "Unexpected output tv of ViewOp: ",
        out_tv->toString());

    const auto& inp_dom =
        TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain());

    // Determine input shape using expr evaluator
    std::vector<int64_t> inp_shape(inp_dom.size(), 0);
    for (const auto i : c10::irange(inp_dom.size())) {
      auto inp_id = inp_dom.at(i);
      // This should have been validated when initially creating reshape
      // op, but just in case
      TORCH_INTERNAL_ASSERT(
          !inp_id->maybePartial(),
          "Invalid domain to reshape: ",
          inp_id->toString());
      auto extent_val = expr_eval->evaluate(inp_id->extent());
      TORCH_INTERNAL_ASSERT(
          extent_val.has_value(),
          "Cannot evaluate the extent of an input domain to reshape: ",
          inp_id->toString());
      TORCH_INTERNAL_ASSERT(
          extent_val->isInt(),
          "Invalid evaluated value of domain extent: ",
          inp_id->toString());
      TORCH_INTERNAL_ASSERT(
          extent_val->as<int64_t>() > 0,
          "Invalid input domain extent: ",
          extent_val->as<int64_t>());
      inp_shape.at(i) = extent_val->as<int64_t>();
    }

    const auto& out_dom = out_tv->getMaybeRFactorDomain();

    // Determine output shape using expr evaluator. Note there may be
    // one domain of extent -1
    std::vector<int64_t> out_shape(out_dom.size(), 0);
    bool extent_m1_found = false;
    for (const auto i : c10::irange(out_dom.size())) {
      auto out_id = out_dom.at(i);
      auto extent_val = expr_eval->evaluate(out_id->extent());
      TORCH_INTERNAL_ASSERT(
          extent_val.has_value(),
          "Cannot evaluate the extent of an output domain to reshape: ",
          out_id->toString());
      TORCH_INTERNAL_ASSERT(
          extent_val->isInt(),
          "Invalid evaluated value of domain extent: ",
          out_id->toString());
      const auto extent_int = extent_val->as<int64_t>();
      if (extent_int == -1) {
        TORCH_INTERNAL_ASSERT(
            !extent_m1_found,
            "Multiple output domains of size -1 not allowed",
            out_tv->toString());
        extent_m1_found = true;
      } else {
        TORCH_INTERNAL_ASSERT(
            extent_int > 0, "Invalid output domain extent: ", extent_int);
      }
      out_shape.at(i) = extent_int;
    }

    auto view_result = analyzeView(inp_tv, inp_shape, out_shape);

    reshape_transforms_.emplace_back(tv_index, view_result);
  }
}

void DynamicTransformConcretizationInfo::analyzeResizes(
    ExpressionEvaluator* expr_eval) {
  const auto& resize_ids = initial_info_->getDynamicResizedIterDomains();
  for (const auto id_index : c10::irange(resize_ids.size())) {
    auto out_id = resize_ids.at(id_index);
    auto op = out_id->definition()->as<Resize>();

    TORCH_CHECK(
        out_id->getIterType() == IterType::Symbolic,
        "Found non-dynamic Resize in initial concretization info: ",
        op->toString());

    auto extent_val = expr_eval->evaluate(out_id->extent());
    TORCH_INTERNAL_ASSERT(
        extent_val.has_value(),
        "Cannot evaluate the extent of a resized domain: ",
        out_id->toString());
    TORCH_INTERNAL_ASSERT(
        extent_val->isInt(),
        "Invalid evaluated value of resized domain extent: ",
        out_id->toString());
    auto extent_int = extent_val->as<int64_t>();
    TORCH_INTERNAL_ASSERT(
        extent_int > 0,
        "Invalid resized domain extent ",
        extent_int,
        " for domain ",
        out_id->toString());

    auto iter_type =
        extent_int == 1 ? IterType::Broadcast : IterType::Iteration;

    resize_itertypes_.emplace_back(id_index, iter_type);
  }
}

bool DynamicTransformConcretizationInfo::operator==(
    const DynamicTransformConcretizationInfo& other) const {
  if (this == &other) {
    return true;
  }

  if (reshape_transforms_.size() != other.reshape_transforms_.size() ||
      resize_itertypes_.size() != other.resize_itertypes_.size() ||
      empty_tensors_.size() != other.empty_tensors_.size()) {
    return false;
  }

  for (const auto i : c10::irange(reshape_transforms_.size())) {
    const auto& analysis = reshape_transforms_.at(i);
    const auto& other_analysis = other.reshape_transforms_.at(i);
    if (analysis != other_analysis) {
      return false;
    }
  }

  for (const auto i : c10::irange(resize_itertypes_.size())) {
    const auto& itertype = resize_itertypes_.at(i);
    const auto& other_itertype = other.resize_itertypes_.at(i);
    if (itertype != other_itertype) {
      return false;
    }
  }

  for (const auto i : c10::irange(empty_tensors_.size())) {
    const auto& et = empty_tensors_.at(i);
    const auto& other_et = other.empty_tensors_.at(i);
    if (et != other_et) {
      return false;
    }
  }

  return true;
}

std::string DynamicTransformConcretizationInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformConcretizationInfo\n";
  std::string indent = "  ";
  ss << indent << "Empty tensors:\n";
  for (const auto& kv : empty_tensors_) {
    ss << indent << indent << initial_info_->lookUpTV(kv.tv_name)->toString()
       << " has zero extent in these axes:";
    for (auto i : kv.empty_axes) {
      ss << " " << i;
    }
    ss << "\n";
  }
  ss << indent << "Reshape:\n";
  for (const auto& [tv_index, analyze_result] : reshape_transforms_) {
    auto tv = initial_info_->getDynamicReshapedTensorViews().at(tv_index);
    ss << indent << indent << tv->toString() << " (index=" << tv_index << "), "
       << analyze_result.toString() << "\n";
  }
  ss << indent << "Resize:\n";
  for (const auto& [id_index, iter_type] : resize_itertypes_) {
    auto id = initial_info_->getDynamicResizedIterDomains().at(id_index);
    ss << indent << indent << id->toString() << " (index=" << id_index << "), "
       << iter_type << "\n";
  }
  return ss.str();
}

//! Concretize a symbolic fusion with concrete transformation info
class DynamicTransformConcretizer : public OptOutMutator {
 public:
  DynamicTransformConcretizer(
      Fusion* fusion,
      const DynamicTransformConcretizationInfo* info)
      : info_(info) {
    TORCH_INTERNAL_ASSERT(
        fusion == info->fusion(),
        "Invalid DynamicTransformInitialInfo. The associated Fusion is different from the given Fusion");
    FusionGuard fg(fusion);
    concretize();
  }

 private:
  void concretize();

  //! removeEmptyBranches sets definitions of empty tensors to full(), and
  //! replaces uses like reductions over empty axes with full calls.
  //!
  //! Consider the following Fusion with input T0, T1 and output T3:
  //!
  //!    T0
  //!     |
  //!    sum
  //!     |
  //!    T2    T1
  //!      \   /
  //!       mul
  //!        |
  //!       T3
  //!
  //! If T1 has any size-zero dimensions, then we know that T3 is also empty,
  //! and T2 may be empty as well (unless it's broadcasting in all the empty
  //! dimensions of T1). In this case, we can replace the entire Fusion with a
  //! single call to full():
  //!
  //!     T0    T1
  //!
  //!       full
  //!        |
  //!       T3
  //!
  //! Notice that the graph is now disconnected since T0 and T1 remain as Fusion
  //! inputs.
  //!
  //! If instead, T1 is not empty, but T0 is, then there are two possibilities:
  //!   a) If any empty axes of T0 are not reduced, then T2 shares those empty
  //!   axes, in which case T3 must also be empty, so we can rewrite the Fusion
  //!   the same way as above, by redefining T3 = full(shape)
  //!
  //!   b) If instead the empty axes of T0 are all being reduced in the sum,
  //!   then T2 is not empty. In this case, since T0 is an input, rewriting it
  //!   as a full() output is not helpful. However, we know that any use of an
  //!   empty tensor does not require computation over T0, so we can rewrite it.
  //!   In this case, we can rewrite the sum as a full(shape, 0) since the sum
  //!   over an empty tensor is 0 (more generally, the initial value of the
  //!   reduction). This leads to the following rewritten Fusion:
  //!
  //!    T0
  //!
  //!    full
  //!     |
  //!    T2    T1
  //!      \   /
  //!       mul
  //!        |
  //!       T3
  //!
  //! After this call, the Fusion will only contain empty tensors if they are
  //! Fusion inputs or outputs. Furthermore, output tensors will have constant
  //! zeros for the extents of empty axes.
  //!
  //! Instead of sum, we may encounter pad or cat ops. These are handled as
  //! follows:
  //!
  //!   Pads of empty tensors are replaced with full() using a fill value equal
  //!   to the pad value.
  //!
  //!   Cat of tensors including some that are empty in the cat dimension are
  //!   simply replaced with a call to cat() that excludes the empty tensors.
  //!   Note that if any non-cat dimensions are empty, then the output will be
  //!   empty as well and the cat becomes dead code, as in the second example
  //!   with empty T0 from above.
  void removeEmptyBranches();

  //! replaceWithFull modifies the Fusion by replacing tv with output of full()
  //! expression in outputs and all uses.
  TensorView* replaceWithFull(
      TensorView* tv,
      std::vector<Val*>& new_shape,
      Val* fill_value = nullptr);

  //! Replace a TensorView with a new one in all uses, and in inputs and
  //! outputs.
  void replaceTV(TensorView* old_tv, TensorView* new_tv);

  void concretizeReshape();

  void concretizeResize();

  //! Use this instead of calling registerMutation directly, since it will also
  //! check that the concretized value is a valid input to all of its uses.
  void registerConcretization(Val* old_val, Val* new_val) {
    checkConcretizedUses(old_val, new_val);
    registerMutation(old_val, new_val);
  }

  //! Check uses of old_val to ensure that new_val does not violate
  //! assumptions. This is currently only used to check that inputs to SqueezeOp
  //! are marked broadcast during concretization.
  void checkConcretizedUses(Val* old_val, Val* new_val) const;

  using OptOutMutator::mutate;

  void mutate(TensorView* tv) final;

  void mutate(TensorDomain* td) final;

  //! Concretizes the root domain of a symbolic consumer tensor from
  //! its producer domains. Returns true if any root ID is concretized.
  bool propagateFromProducerToConsumer(TensorView* consumer);

  TensorView* maybeReplaced(TensorView* tv) {
    auto it = replaced_tvs_.find(tv);
    if (it == replaced_tvs_.end()) {
      return tv;
    }
    return maybeReplaced(it->second);
  };

 private:
  const DynamicTransformConcretizationInfo* info_;

  //! As we replace TensorViews, we want to operate on the replaced values
  //! instead of the originals. This map lets use keep track of multiple
  //! replacements and get the latest one.
  std::unordered_map<TensorView*, TensorView*> replaced_tvs_;
};

void DynamicTransformConcretizer::concretize() {
  // Concretize all dynamic reshape ops
  concretizeReshape();

  // Set output IterTypes for dynamic resize ops
  concretizeResize();

  // Concretize empty tensors last in case some empty tensor are fed into
  // replaced dynamic ops.
  removeEmptyBranches();

  // Finally, propagate concretized domains
  auto all_stmts = StmtSort::getStmts(info_->fusion(), true);
  for (auto stmt : all_stmts) {
    if (stmt->isA<Val>()) {
      mutate(stmt);
    }
  }
}

void DynamicTransformConcretizer::removeEmptyBranches() {
  auto fusion = FusionGuard::getCurFusion();
  for (const auto& empty_tv_descr : info_->getEmptyTensors()) {
    auto tv = info_->initialInfo()->lookUpTV(empty_tv_descr.tv_name);
    auto rfactor = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    std::vector<Val*> new_shape;
    new_shape.reserve(rfactor.size());
    for (auto id : rfactor) {
      new_shape.push_back(id->getMaybeExpandedExtent());
    }
    for (auto ax : empty_tv_descr.empty_axes) {
      // Hard-code zero extent for empty axes. This lets us detect empty input
      // and output tensors during scheduling/execution.
      registerConcretization(new_shape[ax], fusion->zeroVal());
      new_shape[ax] = fusion->zeroVal();
    }

    auto hasEmptyRootReductionAxis = [&empty_tv_descr](TensorView* out_tv) {
      return std::any_of(
          empty_tv_descr.empty_axes.begin(),
          empty_tv_descr.empty_axes.end(),
          [&out_tv](size_t ax) {
            return out_tv->getRootDomain().at(ax)->isReduction();
          });
    };

    // Given a TensorView get a vector of its maybeRFactor maybeExpandedExtents
    auto orig_shape = [](TensorView* out_tv) -> std::vector<Val*> {
      const auto& rfactor =
          TensorDomain::noReductions(out_tv->getMaybeRFactorDomain());
      std::vector<Val*> out_shape;
      out_shape.reserve(rfactor.size());
      for (const auto id : rfactor) {
        out_shape.push_back(id->getMaybeExpandedExtent());
      }
      return out_shape;
    };

    // Replace uses whose outputs might not be empty. Many expressions are
    // guaranteed to have empty outputs if any of the inputs are empty; for
    // example simple unary or binary ops. In those cases, we don't need to
    // doctor the Fusion since they will have an empty tensor downstream which
    // will cut off their dependence, resulting in those uses becoming dead
    // code. For example, suppose we determined tv2 is empty, and we have the
    // following Fusion:
    //
    //   auto tv4 = add(tv2, tv3);
    //   fusion.addOutput(tv4);
    //
    // If we know that tv2 is empty in any dimension, then either tv3 has a
    // matching empty dimension or it is broadcast in that dimension. Either
    // way, the corresponding dimension in tv4 will be empty, so tv4 is an empty
    // tensor. If we replace this expression with
    //
    //   auto tv4 = full(shape, zeroVal());
    //
    // Then the tensors tv2 and tv3 will become dead code if they have no other
    // live uses. In this case tv4 is an output tensor, so we must keep it in
    // the Fusion.
    //
    // Some special expressions can convert an empty tensor into a non-empty
    // tensor; particularly pad, cat, and reduction ops. These ops might have
    // non-empty outputs so in order to guarantee that all non- input or
    // output tensors are removed, we need to replace those ops with an
    // equivalent that does not have any empty inputs. For example
    for (auto use : tv->uses()) {
      // If use is a ReductionOp or WelfordOp over some empty axes, replace it
      // with a call to full().
      if (auto rop = dynamic_cast<ReductionOp*>(use)) {
        auto out = maybeReplaced(rop->out()->as<TensorView>());
        if (hasEmptyRootReductionAxis(out)) {
          auto out_shape = orig_shape(out);
          replaceWithFull(out, out_shape);
        }
      } else if (auto wop = dynamic_cast<WelfordOp*>(use)) {
        auto avg = maybeReplaced(wop->outAvg()->as<TensorView>());
        auto var = maybeReplaced(wop->outVar()->as<TensorView>());
        auto N = maybeReplaced(wop->outN()->as<TensorView>());
        if (hasEmptyRootReductionAxis(avg)) {
          auto out_shape = orig_shape(avg);
          auto nan = IrBuilder::create<Double>(0.0 / 0.0);
          replaceWithFull(avg, out_shape, nan);
          replaceWithFull(var, out_shape, nan);
          replaceWithFull(N, out_shape);
        }
      } else if (auto pop = dynamic_cast<PadOp*>(use)) {
        auto out = maybeReplaced(pop->out()->as<TensorView>());

        // A cat op can have input empty tensors and still output a non-empty
        // tensor. This is only possible if there is more than one input, so we
        // only need to handle those cases. We find the non-empty inputs to cat
        // then replace with another cat (or `set` if n=1).
        //
        // [Detecting cat ops]
        // The `cat` function creates a CatOp object, but its inputs() are not
        // the original inputs. Rather, they are the inputs after padding to the
        // output extent in the concatenated dimension. Thus, in the IR graph,
        // instead of the following:
        //
        //    T0  T1   T2
        //      \  |  /
        //       CatOp
        //         |
        //        T3
        //
        // a cat is represented as:
        //    T0    T1    T2
        //     |     |     |
        //   PadOp PadOp PadOp
        //       \   |   /
        //         CatOp
        //           |
        //          T3
        if (pop->out()->uses().size() == 1 &&
            pop->out()->uses()[0]->isA<CatOp>()) {
          auto cop = pop->out()->uses()[0]->as<CatOp>();
          std::vector<TensorView*> nonempty_inputs;
          for (auto inp : cop->inputs()) {
            // Each "input" to CatOp is a pad() of the corresponding _actual_
            // input. Here we peel off the pad op to collect the non-padded cat
            // inputs.
            auto padded_inp_tv = inp->as<TensorView>();
            TORCH_INTERNAL_ASSERT(
                padded_inp_tv->definition() &&
                    padded_inp_tv->definition()->isA<PadOp>(),
                "Input to cat should have definition that is a PadOp");
            auto inp_tv = padded_inp_tv->definition()
                              ->as<PadOp>()
                              ->in()
                              ->as<TensorView>();

            if (inp_tv != tv) {
              // we could remove other empty tensors here while we're at it.
              // They will get removed by further passes anyway though as tv
              // ranges over all empty tensors.
              nonempty_inputs.push_back(inp_tv);
            }
          }
          auto old_cat = cop->output(0)->as<TensorView>();
          auto new_cat = nonempty_inputs.size() == 1
              ? set(nonempty_inputs[0])
              : cat(nonempty_inputs, cop->concatenatedDim());
          replaceTV(old_cat, new_cat);
        } else { // Replace pads that are not part of CatOps with full()
          auto out_shape = orig_shape(out);
          // Wherever there is a zero in the input, we will replace the original
          // output extent so that we no longer reference the now-zero input
          // extent
          for (auto i : empty_tv_descr.empty_axes) {
            auto pad_widths = pop->getPadWidths((int)i);
            out_shape[i] = add(pad_widths.first, pad_widths.second);
          }
          replaceWithFull(out, out_shape, pop->value());
        }
      }
    }
    if (!tv->isFusionInput()) {
      replaceWithFull(tv, new_shape);
    }
  }
}

TensorView* DynamicTransformConcretizer::replaceWithFull(
    TensorView* tv,
    std::vector<Val*>& new_shape,
    Val* fill_value) {
  TensorView* mut_tv = nullptr;
  if (!tv->definition()) {
    // No definition. Probably an input.
    TORCH_INTERNAL_ASSERT(
        !tv->hasRFactor(),
        "Found RFactor in input TensorView ",
        tv->toString());
    std::vector<bool> expanded(tv->nDims());
    for (int i : c10::irange((int)tv->nDims())) {
      expanded[i] = tv->axis(i)->hasExpandedExtent();
    }
    mut_tv = TensorViewBuilder()
                 .ndims(tv->nDims())
                 .dtype(tv->getDataType().value())
                 .contiguity(tv->getContiguity())
                 .shape(new_shape)
                 .expanded(expanded)
                 .build();
    mut_tv->setMemoryType(MemoryType::Global);
  } else {
    if (!fill_value) {
      fill_value = tv->fusion()->zeroVal();
    }
    if (fill_value->getDataType().value() != tv->getDataType().value()) {
      fill_value = castOp(tv->getDataType().value(), fill_value);
    }
    mut_tv = full(new_shape, fill_value, tv->getDataType().value());
  }
  replaceTV(tv, mut_tv);

  return mut_tv;
}

void DynamicTransformConcretizer::replaceTV(
    TensorView* old_tv,
    TensorView* new_tv) {
  registerConcretization(old_tv, new_tv);
  OptOutMutator::mutate(old_tv);

  for (auto use : old_tv->uses()) {
    ir_utils::replaceValInExpr(use, old_tv, new_tv);
  }

  if (old_tv->isFusionInput()) {
    old_tv->fusion()->replaceInput(old_tv, new_tv);
  }

  if (old_tv->isFusionOutput()) {
    old_tv->fusion()->replaceOutput(old_tv, new_tv);
  }

  replaced_tvs_[old_tv] = new_tv;
}

void DynamicTransformConcretizer::concretizeReshape() {
  // Concretize each reshape op.
  for (const auto& [tv_index, view_analysis] : info_->getReshapeTransforms()) {
    auto incomplete_out_tv =
        info_->initialInfo()->getDynamicReshapedTensorViews().at(tv_index);
    auto view_op = incomplete_out_tv->definition()->as<ViewOp>();
    auto inp_tv = view_op->in()->as<TensorView>();

    auto concrete_reshape_out_tv = reshape(inp_tv, view_analysis);

    // We do the replacement directly here, but we must still check that the
    // replacement is valid
    checkConcretizedUses(incomplete_out_tv, concrete_reshape_out_tv);

    // Replace the old tensor with the new concretized tensor
    for (auto use_of_old_tv : incomplete_out_tv->uses()) {
      ir_utils::replaceValInExpr(
          use_of_old_tv, incomplete_out_tv, concrete_reshape_out_tv);
    }

    if (incomplete_out_tv->isFusionOutput()) {
      incomplete_out_tv->fusion()->replaceOutput(
          incomplete_out_tv, concrete_reshape_out_tv);
    }

    info_->fusion()->removeVal(incomplete_out_tv);
  }
}

void DynamicTransformConcretizer::concretizeResize() {
  // Concretize each resize op.
  for (const auto& [id_index, iter_type] : info_->getResizeIterTypes()) {
    auto id = info_->initialInfo()->getDynamicResizedIterDomains().at(id_index);
    TORCH_CHECK(
        id->definition() && id->definition()->isA<Resize>(),
        "Resized IterDomain must have a Resize definition");
    auto def = id->definition()->as<Resize>();
    auto new_id = IterDomain::resize(
        def->in(),
        def->leftExpand(),
        def->rightExpand(),
        id->isRFactorProduct(),
        iter_type);

    registerConcretization(id, new_id);
  }
}

void DynamicTransformConcretizer::checkConcretizedUses(
    Val* old_val,
    Val* new_val) const {
  for (const auto use : old_val->uses()) {
    use->checkConcretization(old_val, new_val);
  }
}

// Concretizes inherited symbolic domains. Note that when this is
// called, it is assumed that all dynamic ops themselves are
// concretized. Since symbolic IDs may be propagated down to
// consumers, those domains need to be concretized accordingly.
void DynamicTransformConcretizer::mutate(TensorView* tv) {
  if (!tv->domain()->hasSymbolicAxis()) {
    return;
  }

  // First, try to concretize the root domain as there may be symbolic
  // axes inherited from the producers
  propagateFromProducerToConsumer(tv);

  // If no root domain is altered by producer, we don't need to propagate back
  // up to rfactor. We could return early, but instead we go ahead and check the
  // root to rfactor transforms to be sure we have concretized any intermediate
  // IterDomains.

  // At this point, there should be no expr beyond rfactor root
  TORCH_INTERNAL_ASSERT(
      tv->getLeafDomain() == tv->getMaybeRFactorDomain(),
      "Invalid tensor: ",
      tv->toString());

  // If it has an rfactor root domain, the IterTypes of the rfactor
  // IDs may need to be updated as well. Traverse the rfactor exprs
  // and mutate the IterTypes of output IDs if symbolic.
  if (tv->hasRFactor()) {
    // Note that it is assumed that theres's no further expression
    // beyond the rfactor domain as asserted above
    auto all_id_exprs = StmtSort::getExprsBetween(
        tv->fusion(),
        {tv->getRootDomain().begin(), tv->getRootDomain().end()},
        {tv->getMaybeRFactorDomain().begin(),
         tv->getMaybeRFactorDomain().end()});
    for (auto expr : all_id_exprs) {
      // Assume outputs of IterDomain exprs are always IterDomains. If
      // the assumption is invalidated, the logic here would need to
      // be updated. Assert the assumption to immediately detect such
      // a case if happened.
      for (auto out_val : expr->outputs()) {
        TORCH_INTERNAL_ASSERT(
            out_val->isA<IterDomain>(),
            "Unexpected output: ",
            out_val->toString(),
            ". IterDomain was expected.");
      }

      // NOTE: We do not return early if all outputs are concrete as there may
      // still be concrete inputs. For example, a Symbolic IterDomain might be
      // padded with constant pad widths (1, 1), in which case although we do
      // not know the exact extent of the output, we know it is at least as
      // large as the sum of the pad widths, 2. In such cases, the output
      // IterDomain is concrete at definition, since if the extent is >1 we know
      // the IterType is Iteration. In these cases, we must continue to
      // concretize intermediate expressions between the root and R-factor
      // domain. See test DynamicTransform5_CUDA which demonstrates this
      // behavior.
      // NOTE: We also do not assume that if one output ID is symbolic, that
      // they all must be. See test FusionSliceForNanoGPT3_CUDA for an example
      // that does a static split by a factor of 16 of a symbolic input domain.
      // The static split in that case results in a concrete IterDomain with
      // extent 16 along with a symbolic one (extent ceilDiv(n / 16)).

      // Determine the output IterType
      IterType iter_type = IterType::Symbolic;
      for (auto inp_id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
        auto updated_id = maybeMutated(inp_id)->as<IterDomain>();
        iter_type = ops::promoteIterType(iter_type, updated_id->getIterType());
      }
      TORCH_INTERNAL_ASSERT(
          iter_type != IterType::Symbolic,
          "Failed to concretize an output IterType for expression: ",
          expr->toString());

      // Update the IterType of each output
      for (auto out_id : ir_utils::filterByType<IterDomain>(expr->outputs())) {
        if (!out_id->isSymbolic()) {
          continue;
        }
        auto concretized_out_id =
            IterDomainBuilder(out_id).iter_type(iter_type).build();
        registerConcretization(out_id, concretized_out_id);
      }

      // The expr itself needs to be mutated as well in case the outputs are
      // mutated, which can be done by the mutate method
      OptOutMutator::mutate(expr);
    }
  }

  // Root and rfactor domains are updated. First mutate the
  // TensorDomain and then TensorView
  mutate(tv->domain());
  OptOutMutator::mutate(tv);
}

// Almost an exact copy of OptOutMutator::mutate(TensorDomain*), but
// the contiguity vector may need to be updated as well as symbolic
// domains may be mutated to broadcast domains, which means contiguity
// may need to be changed to nullopt
void DynamicTransformConcretizer::mutate(TensorDomain* td) {
  bool mutated = false;

  auto updateIdVec = [&](const std::vector<IterDomain*>& ids) {
    std::vector<IterDomain*> updated_ids;
    for (auto id : ids) {
      auto updated_id = maybeMutated(id)->as<IterDomain>();
      updated_ids.push_back(updated_id);
      if (!updated_id->sameAs(id)) {
        mutated = true;
      }
    }
    return updated_ids;
  };

  std::vector<IterDomain*> root_dom = updateIdVec(td->root());
  std::vector<IterDomain*> rfactor_dom = td->hasRFactor()
      ? updateIdVec(td->maybeRFactor())
      : std::vector<IterDomain*>();
  std::vector<IterDomain*> domain = updateIdVec(td->leaf());

  if (!mutated) {
    return;
  }

  // Update the contiguity vector. Drop the contig val if mutated to broadcast
  auto contig = td->contiguity();

  for (const auto i : c10::irange(td->maybeRFactor().size())) {
    auto original_id = td->maybeRFactor().at(i);
    if (original_id->getIterType() != IterType::Symbolic) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        contig.at(i),
        "Unexpected to have a non-contig symbolic domain: ",
        original_id->toString());

    auto updated_id = td->hasRFactor() ? rfactor_dom.at(i) : root_dom.at(i);

    // If the concretized ID is a broadcast domain, drop the contig val
    if (updated_id->isBroadcast()) {
      contig.at(i) = std::nullopt;
    }
  }

  Val* mutated_val = IrBuilder::create<TensorDomain>(
      td->container(), root_dom, rfactor_dom, domain, contig);
  registerConcretization(td, mutated_val);
}

bool DynamicTransformConcretizer::propagateFromProducerToConsumer(
    TensorView* consumer) {
  if (consumer->definition() == nullptr ||
      !consumer->domain()->hasSymbolicAxis()) {
    return false;
  }

  const auto& root_domain = consumer->getRootDomain();

  auto def = consumer->definition();

  bool is_concretized = false;

  for (const auto i : c10::irange(root_domain.size())) {
    auto root_id = root_domain.at(i);
    if (root_id->getIterType() != IterType::Symbolic) {
      continue;
    }

    // Figure out the right IterType of this consumer root ID from its
    // corresponding producer IDs

    std::optional<IterType> id_type;

    for (auto producer : ir_utils::filterByType<TensorView>(def->inputs())) {
      PairwiseRootDomainMap root_map(producer, consumer);
      auto c2p = root_map.mapConsumerToProducer(
          consumer->domain(), producer->domain());

      TORCH_INTERNAL_ASSERT(
          c2p.find(root_id) != c2p.end(),
          "No input ID found to map with output ID: ",
          root_id->toString());

      auto input_id = c2p.at(root_id);
      TORCH_INTERNAL_ASSERT(
          input_id->getIterType() != IterType::Symbolic,
          "Producer ID not concretized: ",
          input_id->toString());

      if (id_type.has_value()) {
        id_type = ops::promoteIterType(*id_type, input_id->getIterType());
      } else {
        id_type = input_id->getIterType();
      }
    }

    TORCH_INTERNAL_ASSERT(
        id_type.has_value(),
        "Did not find id_type for consumer root domain ",
        root_id->toString(),
        ". Perhaps consumer def has no inputs. Consumer definition = ",
        def->toString());

    TORCH_INTERNAL_ASSERT(
        id_type != IterType::Symbolic,
        "Failed to concretize ",
        root_id->toString(),
        " of ",
        consumer->toString());

    auto concretized_id =
        IterDomainBuilder(root_id).iter_type(*id_type).build();

    registerConcretization(root_id, concretized_id);
    is_concretized = true;
  }

  return is_concretized;
}

DynamicTransformInitialInfo DynamicTransform::getInitialInfo(Fusion* fusion) {
  DynamicTransformInitialInfoBuilder builder(fusion);
  return builder.getInfo();
}

void DynamicTransform::concretizeFusion(
    Fusion* fusion,
    const DynamicTransformConcretizationInfo* info) {
  DynamicTransformConcretizer concretizer(fusion, info);
}

size_t DynamicTransformConcretizationInfo::hash() const {
  size_t hash = 0;
  for (const auto& [tv, view_result] : getReshapeTransforms()) {
    hashCombine(hash, view_result.hash());
  }
  for (const auto& [id, iter_type] : getResizeIterTypes()) {
    hashCombine(hash, (size_t)iter_type);
  }
  return hash;
}

} // namespace nvfuser
