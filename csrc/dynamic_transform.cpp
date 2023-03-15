// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <dynamic_transform.h>
#include <expr_evaluator.h>
#include <lower_utils.h>
#include <root_domain_map.h>
#include <transform_view.h>

namespace nvfuser {

class TORCH_CUDA_CU_API DynamicTransformInfoBuilder : public IterVisitor {
 public:
  static DynamicTransformInfo get(
      Fusion* fusion,
      ExpressionEvaluator* expr_eval);

  DynamicTransformInfoBuilder(Fusion* fusion, ExpressionEvaluator* expr_eval)
      : fusion_(fusion), expr_eval_(expr_eval) {
    TORCH_INTERNAL_ASSERT(
        !fusion->isA<kir::Kernel>(),
        "Invalid container. Kernel container not allowed.\n");

    augmentExprEvaluator();

    traverseTo(fusion, fusion->getTerminatingOutputs(), false, false);
  }

  using IterVisitor::handle;

  void handle(ViewOp* op) override;

  const auto& getInfo() const {
    return info_;
  }

 private:
  void augmentExprEvaluator();

 private:
  Fusion* fusion_ = nullptr;
  ExpressionEvaluator* expr_eval_ = nullptr;

  DynamicTransformInfo info_;
};

// TODO
bool DynamicTransformInfo::operator==(const DynamicTransformInfo& other) const {
  return false;
}

// TODO
std::string DynamicTransformInfo::toString() const {
  std::stringstream ss;
  ss << "DynamicTransformInfo\n";
  std::string indent = "  ";
  ss << indent << "Reshape:\n";
  for (const auto& kv : reshape_transforms_) {
    ss << indent << indent << kv.first->toString() << ", "
       << kv.second.toString() << "\n";
  }
  return ss.str();
}

DynamicTransformInfo DynamicTransformInfo::get(
    Fusion* fusion,
    ExpressionEvaluator* expr_eval) {
  DynamicTransformInfoBuilder builder(fusion, expr_eval);
  return builder.getInfo();
}

void DynamicTransformInfoBuilder::augmentExprEvaluator() {
  const auto mapped_sets = ExactRootDomainMap(fusion_).getMappedSets();

  // std::cerr << "Augmenting ExprEval\n";

  for (const auto& set : mapped_sets.disjointSets()) {
    // std::cerr << "Disjoint set\n";
    int64_t known_size = -1;
    std::vector<Val*> unknown_vals;
    for (const auto id : *set) {
      // std::cerr << "ID: " << id->toString() << std::endl;
      auto eval_val = expr_eval_->evaluate(id->extent());
      if (eval_val.has_value()) {
        TORCH_INTERNAL_ASSERT(eval_val->isInt(), "Invalid extent value");
        int64_t this_size = eval_val->as<int64_t>();
        if (known_size != -1) {
          TORCH_INTERNAL_ASSERT(
              known_size == this_size,
              "Conflicting sizes: ",
              known_size,
              ", ",
              this_size);
        } else {
          known_size = this_size;
        }
      } else {
        unknown_vals.push_back(id->extent());
      }
    }

    if (known_size == -1 || unknown_vals.empty()) {
      continue;
    }

    // Binding unknown vals to known_val
    for (auto unknown_val : unknown_vals) {
      // std::cerr << "Augment: " << unknown_val->toString() << " -> "
      //<< known_size << std::endl;
      expr_eval_->bind(unknown_val, known_size);
    }
  }

  // std::cerr << "Augment done\n";
}

void DynamicTransformInfoBuilder::handle(ViewOp* op) {
  std::cerr << "Reshape: " << op->toString();

  // Determine if this is a dynamic reshape. If the output tv doesn't
  // have an rfactor domain, no view transform is set up, which means
  // its output domain is still just a placeholder

  auto inp_tv = op->in()->as<TensorView>();
  auto out_tv = op->out()->as<TensorView>();

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
    auto extent_val = expr_eval_->evaluate(inp_id->extent());
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

  std::cerr << "Input shape: " << inp_shape << std::endl;

  const auto& out_dom = out_tv->getMaybeRFactorDomain();

  // Determine output shape using expr evaluator. Note there may be
  // one domain of extent -1
  std::vector<int64_t> out_shape(out_dom.size(), 0);
  bool extent_m1_found = false;
  for (const auto i : c10::irange(out_dom.size())) {
    auto out_id = out_dom.at(i);
    auto extent_val = expr_eval_->evaluate(out_id->extent());
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

  std::cerr << "Output shape: " << out_shape << std::endl;

  auto view_result = analyzeView(inp_tv, inp_shape, out_shape);

  info_.reshape_transforms_.emplace_back(out_tv, view_result);
}

} // namespace nvfuser
