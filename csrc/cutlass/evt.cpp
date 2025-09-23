// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "cutlass/gemm.h"
#include <dispatch.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <scheduler/cutlass.h>
#include <type.h>

#include <format>
#include <string>

namespace nvfuser {

namespace cutlass_codegen {

namespace {

//! Find the accumulator which is the direct output of the ScaledMmaOp
TensorView* getAccTv(Fusion* fusion) {
  TensorView* acc = nullptr;
  for (Expr* expr : fusion->exprs()) {
    if (expr->isA<ScaledMmaOp>()) {
      NVF_ERROR(
          acc == nullptr,
          "Found multiple ScaledMmaOps. Cannot determine which accumulator to "
          "return");
      acc = expr->output(0)->as<TensorView>();
    }
  }
  return acc;
}

bool hasEpilogue(Fusion* fusion) {
  TensorView* acc = getAccTv(fusion);
  NVF_ERROR(acc != nullptr);
  return !acc->isFusionOutput() && fusion->outputs().size() == 1;
}

//! This converts the epilogue of a matmul fusion into an Epilogue Visitor Tree
//! (EVT). We model the tree using the EVTModel class above.
//! https://dx.doi.org/doi/10.1145/3620666.3651369
class EVTConverter : OptInDispatch {
 public:
  EVTConverter(Fusion* fusion) : fusion_(fusion) {
    run();
  }

  const EVTModel& model() const {
    return model_;
  }

 private:
  void run() {
    for (Expr* expr :
         StmtSort::getExprsBetween({getAccTv(fusion_)}, fusion_->outputs())) {
      dispatch(expr);
    }
    model_.setRoot(val_nodes_.at(fusion_->outputs().at(0)));
  }

  EVTModel::Node* getNodeFor(Val* val) {
    return val_nodes_.at(val);
  }

  using OptInDispatch::handle;

  void handle(UnaryOp* uop) {
    // TODO: translate all of the supported UnaryOpTypes
  }

  void handle(BinaryOp* bop) {
    // TODO: translate all of the supported BinaryOpTypes
    EVTModel::Node* node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90Compute");
    std::string op_name;
    switch (bop->getBinaryOpType()) {
      case BinaryOpType::Add:
        op_name = "plus";
        break;
      case BinaryOpType::Div:
        op_name = "divides";
        break;
      case BinaryOpType::Mul:
        op_name = "multiplies";
        break;
      case BinaryOpType::Sub:
        op_name = "minus";
        break;
      default:
        NVF_THROW("Unhandled binary op type: ", bop->getBinaryOpType());
    }
    node->inputs.push_back(model_.makeNode("cutlass::" + op_name));

    node->inputs.push_back(getNodeFor(bop->lhs()));
    node->inputs.push_back(getNodeFor(bop->rhs()));

    // https://github.com/NVIDIA/cutlass/blob/2b8dff1f90605452c378c02298dd0cacaf65753c/include/cutlass/numeric_conversion.h#L56
    node->inputs.push_back(
        model_.makeNode("cutlass::FloatRoundStyle::round_toward_zero"));

    val_nodes_.emplace(bop->out(), node);
  }

 private:
  Fusion* fusion_;
  EVTModel model_;
  std::unordered_map<Val*, EVTModel::Node*> val_nodes_;
};

} // namespace

EVTModel EVTModel::copy() const {
  EVTModel new_model;

  std::unordered_map<Node*, Node*> old2new;

  for (const auto& node_up : nodes_up_) {
    Node* new_node = new_model.makeNode(node_up->name);
    old2new.emplace(node_up.get(), new_node);
  }
  // Loop again now that have old2new fully populated
  for (const auto& node_up : nodes_up_) {
    // Fill in new_node->inputs
    Node* new_node = old2new.at(node_up.get());
    new_node->inputs.reserve(node_up->inputs.size());
    for (Node* inp : node_up->inputs) {
      new_node->inputs.push_back(old2new.at(inp));
    }
  }

  new_model.setRoot(old2new.at(root_));
  return new_model;
}

EVTModel extractEVTModel(Fusion* fusion) {
  EVTConverter conv(fusion);
  return conv.model().copy();
}

std::string genEVT(Fusion* fusion) {

  std::stringstream ss;
  ss << "cutlass::epilogue::fusion::Sm90EVT<";
  ss << conv.model().defString();
  ss << ">";
  return ss.str();
}

} // namespace cutlass_codegen

} // namespace nvfuser
