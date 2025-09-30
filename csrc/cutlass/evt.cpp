// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/evt.h>
#include <device_lower/utils.h>
#include <dispatch.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <scheduler/mma_utils.h>
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

//! This converts the epilogue of a matmul fusion into an Epilogue Visitor Tree
//! (EVT). We model the tree using the EVTModel class above.
//! https://dx.doi.org/doi/10.1145/3620666.3651369
class EVTConverter : OptInDispatch {
 public:
  EVTConverter(Fusion* fusion) : fusion_(fusion) {
    run();
  }

  EVTModel& model() {
    return model_;
  }

  const std::string& failureReason() const {
    return failure_reason_;
  }

 private:
  void run() {
    // Start by making nodes for the accumulator and for any epilogue inputs
    TensorView* acc = getAccTv(fusion_);
    val_nodes_.emplace(
        acc, model_.makeNode("cutlass::epilogue::fusion::Sm90AccFetch"));

    // TODO: add load nodes for epilogue inputs

    for (Expr* expr :
         StmtSort::getExprsBetween({getAccTv(fusion_)}, fusion_->outputs())) {
      dispatch(expr);
    }
    model_.setRoot(val_nodes_.at(fusion_->outputs().at(0)));
  }

  EVTModel::Node* getNodeFor(Val* val) {
    return val_nodes_.at(val);
  }

  using OptInDispatch::dispatch;

  void dispatch(Expr* expr) {
    if (!ir_utils::isTvOp(expr)) {
      return;
    }
    OptInDispatch::dispatch(expr);
  }

  void handle(LoadStoreOp* uop) {}

  void handle(UnaryOp* uop) {
    // TODO: translate all of the supported UnaryOpTypes
    std::string op_name;
    switch (uop->getUnaryOpType()) {
      case UnaryOpType::Relu:
        op_name = "epilogue::thread::ReLU";
        break;
      default:
        NVF_THROW("Unhandled unary op type: ", uop->getUnaryOpType());
    }
    // This node and its inputs is essentially a function signature
    EVTModel::Node* func_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90Compute");
    func_node->inputs.push_back(model_.makeNode("cutlass::" + op_name));
    // TODO: infer type of inputs from dtypes
    func_node->inputs.push_back(model_.makeNode("float"));
    // This is the "compute" type of the op
    func_node->inputs.push_back(model_.makeNode("float"));
    // types of inputs
    // rounding mode
    // https://github.com/NVIDIA/cutlass/blob/2b8dff1f90605452c378c02298dd0cacaf65753c/include/cutlass/numeric_conversion.h#L56
    func_node->inputs.push_back(
        model_.makeNode("cutlass::FloatRoundStyle::round_to_nearest"));

    // We combine the signature with tree visitor node
    EVTModel::Node* visitor_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90EVT");
    visitor_node->inputs.push_back(func_node);
    visitor_node->inputs.push_back(getNodeFor(uop->in()));

    val_nodes_.emplace(uop->out(), visitor_node);
  }

  void handle(BinaryOp* bop) {
    // TODO: translate all of the supported BinaryOpTypes
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
    // This node and its inputs is essentially a function signature
    EVTModel::Node* func_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90Compute");
    func_node->inputs.push_back(model_.makeNode("cutlass::" + op_name));
    // TODO: infer type of inputs from dtypes
    func_node->inputs.push_back(model_.makeNode("float"));
    func_node->inputs.push_back(model_.makeNode("float"));
    // types of inputs
    // rounding mode
    // https://github.com/NVIDIA/cutlass/blob/2b8dff1f90605452c378c02298dd0cacaf65753c/include/cutlass/numeric_conversion.h#L56
    func_node->inputs.push_back(
        model_.makeNode("cutlass::FloatRoundStyle::round_to_nearest"));

    // We combine the signature with tree visitor node
    EVTModel::Node* visitor_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90EVT");
    visitor_node->inputs.push_back(func_node);
    visitor_node->inputs.push_back(getNodeFor(bop->lhs()));
    visitor_node->inputs.push_back(getNodeFor(bop->rhs()));

    val_nodes_.emplace(bop->out(), visitor_node);
  }

 private:
  Fusion* fusion_;
  EVTModel model_;
  std::unordered_map<Val*, EVTModel::Node*> val_nodes_;
  std::string failure_reason_;
};

} // namespace

EVTModel::EVTModel(const EVTModel& model) {
  std::unordered_map<Node*, Node*> old2new;

  for (const auto& node_up : model.nodes_up_) {
    Node* new_node = makeNode(node_up->name);
    old2new.emplace(node_up.get(), new_node);
  }
  // Loop again now that have old2new fully populated
  for (const auto& node_up : model.nodes_up_) {
    // Fill in new_node->inputs
    Node* new_node = old2new.at(node_up.get());
    new_node->inputs.reserve(node_up->inputs.size());
    for (Node* inp : node_up->inputs) {
      new_node->inputs.push_back(old2new.at(inp));
    }
  }
  setRoot(old2new.at(model.root()));
}

// TODO: accept a "depth" argument and format the output prettily
std::string EVTModel::defString(Node* node) const {
  if (node == nullptr) {
    node = root_;
  }
  NVF_ERROR(node != nullptr);
  std::stringstream ss;
  ss << node->name;
  if (!node->inputs.empty()) {
    ss << "<";
    bool first = true;
    for (Node* input : node->inputs) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      ss << defString(input);
    }
    ss << ">";
  }
  return ss.str();
}

// TODO: DataWrapperOpt belongs in scheduler_utils
mma_utils::DataWrapperOpt<EVTModel> extractEVTModel(Fusion* fusion) {
  EVTConverter conv(fusion);
  if (!conv.failureReason().empty()) {
    return {conv.failureReason().c_str()};
  }
  return std::move(conv.model());
}

std::string genEVT(Fusion* fusion) {
  EVTConverter conv(fusion);
  std::stringstream ss;
  ss << "cutlass::epilogue::fusion::Sm90EVT<";
  ss << conv.model().defString();
  ss << ">";
  return ss.str();
}

} // namespace cutlass_codegen

} // namespace nvfuser
