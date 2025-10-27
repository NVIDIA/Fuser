// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/codegen.h>
#include <cutlass/evt.h>
#include <device_lower/utils.h>
#include <dispatch.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <scheduler/mma_utils.h>
#include <type.h>

#include <format>
#include <string>

namespace nvfuser {

namespace cutlass_codegen {

namespace {

Expr* getGemmExpr(Fusion* fusion) {
  auto scaled_mma_exprs = fusion->exprs() |
      std::views::filter([](Expr* e) { return e->isA<ScaledMmaOp>(); });
  const int64_t num_exprs = std::ranges::distance(scaled_mma_exprs);
  NVF_ERROR_NE(num_exprs, 0, "No ScaledMmaOps detected");
  NVF_ERROR_EQ(
      num_exprs,
      1,
      "Found multiple ScaledMmaOps. Cannot determine which to return");
  return scaled_mma_exprs.front();
}

//! Find the accumulator which is the direct output of the ScaledMmaOp
TensorView* getAccTv(Fusion* fusion) {
  return getGemmExpr(fusion)->output(0)->as<TensorView>();
}

struct BlockScaledOutputPattern {
  TensorView* prescaled_output;

  TensorView* output;
  TensorView* block_scale_factors;
  TensorView* global_scale_factor;

  int64_t block_size;
};

// This matches a standard block scaling pattern expressed in Fusion IR like
// follows:
std::vector<BlockScaledOutputPattern> findBlockScaledOutputs(Fusion* fusion) {
  std::vector<BlockScaledOutputPattern> patterns;

  for (Val* out_val : fusion->outputs()) {
    auto* out_tv = dynamic_cast<TensorView*>(out_val);
    if (out_tv == nullptr) {
      continue;
    }
    // Match scale factor

    // Match
  }

  return patterns;
}

//! This converts the epilogue of a matmul fusion into an Epilogue Visitor Tree
//! (EVT). We model the tree using the EVTModel class above.
//! https://dx.doi.org/10.1145/3620666.3651369
class EVTConverter : OptInDispatch {
 public:
  static mma_utils::DataWrapperOpt<EVTModel> convert(Fusion* fusion) {
    EVTConverter conv(fusion);
    conv.run();
    if (!conv.failureReason().empty()) {
      return {conv.failureReason().c_str()};
    }
    return std::move(conv.model());
  }

 private:
  EVTConverter(Fusion* fusion) : fusion_(fusion) {}

  EVTModel& model() {
    return model_;
  }

  const std::string& failureReason() const {
    return failure_reason_;
  }

  void run() {
    Expr* mma = getGemmExpr(fusion_);

    auto* scaled_mma = dynamic_cast<ScaledMmaOp*>(mma);
    NVF_ERROR(
        scaled_mma,
        "Only ScaledMmaOp is currently supported for EVT translation");
    TensorView* mma_out = mma->output(0)->as<TensorView>();
    TensorView* alpha = scaled_mma->alpha();
    TensorView* beta = scaled_mma->beta();
    TensorView* bias = scaled_mma->bias();

    // The default kernel uses EpilogueScheduleAuto, which in turn uses
    // LinearCombination as the epilogue. That means an epilogue that looks like
    // this is assumed:
    //
    //   alpha * acc + beta * bias
    //
    // The ScaledMmaOp node has tensor inputs corresponding to these arguments.
    // If some of these are null, we can omit them when building our EVT.
    // Otherwise, we replicate the default EVT defined here:
    // https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp#L118-L134

    NVF_ERROR(beta == nullptr, "Beta not yet supported for EVT translation");
    NVF_ERROR(bias == nullptr, "Bias not yet supported for EVT translation");

    NVF_ERROR(
        scaled_mma->outScale() == nullptr,
        "Output block scale factor not supported for EVT translation");
    NVF_ERROR(
        scaled_mma->outGamma() == nullptr,
        "Output global scale factor not supported for EVT translation");

    // Start by making nodes for the accumulator and for any epilogue inputs
    if (alpha == nullptr && (beta == nullptr || bias == nullptr)) {
      // No epilogue
      val_nodes_.emplace(
          mma_out, model_.makeNode("cutlass::epilogue::fusion::Sm90AccFetch"));
    }

    if (alpha != nullptr) {
      NVF_ERROR(
          alpha->nDims() == 0,
          "Only zero-dimensional alpha is supported for EVT translation");
      NVF_ERROR(
          alpha->dtype() == DataType::Float,
          "Only Float alpha is supported for EVT translation");
      // Broadcast alpha to the same dimensions as the accumulator
      EVTModel::Node* alpha_bcast_node = model_.makeNode(
          "cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>");
      alpha_bcast_node->argument = alpha;
      val_nodes_.emplace(alpha, alpha_bcast_node);

      EVTModel::Node* alpha_acc_node = makeBinaryOpNode(
          BinaryOpType::Mul,
          /*in_type=*/DataType::Float,
          // NOTE: CUTLASS does not have explicit cast EVT nodes.
          /*out_type=*/mma_out->dtype(),
          /*lhs_node=*/alpha_bcast_node,
          /*rhs_node=*/
          model_.makeNode("cutlass::epilogue::fusion::Sm90AccFetch"));

      val_nodes_.emplace(mma_out, alpha_acc_node);
    }

    // TODO: add load nodes for epilogue inputs defined in Fusion (i.e. not as
    // ScaledMmaOp inputs)

    for (Expr* expr :
         StmtSort::getExprsBetween({getAccTv(fusion_)}, fusion_->outputs())) {
      dispatch(expr);
    }
    model_.setRoot(val_nodes_.at(fusion_->outputs().at(0)));
  }

  EVTModel::Node* getNodeFor(Val* val) {
    return val_nodes_.at(val);
  }

  EVTModel::Node* makeBinaryOpNode(
      BinaryOpType op_type,
      DataType in_type,
      DataType out_type,
      EVTModel::Node* lhs_node,
      EVTModel::Node* rhs_node) {
    // TODO: translate all of the supported BinaryOpTypes
    std::string op_name;
    switch (op_type) {
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
        NVF_THROW("Unhandled binary op type: ", op_type);
    }
    // This node and its inputs is essentially a function signature
    EVTModel::Node* func_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90Compute");
    func_node->inputs.push_back(model_.makeNode("cutlass::" + op_name));
    func_node->inputs.push_back(model_.makeNode(dtypeToCutlass(out_type)));
    // Compute type determines what precision the operation will take place in.
    // The op is computed as (out_type)(op((compute_type)x, (compute_type)y))
    func_node->inputs.push_back(model_.makeNode(dtypeToCutlass(in_type)));
    // rounding mode
    // https://github.com/NVIDIA/cutlass/blob/2b8dff1f90605452c378c02298dd0cacaf65753c/include/cutlass/numeric_conversion.h#L56
    func_node->inputs.push_back(
        model_.makeNode("cutlass::FloatRoundStyle::round_to_nearest"));

    // We combine the signature with tree visitor node
    EVTModel::Node* visitor_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90EVT");
    visitor_node->inputs.push_back(func_node);
    visitor_node->inputs.push_back(lhs_node);
    visitor_node->inputs.push_back(rhs_node);

    return visitor_node;
  }

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
    func_node->inputs.push_back(
        model_.makeNode(dtypeToCutlass(uop->out()->dtype())));
    // Compute type determines what precision the operation will take place in.
    // The op is computed as (out_type)(op((compute_type)x))
    func_node->inputs.push_back(
        model_.makeNode(dtypeToCutlass(uop->in()->dtype())));
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
    NVF_ERROR(
        bop->lhs()->dtype() == bop->rhs()->dtype(),
        "We require both inputs to have the same dtype but found ",
        bop->lhs()->dtype(),
        " and ",
        bop->rhs()->dtype());
    val_nodes_.emplace(
        bop->out(),
        makeBinaryOpNode(
            bop->getBinaryOpType(),
            /*in_type=*/bop->lhs()->dtype(),
            /*out_type=*/bop->out()->dtype(),
            getNodeFor(bop->lhs()),
            getNodeFor(bop->rhs())));
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
    if (node_up->argument != nullptr) {
      new_node->argument = node_up->argument;
    }
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

std::string EVTModel::defString(Node* node, int64_t indent_size) const {
  if (node == nullptr) {
    node = root_;
  }
  NVF_ERROR(node != nullptr);
  std::stringstream ss;
  indent(ss, indent_size) << node->name;
  if (!node->inputs.empty()) {
    ss << "<\n";
    bool first = true;
    for (Node* input : node->inputs) {
      if (!first) {
        ss << ",\n";
      }
      first = false;
      ss << defString(input, indent_size + 1);
    }
    ss << ">";
  }
  return ss.str();
}

namespace {

struct CommentedString {
  std::string str;
  std::string comment;
};

// Forward declaration to enable recursion from fine-grained helper functions
CommentedString argStringHelper(EVTModel::Node* node, int64_t indent_size);

CommentedString argumentArgString(EVTModel::Node* node, int64_t indent_size) {
  std::stringstream ss;
  if (node->argument->isA<TensorView>()) {
    indent(ss, indent_size) << "{  // " << node->name << "\n";
    // TODO: If this is an input scalar, we need to obtain its name in the
    // kernel here
    const std::string internal_var_name = "alpha";
    indent(ss, indent_size + 1)
        << ".scalar_ptrs={static_cast<"
        << dtypeToCutlass(node->argument->dtype()) << " const*>("
        << internal_var_name << ".data_ptr)}\n";
    indent(ss, indent_size) << "}";
    return {ss.str(), ""};
  } else {
    // If this is a constant scalar, print its value directly
    if (node->argument->isConstScalar()) {
      return {
          "{.scalars={" + node->argument->toInlineString() + "}}", node->name};
    }
    NVF_ERROR(
        node->argument->isFusionInput(),
        "Non-constant scalars are expected to be fusion inputs for EVT "
        "translation");
    NVF_THROW("Input scalars not yet supported in EVT translation");
    return {ss.str(), ""};
  }
}

// For nodes with no inputs, we print their args like this:
//
//   {}  // node->name
//
// When a node has inputs, we print it like this:
//
//   {  // node->name
//     { ... },  // args for input 1
//     ...
//     { ... }  // args for input N
//   }
CommentedString argStringWithInputs(EVTModel::Node* node, int64_t indent_size) {
  std::stringstream ss;
  indent(ss, indent_size) << "{  // " << node->name << "\n";
  CommentedString prev_cs;
  const auto print_line = [&](bool last) {
    if (prev_cs.str.empty()) {
      return;
    }
    ss << prev_cs.str;
    if (!last) {
      ss << ",";
    }
    if (!prev_cs.comment.empty()) {
      ss << "  // " << prev_cs.comment;
    }
    ss << "\n";
  };
  std::string node_op_name;
  for (EVTModel::Node* input : node->inputs) {
    // TODO: add all other node op names that might appear here
    if (input->name == "cutlass::epilogue::fusion::Sm90Compute") {
      // This just describes what op is being computed in an EVT node. It
      // should not appear in the argument list
      NVF_ERROR(node_op_name.empty());
      node_op_name = input->name;
      continue;
    }
    print_line(false);
    prev_cs = argStringHelper(input, indent_size + 1);
  }
  if (!node_op_name.empty()) {
    // We have a node op. Print its arguments last (there never are any, so
    // don't recurse)
    print_line(false);
    std::stringstream ss_name;
    indent(ss_name, indent_size + 1) << "{}";
    prev_cs = {ss_name.str(), node_op_name};
  }
  print_line(true);
  indent(ss, indent_size) << "}";
  return {ss.str(), ""};
}

CommentedString argStringHelper(EVTModel::Node* node, int64_t indent_size) {
  NVF_ERROR(node != nullptr);
  if (node->inputs.empty()) {
    if (node->argument == nullptr) {
      std::stringstream ss;
      indent(ss, indent_size) << "{}";
      return {ss.str(), node->name};
    } else {
      return argumentArgString(node, indent_size);
    }
  } else {
    return argStringWithInputs(node, indent_size);
  }
}

} // namespace

std::string EVTModel::argString(Node* node, int64_t indent_size) const {
  if (node == nullptr) {
    node = root_;
  }
  const CommentedString cs = argStringHelper(node, indent_size);
  if (cs.comment.empty()) {
    return cs.str;
  } else {
    return cs.str + "  // " + cs.comment;
  }
}

std::string EVTModel::toString() const {
  std::stringstream ss;
  ss << "EVTModel{\n";
  std::unordered_map<Node*, size_t> node_num;
  for (const std::unique_ptr<Node>& node_up : nodes_up_) {
    node_num.emplace(node_up.get(), node_num.size());
  }
  for (const std::unique_ptr<Node>& node_up : nodes_up_) {
    ss << "  " << node_num.at(node_up.get()) << ": " << (void*)node_up.get()
       << " " << node_up->name;
    if (!node_up->inputs.empty()) {
      ss << "(";
      bool first = true;
      for (Node* input : node_up->inputs) {
        if (!first) {
          ss << ", ";
        }
        first = false;
        ss << node_num.at(input);
      }
      ss << ")";
    }
    if (node_up->argument != nullptr) {
      ss << "[" << node_up->argument->toString() << "]";
    }
    ss << "\n";
  }
  ss << "}";
  return ss.str();
}

// TODO: DataWrapperOpt belongs in scheduler_utils
mma_utils::DataWrapperOpt<EVTModel> extractEVTModel(Fusion* fusion) {
  return EVTConverter::convert(fusion);
}

} // namespace cutlass_codegen

} // namespace nvfuser
