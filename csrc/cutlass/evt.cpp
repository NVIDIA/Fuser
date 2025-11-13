// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/block_scaling.h>
#include <cutlass/codegen.h>
#include <cutlass/evt.h>
#include <cutlass/gemm.h>
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

#include <string>

namespace nvfuser {

namespace cutlass_codegen {

namespace {

//! This converts the epilogue of a matmul fusion into an Epilogue Visitor Tree
//! (EVT). We model the tree using the EVTModel class above.
//! https://dx.doi.org/10.1145/3620666.3651369
class EVTConverter : OptInDispatch {
 public:
  static EVTModel convert(Fusion* fusion) {
    EVTConverter conv(fusion);
    conv.run();
    return std::move(conv.model());
  }

 private:
  EVTConverter(Fusion* fusion)
      : fusion_(fusion), pattern_(findCutlassMatmulPattern(fusion)) {
    validatePattern();
    NVF_ERROR_EQ(pattern_.mma->outputs().size(), 1);
    mma_out_ = pattern_.mma->output(0)->as<TensorView>();
  }

  EVTModel& model() {
    return model_;
  }

  //! We pass both inputs and output tensors to the launcher code via a vector
  //! of inputs and outputs, where the outputs are after the inputs. Given a TV,
  //! this function returns something like
  //!
  //!   static_cast<cutlass::bfloat16_t*>(inputs.at(4).data_ptr)
  //!
  std::string getPointerCode(TensorView* tv) {
    int64_t index = -1;
    if (tv->isFusionInput()) {
      index = fusionInputPosition(fusion_, tv);
    } else if (tv->isFusionOutput()) {
      index = fusion_->inputs().size() + fusionOutputPosition(fusion_, tv);
    } else {
      NVF_CUTLASS_REJECT(
          "Cannot get pointer for TV ",
          tv->toString(),
          " which is not a fusion input or output");
    }
    return "static_cast<" + dtypeToCutlass(tv->dtype()) + "*>(inputs.at(" +
        std::to_string(index) + ").data_ptr)";
  }

  std::string getPointerArrayPointerCode(TensorView* tv) {
    // TODO: track
    return getPointerCode(tv);
  }

  void validatePattern() {
    auto check_input = [](TensorView* inp) {
      if (inp == nullptr) {
        // Allow null
        return;
      }
      // Check that input is contiguous
      const std::vector<std::optional<bool>>& contig = inp->getContiguity();
      NVF_CUTLASS_REJECT_IF(
          std::any_of(
              contig.begin(),
              contig.end(),
              [](const std::optional<bool>& c) {
                return c.has_value() && !c.value();
              }),
          "Expected all inputs to ScaledMmaOp to be contiguous but found ",
          inp->toString());
    };
    check_input(pattern_.alpha);
    check_input(pattern_.beta);
    check_input(pattern_.bias);

    // TODO: Grouped gemm entry validation
  }

  // Provide DataType::Float if there is additional fusion required
  EVTModel::Node* makeAlphaAccNode(DataType dtype) {
    EVTModel::Node* acc_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90AccFetch");
    if (pattern_.alpha == nullptr) {
      // TODO: handle casting to dtype when neither alpha or bias is given and
      // there is no epilogue. i.e. simple GEMM
      return acc_node;
    }

    NVF_CUTLASS_REJECT_IF(
        pattern_.alpha->nDims() != 0,
        "Only zero-dimensional alpha is supported for EVT translation");
    NVF_CUTLASS_REJECT_IF(
        pattern_.alpha->dtype() != DataType::Float,
        "Only Float alpha is supported for EVT translation");

    EVTModel::Node* alpha_bcast_node = nullptr;
    if (pattern_.alpha->nDims() == 0) {
      // Broadcast scalar alpha to the same dimensions as the accumulator
      alpha_bcast_node = model_.makeNode(
          "cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>");
      alpha_bcast_node->arguments.emplace_back(
          "scalar_ptrs", "{" + getPointerCode(pattern_.alpha) + "}");
    } else if (pattern_.alpha->nDims() == 1) {
      NVF_CUTLASS_REJECT_IF(
          !pattern_.is_grouped,
          "Non-scalar alpha only supported for grouped GEMM");
      alpha_bcast_node = model_.makeNode(
          "cutlass::epilogue::fusion::Sm90ScalarBroadcastPtrArray<float>");
      alpha_bcast_node->arguments = {
          {"scalars", "{}"},
          {"scalar_ptrs", "{}"},
          {"scalar_ptr_arrays",
           "{" + getPointerArrayPointerCode(pattern_.alpha) + "}"},
      };
    }
    val_nodes_.emplace(pattern_.alpha, alpha_bcast_node);

    return makeBinaryOpNode(
        BinaryOpType::Mul,
        /*in_type=*/DataType::Float,
        // NOTE: CUTLASS does not have explicit cast EVT nodes.
        /*out_type=*/dtype,
        /*lhs_node=*/alpha_bcast_node,
        /*rhs_node=*/acc_node);
  }

  EVTModel::Node* makeBetaBiasNode() {
    if (pattern_.bias == nullptr) {
      return nullptr;
    }

    // Make a node to load the bias
    EVTModel::Node* beta_bias_node = model_.makeNode(
        "cutlass::epilogue::fusion::Sm90SrcFetch<" +
        dtypeToCutlass(pattern_.bias->dtype()) + ">");

    if (pattern_.beta != nullptr) {
      EVTModel::Node* beta_bcast_node = model_.makeNode(
          "cutlass::epilogue::fusion::Sm90ScalarBroadcast<" +
          dtypeToCutlass(pattern_.beta->dtype()) + ">");
      beta_bcast_node->arguments.emplace_back(
          "scalar_ptrs", "{" + getPointerCode(pattern_.beta) + "}");
      // Note: this casts beta and bias to float then multiplies and outputs
      // float, since we will always be adding it straight to alpha*acc
      // anyway
      beta_bias_node = makeBinaryOpNode(
          BinaryOpType::Mul,
          /*in_type=*/DataType::Float,
          /*out_type=*/DataType::Float,
          /*lhs_node=*/beta_bcast_node,
          /*rhs_node=*/beta_bias_node);
    }

    return beta_bias_node;
  }

  void makeMmaOutNode() {
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
    //
    // If there is a bias, then alpha*acc should be Float so that we avoid
    // down-casting until after adding it. Otherwise, we should go ahead and
    // cast to mma_out_'s dtype now.
    EVTModel::Node* mma_out_node = makeAlphaAccNode(
        pattern_.bias == nullptr ? mma_out_->dtype() : DataType::Float);

    if (EVTModel::Node* beta_bias_node = makeBetaBiasNode()) {
      mma_out_node = makeBinaryOpNode(
          BinaryOpType::Add,
          /*in_type=*/DataType::Float,
          /*out_type=*/mma_out_->dtype(),
          /*lhs_node=*/mma_out_node,
          /*rhs_node=*/beta_bias_node);
    }

    val_nodes_.emplace(mma_out_, mma_out_node);
  }

  // Detect block scaled outputs. Any output that is block scaled will
  // have its own special block scaling EVT node, so we don't want to create
  // EVT nodes for the block scaling pattern itself. This function returns all
  // outputs that are not blockscaled, as well as the _inputs_ to block-scaled
  // output before block scaling. This is used to traverse between the
  // accumulator and these outputs so that we don't accidentally create an
  // entire EVT tree for each block scaling portion of the graph.
  std::vector<Val*> getUnquantizedOutputs() {
    std::vector<Val*> unscaled_outputs;
    const std::vector<BlockScaledOutputPattern> scaling_patterns =
        findBlockScaledOutputs(fusion_);
    if (scaling_patterns.empty()) {
      unscaled_outputs.insert(
          unscaled_outputs.end(),
          fusion_->outputs().begin(),
          fusion_->outputs().end());
      model_.setRootTensorView(fusion_->outputs().front()->as<TensorView>());
    } else {
      // This holds all quantized outputs as well as scale factors, so we can
      // skip those outputs
      std::unordered_set<Val*> all_scaling_outputs;
      for (const BlockScaledOutputPattern& pattern : scaling_patterns) {
        block_scaling_patterns_.emplace(pattern.unquantized_output, pattern);
        unscaled_outputs.push_back(pattern.unquantized_output);
        all_scaling_outputs.insert(pattern.quantized_output);
        all_scaling_outputs.insert(pattern.block_scale_factors);
      }
      for (Val* v : fusion_->outputs()) {
        if (!all_scaling_outputs.contains(v)) {
          unscaled_outputs.push_back(v);
        }
      }
      // The first scaling pattern is considered the root
      model_.setRootTensorView(scaling_patterns.front().quantized_output);
    }
    return unscaled_outputs;
  }

  void run() {
    makeMmaOutNode();

    // TODO: add load nodes for epilogue inputs defined in Fusion (i.e. not as
    // ScaledMmaOp inputs)

    const std::vector<Val*> unquantized_outputs = getUnquantizedOutputs();

    // Traverse from the accumulator to the unquantized outputs, creating nodes
    // in the EVT for each of these
    NVF_CUTLASS_REJECT_IF(
        model_.getRootTensorView() == nullptr, "Could not set root TV");
    for (Statement* stmt : StmtSort::getStmtsBetween(
             {pattern_.mma->output(0)}, unquantized_outputs)) {
      dispatch(stmt);
    }
    NVF_CUTLASS_REJECT_IF(
        unquantized_outputs.size() != 1,
        "Only one unquantized output is currently supported");
    model_.setRoot(val_nodes_.at(unquantized_outputs.front()));
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
        NVF_CUTLASS_REJECT("Unhandled binary op type: ", op_type);
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

  using OptInDispatch::dispatch;

  void dispatch(Expr* expr) {
    if (!ir_utils::isTvOp(expr)) {
      return;
    }
    OptInDispatch::dispatch(expr);
  }

  void handle(TensorView* tv) {
    const auto it = block_scaling_patterns_.find(tv);
    if (it == block_scaling_patterns_.end()) {
      return;
    }
    // This is the pre-scaling version of a block-scaled output. Insert a block
    // scaling EVT node which will handle the scaling and outputting the scale
    // factors.
    const BlockScaledOutputPattern& pattern = it->second;
    NVF_CUTLASS_REJECT_IF(
        pattern.global_scale_factor == nullptr ||
            !pattern.global_scale_factor->isFusionInput(),
        "Block-scaled outputs currently require a global scale factor "
        "residing in global memory");
    NVF_CUTLASS_REJECT_IF(
        tv->definition() == nullptr,
        "Must have already processed pre-scaled output's definition but it "
        "has no definition");
    // Assume we have already processed val's definition, so it should have an
    // EVT node
    EVTModel::Node* unquantized_node = getNodeFor(pattern.unquantized_output);
    NVF_CUTLASS_REJECT_IF(
        unquantized_node == nullptr,
        "Could not find EVT node for unquantized output");

    EVTModel::Node* scaling_node = model_.makeNode(
        "cutlass::epilogue::fusion::Sm100BlockScaleFactorRowStore<" +
        std::to_string(pattern.block_size) + ", EpilogueTileShape, " +
        dtypeToCutlass(pattern.quantized_output->dtype()) + ", " +
        dtypeToCutlass(pattern.unquantized_output->dtype()) + ", " +
        dtypeToCutlass(pattern.block_scale_factors->dtype()) +
        ", cutlass::FloatRoundStyle::round_to_nearest>");
    scaling_node->arguments = {
        {"ptr_scale_factor", getPointerCode(pattern.block_scale_factors)},
        {"norm_constant_ptr", getPointerCode(pattern.global_scale_factor)},
        {"norm_constant_stride", "{}"}};

    EVTModel::Node* visitor_node =
        model_.makeNode("cutlass::epilogue::fusion::Sm90EVT");
    visitor_node->inputs.push_back(scaling_node);
    visitor_node->inputs.push_back(unquantized_node);

    val_nodes_[pattern.unquantized_output] = visitor_node;
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
        NVF_CUTLASS_REJECT("Unhandled unary op type: ", uop->getUnaryOpType());
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
    NVF_CUTLASS_REJECT_IF(
        bop->lhs()->dtype() != bop->rhs()->dtype(),
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
  CutlassMatmulPattern pattern_;
  TensorView* mma_out_;

  EVTModel model_;
  std::unordered_map<Val*, EVTModel::Node*> val_nodes_;
  std::unordered_map<Val*, BlockScaledOutputPattern> block_scaling_patterns_;
};

} // namespace

EVTModel::EVTModel(const EVTModel& model) {
  std::unordered_map<Node*, Node*> old2new;

  for (const auto& node_up : model.nodes_up_) {
    Node* new_node = makeNode(node_up->name);
    new_node->arguments = node_up->arguments;
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
  setRootTensorView(model.getRootTensorView());
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
  if (node->arguments.empty()) {
    return {"{}", node->name};
  }
  std::stringstream ss;
  indent(ss, indent_size) << "{  // " << node->name << "\n";

  for (const auto& [i, kv] : enumerate(node->arguments)) {
    indent(ss, indent_size + 1) << "." << kv.first << "=" << kv.second;
    if (i < node->arguments.size() - 1) {
      ss << ",";
    }
    ss << "\n";
  }
  indent(ss, indent_size) << "}";
  return {ss.str(), ""};
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
  if (node->name == "cutlass::epilogue::fusion::Sm90Compute") {
    // Sm90Compute does not require arguments
    // TODO: We should probably not represent Sm90Compute's template parameters
    // as nodes in the EVT
    indent(ss, indent_size) << "{}";
    return {ss.str(), node->name};
  }

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

  // Sm90TreeVisitor is defined like this:
  //
  //   template <class NodeOp, class... ChildOps>
  //   struct Sm90TreeVisitor : Sm90VisitorImpl<ChildOps..., NodeOp>
  //
  // Sm90EVT is just an alias to Sm90TreeVisitor.
  //
  // Notice that NodeOp is the op for the root of the tree and ChildOps are its
  // in-neighbors (producer nodes). When specifying the template parameters
  // NodeOp comes first, but when specifying arguments we need to follow the
  // Sm90VisitorImpl pattern and pass the NodeOp args last instead.
  bool has_node_op =
      node->name == "cutlass::epilogue::fusion::Sm90TreeVisitor" ||
      node->name == "cutlass::epilogue::fusion::Sm90EVT";

  CommentedString node_op_args;
  for (EVTModel::Node* input : node->inputs) {
    if (has_node_op && node_op_args.str.empty()) {
      // Save NodeOp args in order to print them last
      node_op_args = argStringHelper(input, indent_size + 1);
      node_op_args.comment = "(NodeOp arguments last) " + node_op_args.comment;
      continue;
    }
    print_line(false);
    prev_cs = argStringHelper(input, indent_size + 1);
  }
  if (has_node_op) {
    NVF_ERROR(!node_op_args.str.empty(), "Could not find NodeOp");
    // We have a node op. Print its arguments last
    print_line(false);
    std::stringstream ss_name;
    prev_cs = node_op_args;
  }
  print_line(true);
  indent(ss, indent_size) << "}";
  return {ss.str(), ""};
}

CommentedString argStringHelper(EVTModel::Node* node, int64_t indent_size) {
  NVF_ERROR(node != nullptr);
  if (node->inputs.empty()) {
    if (node->arguments.empty()) {
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
    if (!node_up->arguments.empty()) {
      ss << "[";
      bool first = true;
      for (const auto& [k, v] : node_up->arguments) {
        if (!first) {
          ss << ", ";
        }
        first = false;
        ss << k << "=" << v;
      }
      ss << "]";
    }
    ss << "\n";
  }
  ss << "}";
  return ss.str();
}

// TODO: DataWrapperOpt belongs in scheduler_utils
EVTModel extractEVTModel(Fusion* fusion) {
  return EVTConverter::convert(fusion);
}

} // namespace cutlass_codegen

} // namespace nvfuser
