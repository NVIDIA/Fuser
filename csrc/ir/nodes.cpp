// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <dynamic_transform.h>
#include <ir/cloner.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <ops/arith.h>
#include <root_domain_map.h>
#include <transform_iter.h>
#include <transform_rfactor.h>
#include <transform_view.h>
#include <type.h>

#include <c10/util/irange.h>

#include <complex>
#include <iterator>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>

namespace nvfuser {

namespace {

class ScalarCheck : OptInConstDispatch {
 public:
  static bool sameAs(const Val* v1, const Val* v2) {
    if (v1 == v2)
      return true;

    if (v1->getValType() != v2->getValType())
      return false;

    if (v1->getDataType() != v2->getDataType())
      return false;

    ScalarCheck sc(v1, v2);
    return sc.same_;
  }

 private:
  void handle(const Bool* b) final {
    same_ = v1_->as<Bool>()->sameAs(v2_->as<Bool>());
  }

  void handle(const Double* d) final {
    same_ = v1_->as<Double>()->sameAs(v2_->as<Double>());
  }

  void handle(const Int* i) final {
    same_ = v1_->as<Int>()->sameAs(v2_->as<Int>());
  }

  void handle(const NamedScalar* ns) final {
    same_ = v1_->as<NamedScalar>()->sameAs(v2_->as<NamedScalar>());
  }

  ScalarCheck(const Val* _v1, const Val* _v2) : v1_(_v1), v2_(_v2) {
    OptInConstDispatch::handle(v1_);
  }

 private:
  const Val* v1_ = nullptr;
  const Val* v2_ = nullptr;
  bool same_ = false;
};

} // namespace

bool areEqualScalars(Val* v1, Val* v2) {
  return ScalarCheck::sameAs(v1, v2);
}

template class Scalar<bool>;
template class Scalar<int64_t>;
template class Scalar<double>;
template class Scalar<std::complex<double>>;

template Scalar<bool>* IrBuilder::clone<Scalar<bool>>(
    const Scalar<bool>*,
    IrCloner*);
template Scalar<int64_t>* IrBuilder::clone<Scalar<int64_t>>(
    const Scalar<int64_t>*,
    IrCloner*);
template Scalar<double>* IrBuilder::clone<Scalar<double>>(
    const Scalar<double>*,
    IrCloner*);
template Scalar<std::complex<double>>* IrBuilder::clone<
    Scalar<std::complex<double>>>(
    const Scalar<std::complex<double>>*,
    IrCloner*);

FullOp::FullOp(IrBuilderPasskey passkey, Val* out, Val* fill_value)
    : Expr(passkey) {
  if (out->isA<TensorView>()) {
    auto tv_root = out->as<TensorView>()->getRootDomain();
    for (auto id : tv_root) {
      addInput(id->extent());
    }
  }
  addInput(fill_value);
  addOutput(out);
}

std::string FullOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = full({";
  for (auto i : c10::irange(inputs().size())) {
    if (i == inputs().size() - 1) {
      ss << "}";
    }
    if (i > 0) {
      ss << ", ";
    }
    ss << input(i)->toInlineString(indent_size);
  }
  ss << ");\n";
  return ss.str();
}

std::string FullOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(FullOp)

SelectOp::SelectOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    int64_t dim,
    Val* index)
    : Expr(passkey) {
  addInput(in);
  addInput(index);
  addOutput(out);
  addAttribute(
      IrBuilder::create<Attribute<int64_t>>(passkey.ir_container_, dim));
}

std::string SelectOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = select( " << input(0)->toString()
                          << ", axis = " << getIndexedID()
                          << ", index = " << input(1)->toString() << " )\n";
  return ss.str();
}

std::string SelectOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

IterDomain* SelectOp::getIndexedID() const {
  return TensorDomain::noReductions(
             ir_utils::getTvInput(this)->getMaybeRFactorDomain())
      .at(dim());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SelectOp)

IndexSelectOp::IndexSelectOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    int64_t dim,
    Val* indices)
    : Expr(passkey) {
  addInput(in);
  addInput(indices);
  addOutput(out);
  addAttribute(
      IrBuilder::create<Attribute<int64_t>>(passkey.ir_container_, dim));
}

std::string IndexSelectOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = index_select( ";
  ss << input(0)->toString() << ", dim = " << dim() << ", "
     << input(1)->toString() << " )\n";
  return ss.str();
}

std::string IndexSelectOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

IterDomain* IndexSelectOp::getIndexedID() const {
  return TensorDomain::noReductions(
             ir_utils::getTvInput(this)->getMaybeRFactorDomain())
      .at(dim());
}

IterDomain* IndexSelectOp::getConsumerOfIndexedID() const {
  return ir_utils::getTvOutput(this)->getRootDomain().at(dim());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IndexSelectOp)

TorchGatherOp::TorchGatherOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    int64_t dim,
    Val* indices,
    bool exact_sizes)
    : Expr(passkey) {
  addInput(in);
  addInput(indices);
  addOutput(out);
  addAttribute(
      IrBuilder::create<Attribute<int64_t>>(passkey.ir_container_, dim));
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, exact_sizes));
}

std::string TorchGatherOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = "
                          << (exactSizes() ? "take_along_axis" : "torch_gather")
                          << "( " << input(0)->toString();
  if (exactSizes()) {
    ss << ", " << input(1)->toString() << ", dim = " << dim() << " )\n";
  } else {
    ss << ", dim = " << dim() << ", " << input(1)->toString() << " )\n";
  }
  return ss.str();
}

std::string TorchGatherOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

IterDomain* TorchGatherOp::getIndexedID() const {
  return TensorDomain::noReductions(lookupTv()->getMaybeRFactorDomain())
      .at(dim());
}

IterDomain* TorchGatherOp::getConsumerOfIndexedID() const {
  return ir_utils::getTvOutput(this)->getRootDomain().at(dim());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(TorchGatherOp)

ScatterOp::ScatterOp(
    IrBuilderPasskey passkey,
    ScatterOpType type,
    Val* out,
    Val* self,
    int64_t dim,
    Val* index,
    Val* src)
    : Expr(passkey) {
  addInput(self);
  addInput(index);
  addInput(src);
  addOutput(out);
  addAttribute(
      IrBuilder::create<Attribute<int64_t>>(passkey.ir_container_, dim));
  addAttribute(
      IrBuilder::create<Attribute<ScatterOpType>>(passkey.ir_container_, type));
}

std::string ScatterOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " =" << getScatterOpType() << "(";
  ss << "self = " << selfTv()->toString() << ", dim = " << dim()
     << ", src = " << input(2)->toString() << ", idx = " << input(1)->toString()
     << " )\n";
  return ss.str();
}

std::string ScatterOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Scatter op can not be printed inline");
}

IterDomain* ScatterOp::getIndexedID() const {
  return ir_utils::getTvOutput(this)->getRootDomain().at(dim());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ScatterOp)

IotaOp::IotaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* length,
    Val* start,
    Val* step)
    : Expr(passkey) {
  TORCH_CHECK(isIntegralType(*length->getDataType()));
  addInput(length);
  TORCH_CHECK(start->getDataType() == step->getDataType());
  TORCH_CHECK(start->getDataType() == out->getDataType());
  addInput(start);
  addInput(step);
  addOutput(out);
}

std::string IotaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString();
  ss << "\n";
  indent_size++;
  indent(ss, indent_size) << " = iota(" << length()->toString() << ", "
                          << start()->toString() << ", " << step()->toString()
                          << ", " << dtype() << ");\n";
  return ss.str();
}

std::string IotaOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IotaOp)

EyeOp::EyeOp(IrBuilderPasskey passkey, Val* out, DataType dtype)
    : Expr(passkey) {
  if (out->isA<TensorView>()) {
    addInput(out->as<TensorView>()->getRootDomain()[0]->extent());
    if (out->as<TensorView>()->getRootDomain()[1] !=
        out->as<TensorView>()->getRootDomain()[0]) {
      addInput(out->as<TensorView>()->getRootDomain()[1]->extent());
    }
  }
  addOutput(out);
  addAttribute(
      IrBuilder::create<Attribute<DataType>>(passkey.ir_container_, dtype));
}

std::string EyeOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = eye(" << input(0)->toString() << ", "
                          << dtype() << ");\n";
  return ss.str();
}

std::string EyeOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(EyeOp)

UnaryOp::UnaryOp(IrBuilderPasskey passkey, UnaryOpType type, Val* out, Val* in)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addAttribute(
      IrBuilder::create<Attribute<UnaryOpType>>(passkey.ir_container_, type));
}

std::vector<EvaluatorValue> UnaryOp::evaluate(
    const std::vector<EvaluatorValue>& inputs) const {
  using namespace EvaluatorValue_functions;
  const auto& in = inputs.at(0);
  switch (getUnaryOpType()) {
    case UnaryOpType::Neg:
      return {-in};
    case UnaryOpType::Cast:
      if (isIntegralType(*out()->getDataType())) {
        return {EvaluatorValue(in.cast<int64_t>())};
      } else if (isFloatingPointType(*out()->getDataType())) {
        return {EvaluatorValue(in.cast<double>())};
      } else if (out()->getDataType() == DataType::Bool) {
        return {EvaluatorValue(in.cast<bool>())};
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "dtype not supported in evaluator: ", *out()->getDataType());
      }
    case UnaryOpType::Abs:
      return {abs(in)};
      break;
    case UnaryOpType::Not:
      return {notExpr(in)};
      break;
    default:
      TORCH_CHECK(
          false,
          "Unexpected operator type ",
          getUnaryOpType(),
          " in ",
          toString());
  }
}

void UnaryOp::printHelper(std::stringstream& ss, std::string input) const {
  auto op_type = getUnaryOpType();

  if (auto inline_uop = inline_op_str(op_type)) {
    ss << inline_uop.value() << input;
  } else {
    if (op_type == UnaryOpType::Cast) {
      c10::optional<std::string> cast_str = cast_func_str(std::make_pair(
          in()->getDataType().value(), out()->getDataType().value()));
      TORCH_INTERNAL_ASSERT(cast_str != c10::nullopt, "Unsupported Cast");
      ss << cast_str.value();
    } else {
      if (alsoBooleanOperator(op_type) &&
          out()->getDataType().value() == DataType::Bool) {
        ss << stringifyBooleanOp(op_type);
      } else {
        ss << op_type;
      }
      if (out()->getDataType().value() == DataType::Float &&
          needFloatSuffix(op_type)) {
        ss << "f";
      }
    }
    ss << "(" << input << ")";
  }
}

std::string UnaryOp::toString(int indent_size) const {
  std::stringstream ss;
  bool istvop = ir_utils::isTvOp(this);
  indent(ss, indent_size) << out()->toString();
  if (istvop) {
    ss << "\n";
    indent_size++;
    indent(ss, indent_size);
  }
  ss << " = ";
  printHelper(ss, in()->toString());
  ss << ";\n";
  return ss.str();
}

std::string UnaryOp::toInlineString(int indent_size) const {
  checkInlineable(this);
  std::stringstream ss;
  printHelper(ss, in()->toInlineString());
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(UnaryOp)

BinaryOp::BinaryOp(
    IrBuilderPasskey passkey,
    BinaryOpType type,
    Val* out,
    Val* lhs,
    Val* rhs)
    : Expr(passkey) {
  addOutput(out);
  addInput(lhs);
  addInput(rhs);
  addAttribute(
      IrBuilder::create<Attribute<BinaryOpType>>(passkey.ir_container_, type));
}

std::vector<EvaluatorValue> BinaryOp::evaluate(
    const std::vector<EvaluatorValue>& inputs) const {
  using namespace EvaluatorValue_functions;
  const auto& lhs = inputs.at(0);
  const auto& rhs = inputs.at(1);

  switch (getBinaryOpType()) {
    case BinaryOpType::Add:
      return {lhs + rhs};
      break;
    case BinaryOpType::Sub:
      return {lhs - rhs};
      break;
    case BinaryOpType::Mul:
      return {lhs * rhs};
      break;
    case BinaryOpType::Div:
      TORCH_CHECK(rhs != 0);
      return {lhs / rhs};
      break;
    case BinaryOpType::Mod:
      TORCH_CHECK(rhs != 0);
      return {lhs % rhs};
      break;
    case BinaryOpType::CeilDiv:
      TORCH_CHECK(rhs != 0);
      return {ceildiv(lhs, rhs)};
      break;
    case BinaryOpType::And:
      return {lhs && rhs};
      break;
    case BinaryOpType::Or:
      return {lhs || rhs};
      break;
    case BinaryOpType::Xor:
      return {lhs ^ rhs};
      break;
    case BinaryOpType::Eq:
      return {lhs == rhs};
      break;
    case BinaryOpType::NE:
      return {lhs != rhs};
      break;
    case BinaryOpType::GT:
      return {lhs > rhs};
      break;
    case BinaryOpType::GE:
      return {lhs >= rhs};
      break;
    case BinaryOpType::LT:
      return {lhs < rhs};
      break;
    case BinaryOpType::LE:
      return {lhs <= rhs};
      break;
    case BinaryOpType::Max:
      return {max(lhs, rhs)};
      break;
    case BinaryOpType::Min:
      return {min(lhs, rhs)};
      break;
    default:
      TORCH_CHECK(
          false,
          "Unexpected operator type: ",
          getBinaryOpType(),
          " in ",
          toString());
  }
}

void BinaryOp::printHelper(
    std::stringstream& ss,
    int indent_size,
    std::string lhs,
    std::string rhs) const {
  bool istvop = ir_utils::isTvOp(this);
  auto op_type = getBinaryOpType();
  if (auto inline_bop = inline_op_str(op_type)) {
    ss << lhs;
    if (istvop) {
      ss << "\n";
      indent(ss, indent_size);
    }
    ss << " " << inline_bop.value() << " ";
    ss << rhs;
  } else {
    if (alsoBooleanOperator(op_type) &&
        out()->getDataType().value() == DataType::Bool) {
      ss << stringifyBooleanOp(op_type);
    } else {
      ss << op_type;
    }
    if (out()->getDataType().value() == DataType::Float &&
        needFloatSuffix(op_type)) {
      ss << "f";
    }
    ss << "(" << lhs;
    if (istvop) {
      ss << "\n";
      indent(ss, indent_size);
    }
    ss << ", " << rhs << ")";
  }
}

std::string BinaryOp::toString(int indent_size) const {
  std::stringstream ss;
  bool istvop = ir_utils::isTvOp(this);
  indent(ss, indent_size) << out();

  // tensor operations tend to be long, break them up into multiple lines
  if (istvop) {
    ss << "\n";
    indent_size++;
    indent(ss, indent_size);
  }

  ss << " = ";
  printHelper(ss, indent_size, lhs()->toString(), rhs()->toString());
  ss << ";\n";
  return ss.str();
}

std::string BinaryOp::toInlineString(int indent_size) const {
  checkInlineable(this);
  std::stringstream ss;
  printHelper(
      ss, indent_size, lhs()->toInlineString(), rhs()->toInlineString());
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BinaryOp)

TernaryOp::TernaryOp(
    IrBuilderPasskey passkey,
    TernaryOpType type,
    Val* out,
    Val* in1,
    Val* in2,
    Val* in3)
    : Expr(passkey) {
  addOutput(out);
  addInput(in1);
  addInput(in2);
  addInput(in3);
  addAttribute(
      IrBuilder::create<Attribute<TernaryOpType>>(passkey.ir_container_, type));
}

std::vector<EvaluatorValue> TernaryOp::evaluate(
    const std::vector<EvaluatorValue>& inputs) const {
  using namespace EvaluatorValue_functions;
  const auto& in1 = inputs.at(0);
  const auto& in2 = inputs.at(1);
  const auto& in3 = inputs.at(2);
  switch (getTernaryOpType()) {
    case TernaryOpType::Where:
      return {in1.as<bool>() ? in2 : in3};
      break;
    default:
      TORCH_CHECK(
          false,
          "Unexpected operator type: ",
          getTernaryOpType(),
          " in ",
          toString());
  }
}

void TernaryOp::printHelper(
    std::stringstream& ss,
    int indent_size,
    std::string in1,
    std::string in2,
    std::string in3) const {
  bool istvop = ir_utils::isTvOp(this);
  ss << getTernaryOpType() << "(" << in1;
  if (istvop) {
    ss << "\n";
    indent(ss, indent_size);
  }
  ss << ", " << in2;
  if (istvop) {
    ss << "\n";
    indent(ss, indent_size);
  }
  ss << ", " << in3 << ")";
}

std::string TernaryOp::toString(int indent_size) const {
  std::stringstream ss;
  bool istvop = ir_utils::isTvOp(this);
  indent(ss, indent_size);
  ss << out()->toString();

  // tensor operations tend to be long, break them up into multiple lines
  if (istvop) {
    ss << "\n";
    indent_size++;
    indent(ss, indent_size);
  }

  ss << " = ";
  printHelper(
      ss, indent_size, in1()->toString(), in2()->toString(), in3()->toString());
  ss << ";\n";
  return ss.str();
}

std::string TernaryOp::toInlineString(int indent_size) const {
  checkInlineable(this);
  std::stringstream ss;
  printHelper(
      ss,
      indent_size,
      in1()->toInlineString(),
      in2()->toInlineString(),
      in3()->toInlineString());
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(TernaryOp)

RNGOp::RNGOp(
    IrBuilderPasskey passkey,
    RNGOpType type,
    Val* out,
    DataType dtype,
    std::vector<Val*> parameters,
    int rng_offset,
    Val* philox_index)
    : Expr(passkey) {
  if (auto tv_out = dynamic_cast<TensorView*>(out)) {
    for (auto id : tv_out->getRootDomain()) {
      TORCH_CHECK(!id->isReduction(), "Output of RNGOp can not have reduction");
      addInput(id->extent());
    }
  }
  for (auto v : parameters) {
    addInput(v);
  }
  addOutput(out);
  RNGOp::Attributes attr{type, dtype, rng_offset};
  addAttribute(IrBuilder::create<Attribute<RNGOp::Attributes>>(
      passkey.ir_container_, attr));
  addAttribute(philox_index);
}

std::string RNGOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size);
  ss << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size);
  ss << " = ";
  ss << getRNGOpType() << "({" << toDelimitedString(getShape()) << "}, ";
  if (!getParameters().empty()) {
    ss << toDelimitedString(getParameters()) << ", ";
  }
  ss << dtype() << ");\n";
  return ss.str();
}

std::string RNGOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

int64_t RNGOp::getOutputDims() const {
  int64_t ndims = 0;
  if (auto tv_out = dynamic_cast<TensorView*>(output(0))) {
    ndims = (int64_t)tv_out->getRootDomain().size();
  }
  return ndims;
}

NVFUSER_DEFINE_CLONE_AND_CREATE(RNGOp)

BroadcastOp::BroadcastOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    std::vector<bool> is_broadcast_dims)
    : Expr(passkey) {
  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      (out_type == ValType::TensorView && in_type == ValType::TensorView) ||
          (out_type == ValType::TensorIndex && in_type == ValType::TensorIndex),
      "Cannot braodcast a non-tensor object.");

  addOutput(out);
  addInput(in);

  // Validate the broadcast flags when this expr is created with
  // TensorView. Broadcast with TensorIndex only appears after
  // lowering, so it should have already been validated.
  if (out->isA<TensorView>()) {
    TORCH_INTERNAL_ASSERT(in->isA<TensorView>());
    auto in_tv = in->as<TensorView>();
    auto out_tv = out->as<TensorView>();
    auto in_dom = TensorDomain::noReductions(in_tv->getMaybeRFactorDomain());
    auto& out_dom = out_tv->getRootDomain();
    TORCH_INTERNAL_ASSERT(
        is_broadcast_dims.size() == out_dom.size(),
        "The dimensions of output tensor and does not match with is_broadcast_dims");

    auto out_size = is_broadcast_dims.size();
    auto num_new_broadcasts = 0;
    for (const auto i : c10::irange(out_size)) {
      if (is_broadcast_dims[i]) {
        num_new_broadcasts++;
        auto id = out_dom[i];
        TORCH_INTERNAL_ASSERT(
            id->isBroadcast(),
            "New broadcast dimension does not properly set its IterType.");
        TORCH_INTERNAL_ASSERT(
            !id->hasExpandedExtent(),
            "New broadcast dimension can not be expanded.");
        TORCH_INTERNAL_ASSERT(
            id->extent()->isOneInt(),
            "New broadcast dimension must have extent 1");
      } else {
        auto in_id = in_dom[i - num_new_broadcasts];
        auto out_id = out_dom[i];
        TORCH_INTERNAL_ASSERT(
            in_id->sameAs(out_id), "IterDomain does not match in BroadcastOp");
      }
    }
    TORCH_INTERNAL_ASSERT(
        out_size == in_dom.size() + num_new_broadcasts,
        "The dimensions of output tensor and does not match with is_broadcast_dims and input tensor");
  }

  addAttribute(IrBuilder::create<Attribute<std::vector<bool>>>(
      passkey.ir_container_, std::move(is_broadcast_dims)));
}

std::string BroadcastOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size) << "   = broadcast( " << in()->toString() << " )\n";
  return ss.str();
}

std::string BroadcastOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BroadcastOp)

SqueezeOp::SqueezeOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    std::vector<bool> is_squeeze_dims)
    : Expr(passkey) {
  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      in_type == ValType::TensorView,
      "Squeeze input must be a TensorView: ",
      in->toString());

  TORCH_INTERNAL_ASSERT(
      out_type == ValType::TensorView,
      "Squeeze output must be a TensorView: ",
      in->toString());

  addOutput(out);
  addInput(in);

  // Validate the squeeze flags
  auto in_tv = in->as<TensorView>();
  auto out_tv = out->as<TensorView>();
  auto in_dom = TensorDomain::noReductions(in_tv->getMaybeRFactorDomain());
  auto& out_dom = out_tv->getRootDomain();
  TORCH_INTERNAL_ASSERT(
      is_squeeze_dims.size() == in_dom.size(),
      "The dimensions of input tensor and does not match with is_squeeze_dims");

  auto in_size = is_squeeze_dims.size();
  auto num_removed_broadcasts = 0;
  for (const auto i : c10::irange(is_squeeze_dims.size())) {
    if (is_squeeze_dims[i]) {
      num_removed_broadcasts++;
      auto id = in_dom[i];
      TORCH_INTERNAL_ASSERT(
          id->isBroadcast(), "Can not squeeze non-broadcasting dimension(s).");
      TORCH_INTERNAL_ASSERT(
          !id->hasExpandedExtent(), "Can not squeeze expanded dimension(s).");
      TORCH_INTERNAL_ASSERT(
          id->extent()->isOneInt(),
          "Can not squeeze dimension(s) with size != 1.");
    } else {
      auto in_id = in_dom[i];
      auto out_id = out_dom[i - num_removed_broadcasts];
      TORCH_INTERNAL_ASSERT(
          in_id->sameAs(out_id), "IterDomain does not match in BroadcastOp");
    }
  }
  TORCH_INTERNAL_ASSERT(
      in_size == out_tv->nDims() + num_removed_broadcasts,
      "The dimensions of output tensor and does not match with is_squeeze_dims and input tensor");

  addAttribute(IrBuilder::create<Attribute<std::vector<bool>>>(
      passkey.ir_container_, std::move(is_squeeze_dims)));
}

std::string SqueezeOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size) << "   = squeeze( " << in()->toString() << " )\n";
  return ss.str();
}

std::string SqueezeOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SqueezeOp)

ReductionOp::ReductionOp(
    IrBuilderPasskey passkey,
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in,
    bool is_allreduce)
    : Expr(passkey) {
  TORCH_CHECK(
      out->getValType().value() == ValType::TensorView ||
      out->getValType().value() == ValType::TensorIndex);

  TORCH_INTERNAL_ASSERT(
      (in->getValType() == ValType::TensorView &&
       out->getValType() == ValType::TensorView) ||
          (in->getValType() == ValType::TensorIndex &&
           out->getValType() == ValType::TensorIndex),
      "Reduction operation was created that does not have tensor inputs and outputs.");

  if (in->isA<TensorView>()) {
    TORCH_INTERNAL_ASSERT(
        TensorDomain::noReductions(
            in->as<TensorView>()->getMaybeRFactorDomain())
                .size() == out->as<TensorView>()->getRootDomain().size(),
        "Reduction operation created with mismatched domains.");
  }
  TORCH_INTERNAL_ASSERT(
      init->isConstScalar(),
      "Tried to create a reduction operation whith an initial value that isn't a constant.");

  addOutput(out);
  addInput(in);
  addAttribute(init);
  addAttribute(IrBuilder::create<Attribute<BinaryOpType>>(
      passkey.ir_container_, reduction_op_type));
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, is_allreduce));
}

std::string ReductionOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out() << "\n";
  indent(ss, indent_size) << "   = reduction( " << in()->toString()
                          << ", op = " << getReductionOpType()
                          << ", initial value = " << init()->toString()
                          << ", allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string ReductionOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ReductionOp)

GroupedReductionOp::GroupedReductionOp(
    IrBuilderPasskey passkey,
    std::vector<BinaryOpType> reduction_op_types,
    std::vector<Val*> init_vals,
    std::vector<Val*> outputs,
    std::vector<Val*> inputs,
    bool is_fused)
    : Expr(passkey) {
  for (auto out : outputs) {
    addOutput(out);
  }

  for (auto in : inputs) {
    addInput(in);
  }

  addAttribute(IrBuilder::create<Attribute<std::vector<BinaryOpType>>>(
      passkey.ir_container_, std::move(reduction_op_types)));
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, is_fused));

  for (auto init : init_vals) {
    addAttribute(init);
  }
}

std::string GroupedReductionOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedReductionOp(\n";
  ++indent_size;
  for (const auto i : c10::irange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size)
        << output(i)->toString() << " = reduction( " << input(i)->toString()
        << ", op = " << getReductionOpType(i)
        << ", initial value = " << initVal(i)->toString() << " )\n";
  }
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string GroupedReductionOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

int GroupedReductionOp::getExprIndexOfOutput(Val* output_val) const {
  auto it = std::find(outputs().begin(), outputs().end(), output_val);
  if (it != outputs().end()) {
    return (int)std::distance(outputs().begin(), it);
  }

  TORCH_INTERNAL_ASSERT(
      false, "Not an output, ", output_val->toString(), ", of ", toString());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedReductionOp)

c10::optional<WelfordTriplet::ValName> WelfordTriplet::getNameOf(
    Val* val) const {
  auto it = std::find(begin(), end(), val);
  if (it != end()) {
    return indexToValName((int)std::distance(begin(), it));
  }

  return c10::optional<WelfordTriplet::ValName>();
}

bool WelfordTriplet::sameAs(const WelfordTriplet& other) const {
  return this == &other ||
      (avg()->sameAs(other.avg()) && var()->sameAs(other.var()) &&
       N()->sameAs(other.N()));
}

WelfordTriplet WelfordTriplet::clone(IrCloner* ir_cloner) const {
  return transform([&](const Val* val) { return ir_cloner->clone<Val>(val); });
}

std::vector<WelfordTriplet> WelfordTriplet::clone(
    const std::vector<WelfordTriplet>& src,
    IrCloner* ir_cloner) {
  std::vector<WelfordTriplet> cloned(src.size());
  for (const auto i : c10::irange(src.size())) {
    cloned.at(i) = src.at(i).clone(ir_cloner);
  }
  return cloned;
}

WelfordOp::WelfordOp(
    IrBuilderPasskey passkey,
    const WelfordTriplet& output,
    const WelfordTriplet& input,
    const WelfordTriplet& init,
    bool is_fused)
    : Expr(passkey) {
  // Previously, nullptr was accepted and implicitly replaced by
  // default values. Looks like we always pass some non-null values,
  // so removed the implicit default behavior for code simplicity.
  TORCH_INTERNAL_ASSERT(output.avg() != nullptr);
  TORCH_INTERNAL_ASSERT(output.var() != nullptr);
  TORCH_INTERNAL_ASSERT(output.N() != nullptr);
  TORCH_INTERNAL_ASSERT(init.avg() != nullptr);
  TORCH_INTERNAL_ASSERT(init.var() != nullptr);
  TORCH_INTERNAL_ASSERT(init.N() != nullptr);
  TORCH_INTERNAL_ASSERT(input.avg() != nullptr);
  TORCH_INTERNAL_ASSERT(input.var() != nullptr);
  TORCH_INTERNAL_ASSERT(input.N() != nullptr);

  // Check output type
  TORCH_INTERNAL_ASSERT(
      output.avg()->getValType().value() == ValType::TensorView ||
      output.avg()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(
      output.var()->getValType().value() == ValType::TensorView ||
      output.var()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(
      output.N()->getValType().value() == ValType::TensorView ||
      output.N()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(isIntegralType(output.N()->dtype()));

  // check initial value
  TORCH_INTERNAL_ASSERT(init.N()->getValType().value() == ValType::Scalar);
  TORCH_INTERNAL_ASSERT(isIntegralType(init.N()->dtype()));
  if (!init.N()->isZeroInt()) {
    // when initial count is zero, no initial variance or average is needed
    // initial value with a count of 1 is un-common enough that I'll push
    // the responsibility of creating all-zero var tensors to the user
    TORCH_INTERNAL_ASSERT(
        init.avg()->getValType().value() == ValType::TensorView ||
        init.avg()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(
        init.var()->getValType().value() == ValType::TensorView ||
            init.var()->getValType().value() == ValType::TensorIndex,
        "Invalid initial var: ",
        init.var()->toString());
  }

  // check input
  TORCH_INTERNAL_ASSERT(
      input.avg()->getValType().value() == ValType::TensorView ||
          input.avg()->getValType().value() == ValType::TensorIndex,
      input.avg()->getValType().value());
  TORCH_INTERNAL_ASSERT(
      input.N()->getValType().value() == ValType::Scalar ||
      input.N()->getValType().value() == ValType::TensorView ||
      input.N()->getValType().value() == ValType::TensorIndex);
  TORCH_INTERNAL_ASSERT(isIntegralType(input.N()->dtype()));
  if (!input.N()->isOneInt()) {
    // when input is only one value, only the value is required through avg
    // input the var part is implicitly 0 and codegen will handle that.
    TORCH_INTERNAL_ASSERT(
        input.var()->getValType().value() == ValType::TensorView ||
        input.var()->getValType().value() == ValType::TensorIndex);
  } else {
    TORCH_INTERNAL_ASSERT(
        input.var() == nullptr || input.var()->isZeroInt(),
        "Invalid var input, which must be either nullptr or scalar zero when the N input is one.");
  }

  addOutput(output.avg());
  addOutput(output.var());
  addOutput(output.N());

  addInput(input.avg());
  addInput(input.var());
  addInput(input.N());

  addAttribute(init.avg());
  addAttribute(init.var());
  addAttribute(init.N());
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, is_fused));

  TORCH_INTERNAL_ASSERT(attributes().size() == kNumAttrs);
}

WelfordOp::WelfordOp(
    IrBuilderPasskey passkey,
    Val* out_avg,
    Val* out_var,
    Val* out_N,
    Val* in_avg,
    Val* in_var,
    Val* in_N,
    Val* init_avg,
    Val* init_var,
    Val* init_N,
    bool is_fused)
    : WelfordOp(
          passkey,
          WelfordTriplet(out_avg, out_var, out_N),
          WelfordTriplet(in_avg, in_var, in_N),
          WelfordTriplet(init_avg, init_var, init_N),
          is_fused) {}

Val* WelfordOp::getInitValOfOutput(Val* output_val) const {
  auto val_name = outputTriplet().getNameOf(output_val);

  TORCH_INTERNAL_ASSERT(
      val_name.has_value(),
      "Not an output val ",
      output_val->toString(),
      " of ",
      toString());

  return initTriplet().get(*val_name);
}

std::vector<Val*> WelfordOp::getInitVals() const {
  std::vector<Val*> init_vals({initAvg(), initVar(), initN()});
  return init_vals;
}

std::string WelfordOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << outAvg()->toString() << "(Avg),\n"
                          << outVar()->toString() << "(Var),\n"
                          << outN()->toString() << "(Count)"
                          << "\n = Welford ( ";
  if (singleValue()) {
    ss << inAvg()->toString() << "(Avg), ";
  } else {
    ss << inAvg()->toString() << "(Avg)\n  " << inVar()->toString()
       << "(Var)\n  " << inN()->toString() << "(Count)";
  }
  if (hasInit()) {
    ss << "\n  initial value = " << initAvg()->toString() << "(Avg)\n  "
       << initVar()->toString() << "(Var)\n  " << initN()->toString() << "(N)";
  }
  ss << "\n  allreduce = " << (isAllreduce() ? "true" : "false");
  ss << " )\n";
  return ss.str();
}

std::string WelfordOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(WelfordOp)

GroupedWelfordOp::GroupedWelfordOp(
    IrBuilderPasskey passkey,
    std::vector<WelfordTriplet> output_vals,
    std::vector<WelfordTriplet> input_vals,
    std::vector<WelfordTriplet> init_vals,
    bool is_allreduce)
    : Expr(passkey) {
  const auto num_grouped_ops = output_vals.size();

  TORCH_INTERNAL_ASSERT(
      input_vals.size() == num_grouped_ops,
      "Invalid number of input arguments. Expected: ",
      num_grouped_ops,
      ", Given: ",
      input_vals.size());
  TORCH_INTERNAL_ASSERT(
      init_vals.size() == num_grouped_ops,
      "Invalid number of N arguments. Expected: ",
      num_grouped_ops,
      ", Given: ",
      init_vals.size());

  for (const auto i : c10::irange(num_grouped_ops)) {
    // Check output type
    TORCH_INTERNAL_ASSERT(
        output_vals[i].avg()->getValType().value() == ValType::TensorView ||
        output_vals[i].avg()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(
        output_vals[i].var()->getValType().value() == ValType::TensorView ||
        output_vals[i].var()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(
        output_vals[i].N()->getValType().value() == ValType::TensorView ||
        output_vals[i].N()->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(isIntegralType(output_vals[i].N()->dtype()));

    // check initial value
    auto init_avg = init_vals[i].avg();
    auto init_var = init_vals[i].var();
    auto init_N = init_vals[i].N();
    TORCH_INTERNAL_ASSERT(
        init_avg != nullptr && init_var != nullptr && init_N != nullptr,
        "nullptr init vals are not allowed");
    TORCH_INTERNAL_ASSERT(init_N->getValType().value() == ValType::Scalar);
    TORCH_INTERNAL_ASSERT(isIntegralType(init_N->dtype()));
    TORCH_INTERNAL_ASSERT(
        init_avg->getValType().value() == ValType::TensorView ||
            init_avg->getValType().value() == ValType::TensorIndex ||
            (init_N->isZeroInt() &&
             init_avg->getValType().value() == ValType::Scalar),
        "Initial avg must be a tensor or, can be a scalar if initial N is zero.",
        " Initial avg: ",
        init_avg->toString(),
        ". Initial N: ",
        init_N->toString());
    TORCH_INTERNAL_ASSERT(
        init_var->getValType().value() == ValType::TensorView ||
            init_var->getValType().value() == ValType::TensorIndex ||
            (init_N->isZeroInt() &&
             init_var->getValType().value() == ValType::Scalar),
        "Initial var must be a tensor or, can be a scalar if initial N is zero: ",
        init_var->toString());

    // check input
    auto in_avg = input_vals[i].avg();
    auto in_var = input_vals[i].var();
    auto in_N = input_vals[i].N();
    TORCH_INTERNAL_ASSERT(
        in_avg != nullptr && in_var != nullptr && in_N != nullptr,
        "nullptr input vals are not allowed");
    TORCH_INTERNAL_ASSERT(
        in_N->getValType().value() == ValType::Scalar ||
        in_N->getValType().value() == ValType::TensorView ||
        in_N->getValType().value() == ValType::TensorIndex);
    TORCH_INTERNAL_ASSERT(isIntegralType(in_N->dtype()));
    TORCH_INTERNAL_ASSERT(
        in_avg->getValType().value() == ValType::TensorView ||
            in_avg->getValType().value() == ValType::TensorIndex,
        "Invalid input avg argument type: ",
        in_avg->getValType().value());

    if (in_N->isOneInt()) {
      // when input is only one value, only the value is required through avg
      // input the var part must be implicitly 0
      TORCH_INTERNAL_ASSERT(
          in_var->isZeroInt(),
          "Invalid var input, which must be scalar zero when the N input is one: ",
          in_var->toString());
    } else {
      TORCH_INTERNAL_ASSERT(
          in_var->getValType().value() == ValType::TensorView ||
              in_var->getValType().value() == ValType::TensorIndex,
          in_var->getValType().value(),
          ", ",
          in_N->toString());
    }
  }

  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, is_allreduce));
  for (const auto i : c10::irange(num_grouped_ops)) {
    addOutput(output_vals[i].avg());
    addOutput(output_vals[i].var());
    addOutput(output_vals[i].N());
    addInput(input_vals[i].avg());
    addInput(input_vals[i].var());
    addInput(input_vals[i].N());
    addAttribute(init_vals[i].avg());
    addAttribute(init_vals[i].var());
    addAttribute(init_vals[i].N());
  }
}

std::string GroupedWelfordOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedWelford(\n";
  ++indent_size;
  for (const auto i : c10::irange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size) << outAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << outVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << outN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << " = Welford ( ";
    ++indent_size;
    indent(ss, indent_size) << inAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << inVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << inN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << "initial value =\n";
    ++indent_size;
    indent(ss, indent_size) << initAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << initVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << initN(i)->toString() << " (Count) )\n";
    indent_size -= 2;
  }
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string GroupedWelfordOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

int GroupedWelfordOp::getExprIndexOfOutput(Val* output_val) const {
  for (const auto expr_idx : c10::irange(numHorizontallyGroupedExprs())) {
    if (outputVals().at(expr_idx).getNameOf(output_val).has_value()) {
      return (int)expr_idx;
    }
  }

  TORCH_INTERNAL_ASSERT(
      false, "Not an output, ", output_val->toString(), ", of ", toString());
}

Val* GroupedWelfordOp::getInitValOfOutput(Val* output_val) const {
  auto expr_index = getExprIndexOfOutput(output_val);

  auto val_name = outputVals().at(expr_index).getNameOf(output_val).value();

  return initVals().at(expr_index).get(val_name);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedWelfordOp)

//==============================================================================================================================

// MmaOp utils
namespace MmaOpUtils {

// The expected number of concrete domains for gemm
constexpr size_t expected_gemm_cdomains = 2;

// A helper structure used to gather all data created during analysis
struct MmaOpDetails {
  using AxesData = MmaOp::AxesData;
  // Concrete axes from A that are broadcast in B and are not
  //  reduction in output
  AxesData m_axes;
  // Concrete axes from B that are broadcast in A and are not
  //  reduction in output
  AxesData n_axes;
  // Concrete axes from A that are concrete in B and are
  //  reduction in output
  AxesData k_axes;
  // Concrete or broadcast axes that are present in all inputs
  //  and output
  AxesData batch_axes;
  // A placeholder for mma input layout
  std::optional<MmaOptions::MmaLayout> input_layout = std::nullopt;
};

// A helper structure with pieces of information about TensorView
struct TensorViewDetails {
  using AxesData = MmaOp::AxesData;
  // Broadcast domains
  AxesData bcasts;
  // Reduction domains
  AxesData rdomains;
  // Concrete domains
  AxesData cdomains;
};

// A helper for gathering details about TensorView object
TensorViewDetails getDetailsFor(const std::vector<IterDomain*>& dims) {
  TensorViewDetails details;
  for (auto pos : c10::irange((int64_t)dims.size())) {
    const auto axis = dims.at(pos);
    if (axis->isReduction()) {
      details.rdomains.push_back(pos);
      continue;
    }
    if (axis->isBroadcast()) {
      details.bcasts.push_back(pos);
      continue;
    }
    details.cdomains.push_back(pos);
  }
  return details;
}

MmaOptions::MmaLayout getInputLayout(
    const TensorViewDetails& in_a,
    const TensorViewDetails& in_b,
    const MmaOp::AxesData& m_axes,
    const MmaOp::AxesData& n_axes,
    const MmaOp::AxesData& k_axes) {
  // TT layout (b - broadcast, r - reduction):
  // A = [M, K, b]
  // B = [b, K, N]
  // C = [M, r, N] (root domain)
  if ((m_axes.front() < in_a.bcasts.front()) &&
      (k_axes.front() < in_a.bcasts.front()) &&
      (in_b.bcasts.front() < k_axes.front()) &&
      (in_b.bcasts.front() < n_axes.front())) {
    return MmaOptions::MmaLayout::TT;
  }
  // TN layout (b - broadcast, r - reduction):
  // A = [M, b, K]
  // B = [b, N, K]
  // C = [M, N, r] (root domain)
  if ((m_axes.front() < in_a.bcasts.front()) &&
      (in_a.bcasts.front() < k_axes.front()) &&
      (in_b.bcasts.front() < n_axes.front()) &&
      (in_b.bcasts.front() < k_axes.front())) {
    return MmaOptions::MmaLayout::TN;
  }
  // NT layout (b - broadcast, r - reduction):
  // A = [K, M, b]
  // B = [K, b, N]
  // C = [r, M, N] (root domain)
  if ((k_axes.front() < in_a.bcasts.front()) &&
      (m_axes.front() < in_a.bcasts.front()) &&
      (k_axes.front() < in_b.bcasts.front()) &&
      (in_b.bcasts.front() < n_axes.front())) {
    return MmaOptions::MmaLayout::NT;
  }
  // NN layout (b - broadcast, r - reduction):
  // A = [b, K, M]
  // B = [N, K, b]
  // C = [N, r, M] (root domain)
  if ((in_a.bcasts.front() < k_axes.front()) &&
      (k_axes.front() < m_axes.front()) && (n_axes.front() < k_axes.front()) &&
      (k_axes.front() < in_b.bcasts.front())) {
    return MmaOptions::MmaLayout::NN;
  }

  TORCH_INTERNAL_ASSERT(false, "Unsupported input layout");
}

MmaOpDetails getMmaOpDetails(
    TensorView* out,
    TensorView* in_a,
    TensorView* in_b) {
  const auto in_a_details = getDetailsFor(in_a->getMaybeRFactorDomain());
  const auto in_b_details = getDetailsFor(in_b->getMaybeRFactorDomain());
  const auto out_details = getDetailsFor(out->getRootDomain());

  using AxesData = MmaOp::AxesData;

  const auto getMOrNaxes = [](const AxesData& cdomains,
                              const AxesData& bcasts,
                              const AxesData& rdomains) {
    AxesData result;
    // For all concrete domains
    for (const auto& cdomain : cdomains) {
      // That are in broadcast domains but are not in reduction domains
      if ((std::find(bcasts.begin(), bcasts.end(), cdomain) != bcasts.end()) &&
          (std::find(rdomains.begin(), rdomains.end(), cdomain) ==
           rdomains.end())) {
        result.push_back(cdomain);
      }
    }
    return result;
  };

  const auto getKaxes = [](const AxesData& cdomains_a,
                           const AxesData& cdomains_b,
                           const AxesData& rdomains) {
    AxesData result;
    // For all concrete domains from in_a
    for (const auto& cdomain_a : cdomains_a) {
      // That are in concrete domains in in_b and are in reduction domains
      if ((std::find(cdomains_b.begin(), cdomains_b.end(), cdomain_a) !=
           cdomains_b.end()) &&
          (std::find(rdomains.begin(), rdomains.end(), cdomain_a) !=
           rdomains.end())) {
        result.push_back(cdomain_a);
      }
    }
    return result;
  };

  const auto getBatchAxes = [](const TensorViewDetails& in_a_details,
                               const TensorViewDetails& in_b_details,
                               const TensorViewDetails& out_details) {
    AxesData result;
    // Batch candidates:
    //  concrete domains that are in all of inputs and output
    for (const auto& domain : in_a_details.cdomains) {
      if ((std::find(
               in_b_details.cdomains.begin(),
               in_b_details.cdomains.end(),
               domain) != in_b_details.cdomains.end()) &&
          (std::find(
               out_details.cdomains.begin(),
               out_details.cdomains.end(),
               domain) != out_details.cdomains.end())) {
        result.push_back(domain);
      }
    }
    // Batch candidates:
    //  broadcast domains that are in all of inputs and output
    for (const auto& domain : in_a_details.bcasts) {
      if ((std::find(
               in_b_details.bcasts.begin(),
               in_b_details.bcasts.end(),
               domain) != in_b_details.bcasts.end()) &&
          (std::find(
               out_details.bcasts.begin(), out_details.bcasts.end(), domain) !=
           out_details.bcasts.end())) {
        result.push_back(domain);
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  const auto validateInputDetails = [](const TensorViewDetails& details,
                                       const std::string& desc) {
    TORCH_INTERNAL_ASSERT(
        !details.bcasts.empty(), desc, ": has no broadcast domains.");
    TORCH_INTERNAL_ASSERT(
        details.rdomains.empty(), desc, ": has reduction domains.");
    TORCH_INTERNAL_ASSERT(
        details.cdomains.size() >= expected_gemm_cdomains,
        desc,
        ": has unsupported number of concrete domains, expected at least ",
        expected_gemm_cdomains,
        ", got ",
        details.cdomains.size());
  };

  const auto validateOutputDetails = [](const TensorViewDetails& details,
                                        const std::string& desc) {
    // TODO: revise rules when add support for batch gemms
    TORCH_INTERNAL_ASSERT(
        details.bcasts.empty(), desc, ": has broadcast domains.");
    TORCH_INTERNAL_ASSERT(
        !details.rdomains.empty(), desc, ": has no reduction domains.");
    TORCH_INTERNAL_ASSERT(
        (details.cdomains.size() >= expected_gemm_cdomains),
        desc,
        ": has unsupported number of concrete domains, expected at least ",
        expected_gemm_cdomains,
        ", got ",
        details.cdomains.size());
  };

  validateInputDetails(in_a_details, "MmaOp input A");
  validateInputDetails(in_b_details, "MmaOp input B");
  validateOutputDetails(out_details, "MmaOp output");

  MmaOpDetails details;

  // For details, check MmaOpDetails
  details.m_axes = getMOrNaxes(
      in_a_details.cdomains, in_b_details.bcasts, out_details.rdomains);
  details.n_axes = getMOrNaxes(
      in_b_details.cdomains, in_a_details.bcasts, out_details.rdomains);
  details.k_axes = getKaxes(
      in_a_details.cdomains, in_b_details.cdomains, out_details.rdomains);
  details.batch_axes = getBatchAxes(in_a_details, in_b_details, out_details);

  TORCH_INTERNAL_ASSERT(
      !details.m_axes.empty(),
      "MmaOp inputs must define at least a single M dimension");
  TORCH_INTERNAL_ASSERT(
      !details.n_axes.empty(),
      "MmaOp inputs must define at least a single N dimension");
  TORCH_INTERNAL_ASSERT(
      !details.k_axes.empty(),
      "MmaOp inputs must define at least a single K dimension");

  // TODO: for tensor contraction / split-k uses of MmaOp different input layout
  // rules may be needed
  details.input_layout = getInputLayout(
      in_a_details,
      in_b_details,
      details.m_axes,
      details.n_axes,
      details.k_axes);

  return details;
}

}; // namespace MmaOpUtils

MmaOp::MmaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* init)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      out->getValType().value() == ValType::TensorView ||
          out->getValType().value() == ValType::TensorIndex,
      out->getValType().value());

  TORCH_INTERNAL_ASSERT(
      in_a->getValType().value() == ValType::TensorView ||
          in_a->getValType().value() == ValType::TensorIndex,
      in_a->getValType().value());

  TORCH_INTERNAL_ASSERT(
      in_b->getValType().value() == ValType::TensorView ||
          in_b->getValType().value() == ValType::TensorIndex,
      in_b->getValType().value());

  addOutput(out);
  addInput(in_a);
  addInput(in_b);
  // ATTR_POS_INIT
  addAttribute(init);
  // ATTR_POS_OPTS
  addAttribute(
      IrBuilder::create<Attribute<OptionsInMma>>(passkey.ir_container_));
  // ATTR_POS_M_AXES
  addAttribute(IrBuilder::create<Attribute<AxesData>>(passkey.ir_container_));
  // ATTR_POS_N_AXES
  addAttribute(IrBuilder::create<Attribute<AxesData>>(passkey.ir_container_));
  // ATTR_POS_K_AXES
  addAttribute(IrBuilder::create<Attribute<AxesData>>(passkey.ir_container_));
  // ATTR_POS_BATCH_AXES
  addAttribute(IrBuilder::create<Attribute<AxesData>>(passkey.ir_container_));
  // ATTR_POS_INPUT_LAYOUT
  addAttribute(
      IrBuilder::create<Attribute<MmaLayoutOpt>>(passkey.ir_container_));

  MmaOpUtils::MmaOpDetails mma_details;
  // Detailed consistency checks for use case with TensorViews as
  // inputs/output
  if (in_a->isA<TensorView>() && in_b->isA<TensorView>() &&
      out->isA<TensorView>()) {
    mma_details = MmaOpUtils::getMmaOpDetails(
        out->as<TensorView>(), in_a->as<TensorView>(), in_b->as<TensorView>());
  }

  attribute(ATTR_POS_M_AXES)->as<Attribute<AxesData>>()->value =
      std::move(mma_details.m_axes);
  attribute(ATTR_POS_N_AXES)->as<Attribute<AxesData>>()->value =
      std::move(mma_details.n_axes);
  attribute(ATTR_POS_K_AXES)->as<Attribute<AxesData>>()->value =
      std::move(mma_details.k_axes);
  attribute(ATTR_POS_BATCH_AXES)->as<Attribute<AxesData>>()->value =
      std::move(mma_details.batch_axes);
  attribute(ATTR_POS_INPUT_LAYOUT)->as<Attribute<MmaLayoutOpt>>()->value =
      mma_details.input_layout;
}

MmaOp::MmaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* init,
    const OptionsInMma& options,
    const MmaLayoutOpt& input_layout)
    : MmaOp(passkey, out, in_a, in_b, init) {
  attribute(ATTR_POS_OPTS)->as<Attribute<OptionsInMma>>()->value = options;

  const auto input_layout_ =
      attribute(ATTR_POS_INPUT_LAYOUT)->as<Attribute<MmaLayoutOpt>>()->value;
  if (input_layout_.has_value()) {
    TORCH_INTERNAL_ASSERT(input_layout.has_value());
    TORCH_INTERNAL_ASSERT(
        input_layout_.value() == input_layout.value(),
        "Input layout mismatch, infered attribute (",
        nvfuser::toString(input_layout_.value()),
        "), provided attribute (",
        nvfuser::toString(input_layout.value()),
        ")");
  } else {
    attribute(ATTR_POS_INPUT_LAYOUT)->as<Attribute<MmaLayoutOpt>>()->value =
        input_layout;
  }
}

std::string MmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size + 1) << " = mma(" << inA()->toString() << ",\n";
  indent(ss, indent_size + 1) << "       " << inB()->toString() << ")\n";
  return ss.str();
}

std::string MmaOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

void MmaOp::configureOptions(MmaOptions options) {
  OptionsInMma& opt =
      attribute(ATTR_POS_OPTS)->as<Attribute<OptionsInMma>>()->value;
  TORCH_INTERNAL_ASSERT(
      options.macro != MmaOptions::MacroType::NoMMA,
      "Un-configured mma type from options.");
  TORCH_INTERNAL_ASSERT(
      options.accumulator_stride > 0, "Un-configured accumulator stride.");
  opt.accumulator_stride = options.accumulator_stride;
  opt.macro = options.macro;
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MmaOp)

ExpandOp::ExpandOp(
    IrBuilderPasskey passkey,
    TensorView* out,
    TensorView* in,
    std::vector<Val*> _expanded_extents)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  for (auto expanded_extent : _expanded_extents) {
    TORCH_INTERNAL_ASSERT(expanded_extent != nullptr);
    TORCH_INTERNAL_ASSERT(
        expanded_extent->dtype() == DataType::Int,
        "Expanded extents must be of Int type.");
    addInput(expanded_extent);
  }
}

std::string ExpandOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = expand( " << in()
                          << ", {";
  ss << toDelimitedString(expanded_extents());
  ss << "} )\n";
  return ss.str();
}

std::string ExpandOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ExpandOp)

ShiftOp::ShiftOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    std::vector<int> offsets,
    std::vector<int> pad_width)
    : Expr(passkey) {
  // clang-tidy complains about out that it may be null.
  TORCH_INTERNAL_ASSERT(out != nullptr);
  TORCH_INTERNAL_ASSERT(in != nullptr);

  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      out_type == ValType::TensorView && in_type == ValType::TensorView,
      "Cannot shift a non-tensor object.");

  TORCH_INTERNAL_ASSERT(
      offsets.size() ==
          TensorDomain::noReductions(in->as<TensorView>()->getRootDomain())
              .size(),
      "Invalid offset vector: ",
      offsets);

  TORCH_INTERNAL_ASSERT(
      pad_width.size() ==
          TensorDomain::noReductions(in->as<TensorView>()->getRootDomain())
              .size(),
      "Invalid padding width vector: ",
      pad_width);

  addOutput(out);
  addInput(in);
  addAttribute(IrBuilder::create<Attribute<std::vector<int>>>(
      passkey.ir_container_, std::move(offsets)));
  addAttribute(IrBuilder::create<Attribute<std::vector<int>>>(
      passkey.ir_container_, std::move(pad_width)));
}

std::string ShiftOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = shift( "
                          << in()->toString() << ", {" << offsets() << "}, {"
                          << padWidth() << "} )\n";
  return ss.str();
}

std::string ShiftOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ShiftOp)

GatherOp::GatherOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    std::vector<int> window_shape,
    std::vector<std::vector<int>> pad_width)
    : Expr(passkey) {
  // clang-tidy complains about out_ that it may be null.
  TORCH_INTERNAL_ASSERT(out != nullptr);
  TORCH_INTERNAL_ASSERT(in != nullptr);

  auto out_type = out->getValType().value();
  auto in_type = in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      out_type == ValType::TensorView && in_type == ValType::TensorView,
      "Cannot shift a non-tensor object.");

  const auto ndims =
      TensorDomain::noReductions(in->as<TensorView>()->getRootDomain()).size();

  TORCH_INTERNAL_ASSERT(
      window_shape.size() == ndims,
      "Invalid window_shape vector: ",
      window_shape);
  TORCH_INTERNAL_ASSERT(
      pad_width.size() == ndims, "Invalid pad_width vector: ", pad_width);

  for (const auto& pad : pad_width) {
    TORCH_INTERNAL_ASSERT(
        pad.size() == 2, "Padding size for each axis must have two Int vals.");
  }

  addOutput(out);
  addInput(in);
  addAttribute(IrBuilder::create<Attribute<std::vector<int>>>(
      passkey.ir_container_, std::move(window_shape)));
  addAttribute(IrBuilder::create<Attribute<std::vector<std::vector<int>>>>(
      passkey.ir_container_, std::move(pad_width)));
}

std::string GatherOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = gather( "
                          << in()->toString() << ", {";
  ss << toDelimitedString(windowShape()) << "}, {";
  bool no_comma = true;
  for (const auto& pad : padWidth()) {
    if (!no_comma) {
      ss << ", ";
    }
    ss << "{" << pad[0] << ", " << pad[1] << "}";
    no_comma = false;
  }
  ss << "} )\n";
  return ss.str();
}

std::string GatherOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

int64_t GatherOp::gatherAxis(int64_t axis) const {
  if (axis < 0) {
    axis += (int64_t)out()->as<TensorView>()->nDims();
  }
  TORCH_INTERNAL_ASSERT(
      axis >= 0 && axis < (int64_t)windowShape().size(),
      "Invalid axis: ",
      axis);
  return (int64_t)windowShape().size() + axis;
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GatherOp)

ViewAsScalar::ViewAsScalar(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    IterDomain* vector_id,
    Val* index)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addAttribute(vector_id);
  addAttribute(index);
}

std::string ViewAsScalar::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = view_as_scalar( "
                          << in()->toString() << ", " << vector_id()->toString()
                          << " )\n";
  return ss.str();
}

std::string ViewAsScalar::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ViewAsScalar)

ViewOp::ViewOp(IrBuilderPasskey passkey, Val* out, Val* in) : Expr(passkey) {
  addOutput(out);
  addInput(in);
}

std::string ViewOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = view( "
                          << in()->toString() << " )\n";
  return ss.str();
}

std::string ViewOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ViewOp)

LoadStoreOp::LoadStoreOp(
    IrBuilderPasskey passkey,
    LoadStoreOpType op_type,
    Val* out,
    Val* in)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addAttribute(IrBuilder::create<Attribute<LoadStoreOpType>>(
      passkey.ir_container_, op_type));
}

std::vector<EvaluatorValue> LoadStoreOp::evaluate(
    const std::vector<EvaluatorValue>& inputs) const {
  return inputs;
}

std::string LoadStoreOp::toString(int indent_size) const {
  std::stringstream ss;
  std::string optype = load_store_type2string(opType());
  std::string modifier = "";
  { // Get modifier
    TensorView* tv = dynamic_cast<TensorView*>(out());
    if (auto ti = dynamic_cast<kir::TensorIndex*>(out())) {
      tv = ti->view();
    }
    if (tv != nullptr && tv->hasRFactor()) {
      modifier = ".Permute";
    }
  }
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size + 1)
      << " = " << optype << modifier << "( " << in()->toString();
  // Fusion IR does not have predicate
  if (container()->isA<kir::Kernel>() && predicate() != nullptr) {
    ss << ", " << std::endl;
    indent(ss, indent_size + 1)
        << std::string(optype.size() + 5, ' ') << predicate()->toInlineString();
  }
  ss << " )\n";
  return ss.str();
}

std::string LoadStoreOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

bool LoadStoreOp::hasTranspose() const {
  if (auto out_tv = dynamic_cast<TensorView*>(out())) {
    return out_tv->hasRFactor();
  }
  return false;
}

NVFUSER_DEFINE_CLONE_AND_CREATE(LoadStoreOp)

IterDomainBuilder::IterDomainBuilder(Val* _start, Val* _extent)
    : start_(_start), extent_(_extent) {
  TORCH_INTERNAL_ASSERT(
      start_ != nullptr && extent_ != nullptr,
      "Start and extent are required to build an iter domain.");
}

IterDomainBuilder::IterDomainBuilder(const IterDomain* id)
    : start_(id->start()),
      extent_(id->extent()),
      expanded_extent_(
          id->hasExpandedExtent() ? id->expandedExtent() : nullptr),
      stop_offset_(id->stopOffset()),
      parallel_type_(id->getParallelType()),
      iter_type_(id->getIterType()),
      is_rfactor_domain_(id->isRFactorProduct()),
      is_padded_dimension_(id->hasPaddingToMultipleOfWarp()),
      padded_to_size_(id->getMaybeSizeAfterPadding()),
      is_mma_swizzled_(id->isMmaSwizzled()) {}

IterDomainBuilder& IterDomainBuilder::resetSchedulingParams() {
  parallel_type_ = ParallelType::Serial;
  is_rfactor_domain_ = false;
  is_padded_dimension_ = false;
  padded_to_size_ = c10::nullopt;
  is_mma_swizzled_ = false;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::resetRfactor() {
  return is_rfactor_domain(false);
}

IterDomainBuilder& IterDomainBuilder::start(Val* _start) {
  start_ = _start;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::extent(Val* _extent) {
  extent_ = _extent;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::expanded_extent(Val* _expanded_extent) {
  expanded_extent_ = _expanded_extent;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::stop_offset(Val* _stop_offset) {
  stop_offset_ = _stop_offset;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::parallel_type(
    ParallelType _parallel_type) {
  parallel_type_ = _parallel_type;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::iter_type(IterType _iter_type) {
  iter_type_ = _iter_type;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_rfactor_domain(
    bool _is_rfactor_domain) {
  is_rfactor_domain_ = _is_rfactor_domain;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_padded_dimension(
    bool _is_padded_dimension) {
  is_padded_dimension_ = _is_padded_dimension;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::padded_to_size(
    c10::optional<int64_t> _padded_to_size) {
  padded_to_size_ = _padded_to_size;
  return *this;
}

IterDomainBuilder& IterDomainBuilder::is_mma_swizzled(bool _is_mma_swizzled) {
  is_mma_swizzled_ = _is_mma_swizzled;
  return *this;
}

IterDomain* IterDomainBuilder::build() const {
  TORCH_INTERNAL_ASSERT(
      start_ != nullptr && extent_ != nullptr,
      "Start and extent are required to build an iter domain.");
  return IrBuilder::create<IterDomain>(start_->container(), *this);
}

IterDomain::IterDomain(
    IrBuilderPasskey passkey,
    Val* start,
    Val* extent,
    Val* expanded_extent,
    Val* stop_offset,
    ParallelType parallel_type,
    IterType iter_type,
    bool is_rfactor_domain,
    bool is_padded_dimension,
    c10::optional<int64_t> padded_to_size,
    bool is_mma_swizzled)
    : Val(passkey, ValType::IterDomain),
      start_(start),
      extent_(extent),
      expanded_extent_(expanded_extent),
      stop_offset_(
          stop_offset == nullptr ? passkey.ir_container_->zeroVal()
                                 : stop_offset),
      parallel_type_(parallel_type),
      iter_type_(iter_type),
      is_rfactor_domain_(is_rfactor_domain),
      is_padded_dimension_(is_padded_dimension),
      padded_to_size_(padded_to_size),
      is_mma_swizzled_(is_mma_swizzled) {
  // NOTE: We previously asserted !(isRFactorProduct() && isBroadcast()), i.e.
  // that an IterDomain could not be both a broadcast and an rfactor domain.
  // However, since the introduction of the resize op, we now have a legitimate
  // case where this may be true; namely, whenever we resize an IterDomain to
  // size 1, we will mark it as Broadcast, but the resize must lie between root
  // and rfactor.

  TORCH_INTERNAL_ASSERT(
      extent->isIntegralScalar(),
      "Cannot create an iter domain over an extent that is not an int but received ",
      extent,
      " .");

  TORCH_INTERNAL_ASSERT(
      start->isIntegralScalar(),
      "Cannot create an iter domain with a start that is not an int but received ",
      start,
      " .");
}

IterDomain::IterDomain(IrBuilderPasskey passkey, const IterDomainBuilder& args)

    : IterDomain(
          passkey,
          args.start_,
          args.extent_,
          args.expanded_extent_,
          args.stop_offset_,
          args.parallel_type_,
          args.iter_type_,
          args.is_rfactor_domain_,
          args.is_padded_dimension_,
          args.padded_to_size_,
          args.is_mma_swizzled_) {}

IterDomain::IterDomain(const IterDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      start_(ir_cloner->clone(src->start_)),
      extent_(ir_cloner->clone(src->extent_)),
      expanded_extent_(
          src->hasExpandedExtent() ? ir_cloner->clone(src->expandedExtent())
                                   : nullptr),
      stop_offset_(ir_cloner->clone(src->stop_offset_)),
      parallel_type_(src->parallel_type_),
      iter_type_(src->iter_type_),
      is_rfactor_domain_(src->is_rfactor_domain_),
      is_padded_dimension_(src->is_padded_dimension_),
      padded_to_size_(src->padded_to_size_),
      is_mma_swizzled_(src->is_mma_swizzled_) {}

NVFUSER_DEFINE_CLONE(IterDomain)

bool IterDomain::sameAs(const Statement* other) const {
  if (other == this) {
    return true;
  }

  if (!other->isA<IterDomain>()) {
    return false;
  }

  const IterDomain* other_id = other->as<IterDomain>();

  // Here're the data fields of IterDomain:
  // start_
  // extent_
  // expanded_extent_
  // stop_offset_
  // parallel_type_
  // iter_type_
  // is_rfactor_domain_
  // is_padded_dimension_
  // padded_to_size_
  // is_mma_swizzled_

  // Do not take is_rfactor_domain_ into account. IterDomain's are
  // considered the same if they are rfactor or not.

  // TODO: Consider managing them as attributes

  return ScalarCheck::sameAs(start(), other_id->start()) &&
      ScalarCheck::sameAs(extent(), other_id->extent()) &&
      hasExpandedExtent() == other_id->hasExpandedExtent() &&
      (!hasExpandedExtent() ||
       ScalarCheck::sameAs(expandedExtent(), other_id->expandedExtent())) &&
      ScalarCheck::sameAs(stopOffset(), other_id->stopOffset()) &&
      getParallelType() == other_id->getParallelType() &&
      getIterType() == other_id->getIterType() &&
      hasPaddingToMultipleOfWarp() == other_id->hasPaddingToMultipleOfWarp() &&
      getMaybeSizeAfterPadding() == other_id->getMaybeSizeAfterPadding() &&
      isMmaSwizzled() == other_id->isMmaSwizzled();
}

std::string IterDomain::toString(int indent_size) const {
  std::stringstream ss;
  ss << getIterType();
  ss << getParallelType();
  ss << name();
  ss << "{";
  if (!start()->isZeroInt()) {
    ss << start()->toInlineString() << " : ";
  }
  if (stop() != extent()) {
    ss << stop()->toInlineString() << " : ";
  }
  if (isBroadcast() && hasExpandedExtent()) {
    ss << expandedExtent()->toInlineString();
  } else {
    ss << extent()->toInlineString();
  }
  ss << "}";
  if (isRFactorProduct())
    ss << "rf";
  if (hasPaddingToMultipleOfWarp()) {
    ss << "_p";
  }
  return ss.str();
}

std::string IterDomain::toInlineString(int indent_size) const {
  return toString(indent_size);
}

// Returns a new IterDomain matching properties of this except for
// is_rfactor_domain_
IterDomain* IterDomain::cloneWithoutRFactor() const {
  auto cloned = IterDomainBuilder(this).resetRfactor().build();

  return cloned;
}

std::vector<IterDomain*> IterDomain::clone(
    const std::vector<IterDomain*>& domains) {
  std::vector<IterDomain*> cloned_domains;
  std::transform(
      domains.begin(),
      domains.end(),
      std::back_inserter(cloned_domains),
      [](auto id) { return id->cloneWithoutRFactor(); });
  return cloned_domains;
}

// Merging does not propagate the start and stop values of the input
// domains to the merged output domain. The actual range of the
// domains is enforced by predicates. Note that since only root
// domains have valid start and stop, it's not possible to contiguous
// predication.
IterDomain* IterDomain::merge(IterDomain* outer, IterDomain* inner) {
  TORCH_CHECK(
      outer->isReduction() == inner->isReduction(),
      "Merging IterDomains requires that their iteration types match. ",
      "Outer: ",
      outer->toString(),
      ", Inner: ",
      inner->toString());
  TORCH_CHECK(
      (outer->isGather() && inner->isGather()) ||
          (!outer->isGather() && !inner->isGather()),
      "Merging gather and non-gather domains is not supported.");

  TORCH_CHECK(
      !outer->isStride() && !inner->isStride(),
      "No support for merging stride domains");

  Val* merged_id_size = mul(outer->extent(), inner->extent());

  IterType itype = outer->getIterType();

  if (outer->isBroadcast() && inner->isBroadcast()) {
    itype = IterType::Broadcast;
  }

  if ((outer->isBroadcast() || inner->isBroadcast()) &&
      (outer->getIterType() == IterType::Iteration ||
       inner->getIterType() == IterType::Iteration)) {
    itype = IterType::Iteration;
  }

  if ((outer->isBroadcast() || inner->isBroadcast()) &&
      (outer->getIterType() == IterType::GatherScatter ||
       inner->getIterType() == IterType::GatherScatter)) {
    itype = IterType::GatherScatter;
  }

  Val* expanded_extent = nullptr;
  if (outer->hasExpandedExtent() || inner->hasExpandedExtent()) {
    if (outer->hasExpandedExtent() && inner->hasExpandedExtent()) {
      expanded_extent = mul(outer->expandedExtent(), inner->expandedExtent());
    } else if (outer->hasExpandedExtent() && !inner->hasExpandedExtent()) {
      if (inner->isBroadcast()) {
        expanded_extent = outer->expandedExtent();
      } else {
        expanded_extent = mul(outer->expandedExtent(), inner->extent());
      }
    } else if (outer->hasExpandedExtent() && inner->hasExpandedExtent()) {
      if (outer->isBroadcast()) {
        expanded_extent = inner->expandedExtent();
      } else {
        expanded_extent = mul(outer->extent(), inner->expandedExtent());
      }
    }
  }

  IterDomain* merged_id =
      IterDomainBuilder(
          outer->container()->zeroVal(), merged_id_size->as<Int>())
          .parallel_type(outer->getParallelType())
          .expanded_extent(expanded_extent)
          .iter_type(itype)
          .build();

  IrBuilder::create<Merge>(outer->container(), merged_id, outer, inner);

  return merged_id;
}

// Both outer and inner domains do not inherit start and stop
// values as they can't be split. The access range is enforced by
// predicates.
std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor,
    bool inner_split,
    Val* start_offset,
    Val* stop_offset) {
  TORCH_CHECK(
      factor->isIntegralScalar(), "Cannot split by non-integer value ", factor);

  if (factor->getValType() == ValType::Scalar) {
    TORCH_CHECK(
        factor->isConstScalar() ||
            (FusionGuard::getCurFusion() == factor->fusion() &&
             factor->isFusionInput()),
        factor,
        " is not a constant nor an input. It must be one or the other to be used in a split.",
        " If you want a symbolic split based on a thread dimension please use IterDomain::split(IterDomain*, ParallelType);");
  } else if (factor->getValType() == ValType::NamedScalar) {
    TORCH_CHECK(
        factor->as<NamedScalar>()->getParallelDim() != c10::nullopt,
        "Splitting a dimension by a named scalar is only supported on block or grid dimensions but received ",
        factor);
  }

  // outer loop size
  Val* remainder =
      ceilDiv(Split::extent(in->extent(), start_offset, stop_offset), factor);
  Val* expanded_remainder = nullptr;
  if (in->hasExpandedExtent()) {
    expanded_remainder = ceilDiv(
        Split::extent(in->expandedExtent(), start_offset, stop_offset), factor);
  }

  if ((start_offset != nullptr && !start_offset->isZeroInt()) ||
      (stop_offset != nullptr && !stop_offset->isZeroInt())) {
    TORCH_INTERNAL_ASSERT(
        in->definition() == nullptr,
        "Partial split is only allowed with root domains");
  }
  // outer loop IterDomain
  IterDomain* ido =
      IterDomainBuilder(
          in->container()->zeroVal(),
          inner_split ? remainder->as<Int>() : factor)
          .expanded_extent(
              in->hasExpandedExtent() && inner_split ? expanded_remainder
                                                     : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(in->getIterType())
          .build();

  // inner loop IterDomain
  IterDomain* idi =
      IterDomainBuilder(
          in->container()->zeroVal(),
          inner_split ? factor : remainder->as<Int>())
          .expanded_extent(
              in->hasExpandedExtent() && !inner_split ? expanded_remainder
                                                      : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(in->getIterType())
          .build();

  IrBuilder::create<Split>(
      in->container(),
      ido,
      idi,
      in,
      factor,
      inner_split,
      start_offset,
      stop_offset);
  return {ido, idi};
}

std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor,
    bool inner_split,
    bool trim_out_of_bounds) {
  auto start_offset = trim_out_of_bounds ? in->start() : nullptr;
  auto stop_offset = trim_out_of_bounds ? in->stopOffset() : nullptr;
  return IterDomain::split(in, factor, inner_split, start_offset, stop_offset);
}

std::pair<IterDomain*, IterDomain*> IterDomain::stridedSplit(int factor) {
  // Use partial split so that only valid values are retained
  auto split_out = IterDomain::split(
      this, IrBuilder::create<Int>(container(), factor), true, true);

  split_out.second->iter_type_ = IterType::Stride;
  split_out.first->is_rfactor_domain_ = true;
  split_out.second->is_rfactor_domain_ = true;
  return split_out;
}

std::pair<IterDomain*, IterDomain*> IterDomain::swizzle(
    Swizzle2DType swizzle_type,
    IterDomain* in_x,
    IterDomain* in_y,
    SwizzleMode swizzle_mode) {
  TORCH_CHECK(
      !in_x->extent()->isZeroInt() && !in_y->extent()->isZeroInt(),
      "Invalid swizzling of a empty dimension.");

  // TODO: reduction check on swizzle:
  TORCH_CHECK(
      !in_x->isReduction() && !in_y->isReduction(),
      "swizzled reduction not yet supported");

  for (auto input : InputsOf::outputs(in_x->fusion(), {in_x, in_y})) {
    TORCH_CHECK(
        !input->as<IterDomain>()->isBroadcast(),
        "swizzling broadcast axes not yet supported");
  }

  // TODO: gather and shift check on swizzle
  TORCH_INTERNAL_ASSERT(
      !in_x->isGather() && !in_y->isGather(),
      "Swizzled gather not yet supported");

  IterDomain* out_x = IterDomainBuilder(in_x).build();

  IterDomain* out_y = IterDomainBuilder(in_y).build();

  IrBuilder::create<Swizzle2D>(
      in_x->container(), out_x, out_y, in_x, in_y, swizzle_type, swizzle_mode);

  return std::make_pair(out_x, out_y);
}

IterDomain* IterDomain::resize(
    IterDomain* in,
    Val* left_expansion,
    Val* right_expansion,
    bool mark_as_rfactor,
    std::optional<IterType> iter_type_opt) {
  TORCH_CHECK(
      left_expansion->isIntegralScalar(),
      "Expansion factor must be an integer scalar: ",
      left_expansion->toString());
  TORCH_CHECK(
      right_expansion->isIntegralScalar(),
      "Expansion factor must be an integer scalar: ",
      right_expansion->toString());

  TORCH_CHECK(
      in->getIterType() == IterType::Iteration ||
          in->getIterType() == IterType::Broadcast ||
          in->getIterType() == IterType::Symbolic || "Not a valid IterType: ",
      in->getIterType());

  TORCH_CHECK(
      in->start()->isZeroInt(),
      "Non-zero start not supported: ",
      in->toString());
  TORCH_CHECK(
      in->stopOffset()->isZeroInt(),
      "Non-zero stop offset not considered: ",
      in->toString());

  // The overall extent is (in->extent() + left_expansion +
  // right_expansion). This can be simplified for a slice op as
  // the right expansion should look like (slice_end_offset -
  // in->extent()), so the overall extent is left_expansion + slice_end_offset.
  Val* resized_id_size = nullptr;
  if (right_expansion->definition() != nullptr &&
      right_expansion->definition()->isA<BinaryOp>() &&
      right_expansion->definition()->as<BinaryOp>()->getBinaryOpType() ==
          BinaryOpType::Sub &&
      right_expansion->definition()->as<BinaryOp>()->rhs() == in->extent()) {
    resized_id_size = SimplifyingIrBuilder::addExpr(
        left_expansion, right_expansion->definition()->as<BinaryOp>()->lhs());
  } else {
    resized_id_size = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::addExpr(in->extent(), left_expansion),
        right_expansion);
  }

  // If output IterType is provided, use it. Otherwise, if we can prove the
  // resized extent is 1, set to Broadcast, if we can prove it is >1 set to
  // Iteration, and otherwise fall back to Symbolic.
  IterType iter_type = IterType::Symbolic;
  if (iter_type_opt.has_value()) {
    iter_type = iter_type_opt.value();
  } else if (left_expansion->isConstInt() && right_expansion->isConstInt()) {
    if (resized_id_size->isConstInt()) {
      // Means input extent is also known
      auto out_extent = resized_id_size->evaluateInt();
      iter_type = out_extent == 1 ? IterType::Broadcast : IterType::Iteration;
    } else if (
        left_expansion->evaluateInt() + right_expansion->evaluateInt() > 1) {
      // Input extent is non-negative, so we know out_extent > 1
      iter_type = IterType::Iteration;
    }
  }

  auto resized_id =
      IterDomainBuilder(in->container()->zeroVal(), resized_id_size->as<Int>())
          .is_rfactor_domain(mark_as_rfactor)
          .iter_type(iter_type)
          .build();

  IrBuilder::create<Resize>(
      in->container(), resized_id, in, left_expansion, right_expansion);

  return resized_id;
}

// TODO: We should change parallelize interface to be on tensorview or at least
// vectorize should be done on tensorview. This would let us check that we don't
// vectorize to the left of the computeAt domain, and could allow us to do some
// simple validation of vectorize as it's inputs are right most and contiguous.
void IterDomain::parallelize(ParallelType t) {
  if (parallel_type_ == t) {
    // No op, don't do any more checks, it was already set to this value.
    return;
  }

  // assert check that we only parallelize a leaf domain.
  // leaf domains are domains that are not used by any other domains.
  if (t != ParallelType::Serial) {
    TORCH_CHECK(
        uses().empty(),
        "Only allowed to parallelize a leaf domain.",
        " Domain: ",
        toString(),
        ", Parallel type: ",
        t,
        definition() != nullptr ? ", Definition: " + definition()->toString()
                                : "");
  }

  if (t == ParallelType::Unroll || isParallelTypeVectorize(t) ||
      t == ParallelType::Group) {
    TORCH_CHECK(
        start()->isZeroInt() && extent()->isConstScalar(),
        "Vectorization, unrolling, unswitching and grouping are only supported with start = 0 and extent as a const int, but got ",
        "a start of ",
        start(),
        " and extent ",
        extent(),
        " .");
  }

  if (t == ParallelType::Group) {
    TORCH_CHECK(
        getIterType() == IterType::Iteration,
        "Grouping IterDomain of non Iteration type is not allowed. ",
        getIterType());
  }

  if (isMmaSwizzled()) {
    // Mma swizzled axes represent data representation within a warp
    //  so only allow updates that keep the parallelization within
    //  a warp.
    // Note && TODO: this check is actually used to allow indexing path
    //  to make copies of the iterdomains. We might eventually just want
    //  to lock these parallel types and not allowing any changes once
    //  they are swizzled.
    TORCH_CHECK(
        t == ParallelType::Vectorize || t == ParallelType::TIDx ||
            t == ParallelType::Serial || t == ParallelType::Mma,
        "Parallel type other than serial, tidx, vectorize not allowed for mma swizzled ids");
  }

  parallel_type_ = t;
}

bool IterDomain::maybePartial() const {
  return !start()->isZeroInt() || !stopOffset()->isZeroInt();
}

Val* IterDomain::stopOffset() const {
  return stop_offset_;
}

Val* IterDomain::stop() const {
  if (stopOffset()->isZeroInt()) {
    return extent();
  }

  return sub(extent(), stopOffset());
}

namespace {

void validateContiguity(
    const std::vector<IterDomain*>& allocation_domain,
    const std::vector<std::optional<bool>>& contiguity) {
  TORCH_CHECK(
      contiguity.size() == allocation_domain.size(),
      "Invalid contiguity information provided, incorrect size. Received vector of size ",
      contiguity.size(),
      " but needed one of size ",
      allocation_domain.size());
  for (auto i : c10::irange(contiguity.size())) {
    bool expect_null =
        (allocation_domain.at(i)->isBroadcast() ||
         allocation_domain.at(i)->isReduction());
    TORCH_CHECK(
        expect_null != contiguity.at(i).has_value(),
        "The contiguity of a broadcast/reduction dimension must be None. "
        "The contiguity of a non-broadcast/reduction dimension must be true/false");
  }
}

} // namespace

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      leaf_domain_(root_domain_),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)),
      has_reduction_(false) {
  validateContiguity(maybeAllocation(), contiguity_);

  // Just due to clang-tidy, correct value set in resetDomains
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> leaf_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      leaf_domain_(std::move(leaf_domain)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  if (!root_domain_.empty()) {
    TORCH_CHECK(!leaf_domain_.empty(), "Root domain is not empty but leaf is");
    ir_utils::validateDomainEquivalence(root_domain_, leaf_domain_);
  }

  // Just due to clang-tidy, correct value set in resetDomains
  has_reduction_ = false;
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> rfactor_domain,
    std::vector<IterDomain*> leaf_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      rfactor_domain_(std::move(rfactor_domain)),
      leaf_domain_(std::move(leaf_domain)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  if (!root_domain_.empty()) {
    TORCH_CHECK(!leaf_domain_.empty(), "Root domain is not empty but leaf is");
    ir_utils::validateDomainEquivalence(root_domain_, leaf_domain_);
    if (!rfactor_domain_.empty()) {
      ir_utils::validateDomainEquivalence(root_domain_, rfactor_domain_);
      ir_utils::validateDomainEquivalence(rfactor_domain_, leaf_domain_);
    }
  }

  // Just due to clang-tidy, correct value set in resetDomains
  has_reduction_ = false;
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> rfactor_domain,
    std::vector<IterDomain*> allocation_domain,
    std::vector<IterDomain*> leaf_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      rfactor_domain_(std::move(rfactor_domain)),
      allocation_domain_(std::move(allocation_domain)),
      leaf_domain_(std::move(leaf_domain)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  if (!root_domain_.empty()) {
    TORCH_CHECK(!leaf_domain_.empty(), "Root domain is not empty but leaf is");
    ir_utils::validateDomainEquivalence(root_domain_, leaf_domain_);
    if (!rfactor_domain_.empty()) {
      ir_utils::validateDomainEquivalence(root_domain_, rfactor_domain_);
      ir_utils::validateDomainEquivalence(rfactor_domain_, leaf_domain_);
    }
    if (!allocation_domain_.empty()) {
      ir_utils::validateDomainEquivalence(root_domain_, allocation_domain_);
      ir_utils::validateDomainEquivalence(allocation_domain_, leaf_domain_);
    }
  }

  // Just due to clang-tidy, correct value set in resetDomains
  has_reduction_ = false;
  resetDomains();
}

TensorDomain::TensorDomain(const TensorDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      root_domain_(ir_cloner->clone(src->root_domain_)),
      rfactor_domain_(ir_cloner->clone(src->rfactor_domain_)),
      allocation_domain_(ir_cloner->clone(src->allocation_domain_)),
      leaf_domain_(ir_cloner->clone(src->leaf_domain_)),
      no_bcast_domain_(ir_cloner->clone(src->no_bcast_domain_)),
      no_reduction_domain_(ir_cloner->clone(src->no_reduction_domain_)),
      contiguity_(src->contiguity()),
      has_reduction_(src->has_reduction_) {}

NVFUSER_DEFINE_CLONE(TensorDomain)

bool TensorDomain::hasBlockBroadcast() const {
  return std::any_of(
      leaf_domain_.begin(), leaf_domain_.end(), [](IterDomain* id) {
        return id->isBroadcast() && id->isThreadDim();
      });
}

bool TensorDomain::hasGridBroadcast() const {
  return std::any_of(
      leaf_domain_.begin(), leaf_domain_.end(), [](IterDomain* id) {
        return id->isBroadcast() && id->isBlockDim();
      });
}

bool TensorDomain::operator==(const TensorDomain& other) const {
  // Checks equality of each class field. Should not be necessary to
  // check no_bcast_domain_ and no_reduction_domain_ as they are just
  // derived from domain_.
  return root_domain_ == other.root_domain_ &&
      leaf_domain_ == other.leaf_domain_ &&
      rfactor_domain_ == other.rfactor_domain_ &&
      allocation_domain_ == other.allocation_domain_ &&
      contiguity_ == other.contiguity_;
}

bool TensorDomain::sameAs(const Statement* const other) const {
  if (this == other) {
    return true;
  }

  if (!other->isA<TensorDomain>()) {
    return false;
  }

  const TensorDomain* other_td = other->as<TensorDomain>();

  if (nDims() != other_td->nDims()) {
    return false;
  }
  if (root().size() != other_td->root().size()) {
    return false;
  }
  if (rfactor().size() != other_td->rfactor().size()) {
    return false;
  }
  if (allocation().size() != other_td->allocation().size()) {
    return false;
  }

  for (const auto i : c10::irange(nDims())) {
    if (!(axis((int)i)->sameAs(other_td->axis((int)i)))) {
      return false;
    }
  }

  for (const auto i : c10::irange(root().size())) {
    if (!(root()[i]->sameAs(other_td->root()[i]))) {
      return false;
    }
  }

  for (const auto i : c10::irange(rfactor().size())) {
    if (!(rfactor()[i]->sameAs(other_td->rfactor()[i]))) {
      return false;
    }
  }

  for (const auto i : c10::irange(allocation().size())) {
    if (!(allocation()[i]->sameAs(other_td->allocation()[i]))) {
      return false;
    }
  }

  for (const auto i : c10::irange(leaf().size())) {
    if (!(leaf()[i]->sameAs(other_td->leaf()[i]))) {
      return false;
    }
  }

  return true;
}

bool TensorDomain::sameAs(
    const std::vector<IterDomain*>& lhs,
    const std::vector<IterDomain*>& rhs) {
  if (lhs.size() != rhs.size())
    return false;
  size_t i = 0;
  for (auto td_lhs : lhs) {
    if (!td_lhs->sameAs(rhs[i++]))
      return false;
  }
  return true;
}

std::string TensorDomain::toString(int indent_size) const {
  std::stringstream ss;
  if (nDims() == 0) {
    ss << "[ 0 ]";
    return ss.str();
  }
  ss << "[ " << toDelimitedString(leaf()) << " ]";
  return ss.str();
}

std::string TensorDomain::toInlineString(int indent_size) const {
  return toString(indent_size);
}

void TensorDomain::setContiguity(
    const std::vector<std::optional<bool>>& contig) {
  TORCH_INTERNAL_ASSERT(
      maybeAllocation().size() == contig.size(),
      "Invalid size of contiguity vector");
  for (auto i : c10::irange(contig.size())) {
    TORCH_CHECK(
        maybeAllocation().at(i)->isBroadcast() != contig.at(i).has_value(),
        "The contiguity of a broadcast dimension must be None. "
        "The contiguity of a non-broadcast dimension must be true/false");
  }

  contiguity_ = contig;
}

bool TensorDomain::hasBlockReduction() const {
  return std::any_of(
      leaf_domain_.begin(), leaf_domain_.end(), [](IterDomain* id) {
        return id->isReduction() && id->isThreadDim();
      });
}

bool TensorDomain::hasGridReduction() const {
  return std::any_of(
      leaf_domain_.begin(), leaf_domain_.end(), [](IterDomain* id) {
        return id->isReduction() && id->isBlockDim();
      });
}

bool TensorDomain::hasSymbolicAxis() const {
  // If there's any Symbolic axis, there must be one at the root or
  // rfactor domain.
  return std::any_of(
             root().begin(),
             root().end(),
             [](auto id) { return id->getIterType() == IterType::Symbolic; }) ||
      (hasRFactor() &&
       std::any_of(maybeRFactor().begin(), maybeRFactor().end(), [](auto id) {
         return id->getIterType() == IterType::Symbolic;
       }));
}

bool TensorDomain::hasViewLikeRFactor() const {
  if (!hasRFactor()) {
    // Can't have view like rfactor if there is no rfactor domain
    return false;
  }

  // If there's an rfactor domain and no rfactor product is a reduction, this is
  // a view like rfactor
  return std::none_of(
      maybeRFactor().begin(), maybeRFactor().end(), [](IterDomain* id) {
        return (id->isReduction() || id->isStride()) && id->isRFactorProduct();
      });
}

bool TensorDomain::hasVectorize() const {
  return std::any_of(
      leaf_domain_.begin(), leaf_domain_.end(), [](IterDomain* id) {
        return id->getParallelType() == ParallelType::Vectorize ||
            id->getParallelType() == ParallelType::MisalignedVectorize;
      });
}

c10::optional<unsigned int> TensorDomain::getReductionAxis() const {
  auto it = std::find_if(
      leaf_domain_.begin(), leaf_domain_.end(), [](const auto& id) {
        return id->isReduction();
      });
  if (it == leaf_domain_.end()) {
    return c10::optional<unsigned int>();
  } else {
    return c10::optional<unsigned int>(std::distance(leaf_domain_.begin(), it));
  }
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to access an axis in a 0-dim domain");
  if (i < 0) {
    i += (int)nDims();
  }
  TORCH_CHECK(
      i >= 0 && (unsigned int)i < nDims(),
      "Tried to access axis ",
      i,
      " in domain ",
      this);
  return leaf_domain_[i];
}

int64_t TensorDomain::posOf(IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to find an axis in a 0-dim domain");
  int64_t i = 0;
  while (i < (int64_t)leaf_domain_.size()) {
    if (leaf_domain_[i] == id)
      return i;
    i++;
  }
  TORCH_CHECK(false, "Provided id is not part of this domain.");
}

int64_t TensorDomain::rootPosOf(IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(
      !root_domain_.empty(), "Tried to find an axis in a 0-dim root domain");
  auto it = std::find(root_domain_.begin(), root_domain_.end(), id);
  TORCH_INTERNAL_ASSERT(
      it != root_domain_.end(), "Provided id is not part of root domain.");
  return std::distance(root_domain_.begin(), it);
}

void TensorDomain::split(
    int axis_,
    Val* factor,
    bool inner_split,
    bool trim_out_of_bounds) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do split on a 0-dim domain");
  if (axis_ < 0)
    axis_ += (int)nDims();

  TORCH_INTERNAL_ASSERT(
      axis_ >= 0 && (unsigned int)axis_ < nDims(),
      "Tried to split on axis outside TensorDomain's range.");

  IterDomain* id = axis(axis_);

  // partial split is only allowed with root domains
  if (trim_out_of_bounds) {
    TORCH_INTERNAL_ASSERT(
        std::find(root().begin(), root().end(), id) != root().end(),
        "Partial split is only allowed with root domains");
  }

  TORCH_INTERNAL_ASSERT(
      !id->isMmaSwizzled(),
      "Further transformation on warp mapped id's not allowed.");

  auto split_ids =
      IterDomain::split(id, factor, inner_split, trim_out_of_bounds);
  leaf_domain_.erase(leaf_domain_.begin() + axis_);
  leaf_domain_.insert(leaf_domain_.begin() + axis_, split_ids.second);
  leaf_domain_.insert(leaf_domain_.begin() + axis_, split_ids.first);
  resetDomains();
}

// Merge "axis_o" and "axis_i" into 1 dimension
void TensorDomain::merge(int axis_o, int axis_i) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim domain");
  if (axis_o < 0)
    axis_o += (int)nDims();

  if (axis_i < 0)
    axis_i += (int)nDims();

  TORCH_CHECK(
      axis_o >= 0 && (unsigned int)axis_o < nDims() && axis_i >= 0 &&
          (unsigned int)axis_i < nDims(),
      "Invalid merge detected, either one or both axes are outside of TensorView's range.");

  TORCH_CHECK(
      axis_o != axis_i,
      "Invalid merge detected, axes provided are the same axis.");

  if (axis_o > axis_i) {
    auto tmp = axis_i;
    axis_i = axis_o;
    axis_o = tmp;
  }

  IterDomain* first = axis(axis_o);
  IterDomain* second = axis(axis_i);

  TORCH_INTERNAL_ASSERT(
      !first->isMmaSwizzled() && !second->isMmaSwizzled(),
      "Further transformation on warp mapped id's not allowed.");

  IterDomain* merged_id = IterDomain::merge(first, second);

  leaf_domain_.erase(leaf_domain_.begin() + axis_i);
  leaf_domain_.erase(leaf_domain_.begin() + axis_o);
  leaf_domain_.insert(leaf_domain_.begin() + axis_o, merged_id);
  resetDomains();
}

// Reorder axes according to map[old_pos] = new_pos
void TensorDomain::reorder(const std::unordered_map<int, int>& old2new_) {
  TORCH_INTERNAL_ASSERT(
      nDims() != 0 || old2new_.empty(), "Tried to reorder a 0-dim domain");
  leaf_domain_ = orderedAs(leaf_domain_, old2new_);
  resetDomains();
}

std::vector<IterDomain*> TensorDomain::orderedAs(
    const std::vector<IterDomain*>& dom,
    const std::unordered_map<int, int>& old2new_) {
  TORCH_INTERNAL_ASSERT(
      !dom.empty() || old2new_.empty(), "Tried to reorder a 0-dim domain");

  // Eventhough these checks are already in TensorView, we want to redo them as
  // we can enter this function from other places, not through TensorView

  auto new2old = ir_utils::normalizeOld2New(old2new_, dom.size());

  std::vector<IterDomain*> reordered_domain;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_domain),
      [dom](int i) -> IterDomain* { return dom[i]; });

  return reordered_domain;
}

void TensorDomain::swizzle(
    Swizzle2DType swizzle_type,
    int x,
    int y,
    SwizzleMode swizzle_mode) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim domain");

  TORCH_CHECK(
      x >= 0 && (unsigned int)x < nDims(),
      "Invalid swizzle detected, either one or both axes are outside of TensorView's range.");

  TORCH_CHECK(
      y >= 0 && (unsigned int)y < nDims(),
      "Invalid swizzle detected, either one or both axes are outside of TensorView's range.");

  IterDomain* axis_x = axis(x);
  IterDomain* axis_y = axis(y);

  IterDomain* axis_out_x = nullptr;
  IterDomain* axis_out_y = nullptr;

  std::tie(axis_out_x, axis_out_y) =
      IterDomain::swizzle(swizzle_type, axis_x, axis_y, swizzle_mode);

  leaf_domain_.erase(leaf_domain_.begin() + x);
  leaf_domain_.insert(leaf_domain_.begin() + x, axis_out_x);

  leaf_domain_.erase(leaf_domain_.begin() + y);
  leaf_domain_.insert(leaf_domain_.begin() + y, axis_out_y);

  resetDomains();
}

std::vector<IterDomain*> TensorDomain::noReductions(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> noReductionDomain;
  std::copy_if(
      td.begin(),
      td.end(),
      std::back_inserter(noReductionDomain),
      [](IterDomain* id) { return !id->isReduction() && !id->isStride(); });
  return noReductionDomain;
}

std::vector<IterDomain*> TensorDomain::noBroadcasts(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> noBroadcastDomain;
  std::copy_if(
      td.begin(),
      td.end(),
      std::back_inserter(noBroadcastDomain),
      [](IterDomain* id) { return !id->isBroadcast(); });
  return noBroadcastDomain;
}

std::vector<std::optional<bool>> TensorDomain::getContiguityFilledWith(
    const std::vector<IterDomain*>& rfactor_domain,
    bool fill_value) {
  std::vector<std::optional<bool>> contiguity;
  contiguity.reserve(rfactor_domain.size());
  for (auto id : rfactor_domain) {
    if (id->isBroadcast() || id->isReduction()) {
      contiguity.emplace_back(std::nullopt);
    } else {
      contiguity.emplace_back(fill_value);
    }
  }
  return contiguity;
}

bool TensorDomain::hasBroadcast(const std::vector<IterDomain*>& td) {
  for (auto id : td) {
    if (id->isBroadcast()) {
      return true;
    }
  }
  return false;
}

bool TensorDomain::hasReduction(const std::vector<IterDomain*>& td) {
  for (auto id : td) {
    if (id->isReduction()) {
      return true;
    }
  }
  return false;
}

TensorDomain* TensorDomain::view(const AnalyzeViewResult& view_analysis) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to view transform a 0-dim domain");
  return transformView(this, view_analysis);
}

TensorDomain* TensorDomain::flatten(int64_t start_dim, int64_t end_dim) {
  auto inp_domain = noReductions(maybeRFactor());

  if (start_dim < 0) {
    start_dim += (int64_t)inp_domain.size();
  }
  if (end_dim < 0) {
    end_dim += (int64_t)inp_domain.size();
  }
  TORCH_CHECK(
      start_dim >= 0 && start_dim < int64_t(inp_domain.size()),
      "Invalid start_dim ",
      start_dim);
  TORCH_CHECK(
      end_dim >= 0 && end_dim < int64_t(inp_domain.size()),
      "Invalid end_dim ",
      end_dim);
  TORCH_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  std::vector<IterDomain*> new_root_domain;
  new_root_domain.reserve(inp_domain.size());
  for (auto i : c10::irange(inp_domain.size())) {
    bool is_rfactor_dim = i >= size_t(start_dim) && i <= size_t(end_dim);
    auto inp_id = inp_domain[i];
    auto out_id = IterDomainBuilder(inp_id)
                      .is_rfactor_domain(is_rfactor_dim)
                      .extent(
                          (is_rfactor_dim && inp_id->hasExpandedExtent())
                              ? inp_id->expandedExtent()
                              : inp_id->extent())
                      .iter_type(
                          (is_rfactor_dim && inp_id->isBroadcast())
                              ? IterType::Iteration
                              : inp_id->getIterType())
                      .build();
    new_root_domain.push_back(out_id);
  }

  std::vector<IterDomain*> rfactor_domain;
  rfactor_domain.reserve(new_root_domain.size() - (end_dim - start_dim));
  for (auto i : c10::irange(start_dim)) {
    rfactor_domain.push_back(new_root_domain[i]);
  }

  IterDomain* merged_id = new_root_domain[start_dim];
  for (auto i : c10::irange(start_dim + 1, end_dim + 1)) {
    IterDomain* new_merged_id =
        IterDomainBuilder(
            merged_id->container()->zeroVal(),
            mul(merged_id->extent(), new_root_domain[i]->extent()))
            .is_rfactor_domain(true)
            .build();
    IrBuilder::create<Merge>(new_merged_id, merged_id, new_root_domain[i]);
    merged_id = new_merged_id;
  }
  rfactor_domain.push_back(merged_id);

  for (auto i : c10::irange(end_dim + 1, inp_domain.size())) {
    rfactor_domain.push_back(new_root_domain[i]);
  }

  return IrBuilder::create<TensorDomain>(
      new_root_domain,
      rfactor_domain,
      rfactor_domain,
      TensorDomain::getContiguityFilledWith(rfactor_domain, true));
}

// TODO: Rfactor a Welford

// pair is in order where second is the consumer of first
std::pair<TensorDomain*, TensorDomain*> TensorDomain::rFactor(
    const std::vector<int>& axes_) {
  return TransformRFactor::runReplay(this, axes_);
}

void TensorDomain::setAllocationDomain(
    std::vector<IterDomain*> new_allocation_domain,
    std::vector<std::optional<bool>> new_contiguity) {
  validateContiguity(new_allocation_domain, new_contiguity);

  ir_utils::validateDomainEquivalence(root_domain_, new_allocation_domain);
  ir_utils::validateDomainEquivalence(new_allocation_domain, leaf_domain_);

  allocation_domain_ = std::move(new_allocation_domain);
  contiguity_ = std::move(new_contiguity);
}

Split::Split(
    IrBuilderPasskey passkey,
    IterDomain* outer,
    IterDomain* inner,
    IterDomain* in,
    Val* factor,
    bool inner_split,
    Val* start_offset,
    Val* stop_offset)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      factor->isIntegralScalar(),
      "Attempted to create a Split node with a non-integer factor.");
  if (start_offset == nullptr) {
    start_offset = passkey.ir_container_->zeroVal();
  }
  if (stop_offset == nullptr) {
    stop_offset = passkey.ir_container_->zeroVal();
  }
  addOutput(outer);
  addOutput(inner);
  addInput(in);
  // TODO add factor as an input, need to check Split::Split during validation
  // and need to check BestEffortReplay::findFirstMismatchedID addInput(factor);
  addAttribute(factor);
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, inner_split));
  addAttribute(start_offset);
  addAttribute(stop_offset);
}

std::string Split::toString(int indent_size) const {
  std::stringstream ss;
  ss << (innerSplit() ? "Split: " : "Outer split: ");
  ss << in()->toString();
  ss << " by factor " << factor()->toString() << " -> ";
  ss << outer()->toString();
  ss << ", ";
  ss << inner()->toString();
  if (startOffset()) {
    ss << ", start offset: ";
    ss << startOffset()->toString();
  }
  if (stopOffset()) {
    ss << ", stop offset: ";
    ss << stopOffset()->toString();
  }
  ss << "\n";
  return ss.str();
}

std::string Split::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Split can not be printed inline");
}

Val* Split::extent(Val* in_extent, Val* start_offset, Val* stop_offset) {
  TORCH_INTERNAL_ASSERT(in_extent != nullptr);

  if (start_offset != nullptr && !start_offset->isZeroInt()) {
    in_extent = sub(in_extent, start_offset);
  }

  if (stop_offset != nullptr && !stop_offset->isZeroInt()) {
    in_extent = sub(in_extent, stop_offset);
  }

  return in_extent;
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Split)

Merge::Merge(
    IrBuilderPasskey passkey,
    IterDomain* out,
    IterDomain* outer,
    IterDomain* inner)
    : Expr(passkey) {
  addOutput(out);
  addInput(outer);
  addInput(inner);
}

std::string Merge::toString(int indent_size) const {
  std::stringstream ss;
  ss << "Merge: ";
  ss << outer()->toString();
  ss << " and ";
  ss << inner()->toString();
  ss << " -> ";
  ss << out()->toString();
  ss << "\n";
  return ss.str();
}

std::string Merge::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Merge)

Swizzle2D::Swizzle2D(
    IrBuilderPasskey passkey,
    IterDomain* out_x,
    IterDomain* out_y,
    IterDomain* in_x,
    IterDomain* in_y,
    Swizzle2DType swizzle_type,
    SwizzleMode swizzle_mode)
    : Expr(passkey) {
  addOutput(out_x);
  addOutput(out_y);
  addInput(in_x);
  addInput(in_y);
  addAttribute(IrBuilder::create<Attribute<Swizzle2DType>>(
      passkey.ir_container_, swizzle_type));
  addAttribute(IrBuilder::create<Attribute<SwizzleMode>>(
      passkey.ir_container_, swizzle_mode));
}

std::string Swizzle2D::toString(int indent_size) const {
  std::stringstream ss;
  ss << swizzleType() << "(2D): ";
  ss << inX()->toString();
  ss << " , ";
  ss << inY()->toString();
  ss << " -> ";
  ss << outX()->toString();
  ss << " , ";
  ss << outY()->toString();
  ss << "\n";
  return ss.str();
}

std::string Swizzle2D::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Swizzle2D)

Resize::Resize(
    IrBuilderPasskey passkey,
    IterDomain* out,
    IterDomain* in,
    Val* left,
    Val* right)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addAttribute(left);
  addAttribute(right);
}

std::string Resize::toString(int indent_size) const {
  std::stringstream ss;
  ss << "Resize: ";
  ss << in()->toString();
  ss << " by " << leftExpand()->toInlineString() << " and "
     << rightExpand()->toInlineString();
  ss << " -> ";
  ss << out()->toString();
  ss << "\n";
  return ss.str();
}

std::string Resize::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Resize can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Resize)

NamedScalar::NamedScalar(
    IrBuilderPasskey passkey,
    std::string name,
    DataType dtype)
    : Val(passkey, ValType::NamedScalar, dtype), name_(std::move(name)) {}

NamedScalar::NamedScalar(const NamedScalar* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), name_(src->name_) {}

NVFUSER_DEFINE_CLONE(NamedScalar)

bool NamedScalar::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<NamedScalar>()) {
    return false;
  }
  return other->as<NamedScalar>()->name().compare(name()) == 0;
}

bool NamedScalar::isTensorSize() const {
  static const std::regex r(R"(T\d+\.size\[\d+\])");
  return std::regex_match(name(), r);
}

bool NamedScalar::isTensorStride() const {
  static const std::regex r(R"(T\d+\.stride\[\d+\])");
  return std::regex_match(name(), r);
}

NamedScalar* NamedScalar::getParallelDim(ParallelType p_type) {
  TORCH_INTERNAL_ASSERT(
      isParallelTypeThread(p_type),
      "Cannot get parallel dim of non thread type, received: ",
      p_type);
  TORCH_INTERNAL_ASSERT(FusionGuard::getCurFusion() != nullptr);
  std::string parallel_dim = stringifyThreadSize(p_type);
  return IrBuilder::create<NamedScalar>(parallel_dim, DataType::Int);
}

NamedScalar* NamedScalar::getParallelIndex(ParallelType p_type) {
  TORCH_INTERNAL_ASSERT(FusionGuard::getCurFusion() != nullptr);
  std::string parallel_ind = stringifyThread(p_type);
  return IrBuilder::create<NamedScalar>(parallel_ind, DataType::Int);
}

c10::optional<ParallelType> NamedScalar::getParallelDim() const {
  if (stringifyThreadSize(ParallelType::TIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDx);
  } else if (stringifyThreadSize(ParallelType::TIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDy);
  } else if (stringifyThreadSize(ParallelType::TIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDz);
  } else if (stringifyThreadSize(ParallelType::BIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDx);
  } else if (stringifyThreadSize(ParallelType::BIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDy);
  } else if (stringifyThreadSize(ParallelType::BIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDz);
  }
  return c10::nullopt;
}

c10::optional<ParallelType> NamedScalar::getParallelIndex() const {
  if (stringifyThread(ParallelType::TIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDx);
  } else if (stringifyThread(ParallelType::TIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDy);
  } else if (stringifyThread(ParallelType::TIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDz);
  } else if (stringifyThread(ParallelType::BIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDx);
  } else if (stringifyThread(ParallelType::BIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDy);
  } else if (stringifyThread(ParallelType::BIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDz);
  }
  return c10::nullopt;
}

PadOp::PadOp(
    IrBuilderPasskey passkey,
    TensorView* out,
    TensorView* inp,
    const std::vector<Val*>& pad_widths,
    Val* value)
    : Expr(passkey) {
  const auto ndims =
      TensorDomain::noReductions(inp->getMaybeRFactorDomain()).size();
  TORCH_INTERNAL_ASSERT(
      pad_widths.size() % 2 == 0,
      "Invalid size of padding width vector: ",
      pad_widths.size(),
      ". Number of width vals must be even.");
  TORCH_INTERNAL_ASSERT(
      pad_widths.size() == ndims * 2,
      "Invalid size of padding width vector: ",
      pad_widths.size(),
      ". All dimensions, padded or not, must have width vals. Use zero for non non-padded dimensions.");
  addOutput(out);
  addInput(inp);
  addInput(value);
  for (auto width : pad_widths) {
    TORCH_CHECK(width != nullptr, "Padding width must not be nullptr");
    addInput(width);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PadOp)

std::string PadOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size) << "   = pad( " << in()->toString() << ", {"
                          << toDelimitedString(getPadWidths()) << "}"
                          << " )\n";
  return ss.str();
}

std::string PadOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<int> PadOp::getPaddedAxes() const {
  auto num_dims = out()->as<TensorView>()->getRootDomain().size();
  std::vector<int> padded_axes;
  for (const auto i : c10::irange(num_dims)) {
    auto [left_pad, right_pad] = getPadWidths((int)i);
    // Filter out non-padded dimension
    if (left_pad->isZeroInt() && right_pad->isZeroInt()) {
      continue;
    }
    padded_axes.push_back((int)i);
  }
  return padded_axes;
}

std::vector<Val*> PadOp::getPadWidths() const {
  return {getPadWidthInputBegin(), getPadWidthInputEnd()};
}

std::pair<Val*, Val*> PadOp::getPadWidths(int axis) const {
  auto num_dims = (int)out()->as<TensorView>()->getRootDomain().size();

  if (axis < 0) {
    axis += num_dims;
  }

  TORCH_CHECK(axis >= 0 && axis < num_dims, "Invalid axis: ", axis);

  int64_t offset_even = (int64_t)axis * 2;
  int64_t offset_odd = offset_even + 1;
  return std::make_pair(
      (*(getPadWidthInputBegin() + offset_even))->as<Val>(),
      (*(getPadWidthInputBegin() + offset_odd))->as<Val>());
}

SliceOp::SliceOp(
    IrBuilderPasskey passkey,
    TensorView* out,
    TensorView* inp,
    const std::vector<Slice>& ranges)
    : Expr(passkey) {
  const auto ndims =
      TensorDomain::noReductions(inp->getMaybeRFactorDomain()).size();
  TORCH_INTERNAL_ASSERT(
      ndims == ranges.size(),
      "The range vector must have the same number of Slice descriptors. Given: ",
      ranges.size(),
      ", Expected: ",
      ndims);

  addOutput(out);
  addInput(inp);
  for (const auto& range : ranges) {
    TORCH_INTERNAL_ASSERT(range.start != nullptr, "nullptr not allowed");
    TORCH_INTERNAL_ASSERT(range.stop != nullptr, "nullptr not allowed");
    TORCH_INTERNAL_ASSERT(range.step != nullptr, "nullptr not allowed");
    addInput(range.start);
    addInput(range.stop);
    addInput(range.step);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SliceOp)

std::string SliceOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size) << "   = slice( " << in()->toString() << ", {";
  for (const auto& slice : getRanges()) {
    ss << " {"
       << toDelimitedString(std::vector<std::string>{
              slice.start->toString(),
              slice.stop->toString(),
              slice.step->toString()})
       << "}";
  }
  ss << " } )\n";
  return ss.str();
}

std::string SliceOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<Slice> SliceOp::getRanges() const {
  const auto num_range_vals =
      std::distance(getRangeInputBegin(), getRangeInputEnd());
  TORCH_INTERNAL_ASSERT(
      num_range_vals % 3 == 0,
      "Unexpected number of range vals: ",
      num_range_vals);
  auto ndims = num_range_vals / 3;
  std::vector<Slice> ranges(ndims);
  auto range_val_it = getRangeInputBegin();
  for (const auto i : c10::irange(ndims)) {
    ranges.at(i) = Slice{
        .start = *range_val_it,
        .stop = *(range_val_it + 1),
        .step = *(range_val_it + 2)};
    range_val_it += 3;
  }
  return ranges;
}

CatOp::CatOp(
    IrBuilderPasskey passkey,
    Val* out,
    const std::vector<Val*>& inputs,
    int concatenated_dim)
    : Expr(passkey) {
  addOutput(out);
  for (auto inp : inputs) {
    addInput(inp);
  }
  TORCH_INTERNAL_ASSERT(
      concatenated_dim >= 0 &&
          concatenated_dim <
              static_cast<int>(ir_utils::getTv(out)->getRootDomain().size()),
      "Invalid dimension to concatenate: ",
      concatenated_dim);

  addAttribute(IrBuilder::create<Attribute<int>>(
      passkey.ir_container_, concatenated_dim));
}

CatOp::CatOp(
    IrBuilderPasskey passkey,
    Val* out,
    const std::vector<Val*>& inputs,
    int concatenated_dim,
    Val* concatenated_domain_index,
    const std::vector<Bool*>& preds)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "Should only be used for Kernel container.");

  addOutput(out);
  for (auto inp : inputs) {
    addInput(inp);
  }
  addAttribute(IrBuilder::create<Attribute<int>>(
      passkey.ir_container_, concatenated_dim));
  addAttribute(concatenated_domain_index);
  for (auto pred : preds) {
    addAttribute(pred);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CatOp)

std::string CatOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent(ss, indent_size) << "   = cat( ";
  ss << toDelimitedString(inputs());
  ss << ", " << concatenatedDim();
  ss << " )\n";
  return ss.str();
}

std::string CatOp::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

Val* CatOp::getConcatenatedDomainIndex() const {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(),
      "Should only be used for Kernel container.");
  TORCH_INTERNAL_ASSERT(!attributes().empty(), "No attribute found");
  TORCH_INTERNAL_ASSERT(
      attribute(1) != nullptr, "nulllptr attribute is invalid");
  auto idx = attribute(1)->as<Val>();
  return idx;
}

Bool* CatOp::getPred(int input_idx) const {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(),
      "Should only be used for Kernel container.");
  const auto num_input_tensors = static_cast<int>(inputs().size());
  TORCH_INTERNAL_ASSERT(
      input_idx < num_input_tensors, "Invalid input index: ", input_idx);
  const auto attr_idx = input_idx + 2;
  TORCH_INTERNAL_ASSERT(
      attr_idx < static_cast<int>(attributes().size()),
      "Invalid attribute index: ",
      attr_idx,
      ", number of attributes: ",
      attributes().size());
  auto attr = attribute(attr_idx);
  TORCH_INTERNAL_ASSERT(attr != nullptr, "nullptr attribute is invalid");
  TORCH_INTERNAL_ASSERT(
      attr->isA<Bool>(),
      "Attribute must be a Bool val: ",
      attr->toInlineString());
  auto pred = attr->as<Bool>();
  return pred;
}

} // namespace nvfuser
