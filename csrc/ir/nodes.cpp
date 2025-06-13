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
#include <exceptions.h>
#include <host_ir/container.h>
#include <ir/cloner.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <logical_domain_map.h>
#include <multidevice/utils.h>
#include <ops/arith.h>
#include <runtime/allocations.h>
#include <transform_iter.h>
#include <transform_rfactor.h>
#include <transform_view.h>
#include <type.h>

#include <torch/nn/options/embedding.h>

#include <complex>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>

namespace nvfuser {

FullOp::FullOp(IrBuilderPasskey passkey, Val* out, Val* fill_value)
    : Expr(passkey) {
  if (out->isA<TensorView>()) {
    auto tv_logical = out->as<TensorView>()->getLogicalDomain();
    for (auto id : tv_logical) {
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
  for (auto i : arange(inputs().size())) {
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> FullOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  std::vector<int64_t> shape;
  for (auto i : arange(inputs.size() - 1)) {
    shape.push_back(inputs.at(i).as<int64_t>());
  }
  DataType dtype = getFillValue()->getDataType().value();
  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(data_type_to_aten(dtype));
  using namespace PolymorphicValue_functions;
  return {at::full(shape, toScalar(inputs.back()), options)};
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
  addDataAttribute(dim);
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

IterDomain* SelectOp::getIndexedID() const {
  return TensorDomain::noReductions(
             ir_utils::getTvInput(this)->getLogicalDomain())
      .at(dim());
}

std::vector<PolymorphicValue> SelectOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs.at(0).as<at::Tensor>();
  int64_t dimension = dim();
  int64_t index = (int64_t)inputs.at(1);
  return {in.select(dimension, index)};
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
  addDataAttribute(dim);
}

std::string IndexSelectOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = indexSelect( ";
  ss << input(0)->toString() << ", dim = " << dim() << ", "
     << input(1)->toString() << " )\n";
  return ss.str();
}

std::string IndexSelectOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

IterDomain* IndexSelectOp::getIndexedID() const {
  return TensorDomain::noReductions(
             ir_utils::getTvInput(this)->getLogicalDomain())
      .at(dim());
}

IterDomain* IndexSelectOp::getConsumerOfIndexedID() const {
  return ir_utils::getTvOutput(this)->getLogicalDomain().at(dim());
}

std::vector<PolymorphicValue> IndexSelectOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs.at(0).as<at::Tensor>();
  int64_t dimension = dim();
  const auto& indices = inputs.at(1).as<at::Tensor>().squeeze();
  return {at::index_select(in, dimension, indices)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IndexSelectOp)

IndexPutAccumulateOp::IndexPutAccumulateOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* acc,
    Val* index,
    Val* value)
    : Expr(passkey) {
  addInput(acc);
  addInput(index);
  addInput(value);
  addOutput(out);
}

std::string IndexPutAccumulateOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = indexPutAccumulate( ";
  ss << input(0)->toString() << ", " << input(1)->toString() << ", "
     << input(2)->toString() << " )\n";
  return ss.str();
}

std::string IndexPutAccumulateOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

IterDomain* IndexPutAccumulateOp::getIndexingIDOfValue() const {
  return TensorDomain::noReductions(valueTv()->getLogicalDomain()).front();
}

IterDomain* IndexPutAccumulateOp::getIndexingID() const {
  return TensorDomain::noReductions(indexTv()->getLogicalDomain()).front();
}

std::vector<PolymorphicValue> IndexPutAccumulateOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  return {at::index_put(
      /*self=*/inputs.at(0).as<at::Tensor>(),
      // NOTE: aten API doesn't allow the broadcast dimension
      /*indices=*/{inputs.at(1).as<at::Tensor>().squeeze(-1)},
      /*values=*/inputs.at(2).as<at::Tensor>(),
      /*accumulate=*/true)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IndexPutAccumulateOp)

GatherOp::GatherOp(
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
  addDataAttribute(dim);
  addDataAttribute(exact_sizes);
}

std::string GatherOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = "
                          << (exactSizes() ? "takeAlongAxis" : "torchGather")
                          << "( " << input(0)->toString();
  if (exactSizes()) {
    ss << ", " << input(1)->toString() << ", dim = " << dim() << " )\n";
  } else {
    ss << ", dim = " << dim() << ", " << input(1)->toString() << " )\n";
  }
  return ss.str();
}

std::string GatherOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

IterDomain* GatherOp::getIndexedID() const {
  return TensorDomain::noReductions(lookupTv()->getLogicalDomain()).at(dim());
}

IterDomain* GatherOp::getConsumerOfIndexedID() const {
  return ir_utils::getTvOutput(this)->getLogicalDomain().at(dim());
}

std::vector<PolymorphicValue> GatherOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& input = inputs.at(0).as<at::Tensor>();
  const auto& index = inputs.at(1).as<at::Tensor>();
  auto dimension = dim();
  if (exactSizes()) {
    return {at::take_along_dim(input, index, dimension)};
  } else {
    return {at::gather(input, dimension, index)};
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GatherOp)

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
  addDataAttribute(dim);
  addDataAttribute(type);
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
  NVF_CHECK(false, "Scatter op can not be printed inline");
}

IterDomain* ScatterOp::getIndexedID() const {
  return ir_utils::getTvOutput(this)->getLogicalDomain().at(dim());
}

std::vector<PolymorphicValue> ScatterOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& input = inputs.at(0).as<at::Tensor>();
  const auto& index = inputs.at(1).as<at::Tensor>();
  const auto& src = inputs.at(2).as<at::Tensor>();
  auto dimension = dim();
  return {at::scatter(input, dimension, index, src)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ScatterOp)

IotaOp::IotaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* length,
    Val* start,
    Val* step)
    : Expr(passkey) {
  NVF_CHECK(isIntegralType(*length->getDataType()));
  addInput(length);
  NVF_CHECK(start->getDataType() == step->getDataType());
  NVF_CHECK(start->getDataType() == out->getDataType());
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> IotaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(data_type_to_aten(dtype()));
  int64_t length = (int64_t)inputs.at(0);

  if (isIntegralType(dtype())) {
    int64_t start = (int64_t)inputs.at(1);
    int64_t step = (int64_t)inputs.at(2);
    int64_t end = start + step * length;
    return {at::arange(start, end, step, options)};
  } else if (isFloatingPointType(dtype())) {
    double start = (double)inputs.at(1);
    double step = (double)inputs.at(2);
    // Due to rounding error, it can be hard to guarantee the size of
    // the output of arange to be exactly length, so we generate a
    // larger tensor and truncate it to length.
    double end = start + step * ((double)length + 1);
    return {at::arange(start, end, step, options).narrow(0, 0, length)};
  } else {
    NVF_THROW("Unsupported dtype in IotaOp evaluator: ", dtype());
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IotaOp)

EyeOp::EyeOp(IrBuilderPasskey passkey, Val* out, DataType dtype)
    : Expr(passkey) {
  if (out->isA<TensorView>()) {
    addInput(out->as<TensorView>()->getLogicalDomain()[0]->extent());
    if (out->as<TensorView>()->getLogicalDomain()[1] !=
        out->as<TensorView>()->getLogicalDomain()[0]) {
      addInput(out->as<TensorView>()->getLogicalDomain()[1]->extent());
    }
  }
  addOutput(out);
  addDataAttribute(dtype);
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}
std::vector<PolymorphicValue> EyeOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(data_type_to_aten(dtype()));
  int64_t nrows = (int64_t)inputs.at(0);
  if (inputs.size() > 1) {
    int64_t ncols = (int64_t)inputs.at(1);
    return {at::eye(nrows, ncols, options)};
  } else {
    return {at::eye(nrows, options)};
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(EyeOp)

UnaryOp::UnaryOp(IrBuilderPasskey passkey, UnaryOpType type, Val* out, Val* in)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addDataAttribute(type);
}

std::vector<PolymorphicValue> UnaryOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  using namespace PolymorphicValue_functions;

  const auto& in = inputs.at(0);
  if (!in.hasValue()) {
    return {std::monostate{}};
  }

  switch (getUnaryOpType()) {
    case UnaryOpType::Neg:
      return {-in};
    case UnaryOpType::Cast:
      if (in.is<at::Tensor>()) {
        return {PolymorphicValue(
            in.as<at::Tensor>().to(data_type_to_aten(out()->dtype())))};
      } else if (isIntegralType(*out()->getDataType())) {
        return {PolymorphicValue((int64_t)in)};
      } else if (isFloatingPointType(*out()->getDataType())) {
        return {PolymorphicValue((double)in)};
      } else if (out()->getDataType() == DataType::Bool) {
        return {PolymorphicValue((bool)in)};
      } else if (isComplexType(*out()->getDataType())) {
        return {PolymorphicValue((std::complex<double>)in)};
      } else {
        NVF_THROW("dtype not supported in evaluator: ", *out()->getDataType());
      }
    case UnaryOpType::Reciprocal:
      return {1.0 / in};
      break;
    case UnaryOpType::Abs:
      return {abs(in)};
      break;
    case UnaryOpType::LogicalNot:
      return {!in};
      break;
    case UnaryOpType::BitwiseNot:
      return {~in};
      break;
    case UnaryOpType::BitCeil:
      return {static_cast<int64_t>(
          std::bit_ceil(static_cast<uint64_t>(in.as<int64_t>())))};
      break;
    case UnaryOpType::Erf:
      return {erf(in)};
      break;
    case UnaryOpType::ToUnsignedSmemAddr:
      return {(int64_t)(unsigned)in};
      break;
    case UnaryOpType::AdjustPartialLdMatrixAddrInTuring8:
    case UnaryOpType::AdjustPartialLdMatrixAddrInTuring16:
      return {in};
      break;
    case UnaryOpType::Dereference:
      if (*out()->getDataType() == DataType::Float) {
        return {PolymorphicValue((double)*(float*)in)};
      } else {
        NVF_THROW("dtype not supported in evaluator: ", *out()->getDataType());
      }
      break;
    case UnaryOpType::Sigmoid:
      return {in.as<at::Tensor>().sigmoid()};
      break;
    case UnaryOpType::Tanh:
      return {in.as<at::Tensor>().tanh()};
      break;
    case UnaryOpType::Relu:
      return {at::relu(in.as<at::Tensor>())};
      break;
    case UnaryOpType::Gelu:
      return {at::gelu(in.as<at::Tensor>())};
      break;
    case UnaryOpType::Exp:
      return {at::exp(in.as<at::Tensor>())};
      break;
    case UnaryOpType::Sin:
      return {in.as<at::Tensor>().sin()};
      break;
    case UnaryOpType::Signbit:
      return {signbit(in)};
      break;
    case UnaryOpType::Cos:
      return {in.as<at::Tensor>().cos()};
      break;
    case UnaryOpType::BitCast:
      NVF_CHECK(
          dataTypeSizeByte(input(0)->dtype()) ==
              dataTypeSizeByte(out()->dtype()),
          "BitCast only works for types of the same size");
      if (isComplexType(input(0)->dtype()) &&
          std::holds_alternative<ArrayType>(out()->dtype().type)) {
        // view_as_real case.
        auto vec_type = std::get<ArrayType>(out()->dtype().type);
        auto inp_scalar_type = getTypeFromComplexType(input(0)->dtype());
        NVF_CHECK(
            *vec_type.type == inp_scalar_type,
            "Output type must be the same as the scalar type of the complex "
            "input.");
        NVF_CHECK(
            vec_type.size == 2,
            "Expected output to be array of size 2, found array of size ",
            vec_type.size);
        return {in.as<at::Tensor>()};
      } else {
        return {in.as<at::Tensor>().view(data_type_to_aten(out()->dtype()))};
      }
      break;
    case UnaryOpType::Rsqrt:
      return {in.as<at::Tensor>().rsqrt()};
      break;
    case UnaryOpType::Real:
      return {at::real(in.as<at::Tensor>())};
      break;
    case UnaryOpType::Imag:
      return {at::imag(in.as<at::Tensor>())};
      break;
    case UnaryOpType::Tan:
      return {in.as<at::Tensor>().tan()};
      break;
    case UnaryOpType::IsFinite:
      return {at::isfinite(in.as<at::Tensor>())};
      break;
    default:
      NVF_CHECK(
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
      std::optional<std::string> cast_str = cast_func_str(std::make_pair(
          in()->getDataType().value(), out()->getDataType().value()));
      NVF_ERROR(cast_str != std::nullopt, "Unsupported Cast");
      ss << cast_str.value();
    } else {
      ss << op_type;
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

std::string UnaryOp::getGraphvizLabel() const {
  std::stringstream ss;
  ss << getOpString() << "(" << getUnaryOpType() << ")";
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
  addDataAttribute(type);
}

std::vector<PolymorphicValue> BinaryOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  using namespace PolymorphicValue_functions;
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
      NVF_CHECK(
          !rhs.is<int64_t>() || rhs != 0, "Integer division by zero detected");
      return {lhs / rhs};
      break;
    case BinaryOpType::Mod:
      NVF_CHECK(rhs != 0, "Modulo zero detected");
      return {lhs % rhs};
      break;
    case BinaryOpType::Fmod:
      NVF_CHECK(rhs != 0, "Float modulo zero detected");
      return {fmod(lhs, rhs)};
      break;
    case BinaryOpType::CeilDiv:
      NVF_CHECK(rhs != 0, "CeilDiv by zero detected");
      return {ceildiv(lhs, rhs)};
      break;
    case BinaryOpType::LogicalAnd:
      return {lhs && rhs};
      break;
    case BinaryOpType::LogicalOr:
      return {lhs || rhs};
      break;
    case BinaryOpType::BitwiseAnd:
      return {lhs & rhs};
      break;
    case BinaryOpType::BitwiseOr:
      return {lhs | rhs};
      break;
    case BinaryOpType::BitwiseXor:
      return {lhs ^ rhs};
      break;
    case BinaryOpType::Eq:
      return {eq(lhs, rhs)};
      break;
    case BinaryOpType::NE:
      return {ne(lhs, rhs)};
      break;
    case BinaryOpType::GT:
      return {gt(lhs, rhs)};
      break;
    case BinaryOpType::GE:
      return {ge(lhs, rhs)};
      break;
    case BinaryOpType::LT:
      return {lt(lhs, rhs)};
      break;
    case BinaryOpType::LE:
      return {le(lhs, rhs)};
      break;
    case BinaryOpType::Max:
      return {max(lhs, rhs)};
      break;
    case BinaryOpType::Min:
      return {min(lhs, rhs)};
      break;
    case BinaryOpType::Gcd:
      return {gcd(lhs, rhs)};
      break;
    case BinaryOpType::Lshift:
      return {lhs << rhs};
      break;
    case BinaryOpType::Rshift:
      return {lhs >> rhs};
      break;
    case BinaryOpType::Complex:
      return {at::complex(lhs.as<at::Tensor>(), rhs.as<at::Tensor>())};
      break;
    case BinaryOpType::Pow:
      return {pow(lhs, rhs)};
      break;
    default:
      NVF_CHECK(
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
    ss << op_type;
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

std::string BinaryOp::getGraphvizLabel() const {
  std::stringstream ss;
  ss << getOpString() << "(" << getBinaryOpType() << ")";
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
  addDataAttribute(type);
}

std::vector<PolymorphicValue> TernaryOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  using namespace PolymorphicValue_functions;
  const auto& a = inputs.at(0);
  const auto& b = inputs.at(1);
  const auto& c = inputs.at(2);
  switch (getTernaryOpType()) {
    case TernaryOpType::Clamp:
      return {std::min(std::max(a, b), c)};
      break;
    case TernaryOpType::Lerp:
      // This is the same lerp computed in helpers.cu
      // https://math.stackexchange.com/a/1798323
      return {(c < 0.5) ? a + c * (b - a) : b - (b - a) * (1.0 - c)};
      break;
    case TernaryOpType::Threshold:
      return {(a <= b) ? c : a};
      break;
    case TernaryOpType::Where:
      return {a.as<bool>() ? b : c};
      break;
    default:
      NVF_CHECK(
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

std::string TernaryOp::getGraphvizLabel() const {
  std::stringstream ss;
  ss << getOpString() << "(" << getTernaryOpType() << ")";
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(TernaryOp)

ArrayConstruct::ArrayConstruct(
    IrBuilderPasskey passkey,
    Val* output,
    std::vector<Val*> inputs)
    : Expr(passkey) {
  NVF_ERROR(!inputs.empty(), "Cannot create an array with no members.");
  addOutput(output);
  DataType input_dtype = DataType::Null;
  for (auto in : inputs) {
    addInput(in);
    auto in_dtype_opt = in->getDataType();
    NVF_ERROR(in_dtype_opt.has_value());
    if (input_dtype == DataType::Null) {
      input_dtype = *in_dtype_opt;
    } else {
      NVF_CHECK(
          input_dtype == *in_dtype_opt,
          "All inputs to ArrayConstruct must have the same data type");
    }
  }
  auto expected_output_dtype =
      ArrayType{std::make_shared<DataType>(input_dtype), inputs.size()};
  NVF_CHECK(
      output->getDataType() == expected_output_dtype,
      "Output of ArrayConstruct must be an array of the same data type as the "
      "inputs");
}

std::string ArrayConstruct::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = {"
                          << toDelimitedString(inputs()) << "}\n";
  return ss.str();
}

std::string ArrayConstruct::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "{ " << toDelimitedInlineString(inputs()) << " }";
  return ss.str();
}

std::vector<PolymorphicValue> ArrayConstruct::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  return {PolymorphicValue(inputs)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ArrayConstruct)

ReverseArray::ReverseArray(IrBuilderPasskey passkey, Val* output, Val* input)
    : Expr(passkey) {
  NVF_ERROR(
      std::holds_alternative<ArrayType>(input->dtype().type),
      "Cannot reverse a non-array type.");
  NVF_ERROR(
      std::holds_alternative<ArrayType>(output->dtype().type),
      "Cannot reverse a non-array type.");
  auto input_array_type = std::get<ArrayType>(input->dtype().type);
  auto output_array_type = std::get<ArrayType>(output->dtype().type);
  NVF_ERROR(
      input_array_type.type == output_array_type.type,
      "Cannot reverse an array of type ",
      input_array_type.type,
      " into an array of type ",
      output_array_type.type);
  NVF_ERROR(
      input_array_type.size == output_array_type.size,
      "Cannot reverse an array of size ",
      input_array_type.size,
      " into an array of size ",
      output_array_type.size);
  addOutput(output);
  addInput(input);
}

std::string ReverseArray::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = ReverseArray("
                          << in()->toString() << ")\n";
  return ss.str();
}

std::string ReverseArray::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "ReverseArray(" << in()->toInlineString() << ")";
  return ss.str();
}

std::vector<PolymorphicValue> ReverseArray::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 1, "ReverseArray expects 1 input");
  PolymorphicValue array = inputs.at(0);
  auto& vec = array.as<std::vector>();
  std::reverse(vec.begin(), vec.end());
  return {std::move(array)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ReverseArray)

GetItem::GetItem(IrBuilderPasskey passkey, Val* output, Val* array, Val* index)
    : Expr(passkey) {
  addOutput(output);
  addInput(array);
  addInput(index);
  NVF_ERROR(
      *(std::get<ArrayType>(array->dtype().type).type) == output->dtype(),
      "GetItem array input must have a data type");
}

std::string GetItem::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = " << array()->toString()
                          << "[" << index()->toString() << "]\n";
  return ss.str();
}

std::string GetItem::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "(" << array()->toInlineString() << ")[" << index()->toInlineString()
     << "]";
  return ss.str();
}

std::vector<PolymorphicValue> GetItem::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 2, "GetItem expects 2 inputs");
  return {PolymorphicValue(inputs.at(0)[inputs.at(1)])};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GetItem)

StructConstruct::StructConstruct(
    IrBuilderPasskey passkey,
    Val* output,
    const std::vector<std::pair<std::string, Val*>>& fields)
    : Expr(passkey) {
  NVF_ERROR(!fields.empty(), "Cannot create a struct with no members.");
  auto output_dtype = std::get<StructType>(output->dtype().type);
  NVF_ERROR(
      output_dtype.fields.size() == fields.size(),
      "StructConstruct output must have the same number of fields as the "
      "inputs");
  auto it = output_dtype.fields.begin();
  for (const auto& field : fields) {
    NVF_ERROR(
        it->name == field.first,
        "StructConstruct field names must match the output");
    NVF_ERROR(
        *(it->type) == field.second->dtype(),
        "StructConstruct field ",
        field.first,
        " must have the same data type as the output");
    addDataAttribute(field.first);
    addInput(field.second);
    it++;
  }
  addOutput(output);
}

std::string StructConstruct::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = { ";
  for (int64_t i : arange((int64_t)inputs().size())) {
    if (i > 0) {
      ss << ", ";
    }
    ss << attribute<std::string>(i) << " = " << input(i)->toString();
  }
  ss << " }\n";
  return ss.str();
}

std::string StructConstruct::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "{ ";
  for (int64_t i : arange((int64_t)inputs().size())) {
    if (i > 0) {
      ss << ", ";
    }
    ss << attribute<std::string>(i) << " = " << input(i)->toInlineString();
  }
  ss << " }";
  return ss.str();
}

std::vector<PolymorphicValue> StructConstruct::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      this->inputs().size() == inputs.size(),
      "StructConstruct expects ",
      this->inputs().size(),
      " inputs");
  PolymorphicValue struct_ =
      std::get<StructType>(output(0)->dtype().type).create();
  for (int64_t i : arange((int64_t)inputs.size())) {
    struct_->*attribute<std::string>(i) = inputs.at(i);
  }
  return {std::move(struct_)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(StructConstruct)

GetAttr::GetAttr(
    IrBuilderPasskey passkey,
    Val* output,
    Val* struct_,
    std::string attr)
    : Expr(passkey) {
  NVF_ERROR(
      std::get<StructType>(struct_->dtype().type).fieldDataType(attr) ==
          output->dtype(),
      "Data type mismatch for GetAttr");
  addOutput(output);
  addInput(struct_);
  addDataAttribute(std::move(attr));
}

std::string GetAttr::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = " << struct_()->toString()
                          << "." << attr() << "\n";
  return ss.str();
}

std::string GetAttr::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "(" << struct_()->toInlineString() << ")." << attr();
  return ss.str();
}

std::vector<PolymorphicValue> GetAttr::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 1, "GetAttr expects 1 input");
  return {inputs.at(0)->*attr()};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GetAttr)

GetMetaData::GetMetaData(IrBuilderPasskey passkey, Val* output, Val* input)
    : Expr(passkey) {
  addOutput(output);
  addInput(input);
  NVF_ERROR(
      out()->dtype() == metaDataTypeOf(in()),
      "Data type mismatch for GetMetaData")
}

std::string GetMetaData::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = getMetaData("
                          << in()->toString() << ")\n";
  return ss.str();
}

std::string GetMetaData::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "getMetaData(" << ir_utils::varName(in()) << ")";
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GetMetaData)

TensorConstruct::TensorConstruct(
    IrBuilderPasskey passkey,
    TensorView* output,
    Val* input)
    : Expr(passkey) {
  addOutput(output);
  addInput(input);
}

std::string TensorConstruct::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = TensorConstruct("
                          << in()->toString() << ")\n";
  return ss.str();
}

std::string TensorConstruct::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> TensorConstruct::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 1, "TensorConstruct expects 1 input");
  using namespace PolymorphicValue_functions;
  return {toTensor(inputs.at(0))};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(TensorConstruct)

RNGOp::RNGOp(
    IrBuilderPasskey passkey,
    RNGOpType type,
    Val* out,
    DataType dtype,
    std::vector<Val*> parameters,
    Val* philox_seed,
    Val* philox_offset,
    Val* philox_index)
    : Expr(passkey) {
  if (auto tv_out = dynamic_cast<TensorView*>(out)) {
    for (auto id : tv_out->getLogicalDomain()) {
      NVF_CHECK(!id->isReduction(), "Output of RNGOp can not have reduction");
      addInput(id->extent());
    }
  }
  for (auto v : parameters) {
    addInput(v);
  }
  if (philox_seed || philox_offset) {
    NVF_CHECK(
        philox_seed && philox_offset,
        "If either philox_seed or philox_offset is provided, the other must be "
        "also");
    addInput(philox_seed);
    addInput(philox_offset);
  }
  addOutput(out);
  RNGOp::Attributes attr{type, dtype, parameters.size()};
  addDataAttribute(attr);
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
  ss << dtype();
  auto seed = getRNGSeedVal();
  if (seed) {
    ss << ", " << seed->toInlineString();
  }
  ss << ");\n";
  return ss.str();
}

std::string RNGOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

int64_t RNGOp::getOutputDims() const {
  int64_t ndims = 0;
  if (auto tv_out = dynamic_cast<TensorView*>(output(0))) {
    ndims = (int64_t)tv_out->getLogicalDomain().size();
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

  NVF_ERROR(
      (out_type == ValType::TensorView && in_type == ValType::TensorView) ||
          (out_type == ValType::TensorIndex && in_type == ValType::TensorIndex),
      "Cannot braodcast a non-tensor object.");

  addOutput(out);
  addInput(in);

  // Validate the broadcast flags when this expr is created with
  // TensorView. Broadcast with TensorIndex only appears after
  // lowering, so it should have already been validated.
  if (out->isA<TensorView>()) {
    NVF_ERROR(in->isA<TensorView>());
    auto in_tv = in->as<TensorView>();
    auto out_tv = out->as<TensorView>();
    auto in_dom = TensorDomain::noReductions(in_tv->getLogicalDomain());
    auto& out_dom = out_tv->getLogicalDomain();
    NVF_ERROR(
        is_broadcast_dims.size() == out_dom.size(),
        "The dimensions of output tensor and does not match with "
        "is_broadcast_dims");

    auto out_size = is_broadcast_dims.size();
    auto num_new_broadcasts = 0;
    for (const auto i : arange(out_size)) {
      if (is_broadcast_dims[i]) {
        num_new_broadcasts++;
        auto id = out_dom[i];
        NVF_ERROR(
            id->isBroadcast(),
            "New broadcast dimension does not properly set its IterType.");
        NVF_ERROR(
            !id->hasExpandedExtent(),
            "New broadcast dimension can not be expanded.");
        NVF_ERROR(
            id->extent()->isOneInt(),
            "New broadcast dimension must have extent 1");
      } else {
        auto in_id = in_dom[i - num_new_broadcasts];
        auto out_id = out_dom[i];
        NVF_ERROR(
            in_id->sameAs(out_id), "IterDomain does not match in BroadcastOp");
      }
    }
    NVF_ERROR(
        out_size == in_dom.size() + num_new_broadcasts,
        "The dimensions of output tensor and does not match with "
        "is_broadcast_dims and input tensor");
  }

  addDataAttribute(std::move(is_broadcast_dims));
}

std::string BroadcastOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size) << "   = broadcast( " << in()->toString()
                          << ", flags = {";
  bool is_first = true;
  for (const auto f : getBroadcastDimFlags()) {
    if (!is_first) {
      ss << ", ";
    }
    ss << (f ? "true" : "false");
    is_first = false;
  }
  ss << "} )\n";
  return ss.str();
}

std::string BroadcastOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> BroadcastOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      inputs.size() == 1,
      "BroadcastOp expects exactly 1 input, but received ",
      inputs.size());
  std::vector<int64_t> out_shape;
  const auto& in = inputs.at(0).as<at::Tensor>();
  int64_t idx = 0;
  for (bool b : getBroadcastDimFlags()) {
    if (b) {
      out_shape.push_back(1);
    } else {
      out_shape.push_back(in.sizes()[idx++]);
    }
  }
  return {in.view(out_shape)};
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

  NVF_ERROR(
      in_type == ValType::TensorView,
      "Squeeze input must be a TensorView: ",
      in->toString());

  NVF_ERROR(
      out_type == ValType::TensorView,
      "Squeeze output must be a TensorView: ",
      in->toString());

  addOutput(out);
  addInput(in);

  // Validate the squeeze flags
  auto in_tv = in->as<TensorView>();
  auto out_tv = out->as<TensorView>();
  auto in_dom = TensorDomain::noReductions(in_tv->getLogicalDomain());
  auto& out_dom = out_tv->getLogicalDomain();
  NVF_ERROR(
      is_squeeze_dims.size() == in_dom.size(),
      "The dimensions of input tensor and does not match with is_squeeze_dims");

  int64_t in_size = (int64_t)is_squeeze_dims.size();
  auto num_removed_broadcasts = 0;
  for (const auto i : arange(is_squeeze_dims.size())) {
    if (is_squeeze_dims[i]) {
      num_removed_broadcasts++;
      auto id = in_dom[i];
      NVF_ERROR(
          id->isBroadcast() || id->isSymbolic(),
          "Squeeze dimension should be either Symbolic or Broadcast. Found ",
          id->getIterType());
      if (id->isBroadcast()) {
        // Check concrete broadcast extent here. For Symbolic inputs, this check
        // will be deferred to concretization. See dynamic_transform.cpp
        NVF_ERROR(
            id->extent()->isConstScalar() &&
                id->extent()->evaluate().as<int64_t>() == 1,
            "Can not squeeze dimension(s) with size != 1.");
      }
    } else {
      auto in_id = in_dom[i];
      auto out_id = out_dom[i - num_removed_broadcasts];
      NVF_ERROR(
          in_id->sameAs(out_id), "IterDomain does not match in BroadcastOp");
    }
  }
  NVF_ERROR(
      in_size == out_tv->nDims() + num_removed_broadcasts,
      "The dimensions of output tensor and does not match with is_squeeze_dims "
      "and input tensor");

  addDataAttribute(std::move(is_squeeze_dims));
}

std::string SqueezeOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size) << "   = squeeze( " << in()->toString()
                          << ", flags = {";
  bool is_first = true;
  for (const auto f : getSqueezeDimFlags()) {
    if (!is_first) {
      ss << ", ";
    }
    ss << (f ? "true" : "false");
    is_first = false;
  }
  ss << "} )\n";
  return ss.str();
}

std::string SqueezeOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> SqueezeOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      inputs.size() == 1,
      "SqueezeOp expects exactly 1 input, but received ",
      inputs.size());
  std::vector<int64_t> out_shape;
  const auto& in = inputs.at(0).as<at::Tensor>();
  const auto& is_squeeze_dims = getSqueezeDimFlags();
  NVF_ERROR(
      (int64_t)is_squeeze_dims.size() == in.dim(),
      "The dimensions of input tensor and does not match with is_squeeze_dims");
  at::Tensor out = in;
  for (int64_t i : arange((int64_t)is_squeeze_dims.size())) {
    if (is_squeeze_dims[i]) {
      if (in.stride(i) == 0) {
        // If the input dimension is expanded in this dimension, undo the expand
        // by slicing. This ensures that any broadcast dimensions will be
        // unexpanded when we do the final call to view()
        out = out.slice(i, 0, 1);
      }
    } else {
      out_shape.push_back(in.sizes()[i]);
    }
  }
  return {out.view(out_shape)};
}

void SqueezeOp::checkConcretization(Val* old_val, Val* new_val) const {
  Expr::checkConcretization(old_val, new_val); // does nullptr, vtype checks
  NVF_CHECK(
      old_val == in(),
      "Pre-concretized Val ",
      old_val->toString(),
      " does not match input TV ",
      in()->toString());
  auto old_tv = old_val->as<TensorView>();
  auto new_tv = new_val->as<
      TensorView>(); // NOLINT(clang-analyzer-core.CallAndMessage,-warnings-as-errors)
  auto old_logical = old_tv->getLogicalDomain();
  auto new_logical = new_tv->getLogicalDomain();
  NVF_CHECK(
      new_logical.size() == old_tv->getLogicalDomain().size(),
      "New TV ",
      new_tv->toString(),
      " has rfactor of length ",
      new_logical.size(),
      " but expected ",
      old_tv->getLogicalDomain().size());
  auto flags = getSqueezeDimFlags();
  for (auto i : arange(flags.size())) {
    if (!flags.at(i)) {
      continue;
    }
    auto new_id = new_logical.at(i);
    // Check that squeezed dimension concretizes to Broadcast
    NVF_CHECK(
        new_id->getIterType() == IterType::Broadcast,
        "Squeezed IterDomain ",
        new_id->toString(),
        " must concretize to IterType::Broadcast but found ",
        new_id->toString());
    // NOTE: we do not check the extent here. Even if the extent is not a const
    // scalar we know that it would simplify to 1 for these inputs, since this
    // IterDomain is concretized to Broadcast.
  }
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
  NVF_CHECK(
      out->getValType().value() == ValType::TensorView ||
      out->getValType().value() == ValType::TensorIndex);

  NVF_ERROR(
      (in->getValType() == ValType::TensorView &&
       out->getValType() == ValType::TensorView) ||
          (in->getValType() == ValType::TensorIndex &&
           out->getValType() == ValType::TensorIndex),
      "Reduction operation was created that does not have tensor inputs and "
      "outputs.");

  if (in->isA<TensorView>()) {
    NVF_ERROR(
        TensorDomain::noReductions(in->as<TensorView>()->getLogicalDomain())
                .size() == out->as<TensorView>()->getMaybeRootDomain().size(),
        "Reduction operation created with mismatched domains.");
  }
  NVF_ERROR(
      init->isConstScalar(),
      "Tried to create a reduction operation whith an initial value that isn't "
      "a constant.");

  addOutput(out);
  addInput(in);
  addAttribute(init);
  addDataAttribute(reduction_op_type);
  addDataAttribute(is_allreduce);
  addDataAttribute(false); // serial reduction
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ReductionOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& input = inputs.at(0).as<at::Tensor>();
  const auto output = out()->as<TensorView>();

  NVF_ERROR(
      !output->hasRoot(),
      "Evaluation for rFactored reductions is not supported.");

  std::vector<int64_t> reduction_axes;
  for (const auto i : arange(int64_t(output->getLogicalDomain().size()))) {
    auto ax = output->getLogicalDomain().at(i);
    if (ax->isReduction()) {
      reduction_axes.push_back(i);
    }
  }
  switch (getReductionOpType()) {
    case BinaryOpType::Add:
      return {at::sum(input, reduction_axes)};
      break;
    case BinaryOpType::Max:
      return {at::amax(input, reduction_axes)};
      break;
    case BinaryOpType::Min:
      return {at::amin(input, reduction_axes)};
      break;
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          getReductionOpType(),
          " in ",
          toString());
  }
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

  addDataAttribute(std::move(reduction_op_types));
  addDataAttribute(is_fused);

  for (auto init : init_vals) {
    addAttribute(init);
  }
}

std::string GroupedReductionOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedReductionOp(\n";
  ++indent_size;
  for (const auto i : arange(numHorizontallyGroupedExprs())) {
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

int GroupedReductionOp::getExprIndexOfOutput(Val* output_val) const {
  auto it = std::find(outputs().begin(), outputs().end(), output_val);
  if (it != outputs().end()) {
    return (int)std::distance(outputs().begin(), it);
  }

  NVF_THROW("Not an output, ", output_val->toString(), ", of ", toString());
}

std::vector<PolymorphicValue> GroupedReductionOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto num_reductions = numHorizontallyGroupedExprs();
  std::vector<PolymorphicValue> grouped_reduction_out;
  grouped_reduction_out.reserve(num_reductions);
  for (const auto i : arange(num_reductions)) {
    const auto& in_tensor = inputs.at(i).as<at::Tensor>();
    const auto out_tv = output(i)->as<TensorView>();
    NVF_ERROR(
        !out_tv->hasRoot(),
        "Evaluation for rFactored reductions is not supported.");

    std::vector<int64_t> reduction_axes;
    for (const auto id : arange(int64_t(out_tv->getLogicalDomain().size()))) {
      auto ax = out_tv->getLogicalDomain().at(id);
      if (ax->isReduction()) {
        reduction_axes.push_back(id);
      }
    }
    switch (getReductionOpType(i)) {
      case BinaryOpType::Add:
        grouped_reduction_out.emplace_back(at::sum(in_tensor, reduction_axes));
        break;
      case BinaryOpType::Max:
        grouped_reduction_out.emplace_back(at::amax(in_tensor, reduction_axes));
        break;
      default:
        NVF_CHECK(
            false,
            "Unexpected operator type: ",
            getReductionOpType(i),
            " in ",
            toString());
    }
  }
  return grouped_reduction_out;
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedReductionOp)

std::optional<WelfordTriplet::ValName> WelfordTriplet::getNameOf(
    Val* val) const {
  auto it = std::find(begin(), end(), val);
  if (it != end()) {
    return indexToValName((int)std::distance(begin(), it));
  }

  return std::optional<WelfordTriplet::ValName>();
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
  for (const auto i : arange(src.size())) {
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
  NVF_ERROR(output.avg() != nullptr);
  NVF_ERROR(output.var() != nullptr);
  NVF_ERROR(output.N() != nullptr);
  NVF_ERROR(init.avg() != nullptr);
  NVF_ERROR(init.var() != nullptr);
  NVF_ERROR(init.N() != nullptr);
  NVF_ERROR(input.avg() != nullptr);
  NVF_ERROR(input.var() != nullptr);
  NVF_ERROR(input.N() != nullptr);

  // Check output type
  NVF_ERROR(
      output.avg()->getValType().value() == ValType::TensorView ||
      output.avg()->getValType().value() == ValType::TensorIndex);
  NVF_ERROR(
      output.var()->getValType().value() == ValType::TensorView ||
      output.var()->getValType().value() == ValType::TensorIndex);
  NVF_ERROR(
      output.N()->getValType().value() == ValType::TensorView ||
      output.N()->getValType().value() == ValType::TensorIndex);
  NVF_ERROR(isIntegralType(output.N()->dtype()));

  // check initial value
  NVF_ERROR(init.N()->getValType().value() == ValType::Others);
  NVF_ERROR(isIntegralType(init.N()->dtype()));
  if (!init.N()->isZeroInt()) {
    // when initial count is zero, no initial variance or average is needed
    // initial value with a count of 1 is un-common enough that I'll push
    // the responsibility of creating all-zero var tensors to the user
    NVF_ERROR(
        init.avg()->getValType().value() == ValType::TensorView ||
        init.avg()->getValType().value() == ValType::TensorIndex);
    NVF_ERROR(
        init.var()->getValType().value() == ValType::TensorView ||
            init.var()->getValType().value() == ValType::TensorIndex,
        "Invalid initial var: ",
        init.var()->toString());
  }

  // check input
  NVF_ERROR(
      input.avg()->getValType().value() == ValType::TensorView ||
          input.avg()->getValType().value() == ValType::TensorIndex,
      input.avg()->getValType().value());
  NVF_ERROR(
      input.N()->getValType().value() == ValType::Others ||
      input.N()->getValType().value() == ValType::TensorView ||
      input.N()->getValType().value() == ValType::TensorIndex);
  NVF_ERROR(isIntegralType(input.N()->dtype()));
  if (!input.N()->isOneInt()) {
    // when input is only one value, only the value is required through avg
    // input the var part is implicitly 0 and codegen will handle that.
    NVF_ERROR(
        input.var()->getValType().value() == ValType::TensorView ||
        input.var()->getValType().value() == ValType::TensorIndex);
  } else {
    NVF_ERROR(
        input.var() == nullptr || input.var()->isZeroInt(),
        "Invalid var input, which must be either nullptr or scalar zero when "
        "the N input is one.");
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
  addDataAttribute(is_fused);

  NVF_ERROR(attributes().size() == kNumAttrs);
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

  NVF_ERROR(
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> WelfordOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      !hasInit(),
      "Evaluation for WelfordOp is not implemented for non-empty initial "
      "values.");
  const auto& in_tensor = inputs.at(0).as<at::Tensor>();
  const auto out_tv = out()->as<TensorView>();
  NVF_ERROR(
      !out_tv->hasRoot(),
      "Evaluation for WelfordOp is not supported when output is rFactored.");

  int64_t N = 1;
  std::vector<int64_t> reduction_axes;
  for (const auto i : arange(int64_t(out_tv->getLogicalDomain().size()))) {
    auto ax = out_tv->getLogicalDomain().at(i);
    if (ax->isReduction()) {
      reduction_axes.push_back(i);
      N *= in_tensor.size(i);
    }
  }
  const auto [in_var, in_avg] =
      at::var_mean(in_tensor, reduction_axes, false, false);
  return {in_avg, in_var * N, N};
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

  NVF_ERROR(
      input_vals.size() == num_grouped_ops,
      "Invalid number of input arguments. Expected: ",
      num_grouped_ops,
      ", Given: ",
      input_vals.size());
  NVF_ERROR(
      init_vals.size() == num_grouped_ops,
      "Invalid number of N arguments. Expected: ",
      num_grouped_ops,
      ", Given: ",
      init_vals.size());

  for (const auto i : arange(num_grouped_ops)) {
    // Check output type
    NVF_ERROR(
        output_vals[i].avg()->getValType().value() == ValType::TensorView ||
        output_vals[i].avg()->getValType().value() == ValType::TensorIndex);
    NVF_ERROR(
        output_vals[i].var()->getValType().value() == ValType::TensorView ||
        output_vals[i].var()->getValType().value() == ValType::TensorIndex);
    NVF_ERROR(
        output_vals[i].N()->getValType().value() == ValType::TensorView ||
        output_vals[i].N()->getValType().value() == ValType::TensorIndex);
    NVF_ERROR(isIntegralType(output_vals[i].N()->dtype()));

    // check initial value
    auto init_avg = init_vals[i].avg();
    auto init_var = init_vals[i].var();
    auto init_N = init_vals[i].N();
    NVF_ERROR(
        init_avg != nullptr && init_var != nullptr && init_N != nullptr,
        "nullptr init vals are not allowed");
    NVF_ERROR(init_N->getValType().value() == ValType::Others);
    NVF_ERROR(isIntegralType(init_N->dtype()));
    NVF_ERROR(
        init_avg->getValType().value() == ValType::TensorView ||
            init_avg->getValType().value() == ValType::TensorIndex ||
            (init_N->isZeroInt() &&
             init_avg->getValType().value() == ValType::Others),
        "Initial avg must be a tensor or, can be a scalar if initial N is "
        "zero.",
        " Initial avg: ",
        init_avg->toString(),
        ". Initial N: ",
        init_N->toString());
    NVF_ERROR(
        init_var->getValType().value() == ValType::TensorView ||
            init_var->getValType().value() == ValType::TensorIndex ||
            (init_N->isZeroInt() &&
             init_var->getValType().value() == ValType::Others),
        "Initial var must be a tensor or, can be a scalar if initial N is "
        "zero: ",
        init_var->toString());

    // check input
    auto in_avg = input_vals[i].avg();
    auto in_var = input_vals[i].var();
    auto in_N = input_vals[i].N();
    NVF_ERROR(
        in_avg != nullptr && in_var != nullptr && in_N != nullptr,
        "nullptr input vals are not allowed");
    NVF_ERROR(
        in_N->getValType().value() == ValType::Others ||
        in_N->getValType().value() == ValType::TensorView ||
        in_N->getValType().value() == ValType::TensorIndex);
    NVF_ERROR(isIntegralType(in_N->dtype()));
    NVF_ERROR(
        in_avg->getValType().value() == ValType::TensorView ||
            in_avg->getValType().value() == ValType::TensorIndex,
        "Invalid input avg argument type: ",
        in_avg->getValType().value());

    if (in_N->isOneInt()) {
      // when input is only one value, only the value is required through avg
      // input the var part must be implicitly 0
      NVF_ERROR(
          in_var->isZeroInt(),
          "Invalid var input, which must be scalar zero when the N input is "
          "one: ",
          in_var->toString());
    } else {
      NVF_ERROR(
          in_var->getValType().value() == ValType::TensorView ||
              in_var->getValType().value() == ValType::TensorIndex,
          in_var->getValType().value(),
          ", ",
          in_N->toString());
    }
  }

  addDataAttribute(is_allreduce);
  for (const auto i : arange(num_grouped_ops)) {
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
  for (const auto i : arange(numHorizontallyGroupedExprs())) {
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

int GroupedWelfordOp::getExprIndexOfOutput(Val* output_val) const {
  for (const auto expr_idx : arange(numHorizontallyGroupedExprs())) {
    if (outputVals().at(expr_idx).getNameOf(output_val).has_value()) {
      return (int)expr_idx;
    }
  }

  NVF_THROW("Not an output, ", output_val->toString(), ", of ", toString());
}

Val* GroupedWelfordOp::getInitValOfOutput(Val* output_val) const {
  auto expr_index = getExprIndexOfOutput(output_val);

  auto val_name = outputVals().at(expr_index).getNameOf(output_val).value();

  return initVals().at(expr_index).get(val_name);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedWelfordOp)

//==============================================================================================================================

MmaOp::MmaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* init)
    : Expr(passkey) {
  NVF_ERROR(
      out->getValType().value() == ValType::TensorView ||
          out->getValType().value() == ValType::TensorIndex,
      out->getValType().value());

  NVF_ERROR(
      in_a->getValType().value() == ValType::TensorView ||
          in_a->getValType().value() == ValType::TensorIndex,
      in_a->getValType().value());

  NVF_ERROR(
      in_b->getValType().value() == ValType::TensorView ||
          in_b->getValType().value() == ValType::TensorIndex,
      in_b->getValType().value());

  addOutput(out);
  addInput(in_a);
  addInput(in_b);
  // ATTR_POS_INIT
  addAttribute(init);
  // ATTR_POS_MACRO
  addDataAttribute(MmaMacro::NoMMA);
}

MmaOp::MmaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* init,
    const MmaMacro& macro)
    : MmaOp(passkey, out, in_a, in_b, init) {
  attribute<MmaMacro>(ATTR_POS_MACRO) = macro;
}

std::string MmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size + 1) << " = mma(" << inA()->toString() << ",\n";
  indent(ss, indent_size + 1) << "       " << inB()->toString() << ")\n";
  return ss.str();
}

std::string MmaOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

void MmaOp::setMacro(MmaMacro macro) {
  NVF_ERROR(macro != MmaMacro::NoMMA, "Unspecified mma type");
  attribute<MmaMacro>(ATTR_POS_MACRO) = macro;
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
    NVF_ERROR(expanded_extent != nullptr);
    NVF_ERROR(
        expanded_extent->dtype() == DataType::Index,
        "Expanded extents must be of index type.");
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ExpandOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs.at(0).as<at::Tensor>();
  std::vector<int64_t> expanded_size;
  for (auto i : arange(1, inputs.size())) {
    expanded_size.push_back((int64_t)inputs.at(i));
  }
  return {in.expand(expanded_size)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ExpandOp)

RepeatOp::RepeatOp(IrBuilderPasskey passkey, TensorView* out, TensorView* in)
    : Expr(passkey) {
  auto in_domain = TensorDomain::noReductions(in->getLogicalDomain());
  const auto& out_domain = out->getLogicalDomain();

  NVF_ERROR(in_domain.size() == out_domain.size());

  NVF_ERROR(
      std::none_of(
          out->getLogicalDomain().begin(),
          out->getLogicalDomain().end(),
          [](IterDomain* out_logical_id) {
            return out_logical_id->isReduction();
          }),
      "Output should not have reduction IDs.");

  bool repetition_found = false;
  for (const auto i : arange(in_domain.size())) {
    if (in_domain.at(i)->isBroadcast() && !out_domain.at(i)->isBroadcast()) {
      NVF_ERROR(!in_domain.at(i)->hasExpandedExtent());
      NVF_ERROR(in_domain.at(i)->extent()->isOneInt());
      repetition_found = true;
    }
  }

  NVF_ERROR(
      repetition_found,
      "No repetition dim found: ",
      out->toString(),
      ", ",
      in->toString());

  addOutput(out);
  addInput(in);
}

std::string RepeatOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = repeat( " << in()
                          << " )\n";
  return ss.str();
}

std::string RepeatOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> RepeatOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      inputs.size() == 1,
      "RepeatOp expects exactly 1 input, but received ",
      inputs.size());
  auto tensor = inputs.at(0).as<at::Tensor>();
  std::vector<int64_t> multipliers;
  multipliers.reserve(out()->getLogicalDomain().size());
  const auto c2p =
      PairwiseLogicalDomainMap(in(), out()).mapConsumerToProducer();
  for (const auto i : arange(out()->getLogicalDomain().size())) {
    auto out_id = out()->getLogicalDomain().at(i);
    auto inp_id = c2p.at(out_id);
    auto out_extent = ee.evaluate(out_id->extent()).as<int64_t>();
    auto inp_extent = ee.evaluate(inp_id->extent()).as<int64_t>();
    NVF_ERROR(
        out_extent % inp_extent == 0,
        "For dimension ",
        i,
        ", the output extent (",
        out_extent,
        " should be a multiple of the input extent (",
        inp_extent,
        ").");
    multipliers.push_back(out_extent / inp_extent);
  }
  return {tensor.repeat(multipliers)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(RepeatOp)

ViewAsScalar::ViewAsScalar(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    IterDomain* vector_id)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addAttribute(vector_id);
}

std::string ViewAsScalar::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = view_as_scalar( "
                          << in()->toString() << ", " << vector_id()->toString()
                          << " )\n";
  return ss.str();
}

std::string ViewAsScalar::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ViewAsScalar::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const at::Tensor& in = inputs.at(0).as<at::Tensor>();
  return {at::view_as_real(in)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ViewAsScalar)

ViewOp::ViewOp(IrBuilderPasskey passkey, Val* out, Val* in) : Expr(passkey) {
  NVF_ERROR(
      in->isA<TensorView>(),
      in->toString(),
      " is expected to be a TensorView.");
  NVF_ERROR(
      out->isA<TensorView>(),
      out->toString(),
      " is expected to be a TensorView.");
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ViewOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 1);
  const at::Tensor& in_tensor = inputs[0].as<at::Tensor>();

  const auto& [out_shape, _] = inferShapeOfOutput(out(), ee);
  // TODO: check allocation domain and contiguity.

  // Use `at::Tensor::reshape` instead of `at::Tensor::view` because `ViewOp`
  // doesn't always produce an alias. For example, when merging an expanded
  // `IterType::Broadcast` and an `IterType::Iteration`, `ViewOp` has to realize
  // the expand.
  return {in_tensor.reshape(out_shape)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ViewOp)

LoadStoreOp::LoadStoreOp(
    IrBuilderPasskey passkey,
    LoadStoreOpType op_type,
    Val* out,
    Val* in,
    CacheOp cache_op)
    : Expr(passkey) {
  // Pick the default cache operator.
  if (op_type == LoadStoreOpType::CpAsync) {
    if (cache_op == CacheOp::Unspecified) {
      cache_op = CacheOp::AllLevels;
    }
    NVF_CHECK(
        cache_op == CacheOp::Global || cache_op == CacheOp::AllLevels,
        "cp.async only takes .ca or .cg. as cache operator");
  } else if (op_type == LoadStoreOpType::Set) {
    if (cache_op == CacheOp::Unspecified) {
      cache_op = CacheOp::Streaming;
    }
  } else {
    NVF_CHECK(
        cache_op == CacheOp::Unspecified,
        "Only Set and CpAsync take a cache operator.");
  }

  addOutput(out);
  addInput(in);
  addDataAttribute(op_type);
  addDataAttribute(cache_op);
}

std::vector<PolymorphicValue> LoadStoreOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  if (TensorView* out_tv = dynamic_cast<TensorView*>(out())) {
    if (out_tv->hasRoot()) {
      std::optional<std::vector<int64_t>> permutation =
          ir_utils::computePermutation(
              out_tv->getRootDomain(), out_tv->getLogicalDomain());
      NVF_ERROR(
          permutation.has_value(),
          "The logical domain of a Set.Permute is supposed to be a permutation"
          " of the root domain: ",
          out_tv);
      NVF_ERROR(inputs.size() == 1);
      at::Tensor in_tensor = inputs[0].as<at::Tensor>();
      at::Tensor out_tensor = in_tensor.permute(*permutation);
      return {out_tensor};
    }
  }
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
    if (tv != nullptr && tv->hasRoot()) {
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
  if (cacheOp() != CacheOp::Unspecified) {
    ss << ", cache_op=" << cacheOp();
  }
  ss << " )\n";
  return ss.str();
}

std::string LoadStoreOp::toInlineString(int indent_size) const {
  NVF_CHECK(
      !(out()->isA<TensorView>() || in()->isA<TensorView>()),
      "Tensor op can not be printed inline");
  // Set is allowed to have a scalar, e.g. setting the iteration domain
  // of a tensor in pad.
  return in()->toInlineString();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(LoadStoreOp)

IterDomainBuilder::IterDomainBuilder(Val* _start, Val* _extent)
    : start_(_start), extent_(_extent) {
  NVF_ERROR(
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
      padded_to_size_(id->getMaybeSizeAfterPadding()) {}

IterDomainBuilder& IterDomainBuilder::resetSchedulingParams() {
  parallel_type_ = ParallelType::Serial;
  is_rfactor_domain_ = false;
  is_padded_dimension_ = false;
  padded_to_size_ = std::nullopt;
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
    std::optional<int64_t> _padded_to_size) {
  padded_to_size_ = _padded_to_size;
  return *this;
}

IterDomain* IterDomainBuilder::build() const {
  NVF_ERROR(
      start_ != nullptr && extent_ != nullptr,
      "Start and extent are required to build an iter domain.");
  return IrBuilder::createInContainer<IterDomain>(start_->container(), *this);
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
    std::optional<int64_t> padded_to_size)
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
      padded_to_size_(padded_to_size) {
  // NOTE: We previously asserted !(isRFactorProduct() && isBroadcast()), i.e.
  // that an IterDomain could not be both a broadcast and an logical domain.
  // However, since the introduction of the resize op, we now have a legitimate
  // case where this may be true; namely, whenever we resize an IterDomain to
  // size 1, we will mark it as Broadcast, but the resize must lie between root
  // and rfactor.

  NVF_ERROR(
      extent->dtype() == DataType::Index,
      "Cannot create an iter domain over an extent that is not an "
      "nvfuser_index_t but received ",
      extent->dtype(),
      " .");

  NVF_ERROR(
      expanded_extent == nullptr || expanded_extent->dtype() == DataType::Index,
      "Cannot create an iter domain over an expanded_extent that is not an "
      "nvfuser_index_t but received ",
      expanded_extent->dtype(),
      " .");

  NVF_ERROR(
      start->dtype() == DataType::Index,
      "Cannot create an iter domain with a start that is not an "
      "nvfuser_index_t but received ",
      start->dtype(),
      " .");

  NVF_ERROR(
      stop_offset_->dtype() == DataType::Index,
      "Cannot create an iter domain with a stop_offset_ that is not an "
      "nvfuser_index_t but received ",
      stop_offset_->dtype(),
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
          args.padded_to_size_) {}

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
      padded_to_size_(src->padded_to_size_) {}

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

  // Do not take is_rfactor_domain_ into account. IterDomain's are
  // considered the same if they are rfactor or not.

  // TODO: Consider managing them as attributes

  return start()->sameAs(other_id->start()) &&
      extent()->sameAs(other_id->extent()) &&
      hasExpandedExtent() == other_id->hasExpandedExtent() &&
      (!hasExpandedExtent() ||
       expandedExtent()->sameAs(other_id->expandedExtent())) &&
      stopOffset()->sameAs(other_id->stopOffset()) &&
      getParallelType() == other_id->getParallelType() &&
      getIterType() == other_id->getIterType() &&
      hasPaddingToMultipleOfWarp() == other_id->hasPaddingToMultipleOfWarp() &&
      getMaybeSizeAfterPadding() == other_id->getMaybeSizeAfterPadding();
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
  ss << extent()->toInlineString();
  if (hasExpandedExtent()) {
    ss << " ex " << expandedExtent()->toInlineString();
  }
  ss << "}";
  if (isRFactorProduct()) {
    ss << "rf";
  }
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
IterDomain* IterDomain::cloneWithoutRFactor(bool map_with_original) {
  auto cloned = IterDomainBuilder(this).resetRfactor().build();

  if (map_with_original) {
    fusion()->registerExactMapping(this, cloned);
  }

  return cloned;
}

/*static*/ std::vector<IterDomain*> IterDomain::clone(
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
IterDomain* IterDomain::merge(
    IterDomain* outer,
    IterDomain* inner,
    std::optional<bool> rfactor_domain,
    std::optional<IterType> iter_type) {
  NVF_CHECK(
      outer->isReduction() == inner->isReduction(),
      "Merging IterDomains requires that their iteration types match. ",
      "Outer: ",
      outer->toString(),
      ", Inner: ",
      inner->toString());

  NVF_CHECK(
      !outer->isStride() && !inner->isStride(),
      "No support for merging stride domains");

  // By default, if not specified, don't create rfactor
  // outputs. Reshape transformations should propagate the flag, which
  // should explicitly specify the flag
  if (!rfactor_domain.has_value()) {
    rfactor_domain = false;
  }

  Val* merged_id_size =
      SimplifyingIrBuilder::mulExpr(outer->extent(), inner->extent());

  if (!iter_type.has_value()) {
    iter_type = outer->getIterType();

    if (outer->isBroadcast() && inner->isBroadcast()) {
      iter_type = IterType::Broadcast;
    }

    if ((outer->isBroadcast() || inner->isBroadcast()) &&
        (outer->getIterType() == IterType::Iteration ||
         inner->getIterType() == IterType::Iteration)) {
      iter_type = IterType::Iteration;
    }

    if ((outer->isBroadcast() || inner->isBroadcast()) &&
        (outer->getIterType() == IterType::GatherScatter ||
         inner->getIterType() == IterType::GatherScatter)) {
      iter_type = IterType::GatherScatter;
    }
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
    } else if (!outer->hasExpandedExtent() && inner->hasExpandedExtent()) {
      if (outer->isBroadcast()) {
        expanded_extent = inner->expandedExtent();
      } else {
        expanded_extent = mul(outer->extent(), inner->expandedExtent());
      }
    }
  }

  IterDomain* merged_id =
      IterDomainBuilder(outer->container()->zeroVal(), merged_id_size)
          .parallel_type(outer->getParallelType())
          .expanded_extent(expanded_extent)
          .iter_type(*iter_type)
          .is_rfactor_domain(*rfactor_domain)
          .build();

  IrBuilder::createInContainer<Merge>(
      outer->container(), merged_id, outer, inner);

  return merged_id;
}

std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor,
    bool inner_split,
    std::optional<bool> rfactor_domain,
    std::optional<IterType> outer_iter_type,
    std::optional<IterType> inner_iter_type) {
  NVF_CHECK(
      factor->isIntegralScalar(), "Cannot split by non-integer value ", factor);

  // outer loop size
  Val* remainder = SimplifyingIrBuilder::ceilDivExpr(in->extent(), factor);
  Val* expanded_remainder = nullptr;
  if (in->hasExpandedExtent()) {
    expanded_remainder =
        SimplifyingIrBuilder::ceilDivExpr(in->expandedExtent(), factor);
  }

  // By default, if not specified, don't create rfactor
  // outputs. Reshape transformations should propagate the flag, which
  // should explicitly specify the flag
  if (!rfactor_domain.has_value()) {
    rfactor_domain = false;
  }

  // If not specified, inherit these properties from the input iter domain
  if (!outer_iter_type.has_value()) {
    outer_iter_type = in->getIterType();
  }

  if (!inner_iter_type.has_value()) {
    inner_iter_type = in->getIterType();
  }

  // outer loop IterDomain
  IterDomain* ido =
      IterDomainBuilder(
          in->container()->zeroVal(), inner_split ? remainder : factor)
          .expanded_extent(
              in->hasExpandedExtent() && inner_split ? expanded_remainder
                                                     : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(*outer_iter_type)
          .is_rfactor_domain(*rfactor_domain)
          .build();

  // inner loop IterDomain
  IterDomain* idi =
      IterDomainBuilder(
          in->container()->zeroVal(), inner_split ? factor : remainder)
          .expanded_extent(
              in->hasExpandedExtent() && !inner_split ? expanded_remainder
                                                      : nullptr)
          .parallel_type(in->getParallelType())
          .iter_type(*inner_iter_type)
          .is_rfactor_domain(*rfactor_domain)
          .build();

  IrBuilder::createInContainer<Split>(
      in->container(), ido, idi, in, factor, inner_split);
  return {ido, idi};
}

std::pair<IterDomain*, IterDomain*> IterDomain::stridedSplit(int64_t factor) {
  // Use partial split so that only valid values are retained
  auto split_out = IterDomain::split(
      this,
      IrBuilder::createInContainer<Val>(container(), factor, DataType::Index),
      true,
      true);

  split_out.second->iter_type_ = IterType::Stride;
  split_out.first->is_rfactor_domain_ = true;
  split_out.second->is_rfactor_domain_ = true;
  return split_out;
}

std::pair<IterDomain*, IterDomain*> IterDomain::swizzle(
    SwizzleType swizzle_type,
    IterDomain* in_x,
    IterDomain* in_y) {
  NVF_CHECK(
      !in_x->extent()->isZeroInt() && !in_y->extent()->isZeroInt(),
      "Invalid swizzling of a empty dimension.");

  // TODO: reduction check on swizzle:
  NVF_CHECK(
      !in_x->isReduction() && !in_y->isReduction(),
      "swizzled reduction not yet supported");

  for (auto input : InputsOf::outputs({in_x, in_y})) {
    NVF_CHECK(
        !input->as<IterDomain>()->isBroadcast(),
        "swizzling broadcast axes not yet supported");
  }

  IterDomain* out_x = IterDomainBuilder(in_x).build();

  IterDomain* out_y = IterDomainBuilder(in_y).build();

  IrBuilder::createInContainer<Swizzle>(
      in_x->container(), out_x, out_y, in_x, in_y, swizzle_type);

  return std::make_pair(out_x, out_y);
}

std::pair<IterDomain*, IterDomain*> IterDomain::swizzle(
    Swizzle2DType swizzle_type,
    IterDomain* in_x,
    IterDomain* in_y,
    SwizzleMode swizzle_mode) {
  NVF_CHECK(
      !in_x->extent()->isZeroInt() && !in_y->extent()->isZeroInt(),
      "Invalid swizzling of a empty dimension.");

  // TODO: reduction check on swizzle:
  NVF_CHECK(
      !in_x->isReduction() && !in_y->isReduction(),
      "swizzled reduction not yet supported");

  for (auto input : InputsOf::outputs({in_x, in_y})) {
    NVF_CHECK(
        !input->as<IterDomain>()->isBroadcast(),
        "swizzling broadcast axes not yet supported");
  }

  IterDomain* out_x = IterDomainBuilder(in_x).build();

  IterDomain* out_y = IterDomainBuilder(in_y).build();

  IrBuilder::createInContainer<Swizzle2D>(
      in_x->container(), out_x, out_y, in_x, in_y, swizzle_type, swizzle_mode);

  return std::make_pair(out_x, out_y);
}

IterDomain* IterDomain::resize(
    IterDomain* in,
    Val* left_expansion,
    Val* right_expansion,
    bool mark_as_rfactor,
    std::optional<IterType> iter_type_opt) {
  NVF_CHECK(
      left_expansion->isIntegralScalar(),
      "Expansion factor must be an integer scalar: ",
      left_expansion->toString());
  NVF_CHECK(
      right_expansion->isIntegralScalar(),
      "Expansion factor must be an integer scalar: ",
      right_expansion->toString());

  if (left_expansion->isConstInt() && right_expansion->isConstInt()) {
    auto left = left_expansion->evaluate();
    auto right = right_expansion->evaluate();
    if (left == 0 && right == 0) {
      // This is a trivial resize. Check that we are not changing the IterType,
      // then return the input.
      NVF_CHECK(
          !iter_type_opt.has_value() ||
              iter_type_opt.value() == in->getIterType(),
          "If IterType is specified in pad with zero expansion then it must "
          "match input");
      return in;
    }
  }
  NVF_CHECK(
      in->getIterType() == IterType::Iteration ||
          in->getIterType() == IterType::Broadcast ||
          in->getIterType() == IterType::Symbolic,
      "Not a valid IterType: ",
      in->getIterType());

  NVF_CHECK(
      in->start()->isZeroInt(),
      "Non-zero start not supported: ",
      in->toString());
  NVF_CHECK(
      in->stopOffset()->isZeroInt(),
      "Non-zero stop offset not considered: ",
      in->toString());

  // The overall extent is (in->extent() + left_expansion +
  // right_expansion). This can be simplified for a slice op as
  // the right expansion should look like (slice_end_offset -
  // in->extent()), or (slice_end_offset + (- in->extent())), so the
  // overall extent is left_expansion + slice_end_offset.

  // Detect common slice patterns and return a simplified Val
  // representing (in->extent() + right_expansion) if possible
  auto simplify_input_extent_plus_right_expansion = [](Val* right_expansion,
                                                       Val* in_extent) -> Val* {
    auto bop = dynamic_cast<BinaryOp*>(right_expansion->definition());
    if (bop == nullptr) {
      return nullptr;
    }
    Val* sub_rhs = nullptr;
    if (bop->getBinaryOpType() == BinaryOpType::Sub) {
      sub_rhs = bop->rhs();
    } else if (bop->getBinaryOpType() == BinaryOpType::Add) {
      // Note that SimplifyingIrBuilder may turn (a - b) to (a + (- b))
      if (auto uop = dynamic_cast<UnaryOp*>(bop->rhs()->definition());
          uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Neg) {
        sub_rhs = uop->in();
      }
    }
    if (sub_rhs == in_extent) {
      return bop->lhs();
    } else {
      return nullptr;
    }
  };

  Val* resized_id_size = nullptr;
  if (auto simplified_val = simplify_input_extent_plus_right_expansion(
          right_expansion, in->extent())) {
    resized_id_size =
        SimplifyingIrBuilder::addExpr(left_expansion, simplified_val);
  } else {
    resized_id_size = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::addExpr(
            in->getMaybeExpandedExtent(), left_expansion),
        right_expansion);
  }

  // If output IterType is provided, use it. Otherwise, if we can prove the
  // resized extent is 1, set to Broadcast, if we can prove it is >1 set to
  // Iteration, and otherwise fall back to Symbolic.
  IterType iter_type = IterType::Symbolic;
  if (iter_type_opt.has_value()) {
    iter_type = iter_type_opt.value();
  } else if (left_expansion->isConstInt() && right_expansion->isConstInt()) {
    auto left = left_expansion->evaluate();
    auto right = right_expansion->evaluate();
    if (resized_id_size->isConstInt()) {
      // Means input extent is also known
      auto out_extent = resized_id_size->evaluate();
      iter_type = out_extent == 1 ? IterType::Broadcast : IterType::Iteration;
    } else if (left + right > 1) {
      // Input extent is non-negative, so we know out_extent > 1
      iter_type = IterType::Iteration;
    }
  }

  auto resized_id =
      IterDomainBuilder(
          in->container()->zeroVal(),
          // Set immediate constant size of 1 if resize produces broadcast
          iter_type == IterType::Broadcast ? in->fusion()->oneVal()
                                           : resized_id_size)
          .is_rfactor_domain(mark_as_rfactor)
          .iter_type(iter_type)
          .build();

  IrBuilder::createInContainer<Resize>(
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

  if (t == ParallelType::Unroll || isParallelTypeVectorize(t) ||
      t == ParallelType::Group) {
    NVF_CHECK(
        start()->isZeroInt() && extent()->isConstScalar(),
        "Vectorization, unrolling, unswitching and grouping are only supported "
        "with start = 0 and extent as a const int, but got ",
        "a start of ",
        start(),
        " and extent ",
        extent()->toInlineString(),
        " .");
  }

  if (t == ParallelType::Group) {
    NVF_CHECK(
        getIterType() == IterType::Iteration ||
            getIterType() == IterType::GatherScatter,
        "Grouping IterDomain of non Iteration / GatherScatter type is not "
        "allowed. ",
        getIterType());
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
  NVF_CHECK(
      contiguity.size() == allocation_domain.size(),
      "Invalid contiguity information provided, incorrect size. Received "
      "vector of size ",
      contiguity.size(),
      " but needed one of size ",
      allocation_domain.size());
  for (auto i : arange(contiguity.size())) {
    bool expect_null =
        (allocation_domain.at(i)->isBroadcast() ||
         allocation_domain.at(i)->isReduction());
    NVF_CHECK(
        expect_null != contiguity.at(i).has_value(),
        "The contiguity of a broadcast/reduction dimension must be None. "
        "The contiguity of a non-broadcast/reduction dimension must be "
        "true/false. alloation_domain=[",
        toDelimitedString(allocation_domain),
        "], contiguity=[",
        toDelimitedString(contiguity),
        "]");
  }
}

// Check if loop_domain is a valid domain with no
// redundancy. The logical domain is used as a reference to find if
// there's any ID that's not covered by the new loop domain.
void validateLoopDomain(
    const std::vector<IterDomain*>& logical_domain,
    const std::vector<IterDomain*>& loop_domain,
    const std::vector<IterDomain*>& additional_ids) {
  // Skip if there's any symbolic ID
  if (std::any_of(
          logical_domain.begin(),
          logical_domain.end(),
          [](IterDomain* id) { return id->isSymbolic(); }) ||
      std::any_of(
          loop_domain.begin(),
          loop_domain.end(),
          [](IterDomain* id) { return id->isSymbolic(); }) ||
      std::any_of(
          additional_ids.begin(), additional_ids.end(), [](IterDomain* id) {
            return id->isSymbolic();
          })) {
    return;
  }

  std::vector<IterDomain*> reference;
  reference.reserve(logical_domain.size() + additional_ids.size());
  reference.insert(
      reference.end(), logical_domain.begin(), logical_domain.end());
  // additional_ids are also considered part of the reference domain
  reference.insert(
      reference.end(), additional_ids.begin(), additional_ids.end());

  auto [redundant_ids, _, unreachable_reference_ids] =
      ir_utils::compareDomainWithReference(loop_domain, reference);

  auto empty_or_broadcast = [](const auto& ids) {
    return std::all_of(ids.begin(), ids.end(), [](IterDomain* id) {
      return id->isBroadcast();
    });
  };

  NVF_ERROR(
      empty_or_broadcast(redundant_ids),
      "Trying to set a loop domain with non-broadcast redundant IDs: ",
      toDelimitedString(redundant_ids));

  NVF_ERROR(
      empty_or_broadcast(unreachable_reference_ids),
      "Not all logical IDs are covered by loop domain. Loop: ",
      toDelimitedString(loop_domain),
      ". Unreachable logical IDs: ",
      toDelimitedString(unreachable_reference_ids));
}

} // namespace

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> logical_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(logical_domain_),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  // resetDomains initializes other member variables, required by clang-tidy
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> logical_domain,
    std::vector<int64_t> stride_order,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(logical_domain_),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  // setting the proper allocation domain
  if (!stride_order.empty()) {
    auto rank = logical_domain_.size();
    NVF_ERROR(
        rank == stride_order.size(), "Invalid size of stride_order vector");

    // checking stride_order is indeed a permutation
    std::vector<int64_t> inc_vec(rank);
    std::iota(inc_vec.begin(), inc_vec.end(), 0);
    NVF_ERROR(
        std::is_permutation(
            stride_order.begin(), stride_order.end(), inc_vec.begin()),
        "stride_order is not a valid: " + toDelimitedString(stride_order));

    allocation_domain_.resize(rank, nullptr);
    for (auto i : arange(rank)) {
      allocation_domain_[rank - 1 - stride_order[i]] = logical_domain_[i];
    }
  }
  validateContiguity(maybeAllocation(), contiguity_);

  // resetDomains initializes other member variables, required by clang-tidy
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> loop_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(std::move(loop_domain)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  NVF_CHECK(
      loop_domain_.empty() == logical_domain_.empty(),
      "logical domain and loop domain can only be both empty or neither empty");
  validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);

  // resetDomains initializes other member variables, required by clang-tidy
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> loop_domain,
    std::vector<std::optional<bool>> contiguity)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      logical_domain_(std::move(logical_domain)),
      loop_domain_(std::move(loop_domain)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  NVF_CHECK(
      loop_domain_.empty() == logical_domain_.empty(),
      "logical domain and loop domain can only be both empty or neither empty");
  validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);
  if (!root_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, root_domain_, additional_ids_);
  }

  // resetDomains initializes other member variables, required by clang-tidy
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> allocation_domain,
    std::vector<IterDomain*> loop_domain,
    std::vector<std::optional<bool>> contiguity,
    std::vector<IterDomain*> additional_ids)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      logical_domain_(std::move(logical_domain)),
      allocation_domain_(std::move(allocation_domain)),
      loop_domain_(std::move(loop_domain)),
      initial_loop_domain_(loop_domain_),
      additional_ids_(std::move(additional_ids)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  NVF_CHECK(
      loop_domain_.empty() == logical_domain_.empty(),
      "logical domain and loop domain can only be both empty or neither empty");
  validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);
  if (!root_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, root_domain_, additional_ids_);
  }
  if (!allocation_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, allocation_domain_, additional_ids_);
  }

  // resetDomains initializes other member variables, required by clang-tidy
  resetDomains();
}

TensorDomain::TensorDomain(
    IrBuilderPasskey passkey,
    std::vector<IterDomain*> root_domain,
    std::vector<IterDomain*> logical_domain,
    std::vector<IterDomain*> allocation_domain,
    std::vector<IterDomain*> loop_domain,
    std::optional<std::vector<IterDomain*>> alternate_loop_domain,
    std::vector<std::optional<bool>> contiguity,
    std::vector<IterDomain*> additional_ids)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(std::move(root_domain)),
      logical_domain_(std::move(logical_domain)),
      allocation_domain_(std::move(allocation_domain)),
      loop_domain_(std::move(loop_domain)),
      alternate_loop_domain_(alternate_loop_domain),
      initial_loop_domain_(loop_domain_),
      additional_ids_(std::move(additional_ids)),
      contiguity_(
          contiguity.empty() ? getContiguityFilledWith(maybeAllocation(), false)
                             : std::move(contiguity)) {
  validateContiguity(maybeAllocation(), contiguity_);

  NVF_CHECK(
      loop_domain_.empty() == logical_domain_.empty(),
      "logical domain and loop domain can only be both empty or neither empty");
  validateLoopDomain(logical_domain_, loop_domain_, additional_ids_);
  if (!root_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, root_domain_, additional_ids_);
  }
  if (!allocation_domain_.empty()) {
    ir_utils::validateDomainEquivalence(
        logical_domain_, allocation_domain_, additional_ids_);
  }
  if (alternate_loop_domain_.has_value()) {
    validateLoopDomain(
        logical_domain_, alternate_loop_domain_.value(), additional_ids_);
  }

  // resetDomains initializes other member variables, required by clang-tidy
  resetDomains();
}

TensorDomain::TensorDomain(IrBuilderPasskey passkey, const TensorDomain* src)
    : Val(passkey, ValType::TensorDomain, DataType::Null),
      root_domain_(src->root_domain_),
      logical_domain_(src->logical_domain_),
      allocation_domain_(src->allocation_domain_),
      loop_domain_(src->loop_domain_),
      alternate_loop_domain_(src->alternate_loop_domain_),
      initial_loop_domain_(src->initial_loop_domain_),
      additional_ids_(src->additional_ids_),
      no_bcast_domain_(src->no_bcast_domain_),
      no_reduction_domain_(src->no_reduction_domain_),
      contiguity_(src->contiguity_),
      has_reduction_(src->has_reduction_) {}

TensorDomain::TensorDomain(const TensorDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      root_domain_(ir_cloner->clone(src->root_domain_)),
      logical_domain_(ir_cloner->clone(src->logical_domain_)),
      allocation_domain_(ir_cloner->clone(src->allocation_domain_)),
      loop_domain_(ir_cloner->clone(src->loop_domain_)),
      alternate_loop_domain_(ir_cloner->clone(src->alternate_loop_domain_)),
      initial_loop_domain_(ir_cloner->clone(src->initial_loop_domain_)),
      additional_ids_(ir_cloner->clone(src->additional_ids_)),
      no_bcast_domain_(ir_cloner->clone(src->no_bcast_domain_)),
      no_reduction_domain_(ir_cloner->clone(src->no_reduction_domain_)),
      contiguity_(src->contiguity()),
      has_reduction_(src->has_reduction_) {}

NVFUSER_DEFINE_CLONE(TensorDomain)

bool TensorDomain::hasBlockBroadcast() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isBroadcast() && id->isThreadDim();
      });
}

bool TensorDomain::hasGridBroadcast() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isBroadcast() && id->isBlockDim();
      });
}

bool TensorDomain::operator==(const TensorDomain& other) const {
  // Checks equality of each class field. Should not be necessary to
  // check no_bcast_domain_ and no_reduction_domain_ as they are just
  // derived from domain_.
  return root_domain_ == other.root_domain_ &&
      loop_domain_ == other.loop_domain_ &&
      alternate_loop_domain_ == other.alternate_loop_domain_ &&
      logical_domain_ == other.logical_domain_ &&
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
  if (logical().size() != other_td->logical().size()) {
    return false;
  }
  if (allocation().size() != other_td->allocation().size()) {
    return false;
  }

  for (const auto i : arange(nDims())) {
    if (!(axis(i)->sameAs(other_td->axis(i)))) {
      return false;
    }
  }

  for (const auto i : arange(root().size())) {
    if (!(root()[i]->sameAs(other_td->root()[i]))) {
      return false;
    }
  }

  for (const auto i : arange(logical().size())) {
    if (!(logical()[i]->sameAs(other_td->logical()[i]))) {
      return false;
    }
  }

  for (const auto i : arange(allocation().size())) {
    if (!(allocation()[i]->sameAs(other_td->allocation()[i]))) {
      return false;
    }
  }

  for (const auto i : arange(loop().size())) {
    if (!(loop()[i]->sameAs(other_td->loop()[i]))) {
      return false;
    }
  }

  // this_td has_value is not the same as other_td
  if (alternateLoop().has_value() != other_td->alternateLoop().has_value()) {
    return false;
  }

  // has_value is false for both this_td and other_td
  if (!alternateLoop().has_value() && !other_td->alternateLoop().has_value()) {
    return true;
  }

  // has_value is true for both this_td and other_td, so verify that all
  // iterDomains are the same.
  return std::ranges::all_of(
      std::ranges::iota_view{0LL, (int64_t)alternateLoop().value().size()},
      [&](int64_t i) {
        return alternateLoop().value()[i]->sameAs(
            other_td->alternateLoop().value()[i]);
      });
}

bool TensorDomain::sameAs(
    const std::vector<IterDomain*>& lhs,
    const std::vector<IterDomain*>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  size_t i = 0;
  for (auto td_lhs : lhs) {
    if (!td_lhs->sameAs(rhs[i++])) {
      return false;
    }
  }
  return true;
}

std::string TensorDomain::toString(const int indent_size, const bool loop_only)
    const {
  std::stringstream ss;
  if (loop_only) {
    indent(ss, indent_size) << "[" << toDelimitedString(loop()) << "]";
  } else {
    indent(ss, indent_size)
        << "logical=[" << toDelimitedString(logical()) << "]" << std::endl;
    if (hasRoot()) {
      indent(ss, indent_size + 1)
          << "root=[" << toDelimitedString(root()) << "]" << std::endl;
    }
    indent(ss, indent_size + 1)
        << "loop=[" << toDelimitedString(loop()) << "]" << std::endl;
    if (hasAllocation()) {
      indent(ss, indent_size + 1)
          << "allocation=[" << toDelimitedString(allocation()) << "]"
          << std::endl;
    }
    if (alternateLoop().has_value()) {
      indent(ss, indent_size + 1)
          << "alternate_loop=[" << toDelimitedString(alternateLoop().value())
          << "]" << std::endl;
    }
  }
  return ss.str();
}

std::string TensorDomain::toString(const int indent_size) const {
  return toString(indent_size, /*loop_only=*/true);
}

std::string TensorDomain::toInlineString(int indent_size) const {
  return toString(indent_size);
}

void TensorDomain::setContiguity(
    const std::vector<std::optional<bool>>& contig) {
  NVF_ERROR(
      maybeAllocation().size() == contig.size(),
      "Invalid size of contiguity vector");
  for (auto i : arange(contig.size())) {
    NVF_CHECK(
        maybeAllocation().at(i)->isBroadcast() != contig.at(i).has_value(),
        "The contiguity of a broadcast dimension must be None. "
        "The contiguity of a non-broadcast dimension must be true/false");
  }

  contiguity_ = contig;
}

std::vector<int64_t> TensorDomain::strideOrder() const {
  // short-circuit: no allocation domain; default stride-order
  if (allocation_domain_.empty()) {
    return {};
  }

  std::vector<int64_t> stride_order;
  stride_order.reserve(logical_domain_.size());

  for (size_t logical_idx : arange(logical_domain_.size())) {
    IterDomain* logical_id = logical_domain_.at(logical_idx);
    auto alloc_iter = std::find(
        allocation_domain_.begin(), allocation_domain_.end(), logical_id);
    NVF_ERROR(
        alloc_iter != allocation_domain_.end(),
        "Unable to find logical IterDomain in allocation domain.");
    int64_t alloc_idx = std::distance(allocation_domain_.begin(), alloc_iter);
    stride_order.push_back((int64_t)logical_domain_.size() - 1 - alloc_idx);
  }

  return stride_order;
}

bool TensorDomain::hasBlockReduction() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isReduction() && id->isThreadDim();
      });
}

bool TensorDomain::hasGridReduction() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return id->isReduction() && id->isBlockDim();
      });
}

bool TensorDomain::hasSymbolicAxis() const {
  // If there's any Symbolic axis, there must be one at the root or
  // logical domain.
  return (hasRoot() &&
          std::any_of(
              root().begin(),
              root().end(),
              [](auto id) {
                return id->getIterType() == IterType::Symbolic;
              })) ||
      std::any_of(logical().begin(), logical().end(), [](auto id) {
           return id->getIterType() == IterType::Symbolic;
         });
}

bool TensorDomain::hasViewLikeRFactor() const {
  if (!hasRoot()) {
    // Can't have view like rfactor if there is no logical domain
    return false;
  }

  // If there's an logical domain and no rfactor product is a reduction, this is
  // a view like rfactor
  return std::none_of(logical().begin(), logical().end(), [](IterDomain* id) {
    return (id->isReduction() || id->isStride()) && id->isRFactorProduct();
  });
}

bool TensorDomain::hasVectorize() const {
  return std::any_of(
      loop_domain_.begin(), loop_domain_.end(), [](IterDomain* id) {
        return isParallelTypeVectorize(id->getParallelType());
      });
}

std::optional<int64_t> TensorDomain::getReductionAxis() const {
  auto it = std::find_if(
      loop_domain_.begin(), loop_domain_.end(), [](const auto& id) {
        return id->isReduction();
      });
  if (it == loop_domain_.end()) {
    return std::optional<int64_t>();
  } else {
    return std::optional<int64_t>(std::distance(loop_domain_.begin(), it));
  }
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int64_t i) const {
  NVF_ERROR(nDims() > 0, "Tried to access an axis in a 0-dim domain");
  return loop_domain_[wrapDim(i)];
}

int64_t TensorDomain::posOf(IterDomain* id) const {
  NVF_ERROR(nDims() > 0, "Tried to find an axis in a 0-dim domain");
  int64_t i = 0;
  while (i < (int64_t)loop_domain_.size()) {
    if (loop_domain_[i] == id) {
      return i;
    }
    i++;
  }
  NVF_CHECK(false, "Provided id is not part of this domain.");
}

int64_t TensorDomain::rootPosOf(IterDomain* id) const {
  NVF_ERROR(
      !maybeRoot().empty(), "Tried to find an axis in a 0-dim root domain");
  auto it = std::find(maybeRoot().begin(), maybeRoot().end(), id);
  NVF_ERROR(it != maybeRoot().end(), "Provided id is not part of root domain.");
  return std::distance(maybeRoot().begin(), it);
}

void TensorDomain::broadcast(int64_t axis, Val* extent) {
  axis = nvfuser::wrapDim(axis, nDims() + 1);
  IterDomain* id = IterDomainBuilder(fusion()->zeroVal(), extent)
                       .iter_type(IterType::Broadcast)
                       .build();
  loop_domain_.insert(loop_domain_.begin() + axis, id);
  additional_ids_.push_back(id);
}

void TensorDomain::split(int64_t axis, Val* factor, bool inner_split) {
  NVF_ERROR(nDims() > 0, "Tried to do split on a 0-dim domain");
  axis = wrapDim(axis);

  IterDomain* id = this->axis(axis);

  auto split_ids = IterDomain::split(id, factor, inner_split);
  loop_domain_.erase(loop_domain_.begin() + axis);
  loop_domain_.insert(loop_domain_.begin() + axis, split_ids.second);
  loop_domain_.insert(loop_domain_.begin() + axis, split_ids.first);
  resetDomains();
}

// Merge "axis_o" and "axis_i" into 1 dimension
void TensorDomain::merge(int64_t axis_o, int64_t axis_i) {
  NVF_ERROR(nDims() > 0, "Tried to do merge on a 0-dim domain");
  axis_o = wrapDim(axis_o);
  axis_i = wrapDim(axis_i);

  NVF_CHECK(
      axis_o != axis_i,
      "Invalid merge detected, axes provided are the same axis.");

  IterDomain* first = axis(axis_o);
  IterDomain* second = axis(axis_i);

  IterDomain* merged_id = IterDomain::merge(first, second);

  // axis_o is the outer input of this merge but does not
  // automatically mean it's an outer domain in TensorDomain.
  auto td_outer_pos = axis_o < axis_i ? axis_o : axis_i;
  auto td_inner_pos = axis_o < axis_i ? axis_i : axis_o;

  loop_domain_.erase(loop_domain_.begin() + td_inner_pos);
  loop_domain_.erase(loop_domain_.begin() + td_outer_pos);
  loop_domain_.insert(loop_domain_.begin() + td_outer_pos, merged_id);
  resetDomains();
}

// Reorder axes according to map[old_pos] = new_pos
void TensorDomain::reorder(
    const std::unordered_map<int64_t, int64_t>& old2new_) {
  NVF_ERROR(
      nDims() != 0 || old2new_.empty(), "Tried to reorder a 0-dim domain");
  loop_domain_ = orderedAs(loop_domain_, old2new_);
  resetDomains();
}

std::vector<IterDomain*> TensorDomain::orderedAs(
    const std::vector<IterDomain*>& dom,
    const std::unordered_map<int64_t, int64_t>& old2new_) {
  NVF_ERROR(
      !dom.empty() || old2new_.empty(), "Tried to reorder a 0-dim domain");

  // Eventhough these checks are already in TensorView, we want to redo them as
  // we can enter this function from other places, not through TensorView

  auto new2old = ir_utils::normalizeOld2New(old2new_, (int64_t)dom.size());

  std::vector<IterDomain*> reordered_domain;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_domain),
      [dom](int64_t i) -> IterDomain* { return dom[i]; });

  return reordered_domain;
}

void TensorDomain::swizzle(SwizzleType swizzle_type, int64_t x, int64_t y) {
  NVF_ERROR(nDims() > 0, "Tried to do merge on a 0-dim domain");
  x = wrapDim(x);
  y = wrapDim(y);

  IterDomain* axis_x = axis(x);
  IterDomain* axis_y = axis(y);

  IterDomain* axis_out_x = nullptr;
  IterDomain* axis_out_y = nullptr;

  std::tie(axis_out_x, axis_out_y) =
      IterDomain::swizzle(swizzle_type, axis_x, axis_y);

  loop_domain_.erase(loop_domain_.begin() + x);
  loop_domain_.insert(loop_domain_.begin() + x, axis_out_x);

  loop_domain_.erase(loop_domain_.begin() + y);
  loop_domain_.insert(loop_domain_.begin() + y, axis_out_y);

  resetDomains();
}

void TensorDomain::swizzle(
    Swizzle2DType swizzle_type,
    int64_t x,
    int64_t y,
    SwizzleMode swizzle_mode) {
  NVF_ERROR(nDims() > 0, "Tried to do merge on a 0-dim domain");
  x = wrapDim(x);
  y = wrapDim(y);

  IterDomain* axis_x = axis(x);
  IterDomain* axis_y = axis(y);

  IterDomain* axis_out_x = nullptr;
  IterDomain* axis_out_y = nullptr;

  std::tie(axis_out_x, axis_out_y) =
      IterDomain::swizzle(swizzle_type, axis_x, axis_y, swizzle_mode);

  loop_domain_.erase(loop_domain_.begin() + x);
  loop_domain_.insert(loop_domain_.begin() + x, axis_out_x);

  loop_domain_.erase(loop_domain_.begin() + y);
  loop_domain_.insert(loop_domain_.begin() + y, axis_out_y);

  resetDomains();
}

void TensorDomain::resize(
    int64_t axis,
    Val* left_expansion,
    Val* right_expansion,
    std::optional<IterType> iter_type) {
  NVF_ERROR(nDims() > 0, "Tried to do resize on a 0-dim domain");
  axis = wrapDim(axis);

  IterDomain* id = this->axis(axis);

  auto resized_id = IterDomain::resize(
      id,
      left_expansion,
      right_expansion,
      /*mark_as_rfactor=*/false,
      iter_type);
  loop_domain_.at(axis) = resized_id;
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

std::vector<IterDomain*> TensorDomain::noDevices(
    const std::vector<IterDomain*>& td) {
  std::vector<IterDomain*> noDeviceDomain;
  std::copy_if(
      td.begin(),
      td.end(),
      std::back_inserter(noDeviceDomain),
      [](IterDomain* id) { return !id->isDeviceDim(); });
  return noDeviceDomain;
}

/*static*/ std::vector<std::optional<bool>> TensorDomain::
    getContiguityFilledWith(
        const std::vector<IterDomain*>& allocation_domain,
        bool fill_value) {
  std::vector<std::optional<bool>> contiguity;
  contiguity.reserve(allocation_domain.size());
  for (auto id : allocation_domain) {
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
  NVF_ERROR(nDims() > 0, "Tried to view transform a 0-dim domain");
  return transformView(this, view_analysis);
}

TensorDomain* TensorDomain::flatten(int64_t start_dim, int64_t end_dim) {
  auto inp_domain = noReductions(logical());

  if (start_dim < 0) {
    start_dim += (int64_t)inp_domain.size();
  }
  if (end_dim < 0) {
    end_dim += (int64_t)inp_domain.size();
  }
  NVF_CHECK(
      start_dim >= 0 && start_dim < int64_t(inp_domain.size()),
      "Invalid start_dim ",
      start_dim);
  NVF_CHECK(
      end_dim >= 0 && end_dim < int64_t(inp_domain.size()),
      "Invalid end_dim ",
      end_dim);
  NVF_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  std::vector<IterDomain*> new_root_domain;
  new_root_domain.reserve(inp_domain.size());
  for (auto i : arange((int64_t)inp_domain.size())) {
    bool is_rfactor_dim = i >= start_dim && i <= end_dim;
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
                      .expanded_extent(nullptr)
                      .build();
    new_root_domain.push_back(out_id);
  }

  std::vector<IterDomain*> logical_domain;
  logical_domain.reserve(new_root_domain.size() - (end_dim - start_dim));
  for (auto i : arange(start_dim)) {
    logical_domain.push_back(new_root_domain[i]);
  }

  IterDomain* merged_id = new_root_domain[start_dim];
  for (auto i : arange(start_dim + 1, end_dim + 1)) {
    IterDomain* new_merged_id =
        IterDomainBuilder(
            merged_id->container()->zeroVal(),
            mul(merged_id->extent(), new_root_domain[i]->extent()))
            .is_rfactor_domain(true)
            .build();
    IrBuilder::create<Merge>(new_merged_id, merged_id, new_root_domain[i]);
    merged_id = new_merged_id;
  }
  logical_domain.push_back(merged_id);

  for (auto i : arange(end_dim + 1, inp_domain.size())) {
    logical_domain.push_back(new_root_domain[i]);
  }

  return IrBuilder::create<TensorDomain>(
      new_root_domain,
      logical_domain,
      logical_domain,
      TensorDomain::getContiguityFilledWith(logical_domain, true));
}

// TODO: Rfactor a Welford

// pair is in order where second is the consumer of first
std::pair<TensorDomain*, TensorDomain*> TensorDomain::rFactor(
    const std::vector<int64_t>& axes_) {
  return TransformRFactor::runReplay(this, axes_);
}

void TensorDomain::setLoopDomain(std::vector<IterDomain*> new_loop_domain) {
  validateLoopDomain(logical(), new_loop_domain, additionalIDs());
  loop_domain_ = std::move(new_loop_domain);
  initial_loop_domain_ = loop_domain_;
  resetDomains();
}

void TensorDomain::setAlternateLoopDomain(
    std::vector<IterDomain*> new_loop_domain) {
  validateLoopDomain(logical(), new_loop_domain, additionalIDs());
  alternate_loop_domain_ = std::move(new_loop_domain);
}

void TensorDomain::setAllocationDomain(
    std::vector<IterDomain*> new_allocation_domain,
    std::vector<std::optional<bool>> new_contiguity) {
  validateContiguity(new_allocation_domain, new_contiguity);

  ir_utils::validateDomainEquivalence(
      logical_domain_, new_allocation_domain, additional_ids_);

  allocation_domain_ = std::move(new_allocation_domain);
  contiguity_ = std::move(new_contiguity);
}

std::vector<IterDomain*> TensorDomain::allIDs() const {
  std::vector<const std::vector<IterDomain*>*> all_domains = {
      &loop_domain_,
      &logical_domain_,
      &root_domain_,
      &initial_loop_domain_,
      &allocation_domain_,
      &additional_ids_};
  if (alternate_loop_domain_.has_value()) {
    all_domains.push_back(&alternate_loop_domain_.value());
  }
  VectorOfUniqueEntries<IterDomain*> discovered_ids;
  for (auto domain : all_domains) {
    discovered_ids.pushBack(*domain);
  }

  // We only care about IDs on the shortest path between domains
  std::unordered_multimap<IterDomain*, IterDomain*> out2in;
  for (auto i : arange(all_domains.size() - 1)) {
    if (all_domains[i]->empty()) {
      continue;
    }
    for (auto j : arange(i + 1, all_domains.size())) {
      if (all_domains[j]->empty()) {
        continue;
      }
      auto path = getExprsBetween<IRBFS>(
                      {all_domains[i]->begin(), all_domains[i]->end()},
                      {all_domains[j]->begin(), all_domains[j]->end()},
                      false)
                      .first;
      for (auto [expr, _] : path) {
        discovered_ids.pushBack(
            ir_utils::filterByType<IterDomain>(expr->outputs()));
        discovered_ids.pushBack(
            ir_utils::filterByType<IterDomain>(expr->inputs()));
        for (auto in : expr->inputs()) {
          for (auto out : expr->outputs()) {
            out2in.emplace(out->as<IterDomain>(), in->as<IterDomain>());
          }
        }
      }
    }
  }

  // Topological sort all IDs
  std::list<IterDomain*> ids_to_be_sorted(
      discovered_ids.begin(), discovered_ids.end());
  VectorOfUniqueEntries<IterDomain*> sorted_ids;
  while (!ids_to_be_sorted.empty()) {
    auto it = ids_to_be_sorted.begin();
    while (it != ids_to_be_sorted.end()) {
      auto range = out2in.equal_range(*it);
      if (std::all_of(range.first, range.second, [&](const auto& kv) {
            return sorted_ids.has(kv.second);
          })) {
        sorted_ids.pushBack(*it);
        it = ids_to_be_sorted.erase(it);
      } else {
        it++;
      }
    }
  }
  return sorted_ids.vector();
}

std::vector<Expr*> TensorDomain::allExprs() const {
  auto all_ids = allIDs();
  std::unordered_set<Val*> all_id_set{all_ids.begin(), all_ids.end()};

  VectorOfUniqueEntries<Expr*> exprs;
  for (auto id : all_ids) {
    auto def = id->definition();
    if (def == nullptr) {
      continue;
    }

    if (std::all_of(def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
          return all_id_set.find(inp) != all_id_set.end();
        })) {
      exprs.pushBack(def);
    } else {
      NVF_ERROR(std::none_of(
          def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
            return all_id_set.find(inp) != all_id_set.end();
          }));
    }
  }

  return exprs.vector();
}

std::vector<Statement*> TensorDomain::allStatements() const {
  auto all_ids = allIDs();
  std::unordered_set<Val*> all_id_set{all_ids.begin(), all_ids.end()};

  VectorOfUniqueEntries<Statement*> stmts;
  for (auto id : all_ids) {
    // Visit definition if available and all inputs are already visited
    auto def = id->definition();
    if (def != nullptr) {
      if (std::all_of(
              def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
                return all_id_set.find(inp) != all_id_set.end();
              })) {
        stmts.pushBack(def);
      } else {
        NVF_ERROR(std::none_of(
            def->inputs().begin(), def->inputs().end(), [&](Val* inp) {
              return all_id_set.find(inp) != all_id_set.end();
            }));
      }
    }

    stmts.pushBack(id);
  }

  return stmts.vector();
}

Split::Split(
    IrBuilderPasskey passkey,
    IterDomain* outer,
    IterDomain* inner,
    IterDomain* in,
    Val* factor,
    bool inner_split)
    : Expr(passkey) {
  NVF_ERROR(
      factor->isIntegralScalar(),
      "Attempted to create a Split node with a non-integer factor.");
  addOutput(outer);
  addOutput(inner);
  addInput(in);
  // TODO add factor as an input, need to check Split::Split during validation
  // and need to check BestEffortReplay::findFirstMismatchedID addInput(factor);
  addAttribute(factor);
  addDataAttribute(inner_split);
}

Val* Split::isDivisible() const {
  return IrBuilder::isDivisibleExpr(in()->extent(), factor());
}

std::string Split::toString(int indent_size) const {
  std::stringstream ss;
  ss << (innerSplit() ? "Split: " : "Outer split: ");
  ss << in()->toString();
  ss << " by factor " << factor()->toString() << " -> ";
  ss << outer()->toString();
  ss << ", ";
  ss << inner()->toString();
  ss << "\n";
  return ss.str();
}

std::string Split::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Split can not be printed inline");
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Merge)

Swizzle::Swizzle(
    IrBuilderPasskey passkey,
    IterDomain* out_x,
    IterDomain* out_y,
    IterDomain* in_x,
    IterDomain* in_y,
    SwizzleType swizzle_type)
    : Expr(passkey) {
  addOutput(out_x);
  addOutput(out_y);
  addInput(in_x);
  addInput(in_y);
  addDataAttribute(swizzle_type);
}

std::string Swizzle::toString(int indent_size) const {
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

std::string Swizzle::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Swizzle)

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
  addDataAttribute(swizzle_type);
  addDataAttribute(swizzle_mode);
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
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
  NVF_CHECK(false, "Resize can not be printed inline");
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

NamedScalar* NamedScalar::getParallelDim(ParallelType p_type) {
  NVF_ERROR(
      isParallelTypeThread(p_type),
      "Cannot get parallel dim of non thread type, received: ",
      p_type);
  NVF_ERROR(FusionGuard::getCurFusion() != nullptr);
  std::string parallel_dim = stringifyThreadSize(p_type);
  return IrBuilder::create<NamedScalar>(parallel_dim, DataType::Index);
}

NamedScalar* NamedScalar::getParallelIndex(ParallelType p_type) {
  NVF_ERROR(FusionGuard::getCurFusion() != nullptr);
  std::string parallel_ind = stringifyThread(p_type);
  return IrBuilder::create<NamedScalar>(parallel_ind, DataType::Index);
}

std::optional<ParallelType> NamedScalar::getParallelDim() const {
  if (stringifyThreadSize(ParallelType::TIDx).compare(name()) == 0) {
    return ParallelType::TIDx;
  } else if (stringifyThreadSize(ParallelType::TIDy).compare(name()) == 0) {
    return ParallelType::TIDy;
  } else if (stringifyThreadSize(ParallelType::TIDz).compare(name()) == 0) {
    return ParallelType::TIDz;
  } else if (stringifyThreadSize(ParallelType::BIDx).compare(name()) == 0) {
    return ParallelType::BIDx;
  } else if (stringifyThreadSize(ParallelType::BIDy).compare(name()) == 0) {
    return ParallelType::BIDy;
  } else if (stringifyThreadSize(ParallelType::BIDz).compare(name()) == 0) {
    return ParallelType::BIDz;
  }
  return std::nullopt;
}

std::optional<ParallelType> NamedScalar::getParallelIndex() const {
  if (stringifyThread(ParallelType::TIDx).compare(name()) == 0) {
    return ParallelType::TIDx;
  } else if (stringifyThread(ParallelType::TIDy).compare(name()) == 0) {
    return ParallelType::TIDy;
  } else if (stringifyThread(ParallelType::TIDz).compare(name()) == 0) {
    return ParallelType::TIDz;
  } else if (stringifyThread(ParallelType::BIDx).compare(name()) == 0) {
    return ParallelType::BIDx;
  } else if (stringifyThread(ParallelType::BIDy).compare(name()) == 0) {
    return ParallelType::BIDy;
  } else if (stringifyThread(ParallelType::BIDz).compare(name()) == 0) {
    return ParallelType::BIDz;
  }
  return std::nullopt;
}

PadOp::PadOp(
    IrBuilderPasskey passkey,
    TensorView* out,
    TensorView* inp,
    const std::vector<Val*>& pad_widths,
    Val* value)
    : Expr(passkey) {
  const auto ndims = TensorDomain::noReductions(inp->getLogicalDomain()).size();
  NVF_ERROR(
      pad_widths.size() % 2 == 0,
      "Invalid size of padding width vector: ",
      pad_widths.size(),
      ". Number of width vals must be even.");
  NVF_ERROR(
      pad_widths.size() == ndims * 2,
      "Invalid size of padding width vector: ",
      pad_widths.size(),
      ". All dimensions, padded or not, must have width vals. Use zero for non "
      "non-padded dimensions.");
  addOutput(out);
  addInput(inp);
  addInput(value);
  for (auto width : pad_widths) {
    NVF_CHECK(width != nullptr, "Padding width must not be nullptr");
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<int64_t> PadOp::getPaddedAxes() const {
  auto num_dims = (int64_t)out()->as<TensorView>()->getLogicalDomain().size();
  std::vector<int64_t> padded_axes;
  for (const auto i : arange(num_dims)) {
    auto [left_pad, right_pad] = getPadWidths(i);
    // Filter out non-padded dimension
    if (left_pad->isZeroInt() && right_pad->isZeroInt()) {
      continue;
    }
    padded_axes.push_back(i);
  }
  return padded_axes;
}

std::vector<Val*> PadOp::getPadWidths() const {
  return {getPadWidthInputBegin(), getPadWidthInputEnd()};
}

std::pair<Val*, Val*> PadOp::getPadWidths(int64_t axis) const {
  auto num_dims = (int64_t)out()->as<TensorView>()->getLogicalDomain().size();
  axis = wrapDim(axis, num_dims);

  int64_t offset_even = (int64_t)axis * 2;
  int64_t offset_odd = offset_even + 1;
  return std::make_pair(
      (*(getPadWidthInputBegin() + offset_even))->as<Val>(),
      (*(getPadWidthInputBegin() + offset_odd))->as<Val>());
}

std::vector<PolymorphicValue> PadOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs.at(0).as<at::Tensor>();

  std::vector<int64_t> pad_widths;
  auto pad_width_offset = getPadWidthInputOffset();
  auto num_dims = in.dim();

  for (auto i = num_dims - 1; i > -1; i--) {
    auto left_pad = (int64_t)inputs.at(pad_width_offset + 2 * i);
    auto right_pad = (int64_t)inputs.at(pad_width_offset + 2 * i + 1);
    pad_widths.push_back(left_pad);
    pad_widths.push_back(right_pad);
  }

  if (isComplexType(*out()->getDataType())) {
    std::complex<double> value =
        static_cast<std::complex<double>>(inputs.at(1));
    auto real = at::real(in);
    auto imag = at::imag(in);
    auto padded_real = at::pad(real, pad_widths, "constant", value.real());
    auto padded_imag = at::pad(imag, pad_widths, "constant", value.imag());
    return {at::complex(padded_real, padded_imag)};
  } else {
    double value = static_cast<double>(inputs.at(1));
    return {at::pad(in, pad_widths, "constant", value)};
  }
}

SliceOp::SliceOp(
    IrBuilderPasskey passkey,
    TensorView* out,
    TensorView* inp,
    const std::vector<Slice>& ranges)
    : Expr(passkey) {
  size_t ndims = TensorDomain::noReductions(inp->getLogicalDomain()).size();
  NVF_ERROR(
      ndims == ranges.size(),
      "The range vector must have the same number of Slice descriptors. "
      "Given: ",
      ranges.size(),
      ", Expected: ",
      ndims);

  addOutput(out);
  addInput(inp);
  for (const auto& range : ranges) {
    NVF_ERROR(range.start != nullptr, "nullptr not allowed");
    NVF_ERROR(range.stop != nullptr, "nullptr not allowed");
    NVF_ERROR(range.step != nullptr, "nullptr not allowed");
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<Slice> SliceOp::getRanges() const {
  const auto num_range_vals =
      std::distance(getRangeInputBegin(), getRangeInputEnd());
  NVF_ERROR(
      num_range_vals % 3 == 0,
      "Unexpected number of range vals: ",
      num_range_vals);
  auto ndims = num_range_vals / 3;
  std::vector<Slice> ranges(ndims);
  auto range_val_it = getRangeInputBegin();
  for (const auto i : arange(ndims)) {
    ranges.at(i) = Slice{
        .start = *range_val_it,
        .stop = *(range_val_it + 1),
        .step = *(range_val_it + 2)};
    range_val_it += 3;
  }
  return ranges;
}

std::vector<PolymorphicValue> SliceOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs.at(0).as<at::Tensor>();
  std::vector<at::indexing::TensorIndex> ranges;
  auto ranges_offset = getRangeInputOffset();
  auto num_dims = in.dim();
  for (const auto i : arange(num_dims)) {
    auto start = (int64_t)inputs.at(ranges_offset + 3 * i);
    auto stop = (int64_t)inputs.at(ranges_offset + 3 * i + 1);
    auto step = (int64_t)inputs.at(ranges_offset + 3 * i + 2);
    ranges.emplace_back(at::indexing::Slice(start, stop, step));
  }
  return {in.index(ranges)};
}

CatOp::CatOp(
    IrBuilderPasskey passkey,
    Val* out,
    const std::vector<Val*>& inputs,
    int64_t concatenated_dim)
    : Expr(passkey) {
  addOutput(out);
  for (auto inp : inputs) {
    addInput(inp);
  }
  NVF_ERROR(
      concatenated_dim >= 0 &&
          concatenated_dim <
              static_cast<int64_t>(
                  ir_utils::getTv(out)->getLogicalDomain().size()),
      "Invalid dimension to concatenate: ",
      concatenated_dim);

  addDataAttribute(concatenated_dim);
}

CatOp::CatOp(
    IrBuilderPasskey passkey,
    Val* out,
    const std::vector<Val*>& inputs,
    int64_t concatenated_dim,
    Val* concatenated_domain_index,
    const std::vector<Val*>& preds)
    : Expr(passkey) {
  NVF_ERROR(
      passkey.ir_container_ != nullptr,
      "IrContainer must be provided to create a CatOp.");
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "Should only be used for Kernel container.");

  addOutput(out);
  for (auto inp : inputs) {
    addInput(inp);
  }
  addDataAttribute(concatenated_dim);
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
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

Val* CatOp::getConcatenatedDomainIndex() const {
  NVF_ERROR(
      container()->isA<kir::Kernel>(),
      "Should only be used for Kernel container.");
  NVF_ERROR(!attributes().empty(), "No attribute found");
  NVF_ERROR(attribute(1) != nullptr, "nulllptr attribute is invalid");
  auto idx = attribute(1)->as<Val>();
  return idx;
}

Val* CatOp::getPred(int input_idx) const {
  NVF_ERROR(
      container()->isA<kir::Kernel>(),
      "Should only be used for Kernel container.");
  const auto num_input_tensors = static_cast<int64_t>(inputs().size());
  NVF_ERROR(input_idx < num_input_tensors, "Invalid input index: ", input_idx);
  const auto attr_idx = input_idx + 2;
  NVF_ERROR(
      attr_idx < static_cast<int64_t>(attributes().size()),
      "Invalid attribute index: ",
      attr_idx,
      ", number of attributes: ",
      attributes().size());
  auto attr = attributeVal(attr_idx);
  NVF_ERROR(attr != nullptr, "nullptr attribute is invalid");
  NVF_ERROR(
      attr->dtype() == DataType::Bool,
      "Attribute must be a Bool val: ",
      attr->toInlineString());
  auto pred = attr;
  return pred;
}

std::vector<PolymorphicValue> CatOp::evaluate(
    const ExpressionEvaluator& ee,
    std::unordered_map<const Val*, PolymorphicValue>& known_values) const {
  // CatOp is preceded by a PadOp internally.
  // For ATen evaluation, directly compute the unpadded inputs.
  std::vector<at::Tensor> unpadded_inputs;
  unpadded_inputs.reserve(inputs().size());
  int64_t concat_dim = concatenatedDim();
  for (Val* inp : inputs()) {
    NVF_CHECK(
        inp->definition() != nullptr && inp->definition()->isA<PadOp>(),
        "Expected CatOp to be preceded by a PadOp.");
    auto eval_i = ee.evaluate(inp->definition()->input(0), known_values);
    unpadded_inputs.push_back(eval_i.as<at::Tensor>());
  }
  return {at::cat(unpadded_inputs, concat_dim)};
}

MatmulOp::MatmulOp(IrBuilderPasskey passkey, Val* out, Val* in_a, Val* in_b)
    : Expr(passkey) {
  addOutput(out);
  addInput(in_a);
  addInput(in_b);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MatmulOp)

std::string MatmulOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size + 1) << " = matmul(" << inA()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << inB()->toString() << ")\n";
  return ss.str();
}

std::string MatmulOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

namespace {
// When the contracting dimension is sharded, each device has a partial
// matmul output and is followed by an allreduce. For loop split, this is
// represented as an rfactored reduction. For example, for matmul, the local
// logical domain after the rfactor is: i{DIDx}, i{M}, i{N}, r{K//d}. Unsqueeze
// the rfactored DID axis to correctly bind with the logical domain. See
// tests/python/test_multidevice.py/test_matmul_allreduce_loop_split
int64_t getRFactorDeviceDimensionIndex(const TensorView* tv) {
  // Filter out reduction dimensions so the index to `logical` directly maps to
  // an at::Tensor axis.
  auto logical = TensorDomain::noReductions(tv->getLogicalDomain());
  int64_t rfactor_did_idx = -1;
  for (auto idx : arange(static_cast<int64_t>(logical.size()))) {
    IterDomain* id = logical.at(idx);
    if (id->isRFactorProduct() && id->isDeviceDim()) {
      NVF_ERROR(
          rfactor_did_idx == -1,
          "Expected only 1 rfactored DID iterdomain, found at least 2 in ",
          logical);
      rfactor_did_idx = idx;
    }
  }

  return rfactor_did_idx;
}
} // namespace

std::vector<PolymorphicValue> MatmulOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto a = inputs.at(0).as<at::Tensor>();
  const auto b = inputs.at(1).as<at::Tensor>();

  auto matmul_out = at::matmul(a, b);

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    matmul_out = matmul_out.unsqueeze(rfactor_did_idx);
  }

  const auto& [sizes, strides] = inferShapeOfOutput(out(), ee);
  auto meta_out = at::detail::empty_strided_meta(sizes, strides, a.dtype());

  if (meta_out.is_contiguous()) {
    return {matmul_out};
  }

  auto strided_matmul_out = at::empty_strided(sizes, strides, a.options());
  strided_matmul_out = strided_matmul_out.copy_(matmul_out);
  return {strided_matmul_out};
}

LinearOp::LinearOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in_a,
    Val* in_b,
    Val* bias)
    : Expr(passkey) {
  addOutput(out);
  addInput(in_a);
  addInput(in_b);

  if (bias != nullptr) {
    addInput(bias);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(LinearOp)

std::string LinearOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent(ss, indent_size + 1) << " = linear(" << inA()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << inB()->toString();
  if (hasBias()) {
    indent(ss, indent_size + 1) << ",\n          " << bias()->toString();
  }
  indent(ss, indent_size + 1) << ")\n";
  return ss.str();
}

std::string LinearOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> LinearOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto in = inputs.at(0).as<at::Tensor>();
  auto weight = inputs.at(1).as<at::Tensor>();

  auto squeeze_device_dims = [](at::Tensor& t,
                                int64_t num_device_dims) -> void {
    // Record the initial shape for the error message.
    std::vector<int64_t> shape = t.sizes().vec();
    for ([[maybe_unused]] auto _ : arange(num_device_dims)) {
      NVF_CHECK(
          t.size(0) == 1,
          "When the weight is >2D, expect its preceding dimensions and "
          "the bias's preceding dimensions to "
          "be DID-parallel and therefore size-1: ",
          shape);
      t = t.squeeze(0);
    }
  };

  // The squeezes and unsqueezes are currently required to support a sharded
  // linear layer. Remove them after #2563.
  auto num_device_dims = weight.dim() - 2;
  squeeze_device_dims(weight, num_device_dims);

  at::Tensor out_tensor;
  if (hasBias()) {
    auto bias = inputs.at(2).as<at::Tensor>();
    squeeze_device_dims(bias, num_device_dims);
    out_tensor = at::linear(in, weight, bias);
  } else {
    out_tensor = at::linear(in, weight);
  }

  for ([[maybe_unused]] auto _ : arange(num_device_dims)) {
    out_tensor = out_tensor.unsqueeze(0);
  }

  // Handle rFactor DIDs similar to MatmulOp::evaluate.
  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    out_tensor = out_tensor.unsqueeze(rfactor_did_idx);
  }

  return {out_tensor};
}

SdpaFwdOp::SdpaFwdOp(
    IrBuilderPasskey passkey,
    TensorView* output,
    TensorView* log_sumexp,
    TensorView* philox_seed,
    TensorView* philox_offset,
    Val* query,
    Val* key,
    Val* value,
    Val* dropout_p,
    Val* is_causal,
    Val* scale)
    : Expr(passkey) {
  addOutput(output);
  addOutput(log_sumexp);
  addOutput(philox_seed);
  addOutput(philox_offset);

  addInput(query);
  addInput(key);
  addInput(value);
  addInput(dropout_p);
  addInput(is_causal);
  if (scale != nullptr) {
    addInput(scale);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SdpaFwdOp)

std::string SdpaFwdOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << attn_out()->toString() << ",\n";
  indent(ss, indent_size) << logsumexp()->toString() << ",\n";
  indent(ss, indent_size) << philox_seed()->toString() << ",\n";
  indent(ss, indent_size) << philox_offset()->toString() << "\n";
  indent(ss, indent_size + 1) << " = sdpa(" << query()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << key()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << value()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          dropout_p = " << dropout_p()->toInlineString() << ",\n";
  indent(ss, indent_size + 1)
      << "          is_causal = " << is_causal()->toInlineString();
  if (scale() != nullptr) {
    indent(ss, indent_size + 1)
        << ",\n          scale = " << scale()->toInlineString();
  }
  indent(ss, indent_size + 1) << ")\n";
  return ss.str();
}

std::string SdpaFwdOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> SdpaFwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  auto query = inputs.at(0).as<at::Tensor>();
  auto key = inputs.at(1).as<at::Tensor>();
  auto value = inputs.at(2).as<at::Tensor>();

  const auto dropout_p = inputs.at(3).as<double>();
  const auto is_causal = inputs.at(4).as<bool>();

  // Temporary handling of DID parallelization see
  // https://github.com/NVIDIA/Fuser/issues/2563
  bool handle_device_dim = false;
  if (query.dim() == 5) {
    handle_device_dim = true;

    NVF_CHECK(key.dim() == 5 && value.dim() == 5);

    auto query_domain =
        TensorDomain::noReductions(this->query()->getLogicalDomain());
    auto key_domain =
        TensorDomain::noReductions(this->key()->getLogicalDomain());
    auto value_domain =
        TensorDomain::noReductions(this->value()->getLogicalDomain());
    NVF_CHECK(
        query_domain.front()->isDeviceDim(),
        "Only support DID parallelization on outermost axis");
    NVF_CHECK(
        key_domain.front()->isDeviceDim(),
        "Only support DID parallelization on outermost axis");
    NVF_CHECK(
        value_domain.front()->isDeviceDim(),
        "Only support DID parallelization on outermost axis");

    query = query.squeeze(0);
    key = key.squeeze(0);
    value = value.squeeze(0);
  }

  // Flash attention requires the last dimension to be padded to 8.
  // https://github.com/pytorch/pytorch/blob/c27882ffa8c1c7e4cf8ebc6c2f879e5b6c8814ad/aten/src/ATen/native/transformers/attention.cpp#L675-L677
  const auto last_dim_size = query.size(-1);
  auto pad_last_dim = [last_dim_size](
                          at::Tensor inp, int alignment_size) -> at::Tensor {
    if (last_dim_size % alignment_size == 0) {
      return inp;
    }
    auto pad_count = alignment_size - (last_dim_size % alignment_size);
    auto padded_inp = at::pad(inp, {0, pad_count});
    return padded_inp;
  };

  query = pad_last_dim(query, 8);
  key = pad_last_dim(key, 8);
  value = pad_last_dim(value, 8);

  // Conmpute scale using original size of last dimension
  double scale = inputs.size() > 5 ? inputs.back().as<double>()
                                   : 1.0 / std::sqrt(last_dim_size);

  // ATen reference:
  // https://github.com/pytorch/pytorch/blob/c27882ffa8c1c7e4cf8ebc6c2f879e5b6c8814ad/aten/src/ATen/native/transformers/attention.cpp#L680-L681
  auto
      [output,
       log_sumexp,
       cum_seq_q,
       cum_seq_k,
       query_seq_len,
       key_seq_len,
       philox_seed,
       philox_offset,
       debug_attn_mask] =
          at::_scaled_dot_product_flash_attention(
              query,
              key,
              value,
              dropout_p,
              is_causal,
              /*return_debug_mask=*/false,
              scale);

  // If the inputs were padded, slice the output to restore the original
  // size
  if (output.size(-1) != last_dim_size) {
    output = output.slice(-1, 0, last_dim_size);
  }

  // Add back the device dim axis for output.
  if (handle_device_dim) {
    output = output.unsqueeze(0);
    log_sumexp = log_sumexp.unsqueeze(0);
  }

  // We ignore cum_seq_q/k outputs since they are undefined tensors for
  // non-nested tensors. We do not store query/key_seq_len since they can be
  // computed in non-nested tensor directly. debug_attn_mask is ignored
  // since `return_debug_mask=false`.
  return {output, log_sumexp, philox_seed, philox_offset};
}

std::string Scope::toString(int indent_size) const {
  std::stringstream ss;
  for (auto expr : exprs()) {
    ss << expr->toString(indent_size);
  }
  return ss.str();
}

std::vector<Expr*>::iterator Scope::insert(
    std::vector<Expr*>::const_iterator pos,
    Expr* expr) {
  return exprs_.insert(pos, expr);
}

std::vector<Expr*>::iterator Scope::insert_before(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  NVF_ERROR(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " before the reference: ",
      ref,
      " @ ",
      (size_t)ref,
      " however the reference was not found in this scope.");
  return insert(it, expr);
}

std::vector<Expr*>::iterator Scope::insert_after(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  NVF_ERROR(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " after the reference: ",
      ref,
      " however the reference was not found in this scope.");
  return insert(it + 1, expr);
}

std::vector<Expr*>::iterator Scope::insert(size_t pos, Expr* expr) {
  const auto it = exprs_.begin() + (std::ptrdiff_t)pos;
  return insert(it, expr);
}

void Scope::erase(std::vector<Expr*>::const_iterator pos) {
  // Remove the scope of the expr if this is the scope
  [[maybe_unused]] auto expr = *pos;
  exprs_.erase(pos);
}

void Scope::erase(Expr* ref) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  if (it != exprs_.end()) {
    erase(it);
  }
}

void Scope::erase(size_t pos) {
  erase(exprs_.begin() + (std::ptrdiff_t)pos);
}

bool Scope::contains(Expr* expr) const {
  const auto it = std::find(exprs_.begin(), exprs_.end(), expr);
  return it != exprs_.end();
}

void Scope::clear() {
  exprs_.clear();
}

ForLoop::ForLoop(
    IrBuilderPasskey passkey,
    IterDomain* iter_domain,
    Val* index,
    Val* start,
    Val* stop,
    Val* step,
    bool vectorize,
    Val* vectorize_shift,
    bool unroll_required,
    CircularBufferLoopStage circular_buffer_loop_stage,
    int64_t circular_buffer_loop_stage_depth)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>() ||
          passkey.ir_container_->isA<hir::HostIrContainer>(),
      "IR type only valid for Kernel or Host container.");
  NVF_ERROR(isIntegralType(index->dtype()));
  addInput(index);
  addInput(iter_domain);
  if (start == nullptr && iter_domain->isThread()) {
    start = NamedScalar::getParallelIndex(iter_domain->getParallelType());
  }
  if (step == nullptr) {
    if (iter_domain->isThread()) {
      step = NamedScalar::getParallelDim(iter_domain->getParallelType());
    } else {
      step = FusionGuard::getCurFusion()->oneVal();
    }
  }
  NVF_ERROR(
      index->dtype() == DataType::Index, "Loop index must be an index type.");
  NVF_ERROR(
      start == nullptr || start->dtype() == DataType::Index,
      "Loop start must be an index type.");
  NVF_ERROR(
      step->dtype() == DataType::Index, "Loop step must be an index type.");
  NVF_ERROR(
      stop == nullptr || stop->dtype() == DataType::Index,
      "Loop stop must be an index type.");
  addAttribute(start);
  addAttribute(stop);
  addAttribute(step);
  addDataAttribute(vectorize);
  addAttribute(vectorize_shift);
  addDataAttribute(unroll_required);
  addDataAttribute(circular_buffer_loop_stage);
  addDataAttribute(circular_buffer_loop_stage_depth);
  // Storing IR nodes as Attribute is not safe with IrCloner, but
  // fortunately kernel IR does not need this feature.
  addDataAttribute(Scope(this));
}

ForLoop::ForLoop(
    IrBuilderPasskey passkey,
    IterDomain* iter_domain,
    Val* index,
    CircularBufferLoopStage circular_buffer_loop_stage,
    int64_t circular_buffer_loop_stage_depth)
    : ForLoop(
          passkey,
          iter_domain,
          index,
          nullptr,
          nullptr,
          nullptr,
          !iter_domain->isBroadcast() &&
              isParallelTypeVectorize(iter_domain->getParallelType()),
          nullptr,
          false,
          circular_buffer_loop_stage,
          circular_buffer_loop_stage_depth) {}

ForLoop::ForLoop(IrBuilderPasskey passkey, IterDomain* iter_domain)
    : ForLoop(
          passkey,
          iter_domain,
          GpuLower::current()->getLoopIndexVariable(iter_domain),
          CircularBufferLoopStage::NotApplicable,
          0) {}

ForLoop::ForLoop(IrBuilderPasskey passkey, const ForLoop* other)
    : ForLoop(
          passkey,
          other->iter_domain(),
          other->index(),
          other->start(),
          other->stop(),
          other->step(),
          other->vectorize(),
          other->vectorize_shift(),
          other->isUnrollRequired(),
          other->circularBufferLoopStage(),
          other->circularBufferLoopStageDepth()) {}

std::string ForLoop::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "FOR " << index()->toString() << " in "
                          << iter_domain()->toString() << ":\n"
                          << body().toString(indent_size + 1);
  return ss.str();
}

std::string ForLoop::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

bool ForLoop::isUnrollable() const {
  // Start and stop must be constant, must not be a broadcast
  // dimension, cannot be bound to a parallel dimension, must not be
  // vectorized.
  return start()->isConstScalar() && stop()->isConstScalar() &&
      !iter_domain()->isThread() && !iter_domain()->isDeviceDim() &&
      !iter_domain()->isBroadcast() && !vectorize();
}

bool ForLoop::isUnrolled() const {
  if (isUnrollRequired() && !isUnrollable()) {
    // Broadcast and vectorized loops are not generated and do not
    // matter if unrolled or not.
    if (!iter_domain()->isBroadcast() && !vectorize()) {
      TORCH_WARN(
          "Unroll required but not possible. Register allocation disabled. "
          "Loop index: ",
          index()->toString(),
          ", ",
          toString());
    }
    return false;
  }

  // Size-one loop will not be materialized as a loop, so return false
  if (start()->isZeroInt() && stop()->isOneInt()) {
    return false;
  }

  // Unroll if required.
  if (isUnrollRequired()) {
    return true;
  }

  // Don't unroll if not possible
  if (!isUnrollable()) {
    return false;
  }

  // Unrolling is technically possible but avoided
  if (iter_domain()->getParallelType() == ParallelType::Unswitch) {
    // Use ParallelType::Unroll if unrolling is desired. Note that
    // unswitched size-one loops are not unrolled as they are not
    // materialized as actual for-loops.
    return false;
  }

  if (hasRuntimeReductionFunctions()) {
    return false;
  }

  return true;
}

Val* ForLoop::start() const {
  if (attributeVal(0) != nullptr) {
    return attributeVal(0);
  } else {
    // clang-tidy complains without this
    NVF_ERROR(iter_domain() != nullptr);
    return iter_domain()->start();
  }
}

Val* ForLoop::stop() const {
  if (attributeVal(1) != nullptr) {
    return attributeVal(1);
  } else {
    // clang-tidy complains without this
    NVF_ERROR(iter_domain() != nullptr);
    return iter_domain()->extent();
  }
}

Val* ForLoop::step() const {
  NVF_ERROR(attributeVal(2) != nullptr);
  return attributeVal(2);
}

Val* ForLoop::simplifiedStop() const {
  if (simplified_stop_ == nullptr) {
    simplified_stop_ = GpuLower::hasCurrent()
        ? GpuLower::current()->commonScalarMap().hoistScalar(stop(), {})
        : stop();
  }
  return simplified_stop_;
}

bool ForLoop::isTrivial() const {
  // These loops are not materialized
  if (vectorize() || iter_domain()->isBroadcast() ||
      iter_domain()->isStride() || iter_domain()->isMma() ||
      iter_domain()->isBulk() || iter_domain()->isDeviceDim()) {
    return true;
  }

  if (index()->isConstScalar() || index()->definition() != nullptr) {
    return true;
  }

  // By default, a parallelized loop would look like:
  //
  //   for (int x = threadIdx.x; x < stop; x += blockDim.x) {
  //     do_some_comp(x);
  //   }
  //
  // When stop is guaranteed to be smaller or equal to the number of
  // threads, the for-loop is not necessary. In the above case, we
  // would just generate the loop body without the for clause but
  // references to the loop index replaced by the loop start value.
  //
  // When the loop end is the same as the IterDomain extent, the
  // assumption can be safely made. This is more conservative than
  // necessary since the loop stop value just needs to be <= the
  // IterDomain extent. However, at this point, this conservative
  // analysis seems sufficient.
  if (stop() == iter_domain()->extent() && iter_domain()->isThread()) {
    return true;
  }

  // Extent-1 loop: for (int i = 0; i < 1; ++i) {
  if (start()->isZeroInt() && simplifiedStop()->isOneInt() &&
      step()->isOneInt()) {
    return true;
  }

  // Another extent-1 loop: for (int i = N - 1; i < N; ++i) {
  if (start()->definition() != nullptr &&
      start()->definition()->isA<BinaryOp>() &&
      start()->definition()->as<BinaryOp>()->getBinaryOpType() ==
          BinaryOpType::Sub &&
      start()->definition()->as<BinaryOp>()->lhs() == stop() &&
      start()->definition()->as<BinaryOp>()->rhs()->isOneInt()) {
    return true;
  }

  if (start()->isConstScalar() && simplifiedStop()->isConstScalar() &&
      start()->evaluate().as<int64_t>() + 1 ==
          simplifiedStop()->evaluate().as<int64_t>() &&
      step()->isOneInt()) {
    return true;
  }

  return false;
}

namespace {

//! A utility class to check if an expression of a particular type exists
class ExprFinder : kir::ConstIrVisitor {
 public:
  //! True if expr or any of its nested expressions is a type included in
  //! expr_types
  static bool exists(
      const Expr* expr,
      const std::unordered_set<std::type_index>& expr_types) {
    ExprFinder finder(expr_types);
    finder.handle(std::vector<const Expr*>{expr});
    return finder.is_found_;
  }

 private:
  ExprFinder(const std::unordered_set<std::type_index>& expr_types)
      : expr_types_(expr_types) {}

  using kir::ConstIrVisitor::handle;

  void dispatch(const Expr* expr) final {
    if (expr_types_.find(typeid(*expr)) != expr_types_.end()) {
      is_found_ = true;
      return;
    }
    kir::ConstIrVisitor::dispatch(expr);
  }

 private:
  const std::unordered_set<std::type_index>& expr_types_;
  bool is_found_ = false;
};

} // namespace

bool ForLoop::isGroup() const {
  //! True if loop is grouped. The IterDomain of the loop must have
  //! ParallelType::Group, but it isn't sufficient as the loop may be
  //! for an initialization expression, for which the loop shold not
  //! be grouped. Make sure a GroupedGridReduction is found.
  if (iter_domain()->getParallelType() != ParallelType::Group) {
    return false;
  }

  return ExprFinder::exists(
      this,
      {typeid(GroupedReductionOp),
       typeid(kir::GroupedGridReduction),
       typeid(kir::GroupedGridWelford)});
}

namespace {

//! A utility class to check if runtime reduction exists
class RuntimeReductionFinder : kir::ConstIrVisitor {
 public:
  static bool exists(const Expr* expr) {
    NVF_CHECK(expr->container()->isA<kir::Kernel>());
    RuntimeReductionFinder finder;
    finder.handle(std::vector<const Expr*>{expr});
    return finder.is_found_;
  }

 private:
  using kir::ConstIrVisitor::handle;

  void dispatch(const Expr* expr) final {
    if (expr->isA<ReductionOp>() || expr->isA<WelfordOp>() ||
        expr->isA<kir::GridReduction>() ||
        expr->isA<kir::GroupedGridReduction>() ||
        expr->isA<kir::GridWelford>() || expr->isA<kir::GroupedGridWelford>() ||
        expr->isA<GroupedReductionOp>()) {
      is_found_ = true;
      return;
    }
    kir::ConstIrVisitor::dispatch(expr);
  }

 private:
  bool is_found_ = false;
};

std::optional<IterDomain*> returnFirstIfRankThree(const TensorView* tv) {
  const auto& logical_domain =
      TensorDomain::noReductions(tv->getLogicalDomain());
  if (logical_domain.size() == 3) {
    return logical_domain.at(0);
  } else {
    return std::nullopt;
  }
}
} // namespace

bool ForLoop::hasRuntimeReductionFunctions() const {
  return RuntimeReductionFinder::exists(this);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ForLoop)

SdpaBwdOp::SdpaBwdOp(
    IrBuilderPasskey passkey,
    TensorView* grad_query,
    TensorView* grad_key,
    TensorView* grad_value,
    TensorView* grad_output,
    TensorView* query,
    TensorView* key,
    TensorView* value,
    TensorView* output,
    TensorView* log_sumexp,
    Val* dropout_p,
    Val* is_causal,
    TensorView* philox_seed,
    TensorView* philox_offset,
    Val* scale)
    : Expr(passkey) {
  addOutput(grad_query);
  addOutput(grad_key);
  addOutput(grad_value);
  addInput(grad_output);
  addInput(query);
  addInput(key);
  addInput(value);
  addInput(output);
  addInput(log_sumexp);
  addInput(dropout_p);
  addInput(is_causal);
  addInput(philox_seed);
  addInput(philox_offset);
  if (scale != nullptr) {
    addInput(scale);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SdpaBwdOp)

std::string SdpaBwdOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << grad_query()->toString() << ",\n";
  indent(ss, indent_size) << grad_key()->toString() << ",\n";
  indent(ss, indent_size) << grad_value()->toString() << "\n";
  indent(ss, indent_size + 1)
      << " = sdpa_bwd(" << grad_attn()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << query()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << key()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << value()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          " << attn_out()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          logsum_exp = " << logsumexp()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          dropout_p = " << dropout_p()->toInlineString() << ",\n";
  indent(ss, indent_size + 1)
      << "          is_causal = " << is_causal()->toInlineString() << ",\n";
  indent(ss, indent_size + 1)
      << "          philox_seed = " << philox_seed()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          philox_offset = " << philox_offset()->toString() << ",\n";
  if (scale() != nullptr) {
    indent(ss, indent_size + 1)
        << ",\n          scale = " << scale()->toInlineString();
  }
  indent(ss, indent_size + 1) << ")\n";
  return ss.str();
}

std::string SdpaBwdOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> SdpaBwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // Backward tensor inputs: grad_input, query, key, value, output,
  // logsumexp, max_q/k Temporary handling of DID parallelization. See
  // https://github.com/NVIDIA/Fuser/issues/2563
  auto query_domain =
      TensorDomain::noReductions(this->query()->getLogicalDomain());
  bool first_dim_is_did = query_domain.front()->isDeviceDim();
  auto out_grad = inputs[0].as<at::Tensor>();
  if (first_dim_is_did) {
    NVF_CHECK(out_grad.dim() == 5, "Expected 5D but found ", out_grad.sizes());
  } else {
    NVF_CHECK(out_grad.dim() == 4, "Expected 4D but found ", out_grad.sizes());
  }

  std::vector<at::Tensor> bwd_inputs;
  for (auto idx : arange(6)) {
    auto in_tensor = inputs.at(idx).as<at::Tensor>();
    // Removing the size 1 from sharded axis from tensors.
    if (first_dim_is_did) {
      in_tensor = in_tensor.squeeze(0);
    }
    bwd_inputs.push_back(in_tensor);
  }
  const auto dropout_p = inputs.at(6).as<double>();
  const auto is_causal = inputs.at(7).as<bool>();
  const auto philox_seed = inputs.at(8).as<at::Tensor>();
  const auto philox_offset = inputs.at(9).as<at::Tensor>();

  // Flash attention requires the last dimension to be padded to 8.
  // https://github.com/pytorch/pytorch/blob/c27882ffa8c1c7e4cf8ebc6c2f879e5b6c8814ad/aten/src/ATen/native/transformers/attention.cpp#L675-L677
  const auto last_dim_size = bwd_inputs[0].size(-1);
  auto pad_last_dim = [last_dim_size](
                          at::Tensor inp, int alignment_size) -> at::Tensor {
    if (last_dim_size % alignment_size == 0) {
      return inp;
    }
    auto pad_count = alignment_size - (last_dim_size % alignment_size);
    auto padded_inp = at::pad(inp, {0, pad_count});
    return padded_inp;
  };

  // Conmpute scale using original size of last dimension
  double scale = inputs.size() > 10 ? inputs.back().as<double>()
                                    : 1.0 / std::sqrt(last_dim_size);

  // ATen reference:
  // https://github.com/pytorch/pytorch/blob/c27882ffa8c1c7e4cf8ebc6c2f879e5b6c8814ad/aten/src/ATen/native/transformers/attention.cpp#L680-L681
  // cum_seq_q/k are undefined tensors for non-nested input tensors.
  auto [grad_query, grad_key, grad_value] =
      at::_scaled_dot_product_flash_attention_backward(
          /*grad_output=*/pad_last_dim(bwd_inputs[0], 8),
          /*query=*/pad_last_dim(bwd_inputs[1], 8),
          /*key=*/pad_last_dim(bwd_inputs[2], 8),
          /*value=*/pad_last_dim(bwd_inputs[3], 8),
          /*output=*/pad_last_dim(bwd_inputs[4], 8),
          /*logsumexp=*/bwd_inputs[5],
          /*cum_seq_q=*/at::Tensor(),
          /*cum_seq_k=*/at::Tensor(),
          // Note: ATen implementation expects max_q/max_k as scalars.
          /*max_q=*/bwd_inputs[1].size(2),
          /*max_k=*/bwd_inputs[2].size(2),
          /*dropout_p=*/dropout_p,
          /*is_causal=*/is_causal,
          /*philox_seed=*/philox_seed,
          /*philox_offset=*/philox_offset,
          /*scale=*/scale);

  // If the inputs were padded, slice the gradsto restore the original size
  auto slice_last_dim = [last_dim_size](at::Tensor output) -> at::Tensor {
    if (output.size(-1) != last_dim_size) {
      return output;
    }
    return output.slice(-1, 0, last_dim_size);
  };

  // Add device dimension back to outputs.
  if (first_dim_is_did) {
    grad_query = grad_query.unsqueeze(0);
    grad_key = grad_key.unsqueeze(0);
    grad_value = grad_value.unsqueeze(0);
  }

  return {
      slice_last_dim(grad_query),
      slice_last_dim(grad_key),
      slice_last_dim(grad_value)};
}

EmbeddingFwdOp::EmbeddingFwdOp(
    IrBuilderPasskey passkey,
    TensorView* output,
    TensorView* input,
    TensorView* weight,
    Val* padding_idx,
    Val* max_norm,
    Val* norm_type,
    Val* scale_grad_by_freq,
    Val* sparse)
    : Expr(passkey) {
  addOutput(output);

  addInput(input);
  addInput(weight);
  addInput(norm_type);
  addInput(scale_grad_by_freq);
  addInput(sparse);
  if (padding_idx != nullptr) {
    addInput(padding_idx);
    addDataAttribute(true);
  } else {
    addDataAttribute(false);
  }
  if (max_norm != nullptr) {
    addInput(max_norm);
    addDataAttribute(true);
  } else {
    addDataAttribute(false);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(EmbeddingFwdOp)

std::string EmbeddingFwdOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << ",\n";
  indent(ss, indent_size + 1) << " = embedding(" << in()->toString() << ",\n";
  indent(ss, indent_size + 1) << "          " << weight()->toString() << ",\n";
  if (padding_idx() != nullptr) {
    indent(ss, indent_size + 1)
        << "          padding_idx = " << padding_idx()->toString() << ",\n";
  }
  if (max_norm() != nullptr) {
    indent(ss, indent_size + 1)
        << "          max_norm = " << max_norm()->toString() << ",\n";
  }
  indent(ss, indent_size + 1)
      << "          norm_type = " << norm_type()->toString() << ",\n";
  indent(ss, indent_size + 1)
      << "          scale_grad_by_freq = "
      << scale_grad_by_freq()->toInlineString() << ",\n";
  indent(ss, indent_size + 1)
      << "          sparse = " << sparse()->toInlineString() << ")\n";
  return ss.str();
}

std::string EmbeddingFwdOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> EmbeddingFwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  auto input = inputs.at(0).as<at::Tensor>();
  auto weight = inputs.at(1).as<at::Tensor>();
  auto norm_type = inputs.at(2).as<double>();
  auto scale_grad_by_freq = inputs.at(3).as<bool>();
  auto sparse = inputs.at(4).as<bool>();
  std::optional<int64_t> padding_idx = std::nullopt;
  if (has_padding_idx()) {
    padding_idx = inputs.at(5).as<int64_t>();
  }
  std::optional<double> max_norm = std::nullopt;
  if (has_max_norm()) {
    auto idx = 5 + has_padding_idx();
    max_norm = inputs.at(idx).as<double>();
  }

  namespace F = torch::nn::functional;
  return {F::embedding(
      input,
      weight,
      F::EmbeddingFuncOptions()
          .padding_idx(padding_idx)
          .max_norm(max_norm)
          .norm_type(norm_type)
          .scale_grad_by_freq(scale_grad_by_freq)
          .sparse(sparse))};
}

ArgsortOp::ArgsortOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* in,
    int64_t dim,
    bool descending,
    bool stable)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addDataAttribute(dim);
  addDataAttribute(descending);
  addDataAttribute(stable);
}

std::string ArgsortOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = argsort( "
                          << in()->toString() << ", dim = " << dim()
                          << ", descending = "
                          << (isDescending() ? "True" : "False")
                          << ", stable = " << (isStable() ? "True" : "False")
                          << " )\n";
  return ss.str();
}

std::string ArgsortOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ArgsortOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      inputs.size() == 1,
      "ArgsortOp expects 1 input but received ",
      inputs.size());

  const auto& in = inputs[0];
  NVF_ERROR(
      in.is<at::Tensor>(),
      "ArgsortOp expects tensor input but got ",
      in.type().name());

  // at::argsort signature is:
  // Tensor argsort(const Tensor &self, bool stable, int64_t dim, bool
  // descending)
  auto result =
      at::argsort(in.as<at::Tensor>(), isStable(), dim(), isDescending());

  return {result};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ArgsortOp)

TopKOp::TopKOp(
    IrBuilderPasskey passkey,
    Val* out_values,
    Val* out_indices,
    Val* in,
    Val* k,
    int64_t dim,
    bool largest,
    bool sorted)
    : Expr(passkey) {
  addOutput(out_values);
  addOutput(out_indices);
  addInput(in);
  addInput(k);
  addDataAttribute(dim);
  addDataAttribute(largest);
  addDataAttribute(sorted);
}

std::string TopKOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "( " << outValues()->toString() << ", "
                          << outIndices()->toString() << " ) = topk( "
                          << in()->toString() << ", " << k()->toString()
                          << ", dim = " << dim()
                          << ", largest = " << (isLargest() ? "True" : "False")
                          << ", sorted = " << (isSorted() ? "True" : "False")
                          << " )\n";
  return ss.str();
}

std::string TopKOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> TopKOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs[0];
  NVF_ERROR(
      in.is<at::Tensor>(),
      "TopKOp expects tensor input at position 0 but got ",
      in.type().name());

  const auto& k = inputs[1];
  NVF_ERROR(
      k.is<int64_t>(),
      "TopKOp expects int64_t input at position 1 as k but got ",
      k.type().name());

  // at::topk signature is:
  // std::tuple<Tensor, Tensor> topk(const Tensor &self, int64_t k, int64_t dim,
  // bool largest, bool sorted)
  auto result = at::topk(
      in.as<at::Tensor>(), k.as<int64_t>(), dim(), isLargest(), isSorted());

  // at::topk returns a tuple of (values, indices)
  return {std::get<0>(result), std::get<1>(result)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(TopKOp)

GroupedMmaOp::GroupedMmaOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* mat1,
    Val* mat2,
    Val* offsets,
    Val* scale1,
    Val* scale2)
    : Expr(passkey) {
  NVF_ERROR(
      out->getValType().value() == ValType::TensorView,
      "Output must be a TensorView");
  NVF_ERROR(
      mat1->getValType().value() == ValType::TensorView,
      "First input must be a TensorView");
  NVF_ERROR(
      mat2->getValType().value() == ValType::TensorView,
      "Second input must be a TensorView");
  NVF_ERROR(
      offsets->getValType().value() == ValType::TensorView,
      "Offsets must be a TensorView");
  addOutput(out);
  addInput(mat1);
  addInput(mat2);
  addInput(offsets);

  bool has_scale1 = scale1 != nullptr;
  if (has_scale1) {
    NVF_CHECK(
        scale1->getValType().value() == ValType::TensorView,
        "Scale1 must be a TensorView");
    NVF_CHECK(
        scale2->getValType().value() == ValType::TensorView,
        "Scale2 must be a TensorView");
    addInput(scale1);
    addInput(scale2);
  }
  addDataAttribute(has_scale1);
}

std::string GroupedMmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = GroupedMmaOp("
                          << "mat1=" << mat1()->toString() << ", "
                          << "mat2=" << mat2()->toString() << ", "
                          << "offsets=" << offsets()->toString();
  if (hasScale()) {
    ss << ", "
       << "scale1=" << scale1()->toString() << ", "
       << "scale2=" << scale2()->toString();
  }
  ss << ")\n";
  return ss.str();
}

std::string GroupedMmaOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> GroupedMmaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(
      (inputs.size() == 3 && !hasScale()) || (inputs.size() == 5 && hasScale()),
      "GroupedMmaOp expects 3 or 5 inputs but received ",
      inputs.size(),
      " with scale flag: ",
      hasScale() ? "true" : "false");

  const auto& mat1 = inputs[0];
  const auto& mat2 = inputs[1];
  const auto& offsets = inputs[2];

  NVF_ERROR(
      mat1.is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 0 but got ",
      mat1.type().name());

  NVF_ERROR(
      mat2.is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 1 but got ",
      mat2.type().name());

  NVF_ERROR(
      offsets.is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 2 but got ",
      offsets.type().name());

  at::Tensor result;
  if (!hasScale()) {
    result = at::_grouped_mm(
        mat1.as<at::Tensor>(), mat2.as<at::Tensor>(), offsets.as<at::Tensor>());
  } else {
    const auto& scale1 = inputs[3];
    const auto& scale2 = inputs[4];
    NVF_ERROR(
        scale1.is<at::Tensor>(),
        "GroupedMmaOp expects tensor input at position 3 but got ",
        scale1.type().name());
    NVF_ERROR(
        scale2.is<at::Tensor>(),
        "GroupedMmaOp expects tensor input at position 4 but got ",
        scale2.type().name());
    // TODO: at::_scaled_grouped_mm has requirements on mat1 and mat2's memory
    // layout, as well as a different interpretation on broadcast scales. We
    // need to shoe horn it in

    // mat2 needs to be strided to have k dimension as the fastest dimension;
    auto mat1_contiguous = mat1.as<at::Tensor>().contiguous();
    auto mat2_k_last = mat2.as<at::Tensor>().transpose(1, 2).contiguous().transpose(1, 2);

    auto scale1_tensor = scale1.as<at::Tensor>();
    auto scale2_tensor = scale2.as<at::Tensor>();
    // aten kernel limitation
    NVF_CHECK(
        scale1_tensor.size(-1) == 1 && scale2_tensor.size(-2) == 1,
        "Scale1 and scale2 must have size 1 at the k dimension");
    // scale factor handling
    if (out()->nDims() == 3) {
      // case 1, aten API expects collapsed 1D scale with group dimension on the slower side.
      scale1_tensor = scale1_tensor.reshape(-1);
      scale2_tensor = scale2_tensor.reshape(-1);
    } else {
      // case 2 and 3, aten doesn't allow broadcast on k dimension. squeeze k out.
      scale1_tensor = scale1_tensor.squeeze(-1);
      scale2_tensor = scale2_tensor.squeeze(-2);
    }
    result = at::_scaled_grouped_mm(
        mat1_contiguous, mat2_k_last, scale1_tensor, scale2_tensor, offsets.as<at::Tensor>(), std::nullopt, std::nullopt, at::ScalarType::BFloat16);
    result = result.to(data_type_to_aten(out()->dtype()));
  }
  return {result};
}

IterDomain* GroupedMmaOp::getKIDOfMat1() const {
  // mat1 is [g, m, k] or [m, k]
  const auto& logical_domain =
      TensorDomain::noReductions(mat1()->getLogicalDomain());
  return logical_domain.at(logical_domain.size() - 1);
}

IterDomain* GroupedMmaOp::getKIDOfMat2() const {
  // mat2 is [g, k, n] or [k, n]
  const auto& logical_domain =
      TensorDomain::noReductions(mat2()->getLogicalDomain());
  return logical_domain.at(logical_domain.size() - 1);
}

std::optional<IterDomain*> GroupedMmaOp::getGIDOfMat1() const {
  // mat1 is [g, m, k] or [m, k]
  return returnFirstIfRankThree(mat1());
}

std::optional<IterDomain*> GroupedMmaOp::getGIDOfMat2() const {
  // mat2 is [g, k, n] or [k, n]
  return returnFirstIfRankThree(mat2());
}

std::optional<IterDomain*> GroupedMmaOp::getGIDOfOutput() const {
  // mat2 is [g, k, n] or [k, n]
  return returnFirstIfRankThree(out());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedMmaOp)

} // namespace nvfuser
