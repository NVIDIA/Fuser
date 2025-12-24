// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <algorithm>
#include <complex>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOptions.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/core/SymInt.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/options/embedding.h>

#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <ir/cloner.h>
#include <ir/internal_nodes.h>
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
#if NVFUSER_CUTLASS_KERNEL_ENABLED
#include <nvf_cutlass.h>
#endif

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
      /*indices=*/{inputs.at(1).as<at::Tensor>()},
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
    Val* out,
    Val* self,
    int64_t dim,
    Val* index,
    Val* src,
    bool exact_sizes,
    std::optional<BinaryOpType> accumulate_op)
    : Expr(passkey) {
  addInput(self);
  addInput(index);
  addInput(src);
  addOutput(out);
  addDataAttribute(dim);
  addDataAttribute(exact_sizes);
  // is this accumulate?
  addDataAttribute(accumulate_op.has_value());
  if (accumulate_op.has_value()) {
    addDataAttribute(accumulate_op.value());
  }
}

std::string ScatterOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = scatter(";
  ss << "in = " << in()->toString() << ", dim = " << dim()
     << ", src = " << src()->toString() << ", idx = " << index()->toString();
  if (accumulate()) {
    ss << ", accumulate = " << accumulateOp();
  }
  ss << " )\n";
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
  auto dimension = dim();
  if (accumulate()) {
    // Use at::scatter if the src is scalar since at::scatter_reduce
    // doesn't seem to support scalar src. Note that it seems it's
    // deprecated and only supports add and multiply.
    if (src()->isA<TensorView>()) {
      std::string accumulate_op_str;
      switch (accumulateOp()) {
        case BinaryOpType::Add:
          accumulate_op_str = "sum";
          break;
        case BinaryOpType::Mul:
          accumulate_op_str = "prod";
          break;
        case BinaryOpType::Max:
          accumulate_op_str = "amax";
          break;
        case BinaryOpType::Min:
          accumulate_op_str = "amin";
          break;
        default:
          NVF_THROW("Unsupported accumulation op: ", accumulateOp());
      }
      return {at::scatter_reduce(
          input,
          dimension,
          index,
          inputs.at(2).as<at::Tensor>(),
          accumulate_op_str)};
    } else {
      std::string accumulate_op_str;
      switch (accumulateOp()) {
        case BinaryOpType::Add:
          accumulate_op_str = "add";
          break;
        case BinaryOpType::Mul:
          accumulate_op_str = "multiply";
          break;
        default:
          NVF_THROW("Unsupported accumulation op: ", accumulateOp());
      }
      return {at::scatter(
          input,
          dimension,
          index,
          PolymorphicValue_functions::toScalar(inputs.at(2)),
          accumulate_op_str)};
    }
  } else {
    if (src()->isA<TensorView>()) {
      return {
          at::scatter(input, dimension, index, inputs.at(2).as<at::Tensor>())};
    } else {
      return {at::scatter(
          input,
          dimension,
          index,
          PolymorphicValue_functions::toScalar(inputs.at(2)))};
    }
  }
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
    case UnaryOpType::Ceil:
      return {ceil(in)};
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
    case BinaryOpType::FMax:
      return {fmax(lhs, rhs)};
    case BinaryOpType::Max:
      return {max(lhs, rhs)};
      break;
    case BinaryOpType::FMin:
      return {fmin(lhs, rhs)};
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
      return {where(a, b, c)};
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
  // adding nullptr to attributes triggers assert. Though I question if this
  // should be the default behavior and any use of attributes should check for
  // nullptr instead.
  if (philox_index) {
    addAttribute(philox_index);
  }
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
    case BinaryOpType::FMax: {
      // Emulate fmax/fmin NAN behavior, which removes NANs except in the case
      // where the whole set is NANs.
      auto all_nans = at::all(at::isnan(input), reduction_axes);
      auto removed_nans = at::nan_to_num(
          input, /*nan=*/-std::numeric_limits<double>::infinity());
      return {at::where(
          all_nans,
          std::numeric_limits<double>::quiet_NaN(),
          at::amax(removed_nans, reduction_axes))};
    } break;
    case BinaryOpType::Max:
      return {at::amax(input, reduction_axes)};
      break;
    case BinaryOpType::FMin: {
      auto all_nans = at::all(at::isnan(input), reduction_axes);
      auto removed_nans = at::nan_to_num(
          input, /*nan=*/std::numeric_limits<double>::infinity());
      return {at::where(
          all_nans,
          std::numeric_limits<double>::quiet_NaN(),
          at::amin(removed_nans, reduction_axes))};
    } break;
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
    const std::vector<Val*>& expanded_extents)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  for (auto* expanded_extent : expanded_extents) {
    NVF_ERROR(expanded_extent != nullptr);
    NVF_ERROR_EQ(
        expanded_extent->dtype(), DataType::Index, "Found ", expanded_extent);
    addInput(expanded_extent);
  }
}

std::string ExpandOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = expand( " << in() << " )"
                          << std::endl;
  return ss.str();
}

std::string ExpandOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ExpandOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto& in = inputs.at(0).as<at::Tensor>();
  const auto& [out_shape, _] = inferShapeOfOutput(out(), ee);
  return {in.expand(out_shape)};
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

ReshapeOp::ReshapeOp(IrBuilderPasskey passkey, Val* out, Val* in)
    : Expr(passkey) {
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

std::string ReshapeOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = view( "
                          << in()->toString() << " )\n";
  return ss.str();
}

std::string ReshapeOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ReshapeOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 1);
  const at::Tensor& in_tensor = inputs[0].as<at::Tensor>();

  const auto& [out_shape, _] = inferShapeOfOutput(out(), ee);
  // TODO: check allocation domain and contiguity.

  // Use `at::Tensor::reshape` instead of `at::Tensor::view` because `ReshapeOp`
  // doesn't always produce an alias. For example, when merging an expanded
  // `IterType::Broadcast` and an `IterType::Iteration`, `ReshapeOp` has to
  // realize the expand.
  return {in_tensor.reshape(out_shape)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ReshapeOp)

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
    auto* tv = dynamic_cast<TensorView*>(out());
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

Partition::Partition(
    IrBuilderPasskey passkey,
    IterDomain* component,
    RaggedIterDomain* ragged,
    IterDomain* in,
    TensorView* extents)
    : Expr(passkey) {
  addOutput(component);
  addOutput(ragged);
  addInput(in);
  // Note: extents is held as an attribute rather than an input,
  // despite it's a TensorView. Inputs and outputs in the existing
  // IterDomain exprs are always IterDomains. Intuitively, they
  // transform input iteration spaces into output iteration spaces in
  // some way. Since the extents tensor itself is not transformed in the
  // Partition expr, it doesn't seem to be considered as an input. Note that in
  // Split, the split factor is an attribute. However, that said, none
  // of the existing exprs has tensors as attributes, which makes this
  // choice less certain with possible implications.
  addAttribute(extents);
}

std::string Partition::toString(int indent_size) const {
  std::stringstream ss;
  ss << "Partition: ";
  ss << in()->toString();
  ss << " by extents " << extents()->toString();
  ss << " -> component: ";
  ss << component()->toString();
  ss << ", ragged: ";
  ss << ragged()->toString();
  ss << "\n";
  return ss.str();
}

std::string Partition::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Partition can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Partition)

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
                          << toDelimitedString(getPadWidths()) << "} )\n";
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

std::vector<PolymorphicValue> MatmulOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  const auto a = inputs.at(0).as<at::Tensor>();
  const auto b = inputs.at(1).as<at::Tensor>();

  at::Tensor matmul_out;
  // aten::dot does not support meta device. Matmul lowers to dot for 1D @ 1D.
  const bool uses_dot = (a.dim() == 1 && b.dim() == 1);
  if (uses_dot && (a.is_meta() || b.is_meta())) {
    const auto out_scalar_type = at::result_type(a, b);
    auto out_opts = a.options().dtype(out_scalar_type).device(at::kMeta);
    matmul_out = at::empty({}, out_opts);
  } else {
    matmul_out = at::matmul(a, b);
  }

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

namespace spda_meta {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_flash_attention_meta(const at::Tensor& query) {
  const auto sizes = query.sizes();
  const int batch_size = sizes[0];
  int num_heads = sizes[1];
  int seqlen_q = sizes[2];
  auto logsumexp = at::empty(
      {batch_size, num_heads, seqlen_q}, query.options().dtype(at::kFloat));
  // Produce defined meta tensors for philox outputs so downstream segments
  // can bind metadata and types correctly.
  const auto meta_u64 =
      at::TensorOptions().device(at::kMeta).dtype(at::kUInt64);
  // philox_seed/rng_state, see note:
  // https://github.com/pytorch/pytorch/blob/cdc8460f2c76f98ba30556e3f9358e857a2f22f0/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp#L773-L778
  auto rng_state = at::empty({2}, meta_u64);
  auto rng_offset = at::empty({}, meta_u64);
  return std::make_tuple(
      query,
      logsumexp,
      at::Tensor(),
      at::Tensor(),
      c10::SymInt(seqlen_q),
      c10::SymInt(seqlen_q),
      rng_state,
      rng_offset,
      at::Tensor());
}

} // namespace spda_meta

std::vector<PolymorphicValue> SdpaFwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  auto query = inputs.at(0).as<at::Tensor>();
  auto key = inputs.at(1).as<at::Tensor>();
  auto value = inputs.at(2).as<at::Tensor>();

  const auto dropout_p = inputs.at(3).as<double>();
  const auto is_causal = inputs.at(4).as<bool>();

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

  // Compute scale using original size of last dimension
  double scale = inputs.size() > 5 ? inputs.back().as<double>()
                                   : 1.0 / std::sqrt(last_dim_size);

  auto batch_dims = query.sizes().slice(0, query.dim() - 3);
  NVF_CHECK_GE(batch_dims.size(), 1);
  if (batch_dims.size() > 1) {
    query = query.flatten(0, -4);
    NVF_ERROR_EQ(query.dim(), 4);
    key = key.flatten(0, -4);
    value = value.flatten(0, -4);
  }

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
       debug_attn_mask] = query.is_meta()
      ? spda_meta::_scaled_dot_product_flash_attention_meta(query)
      : at::_scaled_dot_product_flash_attention(
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

  auto unflatten_batch_dim = [](at::Tensor t,
                                at::IntArrayRef batch_dims) -> at::Tensor {
    at::DimVector new_shape(batch_dims);
    auto non_batch_dims = t.sizes().slice(1);
    new_shape.append(non_batch_dims.begin(), non_batch_dims.end());
    return t.reshape(new_shape);
  };

  if (batch_dims.size() > 1) {
    output = unflatten_batch_dim(output, batch_dims);
    log_sumexp = unflatten_batch_dim(log_sumexp, batch_dims);
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

Scope::Iterator Scope::insert(Iterator pos, Expr* expr) {
  return exprs_.insert(pos, expr);
}

Scope::Iterator Scope::insert_before(Expr* ref, Expr* expr) {
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

Scope::Iterator Scope::insert_after(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  NVF_ERROR(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " after the reference: ",
      ref,
      " however the reference was not found in this scope.");
  auto insert_pos = std::next(it);
  return insert(insert_pos, expr);
}

void Scope::erase(Iterator pos) {
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

bool Scope::contains(Expr* expr) const {
  const auto it = std::find(exprs_.begin(), exprs_.end(), expr);
  return it != exprs_.end();
}

void Scope::clear() {
  exprs_.clear();
}

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
  at::Tensor grad_query, grad_key, grad_value;
  if (bwd_inputs[0].is_meta()) {
    // Meta path: produce tensors with correct shapes/strides
    grad_query = at::empty_like(bwd_inputs[1]);
    grad_key = at::empty_like(bwd_inputs[2]);
    grad_value = at::empty_like(bwd_inputs[3]);
  } else {
    const auto philox_seed = inputs.at(8).as<at::Tensor>();
    const auto philox_offset = inputs.at(9).as<at::Tensor>();
    std::tie(grad_query, grad_key, grad_value) =
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
  }

  // If the inputs were padded, slice the grads to restore the original size
  auto slice_last_dim = [last_dim_size](at::Tensor output) -> at::Tensor {
    if (output.size(-1) != last_dim_size) {
      return output.slice(-1, 0, last_dim_size);
    }
    return output;
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

  // Meta-safe path: when either input or weight is a Meta tensor, avoid
  // calling into ATen embedding (which dispatches to index_select) and
  // instead synthesize the output shape and strides directly on Meta.
  if (input.is_meta() || weight.is_meta()) {
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(input.dim() + 1);
    for (int64_t d = 0; d < input.dim(); ++d) {
      out_sizes.push_back(input.size(d));
    }
    out_sizes.push_back(weight.size(1));
    auto out = at::empty(
        out_sizes, at::TensorOptions().device(at::kMeta).dtype(weight.dtype()));
    return {out};
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
    Val* out_mat,
    Val* out_scale,
    Val* out_gamma,
    Val* mat1,
    Val* mat2,
    Val* offsets,
    Val* scale1,
    Val* scale2,
    Val* alpha,
    Val* bias,
    Val* beta)
    : Expr(passkey) {
  NVF_ERROR(out_mat->isA<TensorView>(), "Output matrix must be a TensorView");
  NVF_ERROR(mat1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(mat2->isA<TensorView>(), "Second input must be a TensorView");
  NVF_ERROR(offsets->isA<TensorView>(), "Offsets must be a TensorView");
  addOutput(out_mat);
  if (out_scale != nullptr) {
    NVF_ERROR(
        out_scale->isA<TensorView>(), "Output scale must be a TensorView");
    addOutput(out_scale);
  }
  if (out_gamma != nullptr) {
    NVF_ERROR(out_scale != nullptr, "Output gamma requires output scale");
    NVF_ERROR(
        out_gamma->isA<TensorView>(), "Output gamma must be a TensorView");
    addOutput(out_gamma);
  }
  addInput(mat1);
  addInput(mat2);
  addInput(offsets);

  int64_t offset = 3;
  int64_t scale_offset = -1;
  int64_t alpha_offset = -1;
  int64_t bias_offset = -1;
  int64_t beta_offset = -1;

  bool has_scale1 = scale1 != nullptr;
  if (has_scale1) {
    NVF_CHECK(
        scale1->isA<TensorView>(),
        "`scale1` must be a TensorView, but got: ",
        scale1);
    NVF_CHECK(scale2->isA<TensorView>(), "Scale2 must be a TensorView");
    addInput(scale1);
    addInput(scale2);
    scale_offset = offset;
    offset += 2;
  }

  bool has_alpha = alpha != nullptr;
  if (has_alpha) {
    NVF_CHECK(
        alpha->isA<TensorView>(),
        "`alpha` must be a TensorView, but got: ",
        alpha);
    addInput(alpha);
    alpha_offset = offset++;
  }

  bool has_bias = bias != nullptr;
  if (has_bias) {
    NVF_CHECK(
        bias->isA<TensorView>(),
        "`bias` must be a TensorView, but got: ",
        bias);
    addInput(bias);
    bias_offset = offset++;
  }

  bool has_beta = beta != nullptr;
  if (has_beta) {
    NVF_CHECK(
        beta->isA<TensorView>(),
        "`beta` must be a TensorView, but got: ",
        beta);
    addInput(beta);
    beta_offset = offset++;
  }

  addDataAttribute(scale_offset);
  addDataAttribute(alpha_offset);
  addDataAttribute(bias_offset);
  addDataAttribute(beta_offset);
}

std::string GroupedMmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out();
  if (outScale() != nullptr) {
    ss << ", " << outScale();
  }
  if (outGamma() != nullptr) {
    ss << ", " << outGamma();
  }
  ss << " = GroupedMmaOp(" << "mat1=" << matrix1() << ", "
     << "mat2=" << matrix2() << ", " << "offsets=" << offsets();
  if (hasScale()) {
    ss << ", " << "scale1=" << scale1() << ", " << "scale2=" << scale2();
  }
  if (hasAlpha()) {
    ss << ", " << "alpha=" << alpha();
  }
  if (hasBias()) {
    ss << ", " << "bias=" << bias();
  }
  if (hasBeta()) {
    ss << ", " << "beta=" << beta();
  }
  ss << ")\n";
  return ss.str();
}

std::string GroupedMmaOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline.");
}

std::vector<PolymorphicValue> GroupedMmaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // Meta-device fast path outside of torch version guard
  if (inputs.size() >= 3 && inputs[0].is<at::Tensor>() &&
      inputs[1].is<at::Tensor>() && inputs[2].is<at::Tensor>()) {
    const auto& mat1_meta = inputs[0].as<at::Tensor>();
    const auto& mat2_meta = inputs[1].as<at::Tensor>();
    const auto& offsets_meta = inputs[2].as<at::Tensor>();
    if (mat1_meta.is_meta() || mat2_meta.is_meta() || offsets_meta.is_meta()) {
      const int64_t num_groups = offsets_meta.numel();
      std::vector<int64_t> result_sizes;
      if (mat1_meta.dim() == 2 && mat2_meta.dim() == 2) {
        result_sizes = {num_groups, mat1_meta.size(0), mat2_meta.size(-1)};
      } else if (mat1_meta.dim() == 3 && mat2_meta.dim() == 2) {
        result_sizes = {mat1_meta.size(1), mat2_meta.size(-1)};
      } else if (mat1_meta.dim() == 2 && mat2_meta.dim() == 3) {
        result_sizes = {mat1_meta.size(0), mat2_meta.size(-1)};
      } else {
        NVF_THROW(
            "Expect ranks to be <2, 2>, <3, 2> or <2, 3>. Got: mat1 = ",
            mat1_meta.sizes(),
            " and mat2 = ",
            mat2_meta.sizes());
      }

      auto options = mat1_meta.options()
                         .device(c10::Device(c10::kMeta))
                         .dtype(data_type_to_aten(out()->dtype()));
      at::Tensor result = at::empty(result_sizes, options);

      if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
          rfactor_did_idx != -1) {
        result = result.unsqueeze(rfactor_did_idx);
      }

      return {result};
    }
  }

  NVF_ERROR(
      inputs[0].is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 0 but got ",
      inputs[0].type().name());

  NVF_ERROR(
      inputs[1].is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 1 but got ",
      inputs[1].type().name());

  NVF_ERROR(
      inputs[2].is<at::Tensor>(),
      "GroupedMmaOp expects tensor input at position 2 but got ",
      inputs[2].type().name());

  auto mat1 = inputs[0].as<at::Tensor>();
  auto mat2 = inputs[1].as<at::Tensor>();
  auto offsets = inputs[2].as<at::Tensor>();

  at::Tensor alpha;
  at::Tensor bias;
  at::Tensor beta;
  if (hasAlpha()) {
    int alpha_offset = alphaOffset();
    NVF_ERROR(
        inputs[alpha_offset].is<at::Tensor>(),
        "GroupedMmaOp expects tensor alpha at position ",
        alpha_offset,
        " but got ",
        inputs[alpha_offset].type().name());
    alpha = inputs[alpha_offset].as<at::Tensor>();
  }
  if (hasBias()) {
    int bias_offset = biasOffset();
    NVF_ERROR(
        inputs[bias_offset].is<at::Tensor>(),
        "GroupedMmaOp expects tensor bias at position ",
        bias_offset,
        " but got ",
        inputs[bias_offset].type().name());
    bias = inputs[bias_offset].as<at::Tensor>();
  }
  if (hasBeta()) {
    int beta_offset = betaOffset();
    NVF_ERROR(
        inputs[beta_offset].is<at::Tensor>(),
        "GroupedMmaOp expects tensor beta at position ",
        beta_offset,
        " but got ",
        inputs[beta_offset].type().name());
    beta = inputs[beta_offset].as<at::Tensor>();
  }

  // This lambda returns the raw result, which will be postprocessed by the
  // caller, e.g., converting the data type and adding rFactor dimensions.
  auto result = [&]() -> at::Tensor {
    if (hasScale()) {
#if NVF_TORCH_VERSION_NO_LESS(2, 8, 0)
      NVF_ERROR(
          inputs[3].is<at::Tensor>(),
          "GroupedMmaOp expects tensor input at position 3 but got ",
          inputs[3].type().name());
      NVF_ERROR(
          inputs[4].is<at::Tensor>(),
          "GroupedMmaOp expects tensor input at position 4 but got ",
          inputs[4].type().name());

      auto scale1 = inputs[scale1Offset()].as<at::Tensor>();
      auto scale2 = inputs[scale2Offset()].as<at::Tensor>();
      // Note: at::_scaled_grouped_mm requires k dimension to be the fastest on
      // both input matrices.
      auto mat1_k_last = mat1.contiguous();
      auto mat2_k_last = mat2.transpose(-1, -2).contiguous().transpose(-1, -2);

      // at::_scaled_grouped_mm limitation
      NVF_CHECK(
          scale1.size(-1) == 1 && scale2.size(-2) == 1,
          "Scale1 and scale2 must have size 1 at the k dimension. Got ",
          scale1.sizes(),
          " and ",
          scale2.sizes());
      // scale factor handling
      // see NOTE -- [ Grouped Matrix Multiplication semantics ]
      if (TensorDomain::noReductions(out()->getLogicalDomain()).size() == 3) {
        // case 1, aten API expects collapsed 1D scale with group dimension on
        // the slower side.
        scale1 = scale1.reshape(-1);
        scale2 = scale2.reshape(-1);
      } else {
        // case 2 and 3, aten doesn't allow broadcast on k dimension. squeeze k
        // out.
        scale1 = scale1.squeeze(-1);
        scale2 = scale2.squeeze(-2);
      }
      // undefined alpha, bias is not supported by aten API
      NVF_ERROR(!alpha.defined(), "alpha is not supported yet");
      NVF_ERROR(!beta.defined(), "beta is not supported yet");
      NVF_ERROR(!bias.defined(), "bias is not supported yet");
      // NOTE: at::_scaled_grouped_mm only supports bfloat16 as output at this
      // moment, otherwise we should have requested the output dtype directly
      // instead of casting the output afterwards.
      return at::_scaled_grouped_mm(
          mat1_k_last,
          mat2_k_last,
          scale1,
          scale2,
          offsets,
          /*bias=*/std::nullopt,
          /*alpha=*/std::nullopt,
          at::ScalarType::BFloat16);
#else
      NVF_THROW("at::_scaled_grouped_mm does not exist prior to PyTorch 2.8.");
#endif
    }

#if NVFUSER_CUTLASS_KERNEL_ENABLED
    const bool supported_by_cutlass_kernel =
        at::cuda::getCurrentDeviceProperties()->major == 10 &&
        mat1.dim() == 2 && mat2.dim() == 3 && mat1.is_contiguous() &&
        mat2.transpose(-1, -2).is_contiguous();
    if (supported_by_cutlass_kernel) {
      mat2 = mat2.transpose(-1, -2);
      NVF_ERROR(mat2.is_contiguous());
      // [m, k] x [g, n, k]; both contiguous
      const auto k = mat1.size(-1);
      const auto n = mat2.size(1);
      at::Tensor group_sizes = at::diff(
          offsets,
          /*n=*/1,
          /*dim=*/-1,
          /*prepend=*/
          at::zeros({1}, at::dtype(at::kInt).device(offsets.device())));
      at::Tensor ab_strides = at::full_like(offsets, k, at::dtype(at::kLong));
      at::Tensor c_strides = at::full_like(offsets, n, at::dtype(at::kLong));
      at::Tensor problem_sizes =
          at::stack({group_sizes, c_strides, ab_strides}, /*dim=*/-1)
              .to(at::kInt);
      offsets = at::cat(
          {at::zeros({1}, at::dtype(at::kInt).device(offsets.device())),
           offsets.slice(0, 0, -1)});
      return cutlass_kernels::grouped_mm(
          mat1, mat2, ab_strides, c_strides, problem_sizes, offsets);
    }
#endif

    // undefined bias is not supported by aten API
    NVF_ERROR(!alpha.defined(), "alpha is not supported yet");
    NVF_ERROR(!beta.defined(), "beta is not supported yet");
    NVF_ERROR(!bias.defined(), "bias is not supported yet");

    // Compute numbers of tokens per group from offsets.
    NVF_CHECK_EQ(
        c10::cuda::currentStreamCaptureStatusMayInitCtx(),
        c10::cuda::CaptureStatus::None,
        "GroupedMmaOp's fallback implementation below doesn't support CUDA "
        "graph capturing. The shapes of individual matmuls depend on "
        "`offsets`, which is data dependent.");
    at::Tensor offsets_cpu = offsets.cpu();
    NVF_ERROR_EQ(offsets_cpu.dtype(), at::kInt);
    const int* data_ptr = offsets_cpu.data_ptr<int>();
    const int64_t num_groups = offsets_cpu.numel();
    std::vector<int64_t> group_sizes(data_ptr, data_ptr + num_groups);
    for (int64_t i : arange(1, num_groups) | std::views::reverse) {
      group_sizes[i] -= group_sizes[i - 1];
    }

    std::vector<at::Tensor> group_mat1s;
    std::vector<at::Tensor> group_mat2s;
    at::Tensor result;
    std::vector<at::Tensor> group_outs;
    if (mat1.dim() == 2 && mat2.dim() == 2) {
      // [m, k] @ [k, n] => [g, m, n]
      group_mat1s = mat1.split(group_sizes, -1);
      group_mat2s = mat2.split(group_sizes, 0);
      result =
          at::empty({num_groups, mat1.size(0), mat2.size(-1)}, mat1.options());
      group_outs = result.unbind();
    } else if (mat1.dim() == 3 && mat2.dim() == 2) {
      // [g, m, k] @ [k, n] => [m, n]
      group_mat1s = mat1.unbind();
      group_mat2s = mat2.split(group_sizes, -1);
      result = at::empty({mat1.size(1), mat2.size(-1)}, mat1.options());
      group_outs = result.split(group_sizes, -1);
    } else if (mat1.dim() == 2 && mat2.dim() == 3) {
      // [m, k] @ [g, k, n] => [m, n]
      group_mat1s = mat1.split(group_sizes, 0);
      group_mat2s = mat2.unbind();
      result = at::empty({mat1.size(0), mat2.size(-1)}, mat1.options());
      group_outs = result.split(group_sizes, 0);
    } else {
      NVF_THROW(
          "Expect ranks to be <2, 2>, <3, 2> or <2, 3>. Got: mat1 = ",
          mat1.sizes(),
          " and mat2 = ",
          mat2.sizes());
    }

    for (auto [group_mat1, group_mat2, group_out] :
         zip(group_mat1s, group_mat2s, group_outs)) {
      at::matmul_out(group_out, group_mat1, group_mat2);
    }
    return result;
  }();

  // Post-processing
  result = result.to(data_type_to_aten(out()->dtype()));

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    result = result.unsqueeze(rfactor_did_idx);
  }

  return {result};
}

IterDomain* GroupedMmaOp::getKDimOfMatrix1() const {
  // mat1 is [g, m, k] or [m, k]
  const auto& logical_domain =
      TensorDomain::noReductions(matrix1()->getLogicalDomain());
  return logical_domain.at(logical_domain.size() - 1);
}

IterDomain* GroupedMmaOp::getKDimOfMatrix2() const {
  // mat2 is [g, k, n] or [k, n]
  const auto& logical_domain =
      TensorDomain::noReductions(matrix2()->getLogicalDomain());
  return logical_domain.at(logical_domain.size() - 1);
}

namespace {
IterDomain* returnFirstIfRankThree(const TensorView* tv) {
  const auto& logical_domain =
      TensorDomain::noReductions(tv->getLogicalDomain());
  if (logical_domain.size() == 3) {
    return logical_domain.at(0);
  } else {
    return nullptr;
  }
}
} // namespace

IterDomain* GroupedMmaOp::getGroupDimOfMatrix1() const {
  // matrix1 is [g, m, k] or [m, k]
  return returnFirstIfRankThree(matrix1());
}

IterDomain* GroupedMmaOp::getGroupDimOfMatrix2() const {
  // matrix2 is [g, k, n] or [k, n]
  return returnFirstIfRankThree(matrix2());
}

IterDomain* GroupedMmaOp::getGroupDimOfOutput() const {
  // output is [g, m, n] or [m, n]
  return returnFirstIfRankThree(out());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedMmaOp)

ScaledMmaOp::ScaledMmaOp(
    IrBuilderPasskey passkey,
    Val* out_mat,
    Val* out_scale,
    Val* out_gamma,
    Val* mat1,
    Val* mat2,
    Val* scale1,
    Val* scale2,
    Val* alpha,
    Val* bias,
    Val* beta)
    : Expr(passkey) {
  NVF_ERROR(out_mat->isA<TensorView>(), "Output matrix must be a TensorView");
  NVF_ERROR(mat1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(mat2->isA<TensorView>(), "Second input must be a TensorView");
  addOutput(out_mat);
  if (out_scale != nullptr) {
    NVF_ERROR(
        out_scale->isA<TensorView>(), "Output scale must be a TensorView");
    addOutput(out_scale);
  }
  if (out_gamma != nullptr) {
    NVF_ERROR(out_scale != nullptr, "Output gamma requires output scale");
    NVF_ERROR(
        out_gamma->isA<TensorView>(), "Output gamma must be a TensorView");
    addOutput(out_gamma);
  }
  addInput(mat1);
  addInput(mat2);
  NVF_ERROR(scale1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(scale2->isA<TensorView>(), "Second input must be a TensorView");
  addInput(scale1);
  addInput(scale2);

  int64_t offset = 4;
  int64_t alpha_offset = -1;
  int64_t bias_offset = -1;
  int64_t beta_offset = -1;

  bool has_alpha = alpha != nullptr;
  if (has_alpha) {
    NVF_CHECK(
        alpha->isA<TensorView>(),
        "`alpha` must be a TensorView, but got: ",
        alpha);
    addInput(alpha);
    alpha_offset = offset++;
  }

  bool has_bias = bias != nullptr;
  if (has_bias) {
    NVF_CHECK(
        bias->isA<TensorView>(),
        "`bias` must be a TensorView, but got: ",
        bias);
    addInput(bias);
    bias_offset = offset++;
  }

  bool has_beta = beta != nullptr;
  if (has_beta) {
    NVF_CHECK(
        beta->isA<TensorView>(),
        "`beta` must be a TensorView, but got: ",
        beta);
    addInput(beta);
    beta_offset = offset++;
  }

  // Store the offsets as attributes
  addDataAttribute(alpha_offset);
  addDataAttribute(bias_offset);
  addDataAttribute(beta_offset);
}

std::string ScaledMmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out();
  if (outScale() != nullptr) {
    ss << ", " << outScale();
  }
  if (outGamma() != nullptr) {
    ss << ", " << outGamma();
  }
  ss << " = ScaledMmaOp(";
  ss << "mat1=" << matrix1() << ", ";
  ss << "mat2=" << matrix2() << ", ";
  ss << "scale1=" << scale1() << ", ";
  ss << "scale2=" << scale2() << "";
  if (hasAlpha()) {
    ss << ", alpha=" << alpha();
  }
  if (hasBias()) {
    ss << ", bias=" << bias();
  }
  if (hasBeta()) {
    ss << ", beta=" << beta();
  }
  ss << ")\n";
  return ss.str();
}

std::string ScaledMmaOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline.");
}

std::vector<PolymorphicValue> ScaledMmaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  [[maybe_unused]] const auto& mat1 = inputs[0].as<at::Tensor>();
  [[maybe_unused]] const auto& mat2 = inputs[1].as<at::Tensor>();
  [[maybe_unused]] const auto& scale1 = inputs[2].as<at::Tensor>();
  [[maybe_unused]] const auto& scale2 = inputs[3].as<at::Tensor>();

  at::Tensor alpha;
  at::Tensor bias;
  at::Tensor beta;
  if (hasAlpha()) {
    int alpha_offset = alphaOffset();
    NVF_ERROR(
        inputs[alpha_offset].is<at::Tensor>(),
        "ScaledMmaOp expects tensor alpha at position ",
        alpha_offset,
        " but got ",
        inputs[alpha_offset].type().name());
    alpha = inputs[alpha_offset].as<at::Tensor>();
  }
  if (hasBias()) {
    int bias_offset = biasOffset();
    NVF_ERROR(
        inputs[bias_offset].is<at::Tensor>(),
        "ScaledMmaOp expects tensor bias at position ",
        bias_offset,
        " but got ",
        inputs[bias_offset].type().name());
    bias = inputs[bias_offset].as<at::Tensor>();
  }
  if (hasBeta()) {
    int beta_offset = betaOffset();
    NVF_ERROR(
        inputs[beta_offset].is<at::Tensor>(),
        "ScaledMmaOp expects tensor beta at position ",
        beta_offset,
        " but got ",
        inputs[beta_offset].type().name());
    beta = inputs[beta_offset].as<at::Tensor>();
  }

  at::Tensor result;
#if NVFUSER_CUTLASS_KERNEL_ENABLED
  {
    at::ScalarType out_scalar_type = data_type_to_aten(out()->dtype());
    at::Tensor mat1_view = mat1;
    // nvfp4_scaled_mm expected layout
    at::Tensor mat2_view = mat2.t();

    DataType in_dtype = matrix1()->dtype();

    // Debugging: Print shape and strides of mat1_view, mat2_view, scale1, and
    // scale2
    std::cout << "mat1_view shape: " << mat1_view.sizes()
              << ", strides: " << mat1_view.strides() << std::endl;
    std::cout << "mat2_view shape: " << mat2_view.sizes()
              << ", strides: " << mat2_view.strides() << std::endl;
    std::cout << "scale1 shape: " << scale1.sizes()
              << ", strides: " << scale1.strides() << std::endl;
    std::cout << "scale2 shape: " << scale2.sizes()
              << ", strides: " << scale2.strides() << std::endl;

    // NOTE: cutlass nvfp4 kernel doesn't support bias, beta or quantized output
    if (!bias.defined() && !beta.defined() && outputs().size() == 1) {
      bool cutlass_can_run = true;
      // NOTE: this felt ugly. I should go fix up the validate input
      try {
        cutlass_kernels::validateInputsNvfp4ScaledMm(
            mat1_view, mat2_view, scale1, scale2, alpha);
      } catch (...) {
        cutlass_can_run = false;
      }

      if (cutlass_can_run) {
        result = cutlass_kernels::nvfp4_scaled_mm(
            mat1_view,
            mat2_view,
            scale1,
            scale2,
            alpha,
            out_scalar_type,
            /*skip_checks=*/true);
      }
    }
  }
#endif

#if NVF_TORCH_VERSION_NO_LESS(2, 8, 0)
  if (!result.defined()) {
    // TODO: interface with scaled matrix multiplication cutlass kernel. For
    // now, we'll fallback with aten kernels at::_scaled_mm has implementation
    // limitations:
    NVF_CHECK(!beta.defined(), "beta in ScaledMmaOp is not supported yet");
    NVF_CHECK(
        outScale() == nullptr,
        "output block scaling factor in ScaledMmaOp is not supported yet");
    NVF_CHECK(
        outGamma() == nullptr,
        "output global scaling factor in ScaledMmaOp is not supported yet");
    result = at::_scaled_mm(
        mat1,
        mat2,
        scale1,
        scale2,
        bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt,
        alpha.defined() ? std::optional<at::Tensor>(alpha) : std::nullopt,
        data_type_to_aten(out()->dtype()));
  }
#endif

  NVF_CHECK(result.defined(), "Couldn't find fallback kernel for scaled_mm");

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    result = result.unsqueeze(rfactor_did_idx);
  }

  return {result};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ScaledMmaOp)

ScanOp::ScanOp(
    IrBuilderPasskey passkey,
    BinaryOpType op_type,
    Val* init,
    Val* out,
    Val* in,
    int64_t dim)
    : Expr(passkey) {
  addOutput(out);
  addInput(in);
  addAttribute(init);
  addDataAttribute(op_type);
  addDataAttribute(dim);
}

std::string ScanOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString();
  ss << "\n";
  indent(ss, indent_size + 1) << " = scan(" << in()->toString() << ",\n";
  indent(ss, indent_size + 1) << "        dim=" << dim() << ",\n";
  indent(ss, indent_size + 1) << "        op_type=" << opType() << ",\n";
  indent(ss, indent_size + 1)
      << "        init=" << init()->toInlineString() << ")\n";
  return ss.str();
}

std::string ScanOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline");
}

std::vector<PolymorphicValue> ScanOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  auto input = inputs.at(0).as<at::Tensor>();

  NVF_ERROR_EQ(inputs.size(), 1);

  // Meta-safe path: when input is a Meta tensor, avoid invoking ATen ops that
  // may not have Meta kernels (e.g., cummin). Instead, synthesize an output
  // tensor on Meta with the correct shape/strides and dtype.
  if (input.is_meta()) {
    const at::ScalarType out_dtype = data_type_to_aten(out()->dtype());
    auto out_meta = at::empty(
        input.sizes(), at::TensorOptions().device(at::kMeta).dtype(out_dtype));
    return {out_meta};
  }

  at::Tensor out_t;
  switch (opType()) {
    case BinaryOpType::Add:
      out_t = at::cumsum(input, dim());
      break;
    case BinaryOpType::FMax:
    case BinaryOpType::Max:
      out_t = std::get<0>(at::cummax(input, dim()));
      break;
    case BinaryOpType::FMin:
    case BinaryOpType::Min:
      out_t = std::get<0>(at::cummin(input, dim()));
      break;
    case BinaryOpType::Mul:
      out_t = at::cumprod(input, dim());
      break;
    default:
      NVF_THROW("Unhandled opType() ", opType());
  }

  at::ScalarType out_dtype = data_type_to_aten(out()->dtype());
  if (out_t.dtype() != out_dtype) {
    out_t = out_t.to(out_dtype);
  }

  return {out_t};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ScanOp)

CutlassNvfp4GroupedMmaOp::CutlassNvfp4GroupedMmaOp(
    IrBuilderPasskey passkey,
    Val* out_mat,
    Val* mat1,
    Val* mat2,
    Val* scale1,
    Val* scale2,
    Val* alpha,
    Val* problem_sizes,
    Val* expert_offsets,
    Val* sf_offsets)
    : Expr(passkey) {
  NVF_ERROR(out_mat->isA<TensorView>(), "Output matrix must be a TensorView");
  NVF_ERROR(mat1->isA<TensorView>(), "First input must be a TensorView");
  NVF_ERROR(mat2->isA<TensorView>(), "Second input must be a TensorView");
  NVF_ERROR(scale1->isA<TensorView>(), "Scale1 must be a TensorView");
  NVF_ERROR(scale2->isA<TensorView>(), "Scale2 must be a TensorView");
  NVF_ERROR(alpha->isA<TensorView>(), "Alpha must be a TensorView");
  NVF_ERROR(
      problem_sizes->isA<TensorView>(), "Problem sizes must be a TensorView");
  NVF_ERROR(
      expert_offsets->isA<TensorView>(), "Expert offsets must be a TensorView");
  NVF_ERROR(sf_offsets->isA<TensorView>(), "SF offsets must be a TensorView");

  addOutput(out_mat);
  addInput(mat1);
  addInput(mat2);
  addInput(scale1);
  addInput(scale2);
  addInput(alpha);
  addInput(problem_sizes);
  addInput(expert_offsets);
  addInput(sf_offsets);
}

std::string CutlassNvfp4GroupedMmaOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out();
  ss << " = CutlassNvfp4GroupedMmaOp(";
  ss << "mat1=" << matrix1() << ", ";
  ss << "mat2=" << matrix2() << ", ";
  ss << "scale1=" << scale1() << ", ";
  ss << "scale2=" << scale2() << ", ";
  ss << "alpha=" << alpha() << ", ";
  ss << "problem_sizes=" << problemSizes() << ", ";
  ss << "expert_offsets=" << expertOffsets() << ", ";
  ss << "sf_offsets=" << scalingFactorOffsets() << ")\n";
  return ss.str();
}

std::string CutlassNvfp4GroupedMmaOp::toInlineString(int indent_size) const {
  NVF_THROW("Tensor op can not be printed inline.");
}

std::vector<PolymorphicValue> CutlassNvfp4GroupedMmaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
#if NVFUSER_CUTLASS_KERNEL_ENABLED
  const auto& mat1 = inputs[0].as<at::Tensor>();
  const auto& mat2 = inputs[1].as<at::Tensor>();
  const auto& scale1 = inputs[2].as<at::Tensor>();
  const auto& scale2 = inputs[3].as<at::Tensor>();
  const auto& alpha = inputs[4].as<at::Tensor>();
  const auto& problem_sizes = inputs[5].as<at::Tensor>();
  const auto& expert_offsets = inputs[6].as<at::Tensor>();
  const auto& sf_offsets = inputs[7].as<at::Tensor>();
  NVF_CHECK(
      mat1.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 &&
      mat2.scalar_type() == at::ScalarType::Float4_e2m1fn_x2);

  // Validate problem_sizes tensor
  NVF_CHECK(problem_sizes.dim() == 2, "problem_sizes must be a 2D tensor");
  NVF_CHECK(
      problem_sizes.size(1) == 3,
      "problem_sizes must have shape (num_experts, 3)");
  int num_experts = problem_sizes.size(0);

  at::ScalarType out_dtype = data_type_to_aten(out()->dtype());
  const auto options =
      at::TensorOptions().device(mat1.device()).dtype(out_dtype);

  // Calculate proper stride tensors for the cutlass kernel
  // ab_strides: stride information for input matrices A and B
  // c_strides: stride information for output matrix C
  // Note: mat1 is packed fp4x2.
  int k = mat1.size(1) * 2;
  int n = mat2.size(2);
  auto ab_strides =
      at::empty({num_experts}, options.dtype(at::ScalarType::Long));
  auto c_strides =
      at::empty({num_experts}, options.dtype(at::ScalarType::Long));
  // FIXME: this could be done outside and provided as input to avoid two kernel
  // launches.
  ab_strides.fill_(k);
  c_strides.fill_(n);

  // Call the cutlass kernel, note that it expect g,n,k layout on mat2.
  at::Tensor result = cutlass_kernels::nvfp4_scaled_grouped_mm(
      mat1.view(at::ScalarType::Float4_e2m1fn_x2),
      mat2.transpose(-1, -2).view(at::ScalarType::Float4_e2m1fn_x2),
      scale1,
      scale2,
      alpha,
      ab_strides,
      c_strides,
      problem_sizes,
      expert_offsets,
      sf_offsets,
      out_dtype);

  if (const auto rfactor_did_idx = getRFactorDeviceDimensionIndex(out());
      rfactor_did_idx != -1) {
    result = result.unsqueeze(rfactor_did_idx);
  }
  return {result};
#else
  NVF_THROW("CutlassNvfp4GroupedMmaOp requires CUTLASS kernels to be enabled");
#endif
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CutlassNvfp4GroupedMmaOp)

PreprocessGroupedMatmulInputSf::PreprocessGroupedMatmulInputSf(
    IrBuilderPasskey passkey,
    Val* output,
    Val* input,
    Val* input_offsets,
    Val* output_offsets,
    BlockScalingFactorLayout layout,
    Val* k,
    Val* g,
    Val* row_idx,
    Val* col_idx)
    : Expr(passkey) {
  addInput(input);
  addInput(input_offsets);
  addInput(output_offsets);
  addInput(k);
  addInput(g);
  addOutput(output);
  addDataAttribute(layout);
  if (row_idx != nullptr) {
    addAttribute(row_idx);
  }
  if (col_idx != nullptr) {
    addAttribute(col_idx);
  }
}

std::string PreprocessGroupedMatmulInputSf::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = preprocessGroupedMatmulInputSf(\n";
  indent_size++;
  indent(ss, indent_size) << "input = " << in()->toString() << ",\n";
  indent(ss, indent_size) << "input_offsets = " << inputOffsets()->toString()
                          << ",\n";
  indent(ss, indent_size) << "output_offsets = " << outputOffsets()->toString()
                          << ",\n";
  indent(ss, indent_size) << "layout = "
                          << (layout() == BlockScalingFactorLayout::Block128x4
                                  ? "Block128x4"
                                  : "Unknown")
                          << "\n";
  indent_size--;
  indent(ss, indent_size) << ")\n";
  return ss.str();
}

std::string PreprocessGroupedMatmulInputSf::toInlineString(
    int indent_size) const {
  NVF_CHECK(false, "PreprocessGroupedMatmulInputSf can not be printed inline");
}

std::vector<PolymorphicValue> PreprocessGroupedMatmulInputSf::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // This is a placeholder, currently we don't have a fallback kernel available
  NVF_THROW("PreprocessGroupedMatmulInputSf evaluation not yet implemented");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PreprocessGroupedMatmulInputSf)

// Details:
// Currently output_scales is the first input in the constructor even though
// it's the second output. This is because if it's the second output then we hit
// a bug in indexing. The stack trace can be seen here:
// https://gist.github.com/protonu/dc35024c1291625b2b7ce87baa39e2ae
// This happens when creating UnswitchPredicate, probably in the call to
// TensorIndexer::getPredicates. The incorrect predicate_domains for the tv
// in the call to getPredicateDomains.
BlockQuantizationOp::BlockQuantizationOp(
    IrBuilderPasskey passkey,
    Val* output_scales,
    Val* output,
    Val* input,
    Val* logical_index,
    Val* global_scale,
    int64_t block_size,
    bool swizzled_scales)
    : Expr(passkey) {
  addOutput(output);
  addOutput(output_scales);
  addInput(input);
  if (global_scale) {
    addInput(global_scale);
  }
  addAttribute(logical_index);
  addDataAttribute(block_size);
  addDataAttribute(swizzled_scales);
}

std::string BlockQuantizationOp::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "(" << blockScales()->toString() << ",\n "
                          << quantizedOutput()->toString() << ")\n"
                          << " = block_quantize(" << in()->toString() << ")\n";
  return ss.str();
}

std::string BlockQuantizationOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "BlockQuantizationOp can not be printed inline");
}

std::vector<PolymorphicValue> BlockQuantizationOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // This is a placeholder, currently we don't have a fallback kernel available
  NVF_THROW("BlockQuantizationOp evaluation not yet implemented");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockQuantizationOp)

LaunchDependentGridOp::LaunchDependentGridOp(
    IrBuilderPasskey passkey,
    Val* output,
    std::vector<Val*> inputs)
    : Expr(passkey) {
  addOutput(output);
  for (auto input : inputs) {
    addInput(input);
  }
}

std::string LaunchDependentGridOp::toString(int indent_size) const {
  NVF_CHECK_EQ(outputs().size(), 1);
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << " = "
                          << "launchDependentGrid("
                          << toDelimitedString(inputs()) << ")\n";
  return ss.str();
}

std::string LaunchDependentGridOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "LaunchDependentGridOp can not be printed inline");
}

std::vector<PolymorphicValue> LaunchDependentGridOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // This is a placeholder, currently we don't have a fallback kernel available
  NVF_THROW("LaunchDependentGridOp evaluation not yet implemented");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(LaunchDependentGridOp)

WaitForPriorGridOp::WaitForPriorGridOp(
    IrBuilderPasskey passkey,
    Val* output,
    std::vector<Val*> inputs)
    : Expr(passkey) {
  addOutput(output);
  for (auto input : inputs) {
    addInput(input);
  }
}

std::string WaitForPriorGridOp::toString(int indent_size) const {
  NVF_CHECK_EQ(outputs().size(), 1);
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << " = "
                          << "waitForPriorGrid(" << toDelimitedString(inputs())
                          << ")\n";
  return ss.str();
}

std::string WaitForPriorGridOp::toInlineString(int indent_size) const {
  NVF_CHECK(false, "WaitForPriorGridOp can not be printed inline");
}

std::vector<PolymorphicValue> WaitForPriorGridOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  // This is a placeholder, currently we don't have a fallback kernel available
  NVF_THROW("WaitForPriorGridOp evaluation not yet implemented");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(WaitForPriorGridOp)

} // namespace nvfuser
