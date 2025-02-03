// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <kernel_ir.h>
#include <runtime/allocations.h>

namespace nvfuser {

PolymorphicValue Val::evaluate() {
  if (this->value().hasValue()) {
    return this->value();
    FUSER_PERF_SCOPE("Val::evaluate");
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  NVF_ERROR(
      evaluated_val.hasValue(),
      "Detected a const value but failed to infer its value: ",
      toInlineString());
  return evaluated_val;
}

std::vector<PolymorphicValue> Expr::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("Expr::evaluate");
  NVF_THROW(
      "`evaluate` method for expression ",
      getOpString(),
      " is not defined. ",
      "Please override the evaluate method");
}

std::vector<PolymorphicValue> Expr::evaluate(
    const ExpressionEvaluator& ee,
    std::unordered_map<const Val*, PolymorphicValue>& known_values) const {
  FUSER_PERF_SCOPE("Expr::evaluate");
  std::vector<PolymorphicValue> expr_inputs;
  expr_inputs.reserve(inputs().size());
  for (auto inp : inputs()) {
    const auto& eval_i = ee.evaluate(inp, known_values);
    if (!eval_i.hasValue()) {
      return {std::monostate{}};
    }
    expr_inputs.emplace_back(eval_i);
  }
  return this->evaluate(ee, expr_inputs);
}

void Expr::addDataAttribute(PolymorphicValue attr) {
  addAttribute(IrBuilder::createInContainer<Val>(container(), std::move(attr)));
}
std::vector<PolymorphicValue> FullOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("FullOp::evaluate");
  std::vector<int64_t> shape;
  for (auto i : c10::irange(inputs.size() - 1)) {
    shape.push_back(inputs.at(i).as<int64_t>());
  }
  DataType dtype = getFillValue()->getDataType().value();
  const auto options =
      at::TensorOptions().device(at::kCUDA).dtype(data_type_to_aten(dtype));
  using namespace PolymorphicValue_functions;
  return {at::full(shape, toScalar(inputs.back()), options)};
}

std::vector<PolymorphicValue> SelectOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("SelectOp::evaluate");
  const auto& in = inputs.at(0).as<at::Tensor>();
  int64_t dimension = dim();
  int64_t index = (int64_t)inputs.at(1);
  return {in.select(dimension, index)};
}

std::vector<PolymorphicValue> IndexSelectOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("IndexSelectOp::evaluate");
  const auto& in = inputs.at(0).as<at::Tensor>();
  int64_t dimension = dim();
  const auto& indices = inputs.at(1).as<at::Tensor>().squeeze();
  return {at::index_select(in, dimension, indices)};
}

std::vector<PolymorphicValue> TorchGatherOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("TorchGatherOp::evaluate");
  const auto& input = inputs.at(0).as<at::Tensor>();
  const auto& index = inputs.at(1).as<at::Tensor>();
  auto dimension = dim();
  if (exactSizes()) {
    return {at::take_along_dim(input, index, dimension)};
  } else {
    return {at::gather(input, dimension, index)};
  }
}

std::vector<PolymorphicValue> ScatterOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("ScatterOp::evaluate");
  const auto& input = inputs.at(0).as<at::Tensor>();
  const auto& index = inputs.at(1).as<at::Tensor>();
  const auto& src = inputs.at(2).as<at::Tensor>();
  auto dimension = dim();
  return {at::scatter(input, dimension, index, src)};
}

std::vector<PolymorphicValue> IotaOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("IotaOp::evaluate");
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

std::vector<PolymorphicValue> EyeOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("EyeOp::evaluate");
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

std::vector<PolymorphicValue> UnaryOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("UnaryOp::evaluate");
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
          dataTypeSize(input(0)->dtype()) == dataTypeSize(out()->dtype()),
          "BitCast only works for types of the same size");
      if (isComplexType(input(0)->dtype()) &&
          std::holds_alternative<ArrayType>(out()->dtype().type)) {
        // view_as_real case.
        auto vec_type = std::get<ArrayType>(out()->dtype().type);
        auto inp_scalar_type = getTypeFromComplexType(input(0)->dtype());
        NVF_CHECK(
            *vec_type.type == inp_scalar_type,
            "Output type must be the same as the scalar type of the complex input.");
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

std::vector<PolymorphicValue> BinaryOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("BinaryOp::evaluate");
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
      return {lhs / rhs};
      break;
    case BinaryOpType::Mod:
      NVF_CHECK(rhs != 0);
      return {lhs % rhs};
      break;
    case BinaryOpType::Fmod:
      NVF_CHECK(rhs != 0);
      return {fmod(lhs, rhs)};
      break;
    case BinaryOpType::CeilDiv:
      NVF_CHECK(rhs != 0);
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

std::vector<PolymorphicValue> TernaryOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("TernaryOp::evaluate");
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

std::vector<PolymorphicValue> ArrayConstruct::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("ArrayConstruct::evaluate");
  return {PolymorphicValue(inputs)};
}

std::vector<PolymorphicValue> ReverseArray::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("ReverseArray::evaluate");
  NVF_ERROR(inputs.size() == 1, "ReverseArray expects 1 input");
  PolymorphicValue array = inputs.at(0);
  auto& vec = array.as<std::vector>();
  std::reverse(vec.begin(), vec.end());
  return {std::move(array)};
}

std::vector<PolymorphicValue> GetItem::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("GetItem::evaluate");
  NVF_ERROR(inputs.size() == 2, "GetItem expects 2 inputs");
  return {PolymorphicValue(inputs.at(0)[inputs.at(1)])};
}

std::vector<PolymorphicValue> StructConstruct::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("StructConstruct::evaluate");
  NVF_ERROR(
      this->inputs().size() == inputs.size(),
      "StructConstruct expects ",
      this->inputs().size(),
      " inputs");
  PolymorphicValue struct_ =
      std::get<StructType>(output(0)->dtype().type).create();
  for (int64_t i : c10::irange((int64_t)inputs.size())) {
    struct_->*attribute<std::string>(i) = inputs.at(i);
  }
  return {std::move(struct_)};
}

std::vector<PolymorphicValue> GetAttr::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("GetAttr::evaluate");
  NVF_ERROR(inputs.size() == 1, "GetAttr expects 1 input");
  return {inputs.at(0)->*attr()};
}

std::vector<PolymorphicValue> TensorConstruct::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("TensorConstruct::evaluate");
  NVF_ERROR(inputs.size() == 1, "TensorConstruct expects 1 input");
  using namespace PolymorphicValue_functions;
  return {toTensor(inputs.at(0))};
}

std::vector<PolymorphicValue> BroadcastOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("BroadcastOp::evaluate");
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

std::vector<PolymorphicValue> SqueezeOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("SqueezeOp::evaluate");
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
  for (int64_t i : c10::irange((int64_t)is_squeeze_dims.size())) {
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

std::vector<PolymorphicValue> ReductionOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("ReductionOp::evaluate");
  const auto& input = inputs.at(0).as<at::Tensor>();
  const auto output = out()->as<TensorView>();

  NVF_ERROR(
      !output->hasRoot(),
      "Evaluation for rFactored reductions is not supported.");

  std::vector<int64_t> reduction_axes;
  for (const auto i : c10::irange(int64_t(output->getLogicalDomain().size()))) {
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

std::vector<PolymorphicValue> GroupedReductionOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("GroupedReductionOp::evaluate");
  const auto num_reductions = numHorizontallyGroupedExprs();
  std::vector<PolymorphicValue> grouped_reduction_out;
  grouped_reduction_out.reserve(num_reductions);
  for (const auto i : c10::irange(num_reductions)) {
    const auto& in_tensor = inputs.at(i).as<at::Tensor>();
    const auto out_tv = output(i)->as<TensorView>();
    NVF_ERROR(
        !out_tv->hasRoot(),
        "Evaluation for rFactored reductions is not supported.");

    std::vector<int64_t> reduction_axes;
    for (const auto id :
         c10::irange(int64_t(out_tv->getLogicalDomain().size()))) {
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

std::vector<PolymorphicValue> WelfordOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("WelfordOp::evaluate");
  NVF_ERROR(
      !hasInit(),
      "Evaluation for WelfordOp is not implemented for non-empty initial values.");
  const auto& in_tensor = inputs.at(0).as<at::Tensor>();
  const auto out_tv = out()->as<TensorView>();
  NVF_ERROR(
      !out_tv->hasRoot(),
      "Evaluation for WelfordOp is not supported when output is rFactored.");

  int64_t N = 1;
  std::vector<int64_t> reduction_axes;
  for (const auto i : c10::irange(int64_t(out_tv->getLogicalDomain().size()))) {
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

std::vector<PolymorphicValue> ExpandOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("ExpandOp::evaluate");
  const auto& in = inputs.at(0).as<at::Tensor>();
  std::vector<int64_t> expanded_size;
  for (auto i : c10::irange(1, inputs.size())) {
    expanded_size.push_back((int64_t)inputs.at(i));
  }
  return {in.expand(expanded_size)};
}

std::vector<PolymorphicValue> RepeatOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("RepeatOp::evaluate");
  NVF_ERROR(
      inputs.size() == 1,
      "RepeatOp expects exactly 1 input, but received ",
      inputs.size());
  auto tensor = inputs.at(0).as<at::Tensor>();
  std::vector<int64_t> multipliers;
  multipliers.reserve(out()->getLogicalDomain().size());
  const auto c2p =
      PairwiseLogicalDomainMap(in(), out()).mapConsumerToProducer();
  for (const auto i : c10::irange(out()->getLogicalDomain().size())) {
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

std::vector<PolymorphicValue> ViewAsScalar::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("ViewAsScalar::evaluate");
  const at::Tensor& in = inputs.at(0).as<at::Tensor>();
  return {at::view_as_real(in)};
}

std::vector<PolymorphicValue> ViewOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("ViewOp::evaluate");

  NVF_ERROR(inputs.size() == 1);
  const at::Tensor& in_tensor = inputs[0].as<at::Tensor>();

  const std::vector<IterDomain*>& out_logical = out()->getLogicalDomain();
  std::vector<int64_t> out_shape;
  out_shape.reserve(out_logical.size());

  int missing_vals =
      std::count_if(out_logical.begin(), out_logical.end(), [](IterDomain* id) {
        return !id->isDeviceDim() &&
            !id->getMaybeExpandedExtent()->isConstScalar();
      });

  for (IterDomain* id : out_logical) {
    if (id->isDeviceDim()) {
      out_shape.push_back(1);
    } else if (id->getMaybeExpandedExtent()->isConstScalar()) {
      out_shape.push_back(
          id->getMaybeExpandedExtent()->evaluate().as<int64_t>());
    } else {
      if (missing_vals == 1) {
        out_shape.push_back(-1);
      } else {
        out_shape.push_back(
            ee.evaluate(id->getMaybeExpandedExtent()).as<int64_t>());
      }
    }
  }

  // TODO: check allocation domain and contiguity.

  // Use `at::Tensor::reshape` instead of `at::Tensor::view` because `ViewOp`
  // doesn't always produce an alias. For example, when merging an expanded
  // `IterType::Broadcast` and an `IterType::Iteration`, `ViewOp` has to realize
  // the expand.
  if (in_tensor.is_contiguous()) {
    return {in_tensor.view(out_shape)};
  }
  return {in_tensor.reshape(out_shape)};
}

std::vector<PolymorphicValue> LoadStoreOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("LoadStoreOp::evaluate");

  if (TensorView* out_tv = dynamic_cast<TensorView*>(out())) {
    if (out_tv->hasRoot()) {
      std::optional<std::vector<int64_t>> permutation =
          ir_utils::computePermutation(
              out_tv->getRootDomain(), out_tv->getLogicalDomain());
      NVF_ERROR(
          permutation.has_value(),
          "The logical domain of a Set.Permute is supposed to be a permutation of the root domain: ",
          out_tv->toString());
      NVF_ERROR(inputs.size() == 1);
      at::Tensor in_tensor = inputs[0].as<at::Tensor>();
      at::Tensor out_tensor = in_tensor.permute(*permutation);
      return {out_tensor};
    }
  }
  return inputs;
}

std::vector<PolymorphicValue> PadOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("PadOp::evaluate");
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

std::vector<PolymorphicValue> SliceOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("SliceOp::evaluate");
  const auto& in = inputs.at(0).as<at::Tensor>();
  std::vector<at::indexing::TensorIndex> ranges;
  auto ranges_offset = getRangeInputOffset();
  auto num_dims = in.dim();
  for (const auto i : c10::irange(num_dims)) {
    auto start = (int64_t)inputs.at(ranges_offset + 3 * i);
    auto stop = (int64_t)inputs.at(ranges_offset + 3 * i + 1);
    auto step = (int64_t)inputs.at(ranges_offset + 3 * i + 2);
    ranges.emplace_back(at::indexing::Slice(start, stop, step));
  }
  return {in.index(ranges)};
}

std::vector<PolymorphicValue> CatOp::evaluate(
    const ExpressionEvaluator& ee,
    std::unordered_map<const Val*, PolymorphicValue>& known_values) const {
  FUSER_PERF_SCOPE("CatOp::evaluate");
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

std::vector<PolymorphicValue> MatmulOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("MatmulOp::evaluate");
  const auto a = inputs.at(0).as<at::Tensor>();
  const auto b = inputs.at(1).as<at::Tensor>();

  auto matmul_out = at::matmul(a, b);

  // When the contracting dimension is sharded, each device has a partial
  // matmul output and is followed by an allreduce. For loop split, this is
  // represented as an rfactored reduction. The local matmul logical domain
  // after the rfactor is: i{DIDx}, i{M}, i{N}, r{K//d}. Unsqueeze the
  // rfactored DID axis to correctly bind with the logical domain. See
  // tests/python/test_multidevice.py/test_matmul_allreduce_loop_split
  auto out_logical = TensorDomain::noReductions(out()->getLogicalDomain());
  int64_t rfactor_did_idx = -1;
  for (auto idx : c10::irange(static_cast<int64_t>(out_logical.size()))) {
    if (!out_logical.at(idx)->isRFactorProduct() ||
        !out_logical.at(idx)->isDeviceDim()) {
      continue;
    }
    if (rfactor_did_idx != -1) {
      NVF_THROW(
          "Expected only 1 rfactored DID iterdomain, found at least 2 in ",
          out_logical);
    }
    rfactor_did_idx = idx;
  }

  if (rfactor_did_idx != -1) {
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

std::vector<PolymorphicValue> LinearOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("LinearOp::evaluate");
  const auto in = inputs.at(0).as<at::Tensor>();
  auto weight = inputs.at(1).as<at::Tensor>();

  auto squeeze_device_dims = [](at::Tensor& t,
                                int64_t num_device_dims) -> void {
    // Record the initial shape for the error message.
    std::vector<int64_t> shape = t.sizes().vec();
    for ([[maybe_unused]] auto _ : c10::irange(num_device_dims)) {
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

  at::Tensor out;
  if (has_bias()) {
    auto bias = inputs.at(2).as<at::Tensor>();
    squeeze_device_dims(bias, num_device_dims);
    out = at::linear(in, weight, bias);
  } else {
    out = at::linear(in, weight);
  }

  for ([[maybe_unused]] auto _ : c10::irange(num_device_dims)) {
    out = out.unsqueeze(0);
  }
  return {out};
}

std::vector<PolymorphicValue> SdpaFwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("SdpaFwdOp::evaluate");
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

std::vector<PolymorphicValue> SdpaBwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("SdpaBwdOp::evaluate");
  // Backward tensor inputs: grad_input, query, key, value, output,
  // logsumexp, max_q/k Temporary handling of DID parallelization. See
  // https://github.com/NVIDIA/Fuser/issues/2563
  bool first_dim_is_did = this->key()->as<TensorView>()->axis(0)->isDeviceDim();
  auto out_grad = inputs[0].as<at::Tensor>();
  if (first_dim_is_did) {
    NVF_CHECK(out_grad.dim() == 5, "Expected 5D but found ", out_grad.sizes());
  } else {
    NVF_CHECK(out_grad.dim() == 4, "Expected 4D but found ", out_grad.sizes());
  }

  std::vector<at::Tensor> bwd_inputs;
  for (auto idx : c10::irange(6)) {
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

std::vector<PolymorphicValue> EmbeddingFwdOp::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  FUSER_PERF_SCOPE("EmbeddingFwdOp::evaluate");
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

} // namespace nvfuser
