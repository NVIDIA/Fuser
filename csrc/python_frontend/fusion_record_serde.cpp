// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <ops/composite.h>
#include <python_frontend/fusion_record_serde.h>
#include <serde/python_fusion_cache_generated.h>
#include <functional>

namespace nvfuser::python_frontend {

std::vector<State> parseStateArgs(
    const flatbuffers::Vector<const nvfuser::serde::State*>* args) {
  std::vector<State> result;
  for (auto s : *args) {
    result.emplace_back(s->index(), s->type());
  }
  return result;
}

template <typename T>
std::vector<T> parseVector(const flatbuffers::Vector<T>* fb_vector) {
  std::vector<T> result(fb_vector->begin(), fb_vector->end());
  return result;
}

// Flatbuffer stores bool values as uint8_t.
std::vector<bool> parseBoolVector(
    const flatbuffers::Vector<uint8_t>* fb_vector) {
  std::vector<bool> result(fb_vector->begin(), fb_vector->end());
  return result;
}

static PrimDataType mapToNvfuserDtype(serde::DataType t) {
  switch (t) {
    case serde::DataType_Bool:
      return PrimDataType::Bool;
    case serde::DataType_Double:
      return PrimDataType::Double;
    case serde::DataType_Float:
      return PrimDataType::Float;
    case serde::DataType_Half:
      return PrimDataType::Half;
    case serde::DataType_BFloat16:
      return PrimDataType::BFloat16;
    case serde::DataType_Int:
      return PrimDataType::Int;
    case serde::DataType_Int32:
      return PrimDataType::Int32;
    case serde::DataType_ComplexFloat:
      return PrimDataType::ComplexFloat;
    case serde::DataType_ComplexDouble:
      return PrimDataType::ComplexDouble;
    case serde::DataType_None:
      return PrimDataType::Null;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No nvfuser dtype found for serde data type.");
  return PrimDataType::Null;
}

// START deserialize functions

RecordFunctor* deserializeStartRecord(const serde::RecordFunctor* buffer) {
  return new StartRecord();
}

RecordFunctor* deserializeEndRecord(const serde::RecordFunctor* buffer) {
  return new EndRecord();
}

template <class fn_type, class... Signature>
RecordFunctor* deserializeOpRecord(
    const std::unordered_map<std::string, fn_type>& str_to_func_map,
    serde::RecordType record_type,
    const serde::RecordFunctor* buffer) {
  return new OpRecord<Signature...>(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      record_type,
      str_to_func_map.at(buffer->name()->str()));
}

RecordFunctor* deserializeBatchNormRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_BatchNorm();
  return new BatchNormOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      data->training(),
      data->channels_last());
}

RecordFunctor* deserializeBroadcastRecord(const serde::RecordFunctor* buffer) {
  return new BroadcastOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      parseBoolVector(buffer->data_as_Broadcast()->broadcast_dims()));
}

RecordFunctor* deserializeBroadcastInDimRecord(
    const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_BroadcastInDim();
  return new BroadcastInDimOpRecord<int64_t>(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      serde::RecordType_BroadcastInDim,
      parseVector(data->output_shape()),
      parseVector(data->broadcast_dims()));
}

RecordFunctor* deserializeBroadcastInDimSymbolicRecord(
    const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_BroadcastInDimSymbolic();
  return new BroadcastInDimOpRecord<State>(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      serde::RecordType_BroadcastInDimSymbolic,
      parseStateArgs(data->output_shape()),
      parseVector(data->broadcast_dims()));
}

RecordFunctor* deserializeCastTvRecord(const serde::RecordFunctor* buffer) {
  std::function<TensorView*(DataType, TensorView*)> fusion_op =
      static_cast<TensorView* (*)(DataType, TensorView*)>(castOp);
  return new CastOpRecord<TensorView*, TensorView*>(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      serde::RecordType_CastTv,
      fusion_op,
      mapToNvfuserDtype(buffer->data_as_Dtype()->dtype()));
}

RecordFunctor* deserializeCastValRecord(const serde::RecordFunctor* buffer) {
  std::function<Val*(DataType, Val*)> fusion_op =
      static_cast<Val* (*)(DataType, Val*)>(castOp);
  return new CastOpRecord<Val*, Val*>(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      serde::RecordType_CastVal,
      fusion_op,
      mapToNvfuserDtype(buffer->data_as_Dtype()->dtype()));
}

RecordFunctor* deserializeConstantBoolRecord(
    const serde::RecordFunctor* buffer) {
  return new ConstantRecord<Bool, bool>(
      parseStateArgs(buffer->outputs()),
      serde::RecordType_ConstantBool,
      buffer->data_as_Bool()->bool_val(),
      DataType::Bool);
}

RecordFunctor* deserializeConstantDoubleRecord(
    const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Double();
  return new ConstantRecord<Double, double>(
      parseStateArgs(buffer->outputs()),
      serde::RecordType_ConstantDouble,
      data->double_val(),
      mapToNvfuserDtype(data->dtype()));
}

RecordFunctor* deserializeConstantComplexDoubleRecord(
    const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_ComplexDouble();
  return new ConstantRecord<ComplexDouble, std::complex<double>>(
      parseStateArgs(buffer->outputs()),
      serde::RecordType_ConstantComplexDouble,
      std::complex<double>(data->real(), data->imag()),
      mapToNvfuserDtype(data->dtype()));
}

RecordFunctor* deserializeConstantIntRecord(
    const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Int();
  return new ConstantRecord<Int, int64_t>(
      parseStateArgs(buffer->outputs()),
      serde::RecordType_ConstantInt,
      data->int_val(),
      mapToNvfuserDtype(data->dtype()));
}

RecordFunctor* deserializeOutputValRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Output();
  return new OutputRecord<Val>(
      parseStateArgs(buffer->args()),
      serde::RecordType_OutputVal,
      parseVector(data->stride_order()));
}

RecordFunctor* deserializeOutputTvRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Output();
  return new OutputRecord<TensorView>(
      parseStateArgs(buffer->args()),
      serde::RecordType_OutputTv,
      parseVector(data->stride_order()));
}

RecordFunctor* deserializeReductionRecord(
    std::function<
        TensorView*(TensorView*, const std::vector<int>&, bool, DataType)>
        fusion_op,
    serde::RecordType record_type,
    const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Reduction();
  return new ReductionOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      record_type,
      fusion_op,
      parseVector(data->axes()),
      data->keep_dim(),
      mapToNvfuserDtype(data->dtype()));
}

RecordFunctor* deserializeFullRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_TensorCreation();
  return new FullOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      parseVector(data->shape()),
      mapToNvfuserDtype(data->dtype()));
}

RecordFunctor* deserializeIotaRecord(const serde::RecordFunctor* buffer) {
  return new IotaOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      mapToNvfuserDtype(buffer->data_as_Dtype()->dtype()));
}

RecordFunctor* deserializeTorchGatherRecord(
    const serde::RecordFunctor* buffer) {
  return new TorchGatherOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->data_as_Dimension()->dim());
}

RecordFunctor* deserializeIndexSelectRecord(
    const serde::RecordFunctor* buffer) {
  return new IndexSelectOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->data_as_Dimension()->dim());
}

RecordFunctor* deserializePermuteRecord(const serde::RecordFunctor* buffer) {
  return new PermuteOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      parseVector(buffer->data_as_Permute()->dims()));
}

RecordFunctor* deserializeRandomRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_TensorCreationSymbolic();
  return new RandomOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      parseStateArgs(data->shape()),
      buffer->name()->str(),
      mapToNvfuserDtype(data->dtype()));
}

RecordFunctor* deserializeReshapeRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Reshape();
  return new ReshapeOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      parseVector(data->original_shape()),
      parseVector(data->new_shape()));
}

RecordFunctor* deserializeScalarRecord(const serde::RecordFunctor* buffer) {
  return new ScalarRecord(
      parseStateArgs(buffer->outputs()),
      mapToNvfuserDtype(buffer->data_as_Dtype()->dtype()));
}

RecordFunctor* deserializeSqueezeRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Squeeze();
  return new SqueezeOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      parseVector(data->original_shape()),
      parseVector(data->squeeze_dims()));
}

RecordFunctor* deserializeTensorRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Tensor();

  auto mapEnumToOptional = [](int v) -> c10::optional<bool> {
    switch (v) {
      case serde::Contiguity_Strided:
        return c10::optional<bool>(false);
      case serde::Contiguity_Contiguous:
        return c10::optional<bool>(true);
      case serde::Contiguity_None:
        return c10::nullopt;
    }
    TORCH_INTERNAL_ASSERT(false, "Invalid contiguity type.");
    return c10::nullopt;
  };

  std::vector<c10::optional<bool>> contiguous_info;
  std::transform(
      data->contiguity()->cbegin(),
      data->contiguity()->cend(),
      std::back_inserter(contiguous_info),
      mapEnumToOptional);

  return new TensorRecord(
      parseStateArgs(buffer->outputs()),
      parseVector(data->sizes()),
      contiguous_info,
      mapToNvfuserDtype(data->dtype()),
      data->is_cpu());
}

RecordFunctor* deserializeTensorSizesRecord(
    const serde::RecordFunctor* buffer) {
  return new TensorSizesRecord(
      parseStateArgs(buffer->args()), parseStateArgs(buffer->outputs()));
}

RecordFunctor* deserializeVarianceRecord(const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Norm();
  return new VarianceOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      parseVector(data->axes()),
      data->correction(),
      data->keep_dim());
}

RecordFunctor* deserializeVarianceMeanRecord(
    const serde::RecordFunctor* buffer) {
  auto data = buffer->data_as_Norm();
  return new VarianceMeanOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      parseVector(data->axes()),
      data->correction(),
      data->keep_dim());
}

// End deserialize functions

void RecordFunctorFactory::registerAllParsers() {
  registerParser(serde::RecordType_Start, deserializeStartRecord);
  registerParser(serde::RecordType_End, deserializeEndRecord);

  // Unary Ops
  auto unary_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<unary_tv_fn, TensorView*, TensorView*>(
        unary_tv, serde::RecordType_Unary_TV, buffer);
  };
  registerParser(serde::RecordType_Unary_TV, unary_tv_parser);

  auto unary_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<unary_val_fn, Val*, Val*>(
        unary_val, serde::RecordType_Unary_VAL, buffer);
  };
  registerParser(serde::RecordType_Unary_VAL, unary_val_parser);

  // Binary Ops
  auto binary_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        binary_tv_fn,
        TensorView*,
        TensorView*,
        TensorView*>(binary_tv, serde::RecordType_Binary_TV, buffer);
  };
  registerParser(serde::RecordType_Binary_TV, binary_tv_parser);

  auto binary_tv_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        binary_tv_val_fn,
        TensorView*,
        TensorView*,
        Val*>(binary_tv_val, serde::RecordType_Binary_TV_VAL, buffer);
  };
  registerParser(serde::RecordType_Binary_TV_VAL, binary_tv_val_parser);

  auto binary_val_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        binary_val_tv_fn,
        TensorView*,
        Val*,
        TensorView*>(binary_val_tv, serde::RecordType_Binary_VAL_TV, buffer);
  };
  registerParser(serde::RecordType_Binary_VAL_TV, binary_val_tv_parser);

  auto binary_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<binary_val_fn, Val*, Val*, Val*>(
        binary_val, serde::RecordType_Binary_VAL, buffer);
  };
  registerParser(serde::RecordType_Binary_VAL, binary_val_parser);

  // Ternary Ops
  auto ternary_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_fn,
        TensorView*,
        TensorView*,
        TensorView*,
        TensorView*>(ternary_tv, serde::RecordType_Ternary_TV, buffer);
  };
  registerParser(serde::RecordType_Ternary_TV, ternary_tv_parser);

  auto ternary_tv_tv_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_tv_val_fn,
        TensorView*,
        TensorView*,
        TensorView*,
        Val*>(ternary_tv_tv_val, serde::RecordType_Ternary_TV_TV_VAL, buffer);
  };
  registerParser(serde::RecordType_Ternary_TV_TV_VAL, ternary_tv_tv_val_parser);

  auto ternary_tv_val_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_val_tv_fn,
        TensorView*,
        TensorView*,
        Val*,
        TensorView*>(
        ternary_tv_val_tv, serde::RecordType_Ternary_TV_VAL_TV, buffer);
  };
  registerParser(serde::RecordType_Ternary_TV_VAL_TV, ternary_tv_val_tv_parser);

  auto ternary_val_tv_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_val_tv_tv_fn,
        TensorView*,
        Val*,
        TensorView*,
        TensorView*>(
        ternary_val_tv_tv, serde::RecordType_Ternary_VAL_TV_TV, buffer);
  };
  registerParser(serde::RecordType_Ternary_VAL_TV_TV, ternary_val_tv_tv_parser);

  auto ternary_val_val_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_val_val_tv_fn,
        TensorView*,
        Val*,
        Val*,
        TensorView*>(
        ternary_val_val_tv, serde::RecordType_Ternary_VAL_VAL_TV, buffer);
  };
  registerParser(
      serde::RecordType_Ternary_VAL_VAL_TV, ternary_val_val_tv_parser);

  auto ternary_tv_val_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_val_val_fn,
        TensorView*,
        TensorView*,
        Val*,
        Val*>(ternary_tv_val_val, serde::RecordType_Ternary_TV_VAL_VAL, buffer);
  };
  registerParser(
      serde::RecordType_Ternary_TV_VAL_VAL, ternary_tv_val_val_parser);

  auto ternary_val_tv_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_val_tv_val_fn,
        TensorView*,
        Val*,
        TensorView*,
        Val*>(ternary_val_tv_val, serde::RecordType_Ternary_VAL_TV_VAL, buffer);
  };
  registerParser(
      serde::RecordType_Ternary_VAL_TV_VAL, ternary_val_tv_val_parser);

  auto ternary_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<ternary_val_fn, Val*, Val*, Val*, Val*>(
        ternary_val, serde::RecordType_Ternary_VAL, buffer);
  };
  registerParser(serde::RecordType_Ternary_VAL, ternary_val_parser);

  // Ternary-Alpha Ops
  auto ternary_alpha_tv_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_tv_fn,
        TensorView*,
        TensorView*,
        TensorView*,
        TensorView*,
        Val*>(ternary_alpha_tv, serde::RecordType_Ternary_Alpha_TV, buffer);
  };
  registerParser(serde::RecordType_Ternary_Alpha_TV, ternary_alpha_tv_parser);

  auto ternary_alpha_tv_tv_val_parser =
      [&](const serde::RecordFunctor* buffer) {
        return deserializeOpRecord<
            ternary_alpha_tv_tv_val_fn,
            TensorView*,
            TensorView*,
            TensorView*,
            Val*,
            Val*>(
            ternary_alpha_tv_tv_val,
            serde::RecordType_Ternary_Alpha_TV_TV_VAL,
            buffer);
      };
  registerParser(
      serde::RecordType_Ternary_Alpha_TV_TV_VAL,
      ternary_alpha_tv_tv_val_parser);

  auto ternary_alpha_tv_val_tv_parser =
      [&](const serde::RecordFunctor* buffer) {
        return deserializeOpRecord<
            ternary_alpha_tv_val_tv_fn,
            TensorView*,
            TensorView*,
            Val*,
            TensorView*,
            Val*>(
            ternary_alpha_tv_val_tv,
            serde::RecordType_Ternary_Alpha_TV_VAL_TV,
            buffer);
      };
  registerParser(
      serde::RecordType_Ternary_Alpha_TV_VAL_TV,
      ternary_alpha_tv_val_tv_parser);

  auto ternary_alpha_val_tv_tv_parser =
      [&](const serde::RecordFunctor* buffer) {
        return deserializeOpRecord<
            ternary_alpha_val_tv_tv_fn,
            TensorView*,
            Val*,
            TensorView*,
            TensorView*,
            Val*>(
            ternary_alpha_val_tv_tv,
            serde::RecordType_Ternary_Alpha_VAL_TV_TV,
            buffer);
      };
  registerParser(
      serde::RecordType_Ternary_Alpha_VAL_TV_TV,
      ternary_alpha_val_tv_tv_parser);

  auto ternary_alpha_val_val_tv_parser =
      [&](const serde::RecordFunctor* buffer) {
        return deserializeOpRecord<
            ternary_alpha_val_val_tv_fn,
            TensorView*,
            Val*,
            Val*,
            TensorView*,
            Val*>(
            ternary_alpha_val_val_tv,
            serde::RecordType_Ternary_Alpha_VAL_VAL_TV,
            buffer);
      };
  registerParser(
      serde::RecordType_Ternary_Alpha_VAL_VAL_TV,
      ternary_alpha_val_val_tv_parser);

  auto ternary_alpha_tv_val_val_parser =
      [&](const serde::RecordFunctor* buffer) {
        return deserializeOpRecord<
            ternary_alpha_tv_val_val_fn,
            TensorView*,
            TensorView*,
            Val*,
            Val*,
            Val*>(
            ternary_alpha_tv_val_val,
            serde::RecordType_Ternary_Alpha_TV_VAL_VAL,
            buffer);
      };
  registerParser(
      serde::RecordType_Ternary_Alpha_TV_VAL_VAL,
      ternary_alpha_tv_val_val_parser);

  auto ternary_alpha_val_tv_val_parser =
      [&](const serde::RecordFunctor* buffer) {
        return deserializeOpRecord<
            ternary_alpha_val_tv_val_fn,
            TensorView*,
            Val*,
            TensorView*,
            Val*,
            Val*>(
            ternary_alpha_val_tv_val,
            serde::RecordType_Ternary_Alpha_VAL_TV_VAL,
            buffer);
      };
  registerParser(
      serde::RecordType_Ternary_Alpha_VAL_TV_VAL,
      ternary_alpha_val_tv_val_parser);

  auto ternary_alpha_val_parser = [&](const serde::RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_val_fn,
        Val*,
        Val*,
        Val*,
        Val*,
        Val*>(ternary_alpha_val, serde::RecordType_Ternary_Alpha_VAL, buffer);
  };
  registerParser(serde::RecordType_Ternary_Alpha_VAL, ternary_alpha_val_parser);

  registerParser(serde::RecordType_BatchNormOp, deserializeBatchNormRecord);

  registerParser(serde::RecordType_BroadcastOp, deserializeBroadcastRecord);

  registerParser(
      serde::RecordType_BroadcastInDim, deserializeBroadcastInDimRecord);

  registerParser(
      serde::RecordType_BroadcastInDimSymbolic,
      deserializeBroadcastInDimSymbolicRecord);

  registerParser(serde::RecordType_CastTv, deserializeCastTvRecord);

  registerParser(serde::RecordType_CastVal, deserializeCastValRecord);

  registerParser(serde::RecordType_ConstantBool, deserializeConstantBoolRecord);

  registerParser(serde::RecordType_ConstantInt, deserializeConstantIntRecord);

  registerParser(
      serde::RecordType_ConstantDouble, deserializeConstantDoubleRecord);

  registerParser(
      serde::RecordType_ConstantComplexDouble,
      deserializeConstantComplexDoubleRecord);

  registerParser(serde::RecordType_FullOp, deserializeFullRecord);

  registerParser(serde::RecordType_IotaOp, deserializeIotaRecord);

  registerParser(serde::RecordType_TorchGatherOp, deserializeTorchGatherRecord);

  registerParser(serde::RecordType_IndexSelectOp, deserializeIndexSelectRecord);

  registerParser(serde::RecordType_OutputTv, deserializeOutputTvRecord);

  registerParser(serde::RecordType_OutputVal, deserializeOutputValRecord);

  registerParser(serde::RecordType_PermuteOp, deserializePermuteRecord);

  registerParser(serde::RecordType_RandomOp, deserializeRandomRecord);

  auto reduction_max_parser = [](const serde::RecordFunctor* buffer) {
    return deserializeReductionRecord(
        max, serde::RecordType_ReductionMax, buffer);
  };
  registerParser(serde::RecordType_ReductionMax, reduction_max_parser);

  auto reduction_min_parser = [](const serde::RecordFunctor* buffer) {
    return deserializeReductionRecord(
        min, serde::RecordType_ReductionMin, buffer);
  };
  registerParser(serde::RecordType_ReductionMin, reduction_min_parser);

  auto reduction_prod_parser = [](const serde::RecordFunctor* buffer) {
    return deserializeReductionRecord(
        prod, serde::RecordType_ReductionProd, buffer);
  };
  registerParser(serde::RecordType_ReductionProd, reduction_prod_parser);

  auto reduction_sum_parser = [](const serde::RecordFunctor* buffer) {
    return deserializeReductionRecord(
        sum, serde::RecordType_ReductionSum, buffer);
  };
  registerParser(serde::RecordType_ReductionSum, reduction_sum_parser);

  registerParser(serde::RecordType_ReshapeOp, deserializeReshapeRecord);

  registerParser(serde::RecordType_Scalar, deserializeScalarRecord);

  registerParser(serde::RecordType_SqueezeOp, deserializeSqueezeRecord);

  registerParser(serde::RecordType_Tensor, deserializeTensorRecord);

  registerParser(serde::RecordType_TensorSizes, deserializeTensorSizesRecord);

  registerParser(serde::RecordType_VarianceOp, deserializeVarianceRecord);

  registerParser(
      serde::RecordType_VarianceMeanOp, deserializeVarianceMeanRecord);
}

void RecordFunctorFactory::setupFunctionMaps() {
#define NVFUSER_UNARY_TV_OP(op_str, op_name)                                \
  unary_tv.emplace(                                                         \
      ("ops." op_str), static_cast<TensorView* (*)(TensorView*)>(op_name)); \
  unary_val.emplace(("ops." op_str), static_cast<Val* (*)(Val*)>(op_name));

#define NVFUSER_BINARY_TV_OP(op_str, op_name)                           \
  binary_tv.emplace(                                                    \
      ("ops." op_str),                                                  \
      static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name)); \
  binary_val.emplace(                                                   \
      ("ops." op_str), static_cast<Val* (*)(Val*, Val*)>(op_name));     \
  binary_tv_val.emplace(                                                \
      ("ops." op_str),                                                  \
      static_cast<TensorView* (*)(TensorView*, Val*)>(op_name));        \
  binary_val_tv.emplace(                                                \
      ("ops." op_str),                                                  \
      static_cast<TensorView* (*)(Val*, TensorView*)>(op_name));

#define NVFUSER_BINARY_TV_ALPHA_OP(op_str, op_name)                       \
  ternary_val.emplace(                                                    \
      ("ops." op_str), static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)); \
  ternary_tv_tv_val.emplace(                                              \
      ("ops." op_str),                                                    \
      static_cast<                                                        \
                                                                          \
          TensorView* (*)(TensorView*, TensorView*, Val*)>(op_name));     \
  ternary_tv_val_val.emplace(                                             \
      ("ops." op_str),                                                    \
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name));    \
  ternary_val_tv_val.emplace(                                             \
      ("ops." op_str),                                                    \
      static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name));

#define NVFUSER_TERNARY_TV_OP(op_str, op_name)                               \
  ternary_tv.emplace(                                                        \
      ("ops." op_str),                                                       \
      static_cast<                                                           \
                                                                             \
          TensorView* (*)(TensorView*, TensorView*, TensorView*)>(op_name)); \
  ternary_val.emplace(                                                       \
      ("ops." op_str), static_cast<Val* (*)(Val*, Val*, Val*)>(op_name));    \
  ternary_tv_tv_val.emplace(                                                 \
      ("ops." op_str),                                                       \
      static_cast<                                                           \
                                                                             \
          TensorView* (*)(TensorView*, TensorView*, Val*)>(op_name));        \
  ternary_tv_val_tv.emplace(                                                 \
      ("ops." op_str),                                                       \
      static_cast<                                                           \
                                                                             \
          TensorView* (*)(TensorView*, Val*, TensorView*)>(op_name));        \
  ternary_val_tv_tv.emplace(                                                 \
      ("ops." op_str),                                                       \
      static_cast<                                                           \
                                                                             \
          TensorView* (*)(Val*, TensorView*, TensorView*)>(op_name));        \
  ternary_val_val_tv.emplace(                                                \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(Val*, Val*, TensorView*)>(op_name));       \
  ternary_tv_val_val.emplace(                                                \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name));       \
  ternary_val_tv_val.emplace(                                                \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name));

#define NVFUSER_THRESHOLD_TV_OP(op_str, op_name)                          \
  ternary_val.emplace(                                                    \
      ("ops." op_str), static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)); \
  ternary_tv_val_val.emplace(                                             \
      ("ops." op_str),                                                    \
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name));

#define NVFUSER_TERNARY_TV_ALPHA_OP(op_str, op_name)                        \
  ternary_alpha_tv.emplace(                                                 \
      ("ops." op_str),                                                      \
      static_cast<                                                          \
                                                                            \
          TensorView* (*)(TensorView*, TensorView*, TensorView*, Val*)>(    \
          op_name));                                                        \
  ternary_alpha_val.emplace(                                                \
      ("ops." op_str),                                                      \
      static_cast<Val* (*)(Val*, Val*, Val*, Val*)>(op_name));              \
  ternary_alpha_tv_tv_val.emplace(                                          \
      ("ops." op_str),                                                      \
      static_cast<                                                          \
                                                                            \
          TensorView* (*)(TensorView*, TensorView*, Val*, Val*)>(op_name)); \
  ternary_alpha_tv_val_tv.emplace(                                          \
      ("ops." op_str),                                                      \
      static_cast<                                                          \
                                                                            \
          TensorView* (*)(TensorView*, Val*, TensorView*, Val*)>(op_name)); \
  ternary_alpha_val_tv_tv.emplace(                                          \
      ("ops." op_str),                                                      \
      static_cast<                                                          \
                                                                            \
          TensorView* (*)(Val*, TensorView*, TensorView*, Val*)>(op_name)); \
  ternary_alpha_val_val_tv.emplace(                                         \
      ("ops." op_str),                                                      \
      static_cast<                                                          \
                                                                            \
          TensorView* (*)(Val*, Val*, TensorView*, Val*)>(op_name));        \
  ternary_alpha_tv_val_val.emplace(                                         \
      ("ops." op_str),                                                      \
      static_cast<                                                          \
                                                                            \
          TensorView* (*)(TensorView*, Val*, Val*, Val*)>(op_name));        \
  ternary_alpha_val_tv_val.emplace(                                         \
      ("ops." op_str),                                                      \
      static_cast<                                                          \
                                                                            \
          TensorView* (*)(Val*, TensorView*, Val*, Val*)>(op_name));

  NVFUSER_UNARY_TV_OP("abs", abs)
  NVFUSER_UNARY_TV_OP("acos", acos)
  NVFUSER_UNARY_TV_OP("acosh", acosh)
  NVFUSER_UNARY_TV_OP("asin", asin)
  NVFUSER_UNARY_TV_OP("asinh", asinh)
  NVFUSER_UNARY_TV_OP("atan", atan)
  NVFUSER_UNARY_TV_OP("atanh", atanh)
  NVFUSER_UNARY_TV_OP("ceil", ceil)
  NVFUSER_UNARY_TV_OP("cos", cos)
  NVFUSER_UNARY_TV_OP("cosh", cosh)
  NVFUSER_UNARY_TV_OP("exp", exp)
  NVFUSER_UNARY_TV_OP("exp2", exp2)
  NVFUSER_UNARY_TV_OP("expm1", expm1)
  NVFUSER_UNARY_TV_OP("erf", erf)
  NVFUSER_UNARY_TV_OP("erfc", erfc)
  NVFUSER_UNARY_TV_OP("erfinv", erfinv)
  NVFUSER_UNARY_TV_OP("erfcinv", erfcinv)
  NVFUSER_UNARY_TV_OP("floor", floor)
  NVFUSER_UNARY_TV_OP("frac", frac)
  NVFUSER_UNARY_TV_OP("lgamma", lgamma)
  NVFUSER_UNARY_TV_OP("log", log)
  NVFUSER_UNARY_TV_OP("log10", log10)
  NVFUSER_UNARY_TV_OP("log1p", log1p)
  NVFUSER_UNARY_TV_OP("log2", log2)
  NVFUSER_UNARY_TV_OP("neg", neg)
  NVFUSER_UNARY_TV_OP("bitwise_not", bitwise_not)
  NVFUSER_UNARY_TV_OP("relu", relu)
  NVFUSER_UNARY_TV_OP("rand_like", rand_like)
  NVFUSER_UNARY_TV_OP("randn_like", randn_like)
  NVFUSER_UNARY_TV_OP("reciprocal", reciprocal)
  NVFUSER_UNARY_TV_OP("round", round)
  NVFUSER_UNARY_TV_OP("rsqrt", rsqrt)
  NVFUSER_UNARY_TV_OP("set", set)
  NVFUSER_UNARY_TV_OP("sign", sign)
  NVFUSER_UNARY_TV_OP("sigmoid", sigmoid)
  NVFUSER_UNARY_TV_OP("silu", silu)
  NVFUSER_UNARY_TV_OP("sin", sin)
  NVFUSER_UNARY_TV_OP("sinh", sinh)
  NVFUSER_UNARY_TV_OP("sqrt", sqrt)
  NVFUSER_UNARY_TV_OP("tan", tan)
  NVFUSER_UNARY_TV_OP("tanh", tanh)
  NVFUSER_UNARY_TV_OP("trunc", trunc)
  NVFUSER_UNARY_TV_OP("isfinite", isfinite)
  NVFUSER_UNARY_TV_OP("isinf", isinf)
  NVFUSER_UNARY_TV_OP("isnan", isnan)
  NVFUSER_UNARY_TV_OP("isneginf", isneginf)
  NVFUSER_UNARY_TV_OP("isposinf", isposinf)
  NVFUSER_UNARY_TV_OP("isreal", isreal)
  NVFUSER_UNARY_TV_OP("real", real)
  NVFUSER_UNARY_TV_OP("imag", imag)

  NVFUSER_BINARY_TV_OP("add", add)
  NVFUSER_BINARY_TV_OP("atan2", atan2)
  NVFUSER_BINARY_TV_OP("div", div)
  NVFUSER_BINARY_TV_OP("fmod", fmod)
  NVFUSER_BINARY_TV_OP("mul", mul)
  NVFUSER_BINARY_TV_OP("pow", pow)
  NVFUSER_BINARY_TV_OP("remainder", remainder)
  NVFUSER_BINARY_TV_OP("sub", sub)
  NVFUSER_BINARY_TV_OP("mod", mod)
  NVFUSER_BINARY_TV_OP("eq", eq)
  NVFUSER_BINARY_TV_OP("ge", ge)
  NVFUSER_BINARY_TV_OP("gt", gt)
  NVFUSER_BINARY_TV_OP("le", le)
  NVFUSER_BINARY_TV_OP("lt", lt)
  NVFUSER_BINARY_TV_OP("ne", ne)
  NVFUSER_BINARY_TV_OP("bitwise_and", bitwise_and)
  NVFUSER_BINARY_TV_OP("bitwise_or", bitwise_or)
  NVFUSER_BINARY_TV_OP("bitwise_xor", bitwise_xor)
  NVFUSER_BINARY_TV_OP("bitwise_left_shift", bitwise_left_shift)
  NVFUSER_BINARY_TV_OP("bitwise_right_shift", bitwise_left_shift)

  NVFUSER_BINARY_TV_ALPHA_OP("add_alpha", add_alpha)
  NVFUSER_BINARY_TV_ALPHA_OP("sub_alpha", sub_alpha)

  NVFUSER_TERNARY_TV_OP("lerp", lerp)
  NVFUSER_TERNARY_TV_OP("where", where)

  NVFUSER_THRESHOLD_TV_OP("clamp", clamp)
  NVFUSER_THRESHOLD_TV_OP("threshold", threshold)

  NVFUSER_TERNARY_TV_ALPHA_OP("addcmul", addcmul)
}

} // namespace nvfuser::python_frontend
