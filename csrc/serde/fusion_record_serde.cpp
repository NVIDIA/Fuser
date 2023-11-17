// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <ops/composite.h>
#include <python_frontend/fusion_record.h>
#include <serde/fusion_cache_generated.h>
#include <serde/fusion_record_serde.h>
#include <functional>

namespace nvfuser::serde {

std::vector<python_frontend::State> parseStateArgs(
    const flatbuffers::Vector<const State*>* args) {
  std::vector<python_frontend::State> result;
  for (auto s : *args) {
    result.emplace_back(s->index(), s->type());
  }
  return result;
}

std::optional<bool> mapContiguityEnumToOptional(Contiguity v) {
  switch (v) {
    case Contiguity::Strided:
      return std::optional<bool>(false);
    case Contiguity::Contiguous:
      return std::optional<bool>(true);
    case Contiguity::None:
      return std::nullopt;
  }
  NVF_ERROR(false, "Invalid contiguity type.");
  return std::nullopt;
}

template <class fn_type, class... Signature>
python_frontend::RecordFunctor* deserializeOpRecord(
    const std::unordered_map<std::string, fn_type>& str_to_func_map,
    RecordType record_type,
    const RecordFunctor* buffer) {
  NVF_ERROR(
      str_to_func_map.find(buffer->name()->str()) != str_to_func_map.end(),
      "Missing mapping from operation string to nvfuser function in serde deserialization.");
  return new python_frontend::OpRecord<Signature...>(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      record_type,
      str_to_func_map.at(buffer->name()->str()));
}

python_frontend::RecordFunctor* deserializeReductionRecord(
    std::function<TensorView*(
        TensorView*,
        const std::vector<int>&,
        bool,
        nvfuser::DataType)> fusion_op,
    RecordType record_type,
    const RecordFunctor* buffer) {
  auto data = buffer->data_as_Reduction();
  return new python_frontend::ReductionOpRecord(
      parseStateArgs(buffer->args()),
      parseStateArgs(buffer->outputs()),
      buffer->name()->str(),
      record_type,
      fusion_op,
      parseVector(data->axes()),
      data->keep_dim(),
      mapToNvfuserDtype(data->dtype()));
}

void RecordFunctorFactory::registerAllParsers() {
  auto deserializeStartRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::StartRecord();
  };
  registerParser(RecordType::Start, deserializeStartRecord);

  auto deserializeEndRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::EndRecord();
  };
  registerParser(RecordType::End, deserializeEndRecord);

  // Unary Ops
  auto unary_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<unary_tv_fn, TensorView*, TensorView*>(
        unary_tv, RecordType::Unary_TV, buffer);
  };
  registerParser(RecordType::Unary_TV, unary_tv_parser);

  auto unary_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<unary_val_fn, Val*, Val*>(
        unary_val, RecordType::Unary_VAL, buffer);
  };
  registerParser(RecordType::Unary_VAL, unary_val_parser);

  // Binary Ops
  auto binary_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        binary_tv_fn,
        TensorView*,
        TensorView*,
        TensorView*>(binary_tv, RecordType::Binary_TV, buffer);
  };
  registerParser(RecordType::Binary_TV, binary_tv_parser);

  auto binary_tv_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        binary_tv_val_fn,
        TensorView*,
        TensorView*,
        Val*>(binary_tv_val, RecordType::Binary_TV_VAL, buffer);
  };
  registerParser(RecordType::Binary_TV_VAL, binary_tv_val_parser);

  auto binary_val_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        binary_val_tv_fn,
        TensorView*,
        Val*,
        TensorView*>(binary_val_tv, RecordType::Binary_VAL_TV, buffer);
  };
  registerParser(RecordType::Binary_VAL_TV, binary_val_tv_parser);

  auto binary_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<binary_val_fn, Val*, Val*, Val*>(
        binary_val, RecordType::Binary_VAL, buffer);
  };
  registerParser(RecordType::Binary_VAL, binary_val_parser);

  // Ternary Ops
  auto ternary_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_fn,
        TensorView*,
        TensorView*,
        TensorView*,
        TensorView*>(ternary_tv, RecordType::Ternary_TV, buffer);
  };
  registerParser(RecordType::Ternary_TV, ternary_tv_parser);

  auto ternary_tv_tv_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_tv_val_fn,
        TensorView*,
        TensorView*,
        TensorView*,
        Val*>(ternary_tv_tv_val, RecordType::Ternary_TV_TV_VAL, buffer);
  };
  registerParser(RecordType::Ternary_TV_TV_VAL, ternary_tv_tv_val_parser);

  auto ternary_tv_val_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_val_tv_fn,
        TensorView*,
        TensorView*,
        Val*,
        TensorView*>(ternary_tv_val_tv, RecordType::Ternary_TV_VAL_TV, buffer);
  };
  registerParser(RecordType::Ternary_TV_VAL_TV, ternary_tv_val_tv_parser);

  auto ternary_val_tv_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_val_tv_tv_fn,
        TensorView*,
        Val*,
        TensorView*,
        TensorView*>(ternary_val_tv_tv, RecordType::Ternary_VAL_TV_TV, buffer);
  };
  registerParser(RecordType::Ternary_VAL_TV_TV, ternary_val_tv_tv_parser);

  auto ternary_val_val_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_val_val_tv_fn,
        TensorView*,
        Val*,
        Val*,
        TensorView*>(
        ternary_val_val_tv, RecordType::Ternary_VAL_VAL_TV, buffer);
  };
  registerParser(RecordType::Ternary_VAL_VAL_TV, ternary_val_val_tv_parser);

  auto ternary_tv_val_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_tv_val_val_fn,
        TensorView*,
        TensorView*,
        Val*,
        Val*>(ternary_tv_val_val, RecordType::Ternary_TV_VAL_VAL, buffer);
  };
  registerParser(RecordType::Ternary_TV_VAL_VAL, ternary_tv_val_val_parser);

  auto ternary_val_tv_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_val_tv_val_fn,
        TensorView*,
        Val*,
        TensorView*,
        Val*>(ternary_val_tv_val, RecordType::Ternary_VAL_TV_VAL, buffer);
  };
  registerParser(RecordType::Ternary_VAL_TV_VAL, ternary_val_tv_val_parser);

  auto ternary_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<ternary_val_fn, Val*, Val*, Val*, Val*>(
        ternary_val, RecordType::Ternary_VAL, buffer);
  };
  registerParser(RecordType::Ternary_VAL, ternary_val_parser);

  // Ternary-Alpha Ops
  auto ternary_alpha_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_tv_fn,
        TensorView*,
        TensorView*,
        TensorView*,
        TensorView*,
        Val*>(ternary_alpha_tv, RecordType::Ternary_Alpha_TV, buffer);
  };
  registerParser(RecordType::Ternary_Alpha_TV, ternary_alpha_tv_parser);

  auto ternary_alpha_tv_tv_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_tv_tv_val_fn,
        TensorView*,
        TensorView*,
        TensorView*,
        Val*,
        Val*>(
        ternary_alpha_tv_tv_val, RecordType::Ternary_Alpha_TV_TV_VAL, buffer);
  };
  registerParser(
      RecordType::Ternary_Alpha_TV_TV_VAL, ternary_alpha_tv_tv_val_parser);

  auto ternary_alpha_tv_val_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_tv_val_tv_fn,
        TensorView*,
        TensorView*,
        Val*,
        TensorView*,
        Val*>(
        ternary_alpha_tv_val_tv, RecordType::Ternary_Alpha_TV_VAL_TV, buffer);
  };
  registerParser(
      RecordType::Ternary_Alpha_TV_VAL_TV, ternary_alpha_tv_val_tv_parser);

  auto ternary_alpha_val_tv_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_val_tv_tv_fn,
        TensorView*,
        Val*,
        TensorView*,
        TensorView*,
        Val*>(
        ternary_alpha_val_tv_tv, RecordType::Ternary_Alpha_VAL_TV_TV, buffer);
  };
  registerParser(
      RecordType::Ternary_Alpha_VAL_TV_TV, ternary_alpha_val_tv_tv_parser);

  auto ternary_alpha_val_val_tv_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_val_val_tv_fn,
        TensorView*,
        Val*,
        Val*,
        TensorView*,
        Val*>(
        ternary_alpha_val_val_tv, RecordType::Ternary_Alpha_VAL_VAL_TV, buffer);
  };
  registerParser(
      RecordType::Ternary_Alpha_VAL_VAL_TV, ternary_alpha_val_val_tv_parser);

  auto ternary_alpha_tv_val_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_tv_val_val_fn,
        TensorView*,
        TensorView*,
        Val*,
        Val*,
        Val*>(
        ternary_alpha_tv_val_val, RecordType::Ternary_Alpha_TV_VAL_VAL, buffer);
  };
  registerParser(
      RecordType::Ternary_Alpha_TV_VAL_VAL, ternary_alpha_tv_val_val_parser);

  auto ternary_alpha_val_tv_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_val_tv_val_fn,
        TensorView*,
        Val*,
        TensorView*,
        Val*,
        Val*>(
        ternary_alpha_val_tv_val, RecordType::Ternary_Alpha_VAL_TV_VAL, buffer);
  };
  registerParser(
      RecordType::Ternary_Alpha_VAL_TV_VAL, ternary_alpha_val_tv_val_parser);

  auto ternary_alpha_val_parser = [&](const RecordFunctor* buffer) {
    return deserializeOpRecord<
        ternary_alpha_val_fn,
        Val*,
        Val*,
        Val*,
        Val*,
        Val*>(ternary_alpha_val, RecordType::Ternary_Alpha_VAL, buffer);
  };
  registerParser(RecordType::Ternary_Alpha_VAL, ternary_alpha_val_parser);
  // END OpRecord Parsers

  // START Reduction Parsers
  auto reduction_max_parser = [](const RecordFunctor* buffer) {
    return deserializeReductionRecord(max, RecordType::ReductionMax, buffer);
  };
  registerParser(RecordType::ReductionMax, reduction_max_parser);

  auto reduction_min_parser = [](const RecordFunctor* buffer) {
    return deserializeReductionRecord(min, RecordType::ReductionMin, buffer);
  };
  registerParser(RecordType::ReductionMin, reduction_min_parser);

  auto reduction_prod_parser = [](const RecordFunctor* buffer) {
    return deserializeReductionRecord(prod, RecordType::ReductionProd, buffer);
  };
  registerParser(RecordType::ReductionProd, reduction_prod_parser);

  auto reduction_sum_parser = [](const RecordFunctor* buffer) {
    return deserializeReductionRecord(sum, RecordType::ReductionSum, buffer);
  };
  registerParser(RecordType::ReductionSum, reduction_sum_parser);
  // END Reduction Parsers

  auto deserializeBatchNormRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_BatchNorm();
    return new python_frontend::BatchNormOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        data->training(),
        data->channels_last());
  };
  registerParser(RecordType::BatchNormOp, deserializeBatchNormRecord);

  auto deserializeBroadcastRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::BroadcastOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        buffer->name()->str(),
        parseBoolVector(buffer->data_as_Broadcast()->broadcast_dims()));
  };
  registerParser(RecordType::BroadcastOp, deserializeBroadcastRecord);

  auto deserializeCatRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::CatOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        buffer->data_as_Dimension()->dim());
  };
  registerParser(RecordType::CatOp, deserializeCatRecord);

  auto deserializeBroadcastInDimRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_BroadcastInDim();
    return new python_frontend::BroadcastInDimOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        data->output_size(),
        parseVector(data->broadcast_dims()));
  };
  registerParser(RecordType::BroadcastInDim, deserializeBroadcastInDimRecord);

  auto deserializeCastTvRecord = [](const RecordFunctor* buffer) {
    std::function<TensorView*(nvfuser::DataType, TensorView*)> fusion_op =
        static_cast<TensorView* (*)(nvfuser::DataType, TensorView*)>(castOp);
    return new python_frontend::CastOpRecord<TensorView*, TensorView*>(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        buffer->name()->str(),
        RecordType::CastTv,
        fusion_op,
        mapToNvfuserDtype(buffer->data_as_Dtype()->dtype()));
  };
  registerParser(RecordType::CastTv, deserializeCastTvRecord);

  auto deserializeCastValRecord = [](const RecordFunctor* buffer) {
    std::function<Val*(nvfuser::DataType, Val*)> fusion_op =
        static_cast<Val* (*)(nvfuser::DataType, Val*)>(castOp);
    return new python_frontend::CastOpRecord<Val*, Val*>(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        buffer->name()->str(),
        RecordType::CastVal,
        fusion_op,
        mapToNvfuserDtype(buffer->data_as_Dtype()->dtype()));
  };
  registerParser(RecordType::CastVal, deserializeCastValRecord);

  auto deserializeScalarRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::ScalarRecord(
        parseStateArgs(buffer->outputs()),
        deserializePolymorphicValue(buffer->data_as_Scalar()),
        mapToNvfuserDtype(buffer->data_as_Scalar()->dtype()));
  };
  registerParser(RecordType::Scalar, deserializeScalarRecord);

  auto deserializeFullRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_TensorCreationSymbolic();
    return new python_frontend::FullOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        mapToNvfuserDtype(data->dtype()));
  };
  registerParser(RecordType::FullOp, deserializeFullRecord);

  auto deserializeIotaRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::IotaOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        mapToNvfuserDtype(buffer->data_as_Dtype()->dtype()));
  };
  registerParser(RecordType::IotaOp, deserializeIotaRecord);

  auto deserializeTorchGatherRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::TorchGatherOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        buffer->data_as_Dimension()->dim());
  };
  registerParser(RecordType::TorchGatherOp, deserializeTorchGatherRecord);

  auto deserializeTakeAlongAxisRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::TakeAlongAxisOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        buffer->data_as_Dimension()->dim());
  };
  registerParser(RecordType::TakeAlongAxisOp, deserializeTakeAlongAxisRecord);

  auto deserializeIndexSelectRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::IndexSelectOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        buffer->data_as_Dimension()->dim());
  };
  registerParser(RecordType::IndexSelectOp, deserializeIndexSelectRecord);

  auto deserializeOutputTvRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Output();
    return new python_frontend::OutputRecord<TensorView>(
        parseStateArgs(buffer->args()),
        RecordType::OutputTv,
        parseVector(data->stride_order()));
  };
  registerParser(RecordType::OutputTv, deserializeOutputTvRecord);

  auto deserializeOutputValRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Output();
    return new python_frontend::OutputRecord<Val>(
        parseStateArgs(buffer->args()),
        RecordType::OutputVal,
        parseVector(data->stride_order()));
  };
  registerParser(RecordType::OutputVal, deserializeOutputValRecord);

  auto deserializePadRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::PadOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        parseVector(buffer->data_as_Pad()->pad_widths()));
  };
  registerParser(RecordType::PadOp, deserializePadRecord);

  auto deserializePermuteRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::DimsOpRecord<RecordType::PermuteOp>(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        parseVector(buffer->data_as_Dims()->dims()),
        buffer->name()->str());
  };
  registerParser(RecordType::PermuteOp, deserializePermuteRecord);

  auto deserializeStrideOrderRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::DimsOpRecord<RecordType::StrideOrderOp>(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        parseVector(buffer->data_as_Dims()->dims()),
        buffer->name()->str());
  };
  registerParser(RecordType::StrideOrderOp, deserializeStrideOrderRecord);

  auto deserializeNormalDistRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_TensorCreationSymbolic();
    return new python_frontend::RandomDistOpRecord<RecordType::NormalDistOp>(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        mapToNvfuserDtype(data->dtype()));
  };
  registerParser(RecordType::NormalDistOp, deserializeNormalDistRecord);

  auto deserializeUniformDistRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_TensorCreationSymbolic();
    return new python_frontend::RandomDistOpRecord<RecordType::UniformDistOp>(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        mapToNvfuserDtype(data->dtype()));
  };
  registerParser(RecordType::UniformDistOp, deserializeUniformDistRecord);

  auto deserializeReshapeRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::ReshapeOpRecord(
        parseStateArgs(buffer->args()), parseStateArgs(buffer->outputs()));
  };
  registerParser(RecordType::ReshapeOp, deserializeReshapeRecord);

  auto deserializeSliceRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Slice();
    return new python_frontend::SliceOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        parseVector(data->start_indices()),
        parseVector(data->end_indices()),
        parseVector(data->strides()));
  };
  registerParser(RecordType::SliceOp, deserializeSliceRecord);

  auto deserializeSqueezeRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Squeeze();
    return new python_frontend::SqueezeOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        parseVector(data->original_shape()),
        parseVector(data->squeeze_dims()));
  };
  registerParser(RecordType::SqueezeOp, deserializeSqueezeRecord);

  auto deserializeTensorRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Tensor();

    std::vector<std::optional<bool>> contiguity;
    std::transform(
        data->contiguity()->cbegin(),
        data->contiguity()->cend(),
        std::back_inserter(contiguity),
        mapContiguityEnumToOptional);

    return new python_frontend::TensorRecord(
        parseStateArgs(buffer->outputs()),
        parseVector(data->sizes()),
        contiguity,
        mapToNvfuserDtype(data->dtype()),
        data->is_cpu(),
        parseVector(data->stride_order()));
  };
  registerParser(RecordType::Tensor, deserializeTensorRecord);

  auto deserializeTensorSizesRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::TensorSizesRecord(
        parseStateArgs(buffer->args()), parseStateArgs(buffer->outputs()));
  };
  registerParser(RecordType::TensorSizes, deserializeTensorSizesRecord);

  auto deserializeShapeOpRecord = [](const RecordFunctor* buffer) {
    return new python_frontend::ShapeOpRecord(
        parseStateArgs(buffer->args()), parseStateArgs(buffer->outputs()));
  };
  registerParser(RecordType::ShapeOp, deserializeShapeOpRecord);

  auto deserializeSizeOpRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Size();
    return new python_frontend::SizeOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        data->dim());
  };
  registerParser(RecordType::SizeOp, deserializeSizeOpRecord);

  auto deserializeAtOpRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_At();
    return new python_frontend::AtOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        data->index());
  };
  registerParser(RecordType::AtOp, deserializeAtOpRecord);

  auto deserializeVarianceRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Norm();
    return new python_frontend::VarianceOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        parseVector(data->axes()),
        data->correction(),
        data->keep_dim());
  };
  registerParser(RecordType::VarianceOp, deserializeVarianceRecord);

  auto deserializeVarianceMeanRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Norm();
    return new python_frontend::VarianceMeanOpRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        parseVector(data->axes()),
        data->correction(),
        data->keep_dim());
  };
  registerParser(RecordType::VarianceMeanOp, deserializeVarianceMeanRecord);

  auto deserializeVectorRecord = [](const RecordFunctor* buffer) {
    auto data = buffer->data_as_Vector();
    return new python_frontend::VectorRecord(
        parseStateArgs(buffer->args()),
        parseStateArgs(buffer->outputs()),
        mapToNvfuserDtype(data->dtype()));
  };
  registerParser(RecordType::Vector, deserializeVectorRecord);
}

void RecordFunctorFactory::setupFunctionMaps() {
#define NVFUSER_UNARY_TV_OP(op_str, op_name)                                \
  unary_tv.emplace(                                                         \
      ("ops." op_str), static_cast<TensorView* (*)(TensorView*)>(op_name)); \
  unary_val.emplace(("ops." op_str), static_cast<Val* (*)(Val*)>(op_name));

#define NVFUSER_BINARY_TV_ONLY_OP(op_str, op_name) \
  binary_tv.emplace(                               \
      ("ops." op_str),                             \
      static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name));

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

#define NVFUSER_BINARY_TV_ALPHA_OP(op_str, op_name)                           \
  ternary_val.emplace(                                                        \
      ("ops." op_str), static_cast<Val* (*)(Val*, Val*, Val*)>(op_name));     \
  ternary_tv_tv_val.emplace(                                                  \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>(op_name)); \
  ternary_tv_val_val.emplace(                                                 \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name));        \
  ternary_val_tv_val.emplace(                                                 \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name));

#define NVFUSER_TERNARY_TV_OP(op_str, op_name)                                \
  ternary_tv.emplace(                                                         \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(TensorView*, TensorView*, TensorView*)>(    \
          op_name));                                                          \
  ternary_val.emplace(                                                        \
      ("ops." op_str), static_cast<Val* (*)(Val*, Val*, Val*)>(op_name));     \
  ternary_tv_tv_val.emplace(                                                  \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>(op_name)); \
  ternary_tv_val_tv.emplace(                                                  \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(TensorView*, Val*, TensorView*)>(op_name)); \
  ternary_val_tv_tv.emplace(                                                  \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(Val*, TensorView*, TensorView*)>(op_name)); \
  ternary_val_val_tv.emplace(                                                 \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(Val*, Val*, TensorView*)>(op_name));        \
  ternary_tv_val_val.emplace(                                                 \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name));        \
  ternary_val_tv_val.emplace(                                                 \
      ("ops." op_str),                                                        \
      static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name));

#define NVFUSER_THRESHOLD_TV_OP(op_str, op_name)                          \
  ternary_val.emplace(                                                    \
      ("ops." op_str), static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)); \
  ternary_tv_val_val.emplace(                                             \
      ("ops." op_str),                                                    \
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name));

#define NVFUSER_TERNARY_TV_ALPHA_OP(op_str, op_name)                         \
  ternary_alpha_tv.emplace(                                                  \
      ("ops." op_str),                                                       \
      static_cast<                                                           \
          TensorView* (*)(TensorView*, TensorView*, TensorView*, Val*)>(     \
          op_name));                                                         \
  ternary_alpha_val.emplace(                                                 \
      ("ops." op_str),                                                       \
      static_cast<Val* (*)(Val*, Val*, Val*, Val*)>(op_name));               \
  ternary_alpha_tv_tv_val.emplace(                                           \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(TensorView*, TensorView*, Val*, Val*)>(    \
          op_name));                                                         \
  ternary_alpha_tv_val_tv.emplace(                                           \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(TensorView*, Val*, TensorView*, Val*)>(    \
          op_name));                                                         \
  ternary_alpha_val_tv_tv.emplace(                                           \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(Val*, TensorView*, TensorView*, Val*)>(    \
          op_name));                                                         \
  ternary_alpha_val_val_tv.emplace(                                          \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(Val*, Val*, TensorView*, Val*)>(op_name)); \
  ternary_alpha_tv_val_val.emplace(                                          \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(TensorView*, Val*, Val*, Val*)>(op_name)); \
  ternary_alpha_val_tv_val.emplace(                                          \
      ("ops." op_str),                                                       \
      static_cast<TensorView* (*)(Val*, TensorView*, Val*, Val*)>(op_name));

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
  NVFUSER_UNARY_TV_OP("segment_set", segment_set)
  NVFUSER_UNARY_TV_OP("set", set)
  NVFUSER_UNARY_TV_OP("sign", sign)
  NVFUSER_UNARY_TV_OP("sigmoid", sigmoid)
  NVFUSER_UNARY_TV_OP("signbit", signbit)
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

  NVFUSER_BINARY_TV_ONLY_OP("_matmul_nn", _matmul_nn)
  NVFUSER_BINARY_TV_ONLY_OP("_matmul_nt", _matmul_nt)
  NVFUSER_BINARY_TV_ONLY_OP("_matmul_tn", _matmul_tn)
  NVFUSER_BINARY_TV_ONLY_OP("_matmul_tt", _matmul_tt)

  NVFUSER_BINARY_TV_OP("add", add)
  NVFUSER_BINARY_TV_OP("atan2", atan2)
  NVFUSER_BINARY_TV_OP("div", div)
  NVFUSER_BINARY_TV_OP("truediv", truediv)
  NVFUSER_BINARY_TV_OP("fmod", fmod)
  NVFUSER_BINARY_TV_OP("mul", mul)
  NVFUSER_BINARY_TV_OP("nextafter", nextafter)
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
  NVFUSER_BINARY_TV_OP("bitwise_right_shift", bitwise_right_shift)
  NVFUSER_BINARY_TV_OP("logical_right_shift", logical_right_shift)
  NVFUSER_BINARY_TV_OP("gcd", gcd)

  NVFUSER_BINARY_TV_ALPHA_OP("add_alpha", add_alpha)
  NVFUSER_BINARY_TV_ALPHA_OP("sub_alpha", sub_alpha)

  NVFUSER_TERNARY_TV_OP("lerp", lerp)
  NVFUSER_TERNARY_TV_OP("where", where)

  // The following ops behave like TernaryOps but are only TV_VAL_VAL
  ternary_tv_val_val.emplace(
      "ops.rand_like",
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(rand_like));
  ternary_tv_val_val.emplace(
      "ops.randn_like",
      static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(randn_like));

  NVFUSER_THRESHOLD_TV_OP("clamp", clamp)
  NVFUSER_THRESHOLD_TV_OP("threshold", threshold)

  NVFUSER_TERNARY_TV_ALPHA_OP("addcmul", addcmul)
}

} // namespace nvfuser::serde
