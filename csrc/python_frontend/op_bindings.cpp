// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <instrumentation.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_record.h>
#include <python_frontend/python_bindings.h>

namespace nvfuser::python_frontend {

namespace {
template <class ShapeType>
Vector SequenceAsVector(
    ShapeType shape,
    FusionDefinition& fd,
    bool shape_check = true) {
  static_assert(
      std::is_same_v<ShapeType, Vector> ||
      std::is_same_v<ShapeType, py::list> ||
      std::is_same_v<ShapeType, py::tuple>);
  if constexpr (std::is_same_v<ShapeType, Vector>) {
    return shape;
  } else {
    // It's important to call define_vector_fn in the if-else branch.
    //
    // ```
    // if constexpr (std::is_same_v<ShapeType, Vector>) {
    //   return shape;
    // }
    // return define_vector_fn<ShapeType>(fd, shape);
    // ```
    // would not work because the compiler would try to instantiate
    // define_vector_fn<Vector> and fail.
    return define_vector_fn<ShapeType>(
        fd, shape, /*inline_def=*/true, /*shape_check=*/shape_check);
  }
}

template <class ShapeType>
Tensor broadcast_in_dim_fn(
    FusionDefinition::Operators& op,
    Tensor arg,
    ShapeType generic_output_shape,
    std::vector<int64_t>& broadcast_dims) {
  FUSER_PERF_SCOPE("Operators.broadcast_in_dim");
  FusionDefinition* fd = op.fusion_definition;
  NVF_CHECK(op.validUse(), "Attempting to add to a completed definition!");
  Vector output_shape = SequenceAsVector(generic_output_shape, *fd);
  NVF_CHECK(
      output_shape.size >= broadcast_dims.size(),
      "broadcast_dims vector size is too big for output shape!");

  Tensor output = fd->defineTensor(output_shape.size);
  fd->defineRecord(new BroadcastInDimOpRecord(
      {fd->recordingState(arg()), fd->recordingState(output_shape())},
      {fd->recordingState(output())},
      output_shape.size,
      broadcast_dims));
  return output;
}

template <class ShapeType>
Tensor expand_fn(
    FusionDefinition::Operators& op,
    Tensor arg,
    ShapeType generic_output_shape) {
  FUSER_PERF_SCOPE("Operators.expand");
  FusionDefinition* fd = op.fusion_definition;
  NVF_CHECK(op.validUse(), "Attempting to add to a completed definition!");
  Vector output_shape = SequenceAsVector(generic_output_shape, *fd);

  Tensor output = fd->defineTensor(output_shape.size);
  fd->defineRecord(new ExpandOpRecord(
      {fd->recordingState(arg()), fd->recordingState(output_shape())},
      {fd->recordingState(output())}));
  return output;
}

template <class ShapeType>
Tensor full_op_fn(
    FusionDefinition::Operators& self,
    ShapeType generic_output_shape,
    Scalar fill_value,
    PrimDataType dtype) {
  NVF_CHECK(self.validUse(), "Attempting to add to a completed definition!");
  FusionDefinition* fd = self.fusion_definition;
  Vector output_shape = SequenceAsVector(generic_output_shape, *fd);
  Tensor output = fd->defineTensor(output_shape.size);
  fd->defineRecord(new FullOpRecord(
      {fd->recordingState(output_shape()), fd->recordingState(fill_value())},
      {fd->recordingState(output())},
      dtype));
  return output;
}

template <class ShapeType>
Tensor reshape_fn(
    FusionDefinition::Operators& self,
    Tensor arg,
    ShapeType generic_new_shape) {
  NVF_CHECK(self.validUse(), "Attempting to add to a completed definition!");

  FusionDefinition* fd = self.fusion_definition;
  Vector new_shape = SequenceAsVector(generic_new_shape, *fd);

  Tensor output = fd->defineTensor(new_shape.size);
  fd->defineRecord(new ReshapeOpRecord(
      {fd->recordingState(arg()), fd->recordingState(new_shape())},
      {fd->recordingState(output())}));
  return output;
}

template <class ShapeType>
Tensor pad_fn(
    FusionDefinition::Operators& self,
    Tensor arg,
    ShapeType generic_pad_widths,
    std::optional<Scalar> value) {
  NVF_CHECK(self.validUse(), "Attempting to add to a completed definition!");

  FusionDefinition* fd = self.fusion_definition;
  Vector pad_widths =
      SequenceAsVector(generic_pad_widths, *fd, /*shape_check=*/false);

  NVF_CHECK(
      pad_widths.size <= 2 * arg.dims,
      "Number of pad widths must be at most twice the input dimension");

  State value_state = value.has_value() ? fd->recordingState(value.value()())
                                        : State(0, serde::StateType::None);

  Tensor output = fd->defineTensor(arg.dims);
  fd->defineRecord(new PadOpRecord(
      {fd->recordingState(arg()),
       fd->recordingState(pad_widths()),
       value_state},
      {fd->recordingState(output())}));
  return output;
}

template <class ShapeType, serde::RecordType RType>
Tensor random_dist_op_fn(
    FusionDefinition::Operators& self,
    Scalar arg1,
    Scalar arg2,
    ShapeType generic_new_shape,
    std::optional<Scalar> rng_seed,
    std::optional<Scalar> rng_offset,
    PrimDataType dtype) {
  static_assert(
      (RType == serde::RecordType::NormalDistOp) ||
      (RType == serde::RecordType::UniformDistOp));
  NVF_CHECK(self.validUse(), "Attempting to add to a completed definition!");
  NVF_CHECK(
      isFloatingPointType(dtype),
      "Random distributions only create floating point types! ",
      dtype);
  FusionDefinition* fd = self.fusion_definition;
  Vector new_shape = SequenceAsVector(generic_new_shape, *fd);

  Tensor output = fd->defineTensor(new_shape.size);
  std::vector<State> arg_states = {
      fd->recordingState(arg1()),
      fd->recordingState(arg2()),
      fd->recordingState(new_shape()),
  };
  if (rng_seed.has_value() && rng_offset.has_value()) {
    arg_states.push_back(fd->recordingState(rng_seed.value()()));
    arg_states.push_back(fd->recordingState(rng_offset.value()()));
  } else {
    NVF_CHECK(
        !rng_seed.has_value() && !rng_offset.has_value(),
        "rng_seed and rng_offset must be provided together!");
  }

  fd->defineRecord(new RandomDistOpRecord<RType>(
      arg_states, {fd->recordingState(output())}, dtype));

  return output;
}

template <class ShapeType>
Tensor slice_fn(
    FusionDefinition::Operators& self,
    Tensor arg,
    ShapeType start,
    ShapeType end,
    std::optional<ShapeType> strides,
    bool manual_normalization) {
  NVF_CHECK(self.validUse(), "Attempting to add to a completed definition!");

  FusionDefinition* fd = self.fusion_definition;
  Vector new_start = SequenceAsVector(start, *fd, /*shape_check=*/false);
  Vector new_end = SequenceAsVector(end, *fd, /*shape_check=*/false);
  size_t stride_index = 0;

  if (strides.has_value()) {
    Vector new_stride =
        SequenceAsVector(strides.value(), *fd, /*shape_check=*/false);
    NVF_CHECK(
        new_start.size == new_stride.size,
        "Slice start_indices and strides don't match! Start Indices: ",
        new_start.size,
        " Strides: ",
        new_stride.size);
    stride_index = new_stride();
  } else {
    // set stride with default value;
    std::vector<Scalar> stride_vec;
    stride_vec.reserve(new_start.size);
    // Note: we cannot re-use the same ScalarRecord, otherwise, serialized
    // python program uses `define_vector`, which would create multiple
    // ScalarRecord, causing a cache miss.
    for (auto i : c10::irange(new_start.size)) {
      (void)i; // Supress unused variable warning
      Scalar out = fd->defineScalar();
      fd->defineRecord(new ScalarRecord(
          {fd->recordingState(out())},
          1,
          DataType::Int,
          /*inline_def=*/true));
      stride_vec.push_back(out);
    }
    // Cannot inline definition with `Vector` here, since
    // `FusionDefinition.ops.slice` expects start/end/stride to have the same
    // type.
    Vector default_stride = define_vector_base_fn(
        *fd, stride_vec, !std::is_same_v<ShapeType, Vector>);
    stride_index = default_stride();
  }

  NVF_CHECK(
      arg.dims == new_start.size,
      "Number of tensor dimensions does not match slice dimensions! Tensor-dims: ",
      arg.dims,
      " Slice-dims: ",
      new_start.size);
  NVF_CHECK(
      new_start.size == new_end.size,
      "Slice indexing attribute dimensions don't match! Start Indices: ",
      new_start.size,
      " End Indices: ",
      new_end.size);

  Tensor output = fd->defineTensor(arg.dims);
  fd->defineRecord(new SliceOpRecord(
      {fd->recordingState(arg()),
       fd->recordingState(new_start()),
       fd->recordingState(new_end()),
       fd->recordingState(stride_index)},
      {fd->recordingState(output())},
      manual_normalization));
  return output;
}
} // namespace

void bindOps(py::class_<FusionDefinition>& fusion_def) {
  //! The Operators class is a nested class of FusionDefinition to allow the
  //! user to query the class for the list of operators.
  //!
  //! Example:
  //!   help(FusionDefinition.Operators)
  //!
  //! Additional operators are expected to be defined below as needed.  They
  //! may require defining a new RecordFunctor child class if they are unique.
  py::class_<FusionDefinition::Operators> nvf_ops(fusion_def, "Operators");
  nvf_ops.def(py::init<FusionDefinition*>());

// ******************** INSERT OP BINDINGS BELOW HERE ********************
#define OP_PREFIX "Operators."
#define NVFUSER_PYTHON_BINDING_UNARY_OP(op_str, op_name)                      \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self, Tensor input) -> Tensor {         \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(input.dims);                         \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(              \
            {fd->recordingState(input())},                                    \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Unary_TV,                                      \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));             \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self, Scalar input) -> Scalar {         \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new OpRecord<Val*, Val*>(                            \
            {fd->recordingState(input())},                                    \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Unary_VAL,                                     \
            static_cast<Val* (*)(Val*)>(op_name)));                           \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_UNARY_OP("abs", abs)
  NVFUSER_PYTHON_BINDING_UNARY_OP("acos", acos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("acosh", acosh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("asin", asin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("asinh", asinh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atan", atan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("atanh", atanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("ceil", ceil)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cos", cos)
  NVFUSER_PYTHON_BINDING_UNARY_OP("cosh", cosh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("exp", exp)
  NVFUSER_PYTHON_BINDING_UNARY_OP("exp2", exp2)
  NVFUSER_PYTHON_BINDING_UNARY_OP("expm1", expm1)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erf", erf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfc", erfc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfinv", erfinv)
  NVFUSER_PYTHON_BINDING_UNARY_OP("erfcinv", erfcinv)
  NVFUSER_PYTHON_BINDING_UNARY_OP("floor", floor)
  NVFUSER_PYTHON_BINDING_UNARY_OP("frac", frac)
  NVFUSER_PYTHON_BINDING_UNARY_OP("lgamma", lgamma)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log", log)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log10", log10)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log1p", log1p)
  NVFUSER_PYTHON_BINDING_UNARY_OP("log2", log2)
  NVFUSER_PYTHON_BINDING_UNARY_OP("neg", neg)
  NVFUSER_PYTHON_BINDING_UNARY_OP("logical_not", logical_not)
  NVFUSER_PYTHON_BINDING_UNARY_OP("bitwise_not", bitwise_not)
  NVFUSER_PYTHON_BINDING_UNARY_OP("relu", relu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rand_like", rand_like)
  NVFUSER_PYTHON_BINDING_UNARY_OP("randn_like", randn_like)
  NVFUSER_PYTHON_BINDING_UNARY_OP("reciprocal", reciprocal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("round", round)
  NVFUSER_PYTHON_BINDING_UNARY_OP("rsqrt", rsqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("set", set)
  NVFUSER_PYTHON_BINDING_UNARY_OP("segment_set", segment_set)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sign", sign)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sigmoid", sigmoid)
  NVFUSER_PYTHON_BINDING_UNARY_OP("signbit", signbit)
  NVFUSER_PYTHON_BINDING_UNARY_OP("silu", silu)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sin", sin)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sinh", sinh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("sqrt", sqrt)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tan", tan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("tanh", tanh)
  NVFUSER_PYTHON_BINDING_UNARY_OP("trunc", trunc)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isfinite", isfinite)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isinf", isinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isnan", isnan)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isneginf", isneginf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isposinf", isposinf)
  NVFUSER_PYTHON_BINDING_UNARY_OP("isreal", isreal)
  NVFUSER_PYTHON_BINDING_UNARY_OP("real", real)
  NVFUSER_PYTHON_BINDING_UNARY_OP("imag", imag)
#undef NVFUSER_PYTHON_BINDING_UNARY_OP

  // overload to
  nvf_ops.def(
      "stride_order",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& stride_order) -> Tensor {
        FUSER_PERF_SCOPE("Operators.stride_order");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            arg.dims == stride_order.size(),
            "Operator stride_order expects `stride_order` argument to have the same length as input!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new DimsOpRecord<serde::RecordType::StrideOrderOp>(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            std::move(stride_order),
            "ops.stride_order"));
        return output;
      },
      py::arg("arg"),
      py::arg("stride_order"),
      py::return_value_policy::reference);

// rand_like and randn_like are normally used with a single TensorView argument,
// like a UnaryOp. However, they also take an optional pair (rng_seed,
// rng_offset) which converts them to deterministic ops. When those args are
// provided, and they must both be provided if either is, then the op behaves
// like a ternary op. We handle the UnaryOp case above and the TernaryOp case
// here.
#define NVFUSER_PYTHON_BINDING_TERNARY_RANDOM_OP(op_str, op_name)             \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor input,                                                        \
         Scalar rng_seed,                                                     \
         Scalar rng_offset) -> Tensor {                                       \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(input.dims);                         \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(              \
            {fd->recordingState(input()),                                     \
             fd->recordingState(rng_seed()),                                  \
             fd->recordingState(rng_offset())},                               \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_TV_VAL_VAL,                            \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));             \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::kw_only(),                                                          \
      py::arg("rng_seed"),                                                    \
      py::arg("rng_offset"),                                                  \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_RANDOM_OP("rand_like", rand_like)
  NVFUSER_PYTHON_BINDING_TERNARY_RANDOM_OP("randn_like", randn_like)

#undef NVFUSER_PYTHON_BINDING_UNARY_RANDOM_OP

#define NVFUSER_PYTHON_BINDING_MATMUL_OP(op_str, op_name)                      \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        /* Per https://pytorch.org/docs/stable/generated/torch.matmul.html */  \
        size_t out_ndims;                                                      \
        if (arg1.dims <= 2 && arg2.dims <= 2) {                                \
          out_ndims = arg1.dims + arg2.dims - 2;                               \
        } else {                                                               \
          /* batch matmul */                                                   \
          out_ndims = std::max(arg1.dims, arg2.dims);                          \
        }                                                                      \
        Tensor output = fd->defineTensor(out_ndims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Binary_TV,                                      \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);
  NVFUSER_PYTHON_BINDING_MATMUL_OP("matmul", matmul)
#undef NVFUSER_PYTHON_BINDING_MATMUL_OP

  nvf_ops.def(
      "linear",
      [](FusionDefinition::Operators& self,
         Tensor arg1,
         Tensor arg2,
         std::optional<Tensor> bias = std::nullopt) -> Tensor {
        FUSER_PERF_SCOPE("Operators.linear");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        // See newForLinear for how the output rank is computed.
        Tensor output = fd->defineTensor(arg1.dims + arg2.dims - 2);

        if (bias.has_value()) {
          fd->defineRecord(
              new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>(
                  {fd->recordingState(arg1()),
                   fd->recordingState(arg2()),
                   fd->recordingState(bias.value()())},
                  {fd->recordingState(output())},
                  ("ops.linear"),
                  serde::RecordType::Ternary_TV,
                  static_cast<
                      TensorView* (*)(TensorView*, TensorView*, TensorView*)>(
                      linear)));
        } else {
          fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(
              {fd->recordingState(arg1()), fd->recordingState(arg2())},
              {fd->recordingState(output())},
              ("ops.linear"),
              serde::RecordType::Binary_TV,
              static_cast<TensorView* (*)(TensorView*, TensorView*)>(linear)));
        }
        return output;
      },
      py::arg("arg1"),
      py::arg("arg2"),
      py::arg("bias") = std::nullopt,
      py::return_value_policy::reference);

#define NVFUSER_PYTHON_BINDING_BINARY_OP(op_str, op_name)                      \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, TensorView*>(  \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Binary_TV,                                      \
            static_cast<TensorView* (*)(TensorView*, TensorView*)>(op_name))); \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Binary_TV_VAL,                                  \
            static_cast<TensorView* (*)(TensorView*, Val*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*>(         \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Binary_VAL_TV,                                  \
            static_cast<TensorView* (*)(Val*, TensorView*)>(op_name)));        \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*>(                       \
            {fd->recordingState(arg1()), fd->recordingState(arg2())},          \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Binary_VAL,                                     \
            static_cast<Val* (*)(Val*, Val*)>(op_name)));                      \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_OP("add", add)
  NVFUSER_PYTHON_BINDING_BINARY_OP("atan2", atan2)
  NVFUSER_PYTHON_BINDING_BINARY_OP("div", div)
  NVFUSER_PYTHON_BINDING_BINARY_OP("truediv", truediv)
  NVFUSER_PYTHON_BINDING_BINARY_OP("fmod", fmod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP("nextafter", nextafter)
  NVFUSER_PYTHON_BINDING_BINARY_OP("pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP("remainder", remainder)
  NVFUSER_PYTHON_BINDING_BINARY_OP("sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP("minimum", minimum)
  NVFUSER_PYTHON_BINDING_BINARY_OP("maximum", maximum)
  NVFUSER_PYTHON_BINDING_BINARY_OP("mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP("eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP("gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP("lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP("ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP("logical_and", logical_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP("logical_or", logical_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_and", bitwise_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_or", bitwise_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_xor", bitwise_xor)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_left_shift", bitwise_left_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("bitwise_right_shift", bitwise_right_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("logical_right_shift", logical_right_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP("gcd", gcd)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP

#define NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP(op_str, op_name)          \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType::Ternary_TV_TV_VAL,                         \
                static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_TV_VAL_VAL,                            \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg2.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_VAL_TV_VAL,                            \
            static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_VAL,                                   \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("add_alpha", add_alpha)
  NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP("sub_alpha", sub_alpha)
#undef NVFUSER_PYTHON_BINDING_BINARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_OP(op_str, op_name)                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_VAL,                                   \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Tensor arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>( \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType::Ternary_TV,                                \
                static_cast<                                                  \
                    TensorView* (*)(TensorView*, TensorView*, TensorView*)>(  \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType::Ternary_TV_TV_VAL,                         \
                static_cast<TensorView* (*)(TensorView*, TensorView*, Val*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Scalar arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType::Ternary_TV_VAL_TV,                         \
                static_cast<TensorView* (*)(TensorView*, Val*, TensorView*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Tensor arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg2.dims);                          \
        fd->defineRecord(                                                     \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*>(        \
                {fd->recordingState(arg1()),                                  \
                 fd->recordingState(arg2()),                                  \
                 fd->recordingState(arg3())},                                 \
                {fd->recordingState(output())},                               \
                ("ops." op_str),                                              \
                serde::RecordType::Ternary_VAL_TV_TV,                         \
                static_cast<TensorView* (*)(Val*, TensorView*, TensorView*)>( \
                    op_name)));                                               \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Scalar arg2,                                                         \
         Tensor arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg3.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, Val*, Val*, TensorView*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_VAL_VAL_TV,                            \
            static_cast<TensorView* (*)(Val*, Val*, TensorView*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg1,                                                         \
         Scalar arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg1.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_TV_VAL_VAL,                            \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg1,                                                         \
         Tensor arg2,                                                         \
         Scalar arg3) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg2.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, Val*, TensorView*, Val*>(  \
            {fd->recordingState(arg1()),                                      \
             fd->recordingState(arg2()),                                      \
             fd->recordingState(arg3())},                                     \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::Ternary_VAL_TV_VAL,                            \
            static_cast<TensorView* (*)(Val*, TensorView*, Val*)>(op_name))); \
        return output;                                                        \
      },                                                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_OP("lerp", lerp)
  NVFUSER_PYTHON_BINDING_TERNARY_OP("where", where)
#undef NVFUSER_PYTHON_BINDING_TERNARY_OP

#define NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP(op_str, op_name)              \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            !self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*>(                 \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Ternary_VAL,                                    \
            static_cast<Val* (*)(Val*, Val*, Val*)>(op_name)));                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            !self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*, Val*>(   \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Ternary_TV_VAL_VAL,                             \
            static_cast<TensorView* (*)(TensorView*, Val*, Val*)>(op_name)));  \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("clamp", clamp)
  NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP("threshold", threshold)
#undef NVFUSER_PYTHON_BINDING_THRESHOLD_LIKE_OP

#define NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP(op_str, op_name)          \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Scalar {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*, Val*, Val*, Val*>(           \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Ternary_Alpha_VAL,                              \
            static_cast<Val* (*)(Val*, Val*, Val*, Val*)>(op_name)));          \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(new OpRecord<                                         \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         TensorView*,                                          \
                         Val*>(                                                \
            {fd->recordingState(arg1()),                                       \
             fd->recordingState(arg2()),                                       \
             fd->recordingState(arg3()),                                       \
             fd->recordingState(arg4())},                                      \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Ternary_Alpha_TV,                               \
            static_cast<                                                       \
                TensorView* (*)(TensorView*, TensorView*, TensorView*, Val*)>( \
                op_name)));                                                    \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, TensorView*, Val*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType::Ternary_Alpha_TV_TV_VAL,                    \
                static_cast<                                                   \
                    TensorView* (*)(TensorView*, TensorView*, Val*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType::Ternary_Alpha_TV_VAL_TV,                    \
                static_cast<                                                   \
                    TensorView* (*)(TensorView*, Val*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, TensorView*, Val*>(   \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType::Ternary_Alpha_VAL_TV_TV,                    \
                static_cast<                                                   \
                    TensorView* (*)(Val*, TensorView*, TensorView*, Val*)>(    \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Scalar arg2,                                                          \
         Tensor arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg3.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, Val*, TensorView*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType::Ternary_Alpha_VAL_VAL_TV,                   \
                static_cast<TensorView* (*)(Val*, Val*, TensorView*, Val*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Tensor arg1,                                                          \
         Scalar arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg1.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, TensorView*, Val*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType::Ternary_Alpha_TV_VAL_VAL,                   \
                static_cast<TensorView* (*)(TensorView*, Val*, Val*, Val*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  nvf_ops.def(                                                                 \
      op_str,                                                                  \
      [](FusionDefinition::Operators& self,                                    \
         Scalar arg1,                                                          \
         Tensor arg2,                                                          \
         Scalar arg3,                                                          \
         Scalar arg4) -> Tensor {                                              \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        NVF_CHECK(                                                             \
            self.validUse(), "Attempting to add to a completed definition!");  \
        FusionDefinition* fd = self.fusion_definition;                         \
        Tensor output = fd->defineTensor(arg2.dims);                           \
        fd->defineRecord(                                                      \
            new OpRecord<TensorView*, Val*, TensorView*, Val*, Val*>(          \
                {fd->recordingState(arg1()),                                   \
                 fd->recordingState(arg2()),                                   \
                 fd->recordingState(arg3()),                                   \
                 fd->recordingState(arg4())},                                  \
                {fd->recordingState(output())},                                \
                ("ops." op_str),                                               \
                serde::RecordType::Ternary_Alpha_VAL_TV_VAL,                   \
                static_cast<TensorView* (*)(Val*, TensorView*, Val*, Val*)>(   \
                    op_name)));                                                \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP("addcmul", addcmul)
#undef NVFUSER_PYTHON_BINDING_TERNARY_WITH_ALPHA_OP

#define NVFUSER_PYTHON_BINDING_REDUCTION_OP(op_str, op_name, record_type)     \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        size_t ndims = 0;                                                     \
        std::vector<int64_t> dims(arg.dims);                                  \
        std::iota(dims.begin(), dims.end(), 0);                               \
        Tensor output = fd->defineTensor(ndims);                              \
        fd->defineRecord(new ReductionOpRecord(                               \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            record_type,                                                      \
            static_cast<TensorView* (*)(TensorView*,                          \
                                        const std::vector<int64_t>&,          \
                                        bool,                                 \
                                        DataType)>(op_name),                  \
            dims,                                                             \
            false,                                                            \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dtype") = DataType::Null,                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         int dim,                                                             \
         bool keepdim,                                                        \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        size_t ndims = keepdim ? arg.dims : (arg.dims - 1);                   \
        Tensor output = fd->defineTensor(ndims);                              \
        fd->defineRecord(new ReductionOpRecord(                               \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            record_type,                                                      \
            static_cast<TensorView* (*)(TensorView*,                          \
                                        const std::vector<int64_t>&,          \
                                        bool,                                 \
                                        DataType)>(op_name),                  \
            {dim},                                                            \
            keepdim,                                                          \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dim"),                                                         \
      py::arg("keepdim") = false,                                             \
      py::arg("dtype") = DataType::Null,                                      \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         const std::vector<int64_t>& dims,                                    \
         bool keepdim,                                                        \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        size_t ndims = keepdim ? arg.dims : (arg.dims - dims.size());         \
        Tensor output = fd->defineTensor(ndims);                              \
        fd->defineRecord(new ReductionOpRecord(                               \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            record_type,                                                      \
            static_cast<TensorView* (*)(TensorView*,                          \
                                        const std::vector<int64_t>&,          \
                                        bool,                                 \
                                        DataType)>(op_name),                  \
            dims,                                                             \
            keepdim,                                                          \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dims"),                                                        \
      py::arg("keepdim") = false,                                             \
      py::arg("dtype") = DataType::Null,                                      \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "max", max, serde::RecordType::ReductionMax)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "min", min, serde::RecordType::ReductionMin)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "prod", prod, serde::RecordType::ReductionProd)
  NVFUSER_PYTHON_BINDING_REDUCTION_OP(
      "sum", sum, serde::RecordType::ReductionSum)
#undef NVFUSER_PYTHON_BINDING_REDUCTION_OP

#define NVFUSER_PYTHON_BINDING_CAST_OP(op_str, op_name)                       \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Tensor arg,                                                          \
         PrimDataType dtype) -> Tensor {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Tensor output = fd->defineTensor(arg.dims);                           \
        fd->defineRecord(new CastOpRecord<TensorView*, TensorView*>(          \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::CastTv,                                        \
            static_cast<TensorView* (*)(DataType, TensorView*)>(op_name),     \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dtype"),                                                       \
      py::return_value_policy::reference);                                    \
  nvf_ops.def(                                                                \
      op_str,                                                                 \
      [](FusionDefinition::Operators& self,                                   \
         Scalar arg,                                                          \
         PrimDataType dtype) -> Scalar {                                      \
        FUSER_PERF_SCOPE("Operators." op_str);                                \
        NVF_CHECK(                                                            \
            self.validUse(), "Attempting to add to a completed definition!"); \
        FusionDefinition* fd = self.fusion_definition;                        \
        Scalar output = fd->defineScalar();                                   \
        fd->defineRecord(new CastOpRecord<Val*, Val*>(                        \
            {fd->recordingState(arg())},                                      \
            {fd->recordingState(output())},                                   \
            ("ops." op_str),                                                  \
            serde::RecordType::CastVal,                                       \
            static_cast<Val* (*)(DataType, Val*)>(op_name),                   \
            dtype));                                                          \
        return output;                                                        \
      },                                                                      \
      py::arg("arg"),                                                         \
      py::arg("dtype"),                                                       \
      py::return_value_policy::reference);

  NVFUSER_PYTHON_BINDING_CAST_OP("cast", castOp)
#undef NVFUSER_PYTHON_BINDING_CAST_OP

#define NVFUSER_ALL_VECTOR_TYPES(fn, ...) \
  fn(Vector, __VA_ARGS__);                \
  fn(py::list, __VA_ARGS__);              \
  fn(py::tuple, __VA_ARGS__);

#define NVFUSER_RANDOM_DIST_OP_HELPER(             \
    vec_type, op_str, op_type, arg1_str, arg2_str) \
  nvf_ops.def(                                     \
      op_str,                                      \
      random_dist_op_fn<vec_type, op_type>,        \
      py::arg(arg1_str),                           \
      py::arg(arg2_str),                           \
      py::arg("shape"),                            \
      py::kw_only(),                               \
      py::arg("rng_seed") = py::none(),            \
      py::arg("rng_offset") = py::none(),          \
      py::arg("dtype") = DataType::Float,          \
      py::return_value_policy::reference);

#define NVFUSER_PYTHON_BINDING_RANDOM_DIST_OP(...) \
  NVFUSER_ALL_VECTOR_TYPES(NVFUSER_RANDOM_DIST_OP_HELPER, __VA_ARGS__)

  NVFUSER_PYTHON_BINDING_RANDOM_DIST_OP(
      "normal", serde::RecordType::NormalDistOp, "mean", "std")
  NVFUSER_PYTHON_BINDING_RANDOM_DIST_OP(
      "uniform", serde::RecordType::UniformDistOp, "minval", "maxval")
#undef NVFUSER_PYTHON_BINDING_RANDOM_DIST_OP
#undef NVFUSER_RANDOM_DIST_OP_HELPER

#define NVFUSER_FULL_OP_HELPER(vec_type, ...) \
  nvf_ops.def(                                \
      "full",                                 \
      full_op_fn<vec_type>,                   \
      py::arg("shape"),                       \
      py::arg("fill_value"),                  \
      py::arg("dtype"),                       \
      py::return_value_policy::reference);

  // NOTE: The second argument is a dummy to satisfy the macro
  NVFUSER_ALL_VECTOR_TYPES(NVFUSER_FULL_OP_HELPER, false)
#undef NVFUSER_FULL_OP_HELPER

  nvf_ops.def(
      "batch_norm",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::optional<Tensor> weight,
         std::optional<Tensor> bias,
         std::optional<Tensor> running_mean,
         std::optional<Tensor> running_var,
         Scalar momentum,
         Scalar eps,
         bool training,
         bool channels_last) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.batch_norm");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        Tensor mean = fd->defineTensor(1);
        Tensor invstd = fd->defineTensor(1);
        auto weight_state = weight.has_value()
            ? fd->recordingState(weight.value()())
            : State(0, serde::StateType::None);
        auto bias_state = bias.has_value() ? fd->recordingState(bias.value()())
                                           : State(0, serde::StateType::None);
        auto running_mean_state = running_mean.has_value()
            ? fd->recordingState(running_mean.value()())
            : State(0, serde::StateType::None);
        auto running_var_state = running_var.has_value()
            ? fd->recordingState(running_var.value()())
            : State(0, serde::StateType::None);
        fd->defineRecord(new BatchNormOpRecord(
            {fd->recordingState(arg()),
             weight_state,
             bias_state,
             running_mean_state,
             running_var_state,
             fd->recordingState(momentum()),
             fd->recordingState(eps())},
            {fd->recordingState(output()),
             fd->recordingState(mean()),
             fd->recordingState(invstd())},
            training,
            channels_last));
        return std::make_tuple(output, mean, invstd);
      },
      py::arg("arg"),
      py::arg("weight").none(true),
      py::arg("bias").none(true),
      py::arg("running_mean").none(true),
      py::arg("running_var").none(true),
      py::arg("momentum"),
      py::arg("eps"),
      py::arg("training"),
      py::arg("channels_last") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<Vector>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<py::list>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  // NOTE: Tuple support was added to facilitate the direct usage of Pytorch's
  // Tensor.size() function that returns a child class of a Tuple.
  nvf_ops.def(
      "broadcast_in_dim",
      broadcast_in_dim_fn<py::tuple>,
      py::arg("arg"),
      py::arg("shape"),
      py::arg("broadcast_dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "broadcast",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<bool>& is_broadcast_dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.broadcast");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new BroadcastOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            "ops.broadcast",
            std::move(is_broadcast_dim)));
        return output;
      },
      py::arg("arg"),
      py::arg("is_broadcast_dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "cat",
      [](FusionDefinition::Operators& self,
         std::vector<Tensor> tensors,
         int64_t dim,
         bool manual_padding) -> Tensor {
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        NVF_CHECK(
            !tensors.empty(), "Attempting to concatenate empty list of tensors")
        Tensor output = fd->defineTensor(tensors[0].dims);
        std::vector<State> tensor_states;
        tensor_states.reserve(tensors.size());
        for (auto& t : tensors) {
          tensor_states.push_back(fd->recordingState(t()));
        }
        self.fusion_definition->defineRecord(new CatOpRecord(
            tensor_states,
            {fd->recordingState(output())},
            dim,
            manual_padding));
        return output;
      },
      py::arg("tensors"),
      py::arg("dim") = 0,
      py::arg("manual_padding") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "expand",
      expand_fn<Vector>,
      py::arg("arg"),
      py::arg("shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "expand",
      expand_fn<py::list>,
      py::arg("arg"),
      py::arg("shape"),
      py::return_value_policy::reference);
  // NOTE: Tuple support was added to facilitate the direct usage of Pytorch's
  // Tensor.size() function that returns a child class of a Tuple.
  nvf_ops.def(
      "expand",
      expand_fn<py::tuple>,
      py::arg("arg"),
      py::arg("shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "index_select",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.index_select");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new IndexSelectOpRecord(
            {
                fd->recordingState(arg()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "select",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         Scalar index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.select");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new SelectOpRecord(
            {
                fd->recordingState(arg()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      py::arg("arg"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "gather",
      [](FusionDefinition::Operators& self,
         Tensor arg1,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.gather");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            arg1.dims == index.dims,
            "Tensor arguments have different dimensions ",
            arg1.dims,
            " and ",
            index.dims);
        auto num_dims = (int64_t)arg1.dims;
        NVF_CHECK(
            dim >= -num_dims && dim < num_dims,
            "Tensor arguments have dimension ",
            num_dims,
            " so dim argument must satisfy ",
            -num_dims,
            " <= dim < ",
            num_dims,
            ", but received ",
            dim);
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg1.dims);
        fd->defineRecord(new TorchGatherOpRecord(
            {
                fd->recordingState(arg1()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      R"pbdoc(
        Index arg1 in dim at positions given by index.

        The dimension of arg1 and index must match. For all axes other than dim
        the extent of index in that axis need not be equal to its counterpart
        in arg1 but must not be greater than it.

        Args:
            arg1 (Tensor): Tensor of shape `(Ni...,M,Nk...)` where `M` is the
                extent of `arg1` in the dimension `dim`.
            index (Tensor): Tensor of dtype `DataType::Int` of shape
                `(Mi...,J,Mk...)` where all the extents other than `J` are less
                than or equal to their counterparts in `arg1`; for example `Mk
                <= Nk`.
            dim (int): Which position to index along.

        Returns:
            (Tensor): Tensor of same dtype as `arg1` and of shape
                `(Mi...,J,Mk...)` where the element at position `(i...,j,k...)`
                is equal to `arg1[i,...,index[i,...,j,k,...],k,...]`.
      )pbdoc",
      py::arg("arg1"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "pad",
      pad_fn<Vector>,
      py::arg("arg"),
      py::arg("pad_widths"),
      py::arg("value") = py::none(),
      py::return_value_policy::reference);
  nvf_ops.def(
      "pad",
      pad_fn<py::list>,
      py::arg("arg"),
      py::arg("pad_widths"),
      py::arg("value") = py::none(),
      py::return_value_policy::reference);
  nvf_ops.def(
      "pad",
      pad_fn<py::tuple>,
      py::arg("arg"),
      py::arg("pad_widths"),
      py::arg("value") = py::none(),
      py::return_value_policy::reference);
  nvf_ops.def(
      "take_along_axis",
      [](FusionDefinition::Operators& self,
         Tensor arg1,
         Tensor index,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.take_along_axis");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            arg1.dims == index.dims,
            "Tensor arguments have different dimensions ",
            arg1.dims,
            " and ",
            index.dims);
        auto num_dims = (int64_t)arg1.dims;
        NVF_CHECK(
            dim >= -num_dims && dim < num_dims,
            "Tensor arguments have dimension ",
            num_dims,
            " so dim argument must satisfy ",
            -num_dims,
            " <= dim < ",
            num_dims,
            ", but received ",
            dim);
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg1.dims);
        fd->defineRecord(new TakeAlongAxisOpRecord(
            {
                fd->recordingState(arg1()),
                fd->recordingState(index()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      R"pbdoc(
        Index arg1 in dim at positions given by index.

        This operation is very similar to :meth:'gather' but enforces that all
        dimensions other than dim must be equal between arg1 and index.

        Args:
            arg1 (Tensor): Tensor of shape `(Ni...,M,Nk...)` where `M` is the
                extent of `arg1` in the dimension `dim`.
            index (Tensor): Tensor of dtype `DataType::Int` of shape
                `(Ni...,J,Nk...)`.
            dim (int): Which position to index along.

        Returns:
            (Tensor): Tensor of same dtype as `arg1` and of shape
                `(Ni...,J,Nk...)` where the element at position `(i...,j,k...)`
                is equal to `arg1[i,...,index[i,...,j,k,...],k,...]`.
      )pbdoc",
      py::arg("arg1"),
      py::arg("index"),
      py::arg("dim"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "permute",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& dims) -> Tensor {
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            arg.dims == dims.size(),
            "Operator permute expects `dims` argument to have the same length as input!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        self.fusion_definition->defineRecord(
            new DimsOpRecord<serde::RecordType::PermuteOp>(
                {fd->recordingState(arg())},
                {fd->recordingState(output())},
                std::move(dims),
                "ops.permute"));
        return output;
      },
      py::arg("arg"),
      py::arg("dims"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "shape",
      [](FusionDefinition::Operators& self, Tensor arg) -> Vector {
        return shapeDef(arg);
      },
      py::arg("arg"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "size",
      [](FusionDefinition::Operators& self, Tensor arg, int64_t dim) -> Scalar {
        return sizeDef(arg, dim);
      },
      py::arg("arg"),
      py::arg("dim"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "at",
      [](FusionDefinition::Operators& self,
         Vector arg,
         int64_t index) -> Scalar { return atDef(arg, index); },
      py::arg("arg"),
      py::arg("index"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "slice",
      slice_fn<Vector>,
      py::arg("arg"),
      py::arg("start_indices"),
      py::arg("end_indices"),
      py::arg("strides") = py::none(),
      py::arg("manual_normalization") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "slice",
      slice_fn<py::list>,
      py::arg("arg"),
      py::arg("start_indices"),
      py::arg("end_indices"),
      py::arg("strides") = py::none(),
      py::arg("manual_normalization") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "slice",
      slice_fn<py::tuple>,
      py::arg("arg"),
      py::arg("start_indices"),
      py::arg("end_indices"),
      py::arg("strides") = py::none(),
      py::arg("manual_normalization") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "squeeze",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t> dims,
         const bool squeeze_expanded) -> Tensor {
        FUSER_PERF_SCOPE("Operators.squeeze");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims - dims.size());
        fd->defineRecord(new SqueezeOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            std::move(dims),
            squeeze_expanded));
        return output;
      },
      py::arg("arg"),
      py::arg("dims"),
      py::arg("squeeze_expanded") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "tensor_sizes",
      [](FusionDefinition::Operators& self, Tensor arg) -> std::vector<Scalar> {
        FUSER_PERF_SCOPE("Operators.tensor_sizes");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        std::vector<Scalar> outputs;
        std::vector<State> output_state;
        for (const auto idx : c10::irange(arg.dims)) {
          outputs.push_back(fd->defineScalar());
          output_state.push_back(fd->recordingState(outputs[idx]()));
        }
        fd->defineRecord(
            new TensorSizesRecord({fd->recordingState(arg())}, output_state));
        return outputs;
      },
      py::arg("arg"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "reshape",
      reshape_fn<Vector>,
      py::arg("arg"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "reshape",
      reshape_fn<py::list>,
      py::arg("arg"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "reshape",
      reshape_fn<py::tuple>,
      py::arg("arg"),
      py::arg("new_shape"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "iota",
      [](FusionDefinition::Operators& self,
         Scalar length,
         std::optional<Scalar> start,
         std::optional<Scalar> step,
         PrimDataType dtype) -> Tensor {
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(1);
        auto start_state = start.has_value()
            ? fd->recordingState(start.value()())
            : State(0, serde::StateType::None);
        auto step_state = step.has_value() ? fd->recordingState(step.value()())
                                           : State(0, serde::StateType::None);
        fd->defineRecord(new IotaOpRecord(
            {fd->recordingState(length()), start_state, step_state},
            {fd->recordingState(output())},
            dtype));
        return output;
      },
      py::arg("length"),
      py::arg("start").none(true),
      py::arg("step").none(true),
      py::arg("dtype") = DataType::Int,
      py::return_value_policy::reference);
  nvf_ops.def(
      "var",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& dims,
         int64_t correction,
         bool keepdim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.var");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = keepdim ? arg.dims : (arg.dims - dims.size());
        Tensor output = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            std::move(dims),
            correction,
            keepdim));
        return output;
      },
      py::arg("arg"),
      py::arg("dims"),
      py::arg("correction"),
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "var_mean",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         std::vector<int64_t>& dims,
         int64_t correction,
         bool keepdim) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.var_mean");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = keepdim ? arg.dims : (arg.dims - dims.size());
        Tensor var = fd->defineTensor(ndims);
        Tensor mean = fd->defineTensor(ndims);
        fd->defineRecord(new VarianceMeanOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(var()), fd->recordingState(mean())},
            std::move(dims),
            correction,
            keepdim));
        return std::make_tuple(var, mean);
      },
      py::arg("arg"),
      py::arg("dims"),
      py::arg("correction") = 1,
      py::arg("keepdim") = false,
      py::return_value_policy::reference);
  nvf_ops.def(
      "welford",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         const std::vector<int64_t>& dims) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.welford");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = (arg.dims - dims.size());
        Tensor avg = fd->defineTensor(ndims);
        Tensor var_sum = fd->defineTensor(ndims);
        Tensor n = fd->defineTensor(ndims);
        fd->defineRecord(new WelfordOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(avg()),
             fd->recordingState(var_sum()),
             fd->recordingState(n())},
            dims));
        return std::make_tuple(avg, var_sum, n);
      },
      py::arg("arg"),
      py::arg("dims"),
      py::return_value_policy::reference);
  nvf_ops.def(
      "sdpfa_bwd",
      [](FusionDefinition::Operators& self,
         Tensor grad_output,
         Tensor query,
         Tensor key,
         Tensor value,
         Tensor output,
         Tensor log_sumexp,
         std::optional<Scalar> dropout_p,
         std::optional<Scalar> is_causal,
         Tensor philox_seed,
         Tensor philox_offset,
         std::optional<Scalar> scale) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.sdpfa_bwd");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = query.dims;
        Tensor grad_query = fd->defineTensor(/*dims=*/ndims);
        Tensor grad_key = fd->defineTensor(/*dims=*/ndims);
        Tensor grad_value = fd->defineTensor(/*dims=*/ndims);

        auto dropout_p_state = dropout_p.has_value()
            ? fd->recordingState(dropout_p.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto is_causal_state = is_causal.has_value()
            ? fd->recordingState(is_causal.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto scale_state = scale.has_value()
            ? fd->recordingState(scale.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

        fd->defineRecord(new SdpaBwdOpRecord(
            {fd->recordingState(grad_output()),
             fd->recordingState(query()),
             fd->recordingState(key()),
             fd->recordingState(value()),
             fd->recordingState(output()),
             fd->recordingState(log_sumexp()),
             dropout_p_state,
             is_causal_state,
             fd->recordingState(philox_seed()),
             fd->recordingState(philox_offset()),
             scale_state},
            {fd->recordingState(grad_query()),
             fd->recordingState(grad_key()),
             fd->recordingState(grad_value())}));
        return std::make_tuple(grad_query, grad_key, grad_value);
      },
      py::arg("grad_output"),
      py::arg("query"),
      py::arg("key"),
      py::arg("value"),
      py::arg("output"),
      py::arg("log_sumexp"),
      py::arg("dropout_p").none(true) = py::none(),
      py::arg("is_causal").none(true) = py::none(),
      py::arg("philox_seed"),
      py::arg("philox_offset"),
      py::arg("scale").none(true) = py::none(),
      py::return_value_policy::reference);

  nvf_ops.def(
      "sdpfa_fwd",
      [](FusionDefinition::Operators& self,
         Tensor query,
         Tensor key,
         Tensor value,
         std::optional<Scalar> dropout_p,
         std::optional<Scalar> is_causal,
         std::optional<Scalar> scale) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.sdpfa_fwd");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = query.dims;
        Tensor output = fd->defineTensor(/*dims=*/ndims);
        Tensor log_sumexp = fd->defineTensor(/*dims=*/ndims - 1);
        Tensor philox_seed = fd->defineTensor(/*dims=*/0);
        Tensor philox_offset = fd->defineTensor(/*dims=*/0);

        auto dropout_p_state = dropout_p.has_value()
            ? fd->recordingState(dropout_p.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto is_causal_state = is_causal.has_value()
            ? fd->recordingState(is_causal.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto scale_state = scale.has_value()
            ? fd->recordingState(scale.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

        fd->defineRecord(new SdpaFwdOpRecord(
            {fd->recordingState(query()),
             fd->recordingState(key()),
             fd->recordingState(value()),
             dropout_p_state,
             is_causal_state,
             scale_state},
            {fd->recordingState(output()),
             fd->recordingState(log_sumexp()),
             fd->recordingState(philox_seed()),
             fd->recordingState(philox_offset())}));
        return std::make_tuple(output, log_sumexp, philox_seed, philox_offset);
      },
      py::arg("query"),
      py::arg("key"),
      py::arg("value"),
      py::arg("dropout_p").none(true) = py::none(),
      py::arg("is_causal").none(true) = py::none(),
      py::arg("scale").none(true) = py::none(),
      py::return_value_policy::reference);
}

} // namespace nvfuser::python_frontend
