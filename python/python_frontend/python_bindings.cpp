// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <iostream>
#include <optional>
#include <tuple>

#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <debug.h>
#include <fusion_profiler.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <mma_type.h>
#include <multidevice/communicator.h>
#include <ops/all_ops.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_record.h>
#include <python_frontend/python_bindings.h>
#include <python_frontend/translation.h>
#include <python_utils.h>
#include <runtime/fusion_kernel_runtime.h>
#include <scheduler/compile_time_info.h>
#include <scheduler/registry.h>
#include <scheduler/scheduler_types.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <transform_replay.h>

namespace nvfuser::python_frontend {

// Set of local functions that are used to compose python FusionDefinition
// bindings. Ideally, these would be templated lambda functions but those
// are not available without C++20.
namespace {
Vector define_vector_base_fn(
    FusionDefinition& fd,
    std::vector<Scalar>& args,
    bool inline_def = false) {
  FUSER_PERF_SCOPE("python_frontend::define_vector_base_fn");
  NVF_CHECK(!fd.completed(), "Attempting to add to a completed definition!");
  std::vector<State> inputs;
  inputs.reserve(args.size());
  for (const auto& arg : args) {
    inputs.push_back(fd.recordingState(arg()));
  }
  Vector out = fd.defineVector(inputs.size());
  fd.defineRecord(new VectorRecord(
      inputs, {fd.recordingState(out())}, DataType::Int, inline_def));
  return out;
}

template <class ITERABLE>
Vector define_vector_fn(
    FusionDefinition& self,
    ITERABLE& values,
    bool inline_def,
    bool shape_check) {
  FUSER_PERF_SCOPE("python_frontend::define_vector_fn");
  std::vector<Scalar> args;
  size_t idx = 0;
  for (const auto& item : values) {
    if (py::isinstance<py::int_>(item)) {
      auto int_value = py::cast<int64_t>(item);
      NVF_CHECK(
          !shape_check || int_value >= -1,
          "The value ",
          int_value,
          " at index ",
          idx,
          " was neither symbolic(-1), zero_element(0), broadcast(1), or "
          "static(>1).");
      Scalar out = self.defineScalar();
      self.defineRecord(new ScalarRecord(
          {self.recordingState(out())},
          py::cast<int64_t>(item),
          DataType::Int,
          /*inline_def=*/true));
      args.emplace_back(out);
    } else if (py::isinstance<Scalar>(item)) {
      args.emplace_back(py::cast<Scalar>(item));
    } else {
      NVF_CHECK(
          false,
          "Unsupported iterable object type for define_vector! Index:",
          idx);
    }
    ++idx;
  }
  return define_vector_base_fn(self, args, inline_def);
}

template <class ITERABLE>
Vector define_vector_explicit_fn(
    FusionDefinition& self,
    ITERABLE& values,
    PrimDataType dtype = DataType::Int) {
  return define_vector_fn<ITERABLE>(
      self, values, /*inline_def=*/false, /*shape_check=*/true);
}

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
    for (auto i : arange(new_start.size)) {
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
      "Number of tensor dimensions does not match slice dimensions! "
      "Tensor-dims: ",
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

std::vector<std::optional<bool>> computeContiguity(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  NVF_CHECK(
      sizes.size() == strides.size(),
      "compute_contiguity: Sizes and strides must have the same number of "
      "dimensions");
  // Not a broadcast means neither the stride == 0 (size can be non-zero)
  // or the size == 1 that each can indicate a broadcast
  auto not_broadcast = [&](auto i) { return strides[i] != 0 && sizes[i] != 1; };
  // Contiguity defaults to vector of all None's
  std::vector<std::optional<bool>> contiguity(sizes.size(), std::nullopt);
  if (contiguity.empty()) { // zero-dim tensor
    return contiguity;
  }
  int64_t last = (int64_t)sizes.size() - 1; // inner most dimension
  // Contiguity normallly is determined by the current dimension and one
  // dimension to the right.  The innermost dimension, that is not broadcasted,
  // does not have any dimension to it's right and needs to be specially marked
  // contiguous.
  for (; last >= 0; --last) {
    if (not_broadcast(last)) {
      contiguity[last] = (strides.at(last) == 1);
      break;
    }
  }
  // Dimensions are marked contiguous by inspecting the current dimension and
  // one to the right towards the inner dimension while skipping over broadcast
  // dimensions.
  for (int64_t i = 0; i < last;) {
    if (not_broadcast(i)) {
      auto l = i++;
      for (; i <= last; i++) {
        if (not_broadcast(i)) {
          break;
        }
      }
      contiguity[l] = (strides[l] == strides[i] * sizes[i]);
    } else {
      i++;
    }
  }
  return contiguity;
}

// Copy definition from a FusionDefinion's pre-scheduled CPP fusion to a blank
// FusionDefinition. Primarily for testing purposes to check that the
// translation from CPP fusion is correct.
void clone(FusionDefinition& from, FusionDefinition& to) {
  NVF_CHECK(from.completed(), "FusionDefinition definition does not exist!");
  NVF_ERROR(
      !to.completed(), "Expected an incomplete definition before translation.");
  translate(from.preschedFusion(), &to);
}

namespace {
void defineHeuristicParamBindings(py::module& nvfuser) {
  py::class_<LaunchParams> launch_parameters(nvfuser, "LaunchParams");
  launch_parameters.def(
      py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>());
  launch_parameters.def(
      "__repr__", [](const LaunchParams& self) { return self.toString(); });
  launch_parameters.def_property(
      "bdimx",
      [](LaunchParams& self) { return self.bdimx(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDx);
      });
  launch_parameters.def_property(
      "bdimy",
      [](LaunchParams& self) { return self.bdimy(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDy);
      });
  launch_parameters.def_property(
      "bdimz",
      [](LaunchParams& self) { return self.bdimz(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDz);
      });
  launch_parameters.def_property(
      "gdimx",
      [](LaunchParams& self) { return self.gdimx(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDx);
      });
  launch_parameters.def_property(
      "gdimy",
      [](LaunchParams& self) { return self.gdimy(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDy);
      });
  launch_parameters.def_property(
      "gdimz",
      [](LaunchParams& self) { return self.gdimz(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDz);
      });

#define DEFINECLASS(type) py::class_<type>(nvfuser, #type)

#define TOSTRINGTOPLEVEL(type) \
  def("__repr__", [](const type& self) { return toString(self); })
#define TOSTRINGMETHOD(type) \
  def("__repr__", [](const type& self) { return self.toString(); })

#define PARAM(internal_type, name) def_readwrite(#name, &internal_type::name)

  DEFINECLASS(CompileParams)
      .PARAM(CompileParams, index_type)
      .PARAM(CompileParams, maxrregcount)
      .PARAM(CompileParams, enable_magic_zero)
      .PARAM(CompileParams, enable_ptxas_verbose)
      .TOSTRINGMETHOD(CompileParams);

  DEFINECLASS(GemmTile)
      .def(py::init<int64_t, int64_t, int64_t>())
      .PARAM(GemmTile, m)
      .PARAM(GemmTile, n)
      .PARAM(GemmTile, k)
      .TOSTRINGTOPLEVEL(GemmTile);

  DEFINECLASS(MatMulTileOptions)
      .def(py::init<GemmTile, GemmTile>())
      .PARAM(MatMulTileOptions, cta_tile)
      .PARAM(MatMulTileOptions, warp_tile)
      .TOSTRINGTOPLEVEL(MatMulTileOptions);

  py::class_<MatmulParams::CircularBufferOptions>(
      nvfuser, "CircularBufferOptions")
      .def(py::init<bool, bool, int, int>())
      .PARAM(MatmulParams::CircularBufferOptions, circular_buffer_smem_read)
      .PARAM(MatmulParams::CircularBufferOptions, circular_buffer_smem_write)
      .PARAM(MatmulParams::CircularBufferOptions, smem_circular_buffer_stage)
      .PARAM(
          MatmulParams::CircularBufferOptions,
          smem_circular_buffer_prefetch_gap)
      .TOSTRINGMETHOD(MatmulParams::CircularBufferOptions);

  py::class_<MatmulParams::SupportedVectorization>(
      nvfuser, "SupportedVectorization")
      .def(py::init<int64_t, int64_t, int64_t>())
      .PARAM(MatmulParams::SupportedVectorization, a)
      .PARAM(MatmulParams::SupportedVectorization, b)
      .PARAM(MatmulParams::SupportedVectorization, epilogue)
      .TOSTRINGMETHOD(MatmulParams::SupportedVectorization);

  py::enum_<MatmulParams::TileRasterizationOrder>(
      nvfuser, "MatmulTileRasterizationOrder")
      .value("column_major", MatmulParams::TileRasterizationOrder::ColumnMajor)
      .value("row_major", MatmulParams::TileRasterizationOrder::RowMajor);

  py::class_<MatmulParams::ClusterDims>(nvfuser, "ClusterDims")
      .def(py::init<int64_t, int64_t, int64_t>())
      .PARAM(MatmulParams::ClusterDims, x)
      .PARAM(MatmulParams::ClusterDims, y)
      .PARAM(MatmulParams::ClusterDims, z)
      .TOSTRINGMETHOD(MatmulParams::ClusterDims);

  py::enum_<MmaMacroEncode::Arch>(nvfuser, "MmaMacroArch")
      .value("no_mma", MmaMacroEncode::Arch::NoMma)
      .value("volta", MmaMacroEncode::Arch::Volta)
      .value("turing", MmaMacroEncode::Arch::Turing)
      .value("ampere", MmaMacroEncode::Arch::Ampere)
      .value("hopper", MmaMacroEncode::Arch::Hopper);

  DEFINECLASS(MmaMacroEncode)
      .def(py::init<MmaMacroEncode::Arch, uint16_t, uint16_t, uint16_t>())
      .def("mma_macro", &MmaMacroEncode::operator MmaMacro)
      .PARAM(MmaMacroEncode, arch)
      .PARAM(MmaMacroEncode, m)
      .PARAM(MmaMacroEncode, n)
      .PARAM(MmaMacroEncode, k);

  // NOTE: MmaMacro is a uint64_t. To modify it, we convert to and from
  // MmaMacroEncode
#define MMAMACROPROP(prop, type)                                      \
  def_property(                                                       \
      #prop,                                                          \
      [](const MmaMacro& self) { return MmaMacroEncode(self).prop; }, \
      [](MmaMacro& self, type x) {                                    \
        auto enc = MmaMacroEncode(self);                              \
        enc.prop = x;                                                 \
        self = enc;                                                   \
      })
  DEFINECLASS(MmaMacro)
      .MMAMACROPROP(arch, MmaMacroEncode::Arch)
      .MMAMACROPROP(m, uint16_t)
      .MMAMACROPROP(n, uint16_t)
      .MMAMACROPROP(k, uint16_t)
      .TOSTRINGTOPLEVEL(MmaMacro);
#undef MMAMACROPROP

  py::enum_<MatmulParams::TilingStrategy>(nvfuser, "MatmulTilingStrategy")
      .value("one_tile_per_cta", MatmulParams::TilingStrategy::OneTilePerCTA)
      .value(
          "distribute_tiles_across_sms",
          MatmulParams::TilingStrategy::DistributeTilesAcrossSMs)
      .value(
          "distribute_stages_across_sms",
          MatmulParams::TilingStrategy::DistributeStagesAcrossSMs);
  py::enum_<MatmulParams::BufferingLoopLevel>(
      nvfuser, "MatmulBufferingLoopLevel")
      .value("cta_tiles", MatmulParams::BufferingLoopLevel::CTATiles)
      .value("warp_tiles", MatmulParams::BufferingLoopLevel::WarpTiles);
  py::enum_<MatmulParams::CircularBufferingStrategy>(
      nvfuser, "MatmulCircularBufferingStrategy")
      .value("pipelined", MatmulParams::CircularBufferingStrategy::Pipelined)
      .value(
          "warp_specialized",
          MatmulParams::CircularBufferingStrategy::WarpSpecialized);

  // Base class for scheduler parameters
  DEFINECLASS(HeuristicParams)
      .TOSTRINGMETHOD(HeuristicParams)
      .PARAM(HeuristicParams, lparams)
      .PARAM(HeuristicParams, cparams);

#define INITHEURISTICPARAMS(internal_type)                            \
  py::class_<internal_type, HeuristicParams>(nvfuser, #internal_type) \
      .def(py::init())                                                \
      .def("__repr__", [](const internal_type& self) {                \
        return self.toString();                                       \
      })

  // Pointwise scheduler parameters
  INITHEURISTICPARAMS(PointwiseParams)
      .PARAM(PointwiseParams, break_point)
      .PARAM(PointwiseParams, split_block)
      .PARAM(PointwiseParams, split_grid_y_dim)
      .PARAM(PointwiseParams, flip_grid_binding)
      .PARAM(PointwiseParams, vectorization_factor)
      .PARAM(PointwiseParams, unroll_factor_inner)
      .PARAM(PointwiseParams, unroll_factor_outer);

  // Reduction scheduler parameters
  INITHEURISTICPARAMS(ReductionParams)
      .PARAM(ReductionParams, fastest_dim)
      .PARAM(ReductionParams, persistent_kernel)
      .PARAM(ReductionParams, project_persistent_buffers)
      .PARAM(ReductionParams, schedule_3D)
      .PARAM(ReductionParams, flip_grid)
      .PARAM(ReductionParams, cross_block_inner_reduction)
      .PARAM(ReductionParams, cross_grid_inner_reduction)
      .PARAM(ReductionParams, unroll_factor_inner_reduction)
      .PARAM(ReductionParams, unroll_factor_top_of_vectorization)
      .PARAM(ReductionParams, vectorize_inner_reduction)
      .PARAM(ReductionParams, split_grid_dim_inner_reduction)
      .PARAM(ReductionParams, pad_inner_reduction_to_warp)
      .PARAM(ReductionParams, batches_per_block_inner_reduction)
      .PARAM(ReductionParams, block_dim_inner_reduction)
      .PARAM(ReductionParams, grid_dim_inner_reduction)
      .PARAM(ReductionParams, multiple_reds_per_blk)
      .PARAM(ReductionParams, unroll_factor_iter_dom)
      .PARAM(ReductionParams, vectorize_iter_dom)
      .PARAM(ReductionParams, split_grid_dim_iter_dom_inner)
      .PARAM(ReductionParams, split_grid_dim_iter_dom_outer)
      .PARAM(ReductionParams, block_dim_iter_dom)
      .PARAM(ReductionParams, grid_dim_iter_dom)
      .PARAM(ReductionParams, cross_block_outer_reduction)
      .PARAM(ReductionParams, cross_grid_outer_reduction)
      .PARAM(ReductionParams, batches_per_block_outer_reduction)
      .PARAM(ReductionParams, unroll_factor_outer_reduction)
      .PARAM(ReductionParams, block_dim_outer_reduction)
      .PARAM(ReductionParams, grid_dim_outer_reduction)
      .PARAM(ReductionParams, compute_persistent_buffer_with_first_consumer)
      .PARAM(ReductionParams, static_bdimx)
      .PARAM(ReductionParams, static_bdimy)
      .PARAM(ReductionParams, combined_inner_outer)
      .PARAM(ReductionParams, tidx_for_outer_reduction)
      .PARAM(ReductionParams, pad_outer_reduction_to_warp)
      .PARAM(ReductionParams, combined_split_grid_inner_dim)
      .PARAM(ReductionParams, vectorization_factor_outer)
      .PARAM(ReductionParams, vectorization_factor_tmp_gmem_write)
      .PARAM(ReductionParams, block_dim_inner_reduction_extra);

  // Matmul scheduler parameters
  INITHEURISTICPARAMS(MatmulParams)
      .PARAM(MatmulParams, tile_sizes)
      .PARAM(MatmulParams, circular_buffer_options)
      .PARAM(MatmulParams, supported_vec_size)
      .PARAM(MatmulParams, async_gmem_load_operands)
      .PARAM(MatmulParams, grid_traversal_factor)
      .PARAM(MatmulParams, use_smem_epilogue)
      .PARAM(MatmulParams, use_ldst_matrix)
      .PARAM(MatmulParams, promote_prologue_smem_reuse)
      .PARAM(MatmulParams, splitk_factor)
      .PARAM(MatmulParams, tiling_strategy)
      .PARAM(MatmulParams, buffering_loop_level)
      .PARAM(MatmulParams, circular_buffering_strategy)
      .PARAM(MatmulParams, cta_order)
      .PARAM(MatmulParams, cluster_dims)
      .PARAM(MatmulParams, mma_macro);

#undef PARAM
#undef INITPARAMS
}

} // namespace

void initNvFuserPythonBindings(PyObject* module) {
  auto nvfuser = py::handle(module).cast<py::module>();

  nvfuser.def("clone", clone);

  //! DataTypes supported by nvFuser in the FusionDefinition
  py::enum_<PrimDataType>(nvfuser, "DataType")
      .value("Double", DataType::Double)
      .value("Float", DataType::Float)
      .value("Half", DataType::Half)
      .value("Int", DataType::Int)
      .value("Int32", DataType::Int32)
      .value("UInt64", DataType::UInt64)
      .value("Bool", DataType::Bool)
      .value("BFloat16", DataType::BFloat16)
      .value("Float8_e4m3fn", DataType::Float8_e4m3fn)
      .value("Float8_e5m2", DataType::Float8_e5m2)
      .value("Float8_e8m0fnu", DataType::Float8_e8m0fnu)
      .value("ComplexFloat", DataType::ComplexFloat)
      .value("ComplexDouble", DataType::ComplexDouble)
      .value("Null", DataType::Null);

  //! ParallelType used for scheduling
  py::enum_<ParallelType>(nvfuser, "ParallelType")
      .value("mesh_x", ParallelType::DIDx)
      .value("grid_x", ParallelType::BIDx)
      .value("grid_y", ParallelType::BIDy)
      .value("grid_z", ParallelType::BIDz)
      .value("block_x", ParallelType::TIDx)
      .value("block_y", ParallelType::TIDy)
      .value("block_z", ParallelType::TIDz)
      .value("mma", ParallelType::Mma)
      .value("serial", ParallelType::Serial)
      .value("tma", ParallelType::Bulk)
      .value("unroll", ParallelType::Unroll)
      .value("unswitch", ParallelType::Unswitch)
      .value("vectorize", ParallelType::Vectorize)
      .value("stream", ParallelType::Stream);

  //! LoadStoreOpType used for scheduling
  py::enum_<LoadStoreOpType>(nvfuser, "LoadStoreOpType")
      .value("set", LoadStoreOpType::Set)
      .value("load_matrix", LoadStoreOpType::LdMatrix)
      .value("cp_async", LoadStoreOpType::CpAsync)
      .value("tma", LoadStoreOpType::CpAsyncBulkTensorTile);

  //! CacheOp used for scheduling
  py::enum_<CacheOp>(nvfuser, "CacheOp")
      .value("unspecified", CacheOp::Unspecified)
      .value("all_levels", CacheOp::AllLevels)
      .value("streaming", CacheOp::Streaming)
      .value("global", CacheOp::Global);

  //! MemoryType used for scheduling
  py::enum_<MemoryType>(nvfuser, "MemoryType")
      .value("local", MemoryType::Local)
      .value("shared", MemoryType::Shared)
      .value("global", MemoryType::Global);

  //! Scheduler Type for scheduling
  py::enum_<SchedulerType>(nvfuser, "SchedulerType")
      .value("none", SchedulerType::None)
      .value("no_op", SchedulerType::NoOp)
      .value("pointwise", SchedulerType::PointWise)
      .value("matmul", SchedulerType::Matmul)
      .value("reduction", SchedulerType::Reduction)
      .value("inner_persistent", SchedulerType::InnerPersistent)
      .value("inner_outer_persistent", SchedulerType::InnerOuterPersistent)
      .value("outer_persistent", SchedulerType::OuterPersistent)
      .value("transpose", SchedulerType::Transpose)
      .value("expr_eval", SchedulerType::ExprEval)
      .value("resize", SchedulerType::Resize);

  py::enum_<CommunicatorBackend>(nvfuser, "CommunicatorBackend")
      .value("nccl", CommunicatorBackend::kNccl)
      .value("ucc", CommunicatorBackend::kUcc);

  nvfuser.def("compute_contiguity", computeContiguity);
  nvfuser.def("compute_tensor_descriptor", computeTensorDescriptor);
  nvfuser.def("serialize", serialize);

  //! Binding the FusionCache that holds a cache of Fusions
  //! This is only bound to provide an interface to get the number of fusions
  //! that are cached.
  py::class_<FusionCache> fusion_cache(nvfuser, "FusionCache");
  fusion_cache
      .def_static(
          "get",
          &FusionCache::get,
          py::arg("max_fusions") = int(16384),
          py::arg("selected_device") = py::none(),
          py::arg("load_from_default_workspace") = true,
          py::return_value_policy::reference)
      .def("num_fusions", &FusionCache::numFusions)
      .def_static(
          "reset", &FusionCache::reset, py::return_value_policy::reference)
      .def(
          "serialize",
          [](FusionCache& self, std::string filename) {
            FUSER_PERF_SCOPE("FusionCache.serialize (string)");
            self.serialize(filename);
          },
          py::arg("filename"))
      .def(
          "deserialize",
          [](FusionCache& self, std::string filename) {
            FUSER_PERF_SCOPE("FusionCache.deserialize (string)");
            self.deserialize(filename);
          },
          py::arg("filename"))
      .def(
          "__repr__",
          [](FusionCache& self) {
            std::stringstream ss;
            self.print(ss);
            return ss.str();
          })
      .def("stats", [](FusionCache& self) {
        std::stringstream ss;
        self.stats(ss);
        return ss.str();
      });

  defineHeuristicParamBindings(nvfuser);

  py::class_<scheduler_utils::SchedulerHyperParameters> hyperparameters(
      nvfuser, "SchedulerHyperParameters");
  hyperparameters.def(py::init<int64_t, int64_t, int64_t, int64_t, bool>());
  hyperparameters.def_property(
      "vectorize_factor",
      [](scheduler_utils::SchedulerHyperParameters& self) {
        return self.vectorize_factor;
      },
      [](scheduler_utils::SchedulerHyperParameters& self,
         int64_t vectorize_factor_) {
        self.vectorize_factor = vectorize_factor_;
      });
  hyperparameters.def_property(
      "unroll_factor",
      [](scheduler_utils::SchedulerHyperParameters& self) {
        return self.unroll_factor;
      },
      [](scheduler_utils::SchedulerHyperParameters& self,
         int64_t unroll_factor_) { self.unroll_factor = unroll_factor_; });
  hyperparameters.def_property(
      "threads_per_block_min",
      [](scheduler_utils::SchedulerHyperParameters& self) {
        return self.threads_per_block_min;
      },
      [](scheduler_utils::SchedulerHyperParameters& self,
         int64_t threads_per_block_min_) {
        self.threads_per_block_min = threads_per_block_min_;
      });
  hyperparameters.def_property(
      "threads_per_block_max",
      [](scheduler_utils::SchedulerHyperParameters& self) {
        return self.threads_per_block_max;
      },
      [](scheduler_utils::SchedulerHyperParameters& self,
         int64_t threads_per_block_max_) {
        self.threads_per_block_max = threads_per_block_max_;
      });
  hyperparameters.def_property(
      "is_warp_specialized",
      [](scheduler_utils::SchedulerHyperParameters& self) {
        return self.is_warp_specialized;
      },
      [](scheduler_utils::SchedulerHyperParameters& self,
         int64_t is_warp_specialized_) {
        self.is_warp_specialized = is_warp_specialized_;
      });
  //! KernelProfiles are encapsulated in FusionProfiles where each KP
  //! is associated with a segment.
  py::class_<KernelProfile> kernel_prof(nvfuser, "KernelProfile");
  kernel_prof.def_property_readonly(
      "name", [](KernelProfile& self) { return self.name; });
  kernel_prof.def_property_readonly(
      "segment_id", [](KernelProfile& self) { return self.segment_id; });
  kernel_prof.def_property_readonly(
      "device", [](KernelProfile& self) { return self.device; });
  kernel_prof.def_property_readonly(
      "stream", [](KernelProfile& self) { return self.stream; });
  kernel_prof.def_property_readonly("correlation_id", [](KernelProfile& self) {
    return self.correlation_id;
  });
  kernel_prof.def_property_readonly("compile_time_ms", [](KernelProfile& self) {
    return self.compile_time_ms;
  });
  kernel_prof.def_property_readonly(
      "time_ms", [](KernelProfile& self) { return self.time_ms; });
  kernel_prof.def_property_readonly(
      "effective_bandwidth_gbs",
      [](KernelProfile& self) { return self.effective_bandwidth_gbs; });
  kernel_prof.def_property_readonly(
      "percentage_peak_bandwidth",
      [](KernelProfile& self) { return self.percentage_peak_bandwidth; });
  kernel_prof.def_property_readonly(
      "grid_str", [](KernelProfile& self) { return self.grid_str; });
  kernel_prof.def_property_readonly(
      "block_str", [](KernelProfile& self) { return self.block_str; });
  kernel_prof.def_property_readonly(
      "cluster_str", [](KernelProfile& self) { return self.cluster_str; });
  kernel_prof.def_property_readonly("shared_mem_str", [](KernelProfile& self) {
    return self.shared_mem_str;
  });
  kernel_prof.def_property_readonly(
      "registers", [](KernelProfile& self) { return self.registers; });
  kernel_prof.def_property_readonly(
      "input_bytes", [](KernelProfile& self) { return self.input_bytes; });
  kernel_prof.def_property_readonly(
      "output_bytes", [](KernelProfile& self) { return self.output_bytes; });
  kernel_prof.def_property_readonly(
      "scheduler", [](KernelProfile& self) { return self.scheduler; });

  //! A fusion profile is generated for FusionDefinition.
  py::class_<FusionProfile> fusion_prof(nvfuser, "FusionProfile");
  fusion_prof.def_property_readonly(
      "verbose", [](FusionProfile& self) { return self.verbose; });
  fusion_prof.def_property_readonly(
      "fusion_id", [](FusionProfile& self) { return self.fusion_id; });
  fusion_prof.def_property_readonly(
      "segments", [](FusionProfile& self) { return self.segments; });
  fusion_prof.def_property_readonly(
      "cuda_evt_time_ms",
      [](FusionProfile& self) { return self.cuda_evt_time_ms; });
  fusion_prof.def_property_readonly(
      "host_time_ms", [](FusionProfile& self) { return self.host_time_ms; });
  fusion_prof.def_property_readonly("compile_time_ms", [](FusionProfile& self) {
    return self.compile_time_ms;
  });
  fusion_prof.def_property_readonly("kernel_time_ms", [](FusionProfile& self) {
    return self.kernel_time_ms;
  });
  fusion_prof.def_property_readonly(
      "effective_bandwidth_gbs",
      [](FusionProfile& self) { return self.effective_bandwidth_gbs; });
  fusion_prof.def_property_readonly(
      "percentage_peak_bandwith",
      [](FusionProfile& self) { return self.percentage_peak_bandwidth; });
  fusion_prof.def_property_readonly(
      "input_bytes", [](FusionProfile& self) { return self.input_bytes; });
  fusion_prof.def_property_readonly(
      "output_bytes", [](FusionProfile& self) { return self.output_bytes; });
  fusion_prof.def_property_readonly("kernel_profiles", [](FusionProfile& self) {
    return self.kernel_profiles;
  });

  //! These are the FusionDefinition supported object types that are either
  //! defined as inputs or the output of an operation.
  py::class_<Tensor> tensor_class(nvfuser, "Tensor");
  tensor_class.def("__repr__", [](Tensor& self) {
    std::stringstream ss;
    ss << "Tensor(index=" << self.index << ", ndim=" << self.dims << ")";
    return ss.str();
  });
  tensor_class.def_property_readonly(
      "ndim",
      [](Tensor& self) { return self.dims; },
      "Returns the rank of the tensor.");
  tensor_class.def_property_readonly(
      "index",
      [](Tensor& self) { return self.index; },
      "Returns the index of the tensor as in "
      "FusionDefinition.sched.tensors().");
  tensor_class.def("_get_fusion_definition", [](Tensor& self) {
    return self.fusion_definition;
  });
  tensor_class.def(pybind11::self == pybind11::self);
  tensor_class.def(pybind11::self != pybind11::self);

  py::class_<Scalar> scalar_class(nvfuser, "Scalar");
  scalar_class.def("__repr__", [](Scalar& self) {
    std::stringstream ss;
    ss << "Scalar(index=" << self.index << ")";
    return ss.str();
  });
  scalar_class.def(pybind11::self == pybind11::self);
  scalar_class.def(pybind11::self != pybind11::self);

  py::class_<Vector> vector_class(nvfuser, "Vector");
  vector_class.def("__repr__", [](Vector& self) {
    std::stringstream ss;
    ss << "Vector(index=" << self.index << ", size=" << self.size << ")";
    return ss.str();
  });
  vector_class.def_property_readonly(
      "size", [](Vector& self) { return self.size; });
  vector_class.def(pybind11::self == pybind11::self);
  vector_class.def(pybind11::self != pybind11::self);

  //! The FusionDefinition is a context manager in Python where the user will
  //! define the set the operations and connections between operations for
  //! nvFuser to create.
  py::class_<FusionDefinition> fusion_def(nvfuser, "_FusionDefinition");
  fusion_def
      .def(
          py::init<std::optional<size_t>, size_t, bool, CommunicatorBackend>(),
          py::arg("id") = py::none(),
          py::arg("max_length") = int(1024),
          py::arg("use_multidevice_executor") = false,
          py::arg("backend_type") = CommunicatorBackend::kNccl)
      .def_readwrite("ops", &FusionDefinition::ops)
      .def_readwrite("sched", &FusionDefinition::sched)
      .def(
          "_setup_definition",
          [](FusionDefinition& self) -> FusionDefinition* {
            // Instrumentation to mark the beginning of a FusionDefinition
            inst::Trace::instance()->beginEvent("FusionDefinition Definition");
            return self.setupDefinition();
          })
      .def(
          "_finalize_definition",
          [](FusionDefinition& self) {
            self.finalizeDefinition();
            // Mark the end of a definition
            inst::Trace::instance()->endEvent("FusionDefinition Definition");
          })
      .def(
          "_exist_schedule",
          [](FusionDefinition& self, const py::iterable& iter) {
            KernelArgumentHolder args;
            for (py::handle obj : iter) {
              args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            return self.existSchedule(args);
          })
      .def(
          "_setup_schedule",
          [](FusionDefinition& self,
             const py::iterable& iter,
             bool overwrite_existing_schedule) {
            // Instrumentation to mark the beginning of a schedule
            inst::Trace::instance()->beginEvent("FusionDefinition Schedule");
            KernelArgumentHolder args;
            for (py::handle obj : iter) {
              args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            self.setupSchedule(args, overwrite_existing_schedule);
          },
          py::arg("inputs"),
          py::kw_only(),
          py::arg("overwrite_existing_schedule") = false)
      .def(
          "_finalize_schedule",
          [](FusionDefinition& self, const py::iterable& iter) {
            KernelArgumentHolder args;
            for (py::handle obj : iter) {
              args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            self.finalizeSchedule(args);
            // Mark the end of a schedule
            inst::Trace::instance()->endEvent(nullptr);
          })
      .def(
          "_setup_multidevice_schedule",
          [](FusionDefinition& self) { self.setupMultideviceSchedule(); })
      .def(
          "_finalize_multidevice_schedule",
          [](FusionDefinition& self) { self.finalizeMultideviceSchedule(); })
      .def("inputs", [](FusionDefinition& self) { return self.inputs(); })
      .def("outputs", [](FusionDefinition& self) { return self.outputs(); })
      .def("extents", [](FusionDefinition& self) { return self.extents(); })
      .def(
          "_setup_segmentation",
          [](FusionDefinition& self, const py::iterable& iter) {
            // Instrumentation to mark the beginning of segmentation
            inst::Trace::instance()->beginEvent(
                "FusionDefinition Segmentation");
            KernelArgumentHolder args;
            for (py::handle obj : iter) {
              // Allows for a Vector of Sizes to be inputed as a list/tuple
              if (py::isinstance<py::list>(obj) ||
                  py::isinstance<py::tuple>(obj)) {
                for (py::handle item : obj) {
                  args.push(torch::jit::toIValue(item, c10::AnyType::get()));
                }
              } else {
                args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
              }
            }
            return self.setupSegmentation(args);
          })
      .def(
          "_build_segment",
          [](FusionDefinition& self,
             FusionDefinition& other,
             int64_t segment_id) {
            return self.buildSegment(other, segment_id);
          })
      .def(
          "_finalize_segmentation",
          [](FusionDefinition& self) {
            self.finalizeSegmentation();
            // Mark the end of segmentation
            inst::Trace::instance()->endEvent(nullptr);
          })
      .def("inputs", [](FusionDefinition& self) { return self.inputs(); })
      .def("outputs", [](FusionDefinition& self) { return self.outputs(); })
      .def("extents", [](FusionDefinition& self) { return self.extents(); })
      .def(
          "__repr__",
          [](FusionDefinition& self) {
            std::stringstream ss;
            self.print(ss);
            return ss.str();
          })
      .def(
          "_execute",
          [](FusionDefinition& self,
             const py::iterable& iter,
             std::optional<int64_t> device,
             bool override_user_schedule,
             bool capture_debug_output,
             bool profile,
             std::vector<std::string> _enable_options,
             std::vector<std::string> _disable_options)
              -> std::pair<std::vector<at::Tensor>, std::vector<Sharding>> {
            KernelArgumentHolder ins;
            for (py::handle obj : iter) {
              // Allows for a Vector of Sizes to be inputed as a list/tuple
              if (py::isinstance<py::list>(obj) ||
                  py::isinstance<py::tuple>(obj)) {
                for (py::handle item : obj) {
                  ins.push(torch::jit::toIValue(item, c10::AnyType::get()));
                }
              } else {
                ins.push(torch::jit::toIValue(obj, c10::AnyType::get()));
              }
            }
            std::optional<int8_t> int8_device = std::nullopt;
            if (device.has_value()) {
              NVF_CHECK(device.value() < 256, "Maximum device index is 255");
              int8_device = (int8_t)device.value();
            }
            auto&& [outs, out_shardings] = self.execute(
                ins,
                int8_device,
                override_user_schedule,
                capture_debug_output,
                profile,
                _enable_options,
                _disable_options);

            std::vector<at::Tensor> out_tensors;
            out_tensors.reserve(outs.size());
            for (const auto& out : outs) {
              // Should we append toIValue(out) instead?
              out_tensors.push_back(out.as<at::Tensor>());
            }
            return std::make_pair(
                std::move(out_tensors), std::move(out_shardings));
          },
          py::arg("inputs"),
          py::kw_only(),
          py::arg("device") = py::none(),
          py::arg("override_user_schedule") = false,
          py::arg("capture_debug_output") = false,
          py::arg("profile") = false,
          py::arg("_enable_options") = py::none(),
          py::arg("_disable_options") = py::none(),
          py::return_value_policy::reference)
      .def_static(
          "_profile",
          &FusionProfiler::profile,
          py::return_value_policy::reference)
      .def(
          "_debug_output",
          [](FusionDefinition& self) { return self.getDebugOutput(); },
          py::return_value_policy::reference)
      .def(
          "_fusion_ir",
          [](FusionDefinition& self) { return self.fusionIr(); },
          py::return_value_policy::reference)
      .def(
          "_user_schedule_ir",
          [](FusionDefinition& self) { return self.userScheduleIr(); },
          py::return_value_policy::reference)
      .def(
          "_last_cuda_code",
          [](FusionDefinition& self,
             bool intrinsic_code,
             bool override_user_schedule) {
            return self.lastCudaCode(intrinsic_code, override_user_schedule);
          },
          py::arg("intrinsic_code") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "_cuda_code_for",
          [](FusionDefinition& self,
             const py::iterable& iter,
             bool intrinsic_code,
             bool override_user_schedule) {
            KernelArgumentHolder args;
            for (py::handle obj : iter) {
              args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            return self.cudaCodeFor(
                args, intrinsic_code, override_user_schedule);
          },
          py::arg("inputs"),
          py::arg("intrinsic_code") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "_last_scheduled_fusion_ir",
          [](FusionDefinition& self,
             bool tensor_transforms,
             bool override_user_schedule) {
            return self.lastScheduledFusionIr(
                tensor_transforms, override_user_schedule);
          },
          py::arg("tensor_transforms") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "_scheduled_fusion_ir_for",
          [](FusionDefinition& self,
             const py::iterable& iter,
             bool tensor_transforms,
             bool override_user_schedule) {
            KernelArgumentHolder args;
            for (py::handle obj : iter) {
              args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
            }
            return self.scheduledFusionIrFor(
                args, tensor_transforms, override_user_schedule);
          },
          py::arg("inputs"),
          py::arg("tensor_transforms") = false,
          py::arg("override_user_schedule") = false,
          py::return_value_policy::reference)
      .def(
          "id",
          [](FusionDefinition& self) -> std::optional<size_t> {
            return self.id();
          })
      .def(
          "add_output",
          [](FusionDefinition& self, Scalar output) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (scalar)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            self.defineRecord(new OutputRecord<Val>(
                {self.recordingState(output())}, serde::RecordType::OutputVal));
          },
          py::arg("output"))
      .def(
          "add_output",
          [](FusionDefinition& self,
             Tensor output,
             std::optional<Tensor> alias_input = std::nullopt) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (tensor)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            if (alias_input.has_value()) {
              self.defineRecord(new OutputRecord<TensorView>(
                  {self.recordingState(output()),
                   self.recordingState(alias_input.value()())},
                  serde::RecordType::OutputTv));
            } else {
              self.defineRecord(new OutputRecord<TensorView>(
                  {self.recordingState(output())},
                  serde::RecordType::OutputTv));
            }
          },
          py::arg("output"),
          py::arg("alias_input") = py::none())
      .def(
          "add_output",
          [](FusionDefinition& self,
             Tensor output,
             std::vector<int64_t> stride_order) {
            FUSER_PERF_SCOPE("FusionDefinition.add_output (tensor)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            NVF_CHECK(
                stride_order.empty() || output.dims == stride_order.size(),
                "stride_order needs to be either empty or the same length of "
                "Tensor `output`");
            int64_t duplicate_check = 0;
            for (const auto& v : stride_order) {
              NVF_CHECK(
                  v >= 0 && v < (int64_t)stride_order.size(),
                  "stride_order elements need to be within [0, "
                  "stride_order.size())");
              duplicate_check |= 1 << v;
            }
            NVF_CHECK(
                duplicate_check == (1 << stride_order.size()) - 1,
                "duplicated elements in stride_order detected!");
            self.defineRecord(new OutputRecord<TensorView>(
                {self.recordingState(output())},
                serde::RecordType::OutputTv,
                stride_order));
          },
          py::arg("output"),
          py::arg("stride_order"))
      // This version of define_tensor is the canonical version
      // that displays the values as they are passed to the IR's
      // TensorViewBuilder.
      // Each dimension can be of value:
      // -1 : Symbolic for Dynamic usage
      //  0 : Zero-element
      //  1 : Broadcast
      // >1 : Static size
      // NOTE: A Tensor defined for dynamic shape usage should only
      // contain either symbolic(-1) or broadcast(1) defined dimensions.
      .def(
          "define_tensor",
          [](FusionDefinition& self,
             const std::vector<int64_t>& shape,
             const std::vector<std::optional<bool>>& contiguity,
             const PrimDataType dtype = DataType::Float,
             const bool is_cpu = false,
             const std::vector<int64_t>& stride_order = {}) -> Tensor {
            FUSER_PERF_SCOPE(
                "FusionDefinition.define_tensor (contiguity as vector)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");

            verifyShape(shape);

            Tensor out = self.defineTensor(shape.size());
            self.defineRecord(new TensorRecord(
                {self.recordingState(out())},
                shape,
                contiguity,
                dtype,
                is_cpu,
                stride_order));

            return out;
          },
          py::arg("shape"),
          py::arg("contiguity"),
          py::arg("dtype") = DataType::Float,
          py::arg("is_cpu") = false,
          py::arg("stride_order") = py::list(),
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](FusionDefinition& self,
             const std::vector<int64_t>& shape,
             // Contiguity for non-broadcast dimensions.
             const bool contiguity = false,
             const PrimDataType dtype = DataType::Float,
             const bool is_cpu = false,
             const std::vector<int64_t>& stride_order = {}) -> Tensor {
            FUSER_PERF_SCOPE(
                "FusionDefinition.define_tensor (contiguity as bool)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");

            verifyShape(shape);
            Tensor out = self.defineTensor(shape.size());
            self.defineRecord(new TensorRecord(
                {self.recordingState(out())},
                shape,
                getContiguityVec(shape, stride_order, contiguity),
                dtype,
                is_cpu,
                stride_order));

            return out;
          },
          py::arg("shape"),
          py::arg("contiguity") = false,
          py::arg("dtype") = DataType::Float,
          py::arg("is_cpu") = false,
          py::arg("stride_order") = py::list(),
          py::return_value_policy::reference)
      .def(
          "define_tensor",
          [](FusionDefinition& self,
             const std::vector<int64_t>& sizes,
             const std::vector<int64_t>& strides,
             const PrimDataType dtype = DataType::Float,
             const bool static_sizes = false,
             const bool is_cpu = false) -> Tensor {
            FUSER_PERF_SCOPE("FusionDefinition.define_tensor (integration)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            NVF_CHECK(
                sizes.size() == strides.size(),
                "The number of sizes does not match the number of strides.",
                sizes.size(),
                strides.size());
            Tensor out = self.defineTensor(sizes.size());
            std::vector<std::optional<bool>> contiguity;
            std::vector<int64_t> stride_order;
            std::tie(contiguity, stride_order) =
                computeTensorDescriptor(sizes, strides);
            self.defineRecord(new TensorRecord(
                {self.recordingState(out())},
                getTensorViewBuilderSizes(sizes, static_sizes),
                contiguity,
                dtype,
                is_cpu,
                stride_order));
            return out;
          },
          py::arg("sizes"),
          py::arg("strides"),
          py::arg("dtype") = DataType::Float,
          py::arg("static_sizes") = false,
          py::arg("is_cpu") = false,
          py::return_value_policy::reference)
      .def(
          "define_scalar",
          [](FusionDefinition& self,
             PrimDataType dtype = DataType::Double) -> Scalar {
            FUSER_PERF_SCOPE("FusionDefinition.define_scalar (input_specific)");
            NVF_CHECK(
                !self.completed(),
                "Attempting to add to a completed definition!");
            Scalar out = self.defineScalar();
            self.defineRecord(new ScalarRecord(
                {self.recordingState(out())}, std::monostate{}, dtype));
            return out;
          },
          py::arg("dtype") = DataType::Double,
          py::return_value_policy::reference);
  fusion_def.def(
      "define_scalar",
      [](FusionDefinition& self,
         PolymorphicValue::VariantType value,
         std::optional<PrimDataType> dtype) -> Scalar {
        FUSER_PERF_SCOPE("FusionDefinition.define_scalar");
        Scalar out = self.defineScalar();
        self.defineRecord(
            new ScalarRecord({self.recordingState(out())}, value, dtype));
        return out;
      },
      py::arg("value"),
      py::arg("dtype") = std::nullopt,
      py::return_value_policy::reference);
  fusion_def.def(
      "define_constant",
      [](FusionDefinition& self,
         PolymorphicValue::VariantType value,
         std::optional<PrimDataType> dtype) -> Scalar {
        FUSER_PERF_SCOPE("FusionDefinition.define_contant");
        TORCH_WARN_ONCE(
            "Deprecating define_constant functions in favor of define_scalar "
            "for constants.");
        Scalar out = self.defineScalar();
        self.defineRecord(
            new ScalarRecord({self.recordingState(out())}, value, dtype));
        return out;
      },
      py::arg("value"),
      py::arg("dtype") = std::nullopt,
      py::return_value_policy::reference);

  // This is the input version of define_vector
  fusion_def.def(
      "define_vector",
      [](FusionDefinition& self, size_t size) -> Vector {
        std::vector<Scalar> args;
        args.reserve(size);
        for (size_t i = 0; i < size; ++i) {
          Scalar out = self.defineScalar();
          self.defineRecord(new ScalarRecord(
              {self.recordingState(out())}, std::monostate{}, DataType::Int));
          args.emplace_back(out);
        }
        return define_vector_base_fn(self, args);
      },
      py::arg("size"),
      py::return_value_policy::reference);
  // This is the constant version of define_vector when given a vector
  // of constant values.
  fusion_def.def(
      "define_vector",
      define_vector_explicit_fn<py::list>,
      py::arg("values"),
      py::arg("dtype") = DataType::Int,
      py::return_value_policy::reference);
  fusion_def.def(
      "define_vector",
      define_vector_explicit_fn<py::tuple>,
      py::arg("values"),
      py::arg("dtype") = DataType::Int,
      py::return_value_policy::reference);

  fusion_def.def(
      "getValTolerances",
      [](FusionDefinition& self, const py::iterable& input_iter) {
        KernelArgumentHolder args;
        for (py::handle obj : input_iter) {
          args.push(torch::jit::toIValue(obj, c10::AnyType::get()));
        }
        return self.getValTolerances(args);
      },
      py::return_value_policy::reference);

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

  nvf_ops.def(
      "triu",
      [](FusionDefinition::Operators& self,
         Tensor input,
         int64_t diagonal) -> Tensor {
        FUSER_PERF_SCOPE("Operators.triu");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(input.dims);

        auto diagonal_ = fd->defineScalar();
        fd->defineRecord(new ScalarRecord(
            {fd->recordingState(diagonal_())}, diagonal, DataType::Int, true));

        fd->defineRecord(new OpRecord<TensorView*, TensorView*, Val*>(
            {fd->recordingState(input()), fd->recordingState(diagonal_())},
            {fd->recordingState(output())},
            ("ops.triu"),
            serde::RecordType::Binary_TV_VAL,
            static_cast<TensorView* (*)(TensorView*, Val*)>(triu)));

        return output;
      },
      py::arg("input"),
      py::arg("diagonal") = 0,
      py::return_value_policy::reference,
      R"doc(
    Returns the upper triangular part of a 2+D tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    diagonal : int, optional
        The diagonal to consider. Default is 0.

    Returns
    -------
    Tensor
        The upper triangular part of the input tensor.

    >>> a = torch.randn(3, 3)
    >>> fd.ops.triu(a)
    )doc");

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
            "Operator stride_order expects `stride_order` argument to have the "
            "same length as input!");
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

#define NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL(op_str, op_name)               \
  tensor_class.def(                                                            \
      "__" op_str "__",                                                        \
      [](Tensor input) -> Tensor {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        NVF_CHECK(                                                             \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Tensor output = fd->defineTensor(input.dims);                          \
        fd->defineRecord(new OpRecord<TensorView*, TensorView*>(               \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Unary_TV,                                       \
            static_cast<TensorView* (*)(TensorView*)>(op_name)));              \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);                                     \
  scalar_class.def(                                                            \
      "__" op_str "__",                                                        \
      [](Scalar input) -> Scalar {                                             \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = input.fusion_definition;                        \
        NVF_CHECK(                                                             \
            !fd->completed(), "Attempting to add to a completed definition!"); \
        Scalar output = fd->defineScalar();                                    \
        fd->defineRecord(new OpRecord<Val*, Val*>(                             \
            {fd->recordingState(input())},                                     \
            {fd->recordingState(output())},                                    \
            ("ops." op_str),                                                   \
            serde::RecordType::Unary_VAL,                                      \
            static_cast<Val* (*)(Val*)>(op_name)));                            \
        return output;                                                         \
      },                                                                       \
      py::return_value_policy::reference);
  NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL("abs", abs)
  NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL("neg", neg)
#undef NVFUSER_PYTHON_BINDING_UNARY_OP_SPECIAL

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

#define NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(py_op, op_str, op_name)       \
  tensor_class.def(                                                            \
      py_op,                                                                   \
      [](Tensor arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
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
  tensor_class.def(                                                            \
      py_op,                                                                   \
      [](Tensor arg1, Scalar arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
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
  scalar_class.def(                                                            \
      py_op,                                                                   \
      [](Scalar arg1, Tensor arg2) -> Tensor {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg1.fusion_definition;                         \
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
  scalar_class.def(                                                            \
      py_op,                                                                   \
      [](Scalar arg1, Scalar arg2) -> Scalar {                                 \
        FUSER_PERF_SCOPE("Operators." op_str);                                 \
        FusionDefinition* fd = arg2.fusion_definition;                         \
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

  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__add__", "add", add)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__mul__", "mul", mul)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__pow__", "pow", pow)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__sub__", "sub", sub)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__mod__", "mod", mod)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__eq__", "eq", eq)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__ge__", "ge", ge)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__gt__", "gt", gt)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__le__", "le", le)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__lt__", "lt", lt)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__ne__", "ne", ne)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__and__", "bitwise_and", bitwise_and)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__or__", "bitwise_or", bitwise_or)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__xor__", "bitwise_xor", bitwise_xor)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__lshift__", "bitwise_left_shift", bitwise_left_shift)
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL(
      "__rshift__", "bitwise_right_shift", bitwise_right_shift)
  // In python, __truediv__ (/) always returns a float regardless of whether
  // the input arguments are float or integer. __truediv__ (/) corresponds with
  // pytorch torch.true_divide(a, b). The __div__ operator is deprecated in
  // python 3.
  //
  // In nvfuser, truediv function in csrc/ops/arith.h has the same semantics as
  // python's operator __truediv__ (/). The div function in csrc/ops/arith.h
  // truncates the result instead of promoting it to float. It has the same
  // semantics as the C++'s (/) operator. In pytorch,
  // torch.div(a, b, rounding_mode='trunc') corresponds C-style integer
  // division.
  //
  // Hence, in the python frontend, the __truediv__ (/) python operator maps to
  // trunc division.
  NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL("__truediv__", "div", div)
#undef NVFUSER_PYTHON_BINDING_BINARY_OP_SPECIAL

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
      "index_put_accumulate",
      [](FusionDefinition::Operators& self,
         Tensor acc,
         Tensor index,
         Tensor value) -> Tensor {
        FUSER_PERF_SCOPE("Operators.index_put_accumulate");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(acc.dims);
        fd->defineRecord(new IndexPutAccumulateOpRecord(
            {
                fd->recordingState(acc()),
                fd->recordingState(index()),
                fd->recordingState(value()),
            },
            {fd->recordingState(output())}));
        return output;
      },
      py::arg("acc"),
      py::arg("index"),
      py::arg("value"),
      py::return_value_policy::reference,
      R"doc(
        Accumulates values into a tensor at specified indices.

        This function performs a restricted version of `torch.index_put`.
        It adds the values from `value_tv` to the elements of `acc_tv` at the indices
        specified by `index_tv`.

        acc_tv: The tensor to accumulate into (in-place modification).
        index_tv: The tensor containing the indices.
        value_tv: The tensor containing the values to accumulate.

        Returns:
            An alias to the modified `acc_tv` tensor.

        Note:
            This is a restricted version and may not support all features of the
            full `torch.index_put(..., accumulate=true)` function.
    )doc");
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
      "scatter",
      [](FusionDefinition::Operators& self,
         Tensor arg1,
         Tensor index,
         Tensor src,
         int64_t dim) -> Tensor {
        FUSER_PERF_SCOPE("Operators.scatter");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        NVF_CHECK(
            arg1.dims == index.dims && arg1.dims == src.dims,
            "Tensor arguments have different dimensions ",
            arg1.dims,
            ", ",
            index.dims,
            " and ",
            src.dims);
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
        Tensor output = fd->defineTensor(num_dims);
        fd->defineRecord(new ScatterOpRecord(
            {
                fd->recordingState(arg1()),
                fd->recordingState(index()),
                fd->recordingState(src()),
            },
            {fd->recordingState(output())},
            dim));
        return output;
      },
      py::arg("arg1"),
      py::arg("index"),
      py::arg("src"),
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
        fd->defineRecord(new GatherOpRecord(
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
            "Operator permute expects `dims` argument to have the same length "
            "as input!");
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

  auto shape_def = [](Tensor arg) -> Vector {
    FUSER_PERF_SCOPE("Operators.shape");
    auto fd = arg.fusion_definition;
    NVF_CHECK(
        fd->ops.validUse(), "Attempting to add to a completed definition!");
    Vector output = fd->defineVector(arg.dims);
    fd->defineRecord(new ShapeOpRecord(
        {fd->recordingState(arg())}, {fd->recordingState(output())}));
    return output;
  };

  tensor_class.def(
      "shape",
      [&shape_def](Tensor arg) -> Vector { return shape_def(arg); },
      py::return_value_policy::reference);
  nvf_ops.def(
      "shape",
      [&shape_def](FusionDefinition::Operators& self, Tensor arg) -> Vector {
        return shape_def(arg);
      },
      py::arg("arg"),
      py::return_value_policy::reference);

  auto size_def = [](Tensor arg, int64_t dim) -> Scalar {
    FUSER_PERF_SCOPE("Operators.size");
    auto fd = arg.fusion_definition;
    NVF_CHECK(
        fd->ops.validUse(), "Attempting to add to a completed definition!");
    Scalar output = fd->defineScalar();
    fd->defineRecord(new SizeOpRecord(
        {fd->recordingState(arg())}, {fd->recordingState(output())}, dim));
    return output;
  };

  tensor_class.def(
      "size",
      [&size_def](Tensor arg, int64_t dim) -> Scalar {
        return size_def(arg, dim);
      },
      py::return_value_policy::reference);
  nvf_ops.def(
      "size",
      [&size_def](FusionDefinition::Operators& self, Tensor arg, int64_t dim)
          -> Scalar { return size_def(arg, dim); },
      py::arg("arg"),
      py::arg("dim"),
      py::return_value_policy::reference);

  auto at_def = [](Vector arg, int64_t index) -> Scalar {
    FUSER_PERF_SCOPE("Operators.at");
    auto fd = arg.fusion_definition;
    NVF_CHECK(
        fd->ops.validUse(), "Attempting to add to a completed definition!");
    Scalar output = fd->defineScalar();
    fd->defineRecord(new AtOpRecord(
        {fd->recordingState(arg())}, {fd->recordingState(output())}, index));
    return output;
  };

  vector_class.def(
      "at",
      [&at_def](Vector arg, int64_t index) -> Scalar {
        return at_def(arg, index);
      },
      py::return_value_policy::reference);
  vector_class.def(
      "__getitem__",
      [&at_def](Vector arg, int64_t index) -> Scalar {
        return at_def(arg, index);
      },
      py::return_value_policy::reference);
  nvf_ops.def(
      "at",
      [&at_def](FusionDefinition::Operators& self, Vector arg, int64_t index)
          -> Scalar { return at_def(arg, index); },
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
        for (const auto idx : arange(arg.dims)) {
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
#if NVF_TORCH_VERSION_NO_LESS(2, 7, 0)
        int64_t philox_ndims = 1;
#else
        int64_t philox_ndims = 0;
#endif
        Tensor philox_seed = fd->defineTensor(philox_ndims);
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

  nvf_ops.def(
      "embedding_fwd",
      [](FusionDefinition::Operators& self,
         Tensor input,
         Tensor weight,
         std::optional<Scalar> padding_idx,
         std::optional<Scalar> max_norm,
         std::optional<Scalar> norm_type,
         std::optional<Scalar> scale_grad_by_freq,
         std::optional<Scalar> sparse) -> decltype(auto) {
        FUSER_PERF_SCOPE("Operators.embedding_fwd");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        size_t ndims = input.dims + 1;
        Tensor output = fd->defineTensor(/*dims=*/ndims);

        auto padding_idx_state = padding_idx.has_value()
            ? fd->recordingState(padding_idx.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto max_norm_state = max_norm.has_value()
            ? fd->recordingState(max_norm.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto norm_type_state = norm_type.has_value()
            ? fd->recordingState(norm_type.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto scale_grad_by_freq_state = scale_grad_by_freq.has_value()
            ? fd->recordingState(scale_grad_by_freq.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);
        auto sparse_state = sparse.has_value()
            ? fd->recordingState(sparse.value()())
            : State(/*_index=*/0, /*_stype=*/serde::StateType::None);

        fd->defineRecord(new EmbeddingFwdOpRecord(
            {fd->recordingState(input()),
             fd->recordingState(weight()),
             padding_idx_state,
             max_norm_state,
             norm_type_state,
             scale_grad_by_freq_state,
             sparse_state},
            {fd->recordingState(output())}));
        return output;
      },
      py::arg("input"),
      py::arg("weight"),
      py::arg("padding_idx").none(true) = py::none(),
      py::arg("max_norm").none(true) = py::none(),
      py::arg("norm_type").none(true) = py::none(),
      py::arg("scale_grad_by_freq").none(true) = py::none(),
      py::arg("sparse").none(true) = py::none(),
      py::return_value_policy::reference);

  nvf_ops.def(
      "argsort",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         int64_t dim,
         bool descending,
         bool stable) -> Tensor {
        FUSER_PERF_SCOPE("Operators.argsort");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;
        Tensor output = fd->defineTensor(arg.dims);
        fd->defineRecord(new ArgsortOpRecord(
            {fd->recordingState(arg())},
            {fd->recordingState(output())},
            dim,
            descending,
            stable));
        return output;
      },
      py::arg("arg"),
      py::arg("dim"),
      py::arg("descending") = false,
      py::arg("stable") = false,
      py::return_value_policy::reference);

  nvf_ops.def(
      "grouped_mm",
      [](FusionDefinition::Operators& self,
         Tensor mat1,
         Tensor mat2,
         Tensor offsets) -> Tensor {
        FUSER_PERF_SCOPE("Operators.grouped_mm");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;

        // Calculate output dimensions based on mat1 & mat2 rank
        size_t output_dims = mat1.dims == 2 && mat2.dims == 2 ? 3 : 2;
        Tensor output = fd->defineTensor(output_dims);
        fd->defineRecord(
            new OpRecord<TensorView*, TensorView*, TensorView*, TensorView*>(
                {fd->recordingState(mat1()),
                 fd->recordingState(mat2()),
                 fd->recordingState(offsets())},
                {fd->recordingState(output())},
                ("ops.grouped_mm"),
                serde::RecordType::Ternary_TV,
                static_cast<
                    TensorView* (*)(TensorView*, TensorView*, TensorView*)>(
                    [](TensorView* mat1,
                       TensorView* mat2,
                       TensorView* offsets) {
                      return grouped_mm(mat1, mat2, offsets);
                    })));
        return output;
      },
      R"(
      Grouped matrix multiplication.

      Performs matrix multiplication on grouped sets of matrices using offsets
      to define variable-sized groups.

      Args:
          mat1 (Tensor): First set of matrices
          mat2 (Tensor): Second set of matrices
          offsets (Tensor): Offsets tensor defining group boundaries

      Returns:
          Tensor: Result of grouped matrix multiplication
      )",
      py::arg("mat1"),
      py::arg("mat2"),
      py::arg("offsets"),
      py::return_value_policy::reference);

  nvf_ops.def(
      "grouped_mm",
      [](FusionDefinition::Operators& self,
         Tensor mat1,
         Tensor mat2,
         Tensor offsets,
         Tensor scale1,
         Tensor scale2,
         std::optional<PrimDataType> dtype) -> Tensor {
        FUSER_PERF_SCOPE("Operators.grouped_mm");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;

        // Calculate output dimensions based on mat1 & mat2 rank
        size_t output_dims = mat1.dims == 2 && mat2.dims == 2 ? 3 : 2;
        Tensor output = fd->defineTensor(output_dims);

        fd->defineRecord(new ScaledGroupedMmaOpRecord(
            {fd->recordingState(mat1()),
             fd->recordingState(mat2()),
             fd->recordingState(offsets()),
             fd->recordingState(scale1()),
             fd->recordingState(scale2())},
            {fd->recordingState(output())},
            dtype));
        return output;
      },
      R"(
      Scaled Grouped matrix multiplication.

      Performs matrix multiplication on grouped sets of matrices using offsets
      to define variable-sized groups.

      Args:
          mat1 (Tensor): First set of matrices
          mat2 (Tensor): Second set of matrices
          offsets (Tensor): Offsets tensor defining group boundaries
          scale1 (Tensor): Scale tensor for mat1
          scale2 (Tensor): Scale tensor for mat2
          dtype (ScalarType): Output tensor type [optional]

      Returns:
          Tensor: Result of grouped matrix multiplication
      )",
      py::arg("mat1"),
      py::arg("mat2"),
      py::arg("offsets"),
      py::arg("scale1"),
      py::arg("scale2"),
      py::arg("dtype") = std::nullopt,
      py::return_value_policy::reference);

  nvf_ops.def(
      "topk",
      [](FusionDefinition::Operators& self,
         Tensor arg,
         Scalar k,
         int64_t dim,
         bool largest,
         bool sorted) -> py::tuple {
        FUSER_PERF_SCOPE("Operators.topk");
        NVF_CHECK(
            self.validUse(), "Attempting to add to a completed definition!");
        FusionDefinition* fd = self.fusion_definition;

        Tensor values = fd->defineTensor(arg.dims);
        Tensor indices = fd->defineTensor(arg.dims);

        fd->defineRecord(new TopKOpRecord(
            {fd->recordingState(arg()), fd->recordingState(k())},
            {fd->recordingState(values()), fd->recordingState(indices())},
            dim,
            largest,
            sorted));

        return py::make_tuple(values, indices);
      },
      R"(
      Find the k largest or smallest elements along a dimension.

      Args:
          arg (Tensor): Input tensor
          k (Scalar): Number of elements to return
          dim (int, optional): Dimension along which to find top-k. Defaults to -1.
          largest (bool, optional): If True, return largest elements. Defaults to True.
          sorted (bool, optional): If True, return elements in sorted order. Defaults to False.

      Returns:
          tuple[Tensor, Tensor]: A tuple of (values, indices) where values contains
                                the k largest/smallest elements and indices contains
                                their positions in the original tensor.
      )",
      py::arg("arg"),
      py::arg("k"),
      py::arg("dim") = -1,
      py::arg("largest") = true,
      py::arg("sorted") = false,
      py::return_value_policy::reference);

  bindSchedule(fusion_def);

  bindMultidevice(nvfuser);
}

void cleanup() {
  Communicator::getInstance().cleanup();
}

} // namespace nvfuser::python_frontend
