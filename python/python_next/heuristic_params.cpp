// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>

#include <scheduler/matmul_heuristic.h>
#include <scheduler/pointwise_heuristic.h>
#include <scheduler/reduction_heuristic.h>

namespace nvfuser::python {

void bindHeuristicParams(py::module& nvfuser) {
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

} // namespace nvfuser::python
