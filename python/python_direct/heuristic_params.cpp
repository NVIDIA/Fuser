// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <bindings.h>

#include <scheduler/matmul_heuristic.h>
#include <scheduler/pointwise_heuristic.h>
#include <scheduler/reduction_heuristic.h>

// Tell nanobind to not use automatic type caster for MmaMacro enum
// so we can bind it as a class with properties
NB_MAKE_OPAQUE(nvfuser::MmaMacro);

namespace nvfuser::python {

void bindHeuristicParams(nb::module_& nvfuser) {
  nb::class_<LaunchParams>(nvfuser, "LaunchParams")
      .def(nb::init<>())
      .def(nb::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>())
      .def("__repr__", [](const LaunchParams& self) { return self.toString(); })
      .def_prop_rw(
          "bdimx",
          [](LaunchParams& self) { return self.bdimx(); },
          [](LaunchParams& self, int64_t val) {
            self.bindUnsafe(val, ParallelType::TIDx);
          },
          R"(The number of threads in the x dimension of the block.)")
      .def_prop_rw(
          "bdimy",
          [](LaunchParams& self) { return self.bdimy(); },
          [](LaunchParams& self, int64_t val) {
            self.bindUnsafe(val, ParallelType::TIDy);
          },
          R"(The number of threads in the y dimension of the block.)")
      .def_prop_rw(
          "bdimz",
          [](LaunchParams& self) { return self.bdimz(); },
          [](LaunchParams& self, int64_t val) {
            self.bindUnsafe(val, ParallelType::TIDz);
          },
          R"(The number of threads in the z dimension of the block.)")
      .def_prop_rw(
          "gdimx",
          [](LaunchParams& self) { return self.gdimx(); },
          [](LaunchParams& self, int64_t val) {
            self.bindUnsafe(val, ParallelType::BIDx);
          },
          R"(
          The number of blocks in the x dimension of the grid.
      )")
      .def_prop_rw(
          "gdimy",
          [](LaunchParams& self) { return self.gdimy(); },
          [](LaunchParams& self, int64_t val) {
            self.bindUnsafe(val, ParallelType::BIDy);
          },
          R"(The number of blocks in the y dimension of the grid.)")
      .def_prop_rw(
          "gdimz",
          [](LaunchParams& self) { return self.gdimz(); },
          [](LaunchParams& self, int64_t val) {
            self.bindUnsafe(val, ParallelType::BIDz);
          },
          R"(
          The number of blocks in the z dimension of the grid.
      )");

  nb::class_<CompileParams>(nvfuser, "CompileParams")
      .def(
          "__init__",
          [](CompileParams* self,
             std::optional<PrimDataType> index_type,
             int64_t maxrregcount,
             bool enable_magic_zero,
             bool enable_ptxas_verbose,
             std::optional<c10::Device> device,
             std::vector<std::string> include_paths) {
            new (self) CompileParams(
                index_type,
                maxrregcount,
                enable_magic_zero,
                enable_ptxas_verbose,
                device,
                include_paths);
          },
          nb::kw_only(),
          nb::arg("index_type") = nb::none(),
          nb::arg("maxrregcount") = 255,
          nb::arg("enable_magic_zero") = true,
          nb::arg("enable_ptxas_verbose") = false,
          nb::arg("device") = nb::none(),
          nb::arg("include_paths") = nb::list(),
          R"(
              Parameters
              ----------
              index_type : PrimDataType, optional
                The index type to use for the kernel.
              maxrregcount : int, optional
                The maximum number of registers to use for the kernel.
              enable_magic_zero : bool, optional
                Whether to enable magic zero for the kernel.
              enable_ptxas_verbose : bool, optional
                Whether to enable verbose output for the kernel.
              device : c10::Device, optional
                The device to use for the kernel.
              include_paths : list of str, optional
                The additional include paths to use for the kernel.

              Returns
              -------
              CompileParams
                The parameters used to compile a kernel with NVRTC.
            )")
      .def(
          "__repr__", [](const CompileParams& self) { return self.toString(); })
      .def_rw("index_type", &CompileParams::index_type, R"(
                The index type to use for the kernel.
              )")
      .def_rw("maxrregcount", &CompileParams::maxrregcount, R"(
                The maximum number of registers to use for the kernel.
              )")
      .def_rw("enable_magic_zero", &CompileParams::enable_magic_zero, R"(
                Whether to enable magic zero for the kernel.
              )")
      .def_rw("enable_ptxas_verbose", &CompileParams::enable_ptxas_verbose, R"(
                Whether to enable verbose output for the kernel.
              )")
      .def_rw("device", &CompileParams::device, R"(
                The device to use for the kernel.
              )")
      .def_rw("include_paths", &CompileParams::include_paths, R"(
                The additional include paths to use for the kernel.
              )");

  nb::class_<HeuristicParams>(nvfuser, "HeuristicParams")
      .def(
          "__repr__",
          [](const HeuristicParams& self) { return self.toString(); })
      .def("__eq__", &HeuristicParams::sameAs, R"(
                Whether the heuristic parameters are the same.
              )")
      .def_rw("lparams", &HeuristicParams::lparams, R"(
                The launch parameters for the kernel.
              )")
      .def_rw("cparams", &HeuristicParams::cparams, R"(
                The compile parameters for the kernel.
              )")
      .def_ro("scheduler_type", &HeuristicParams::scheduler_type, R"(
                The type of scheduler that generated these parameters.
              )")
      .def("hash", &HeuristicParams::hash, R"(
                The hash of the heuristic parameters.
              )");

  nb::class_<PointwiseParams, HeuristicParams>(nvfuser, "PointwiseParams")
      .def(nb::init())
      .def(
          "__repr__",
          [](const PointwiseParams& self) { return self.toString(); })
      .def_rw("break_point", &PointwiseParams::break_point, R"(
                Split point from left to right of domain for 2D scheduling.
              )")
      .def_rw("split_block", &PointwiseParams::split_block, R"(
                Split block across left and right dimension.
              )")
      .def_rw("split_grid_y_dim", &PointwiseParams::split_grid_y_dim, R"(
                Split grid y dimension if too large.
              )")
      .def_rw("flip_grid_binding", &PointwiseParams::flip_grid_binding, R"(
                Bind BIDy on innermost dimension for broadcast performance.
              )")
      .def_rw(
          "vectorization_factor", &PointwiseParams::vectorization_factor, R"(
                Vectorization factor.
              )")
      .def_rw("unroll_factor_inner", &PointwiseParams::unroll_factor_inner, R"(
                Unroll factor for inner dimension.
              )")
      .def_rw("unroll_factor_outer", &PointwiseParams::unroll_factor_outer, R"(
                Unroll factor for outer dimension to reuse loaded data.
              )");

  nb::class_<ReductionParams, HeuristicParams>(nvfuser, "ReductionParams")
      .def(nb::init())
      .def(
          "__repr__",
          [](const ReductionParams& self) { return self.toString(); })
      .def_rw("fastest_dim", &ReductionParams::fastest_dim, R"(
                Reduce on innermost dimension.
              )")
      .def_rw("persistent_kernel", &ReductionParams::persistent_kernel, R"(
                Store input in shared memory or registers to reduce global memory reads.
              )")
      .def_rw(
          "project_persistent_buffers",
          &ReductionParams::project_persistent_buffers,
          R"(Project persistent buffers back to inputs.
              )")
      .def_rw("schedule_3D", &ReductionParams::schedule_3d, R"(
                Use 3D scheduling for patterns like [reduction, iteration, reduction].
              )")
      .def_rw("flip_grid", &ReductionParams::flip_grid, R"(
                Swap gdimx and gdimy bindings for outer reductions.)")
      .def_rw(
          "cross_block_inner_reduction",
          &ReductionParams::cross_block_inner_reduction,
          R"(Reduce across the block for inner reduction.)")
      .def_rw(
          "cross_grid_inner_reduction",
          &ReductionParams::cross_grid_inner_reduction,
          R"(Reduce across the grid for inner reduction.)")
      .def_rw(
          "unroll_factor_inner_reduction",
          &ReductionParams::unroll_factor_inner_reduction,
          R"(Unrolling/vectorization factor for inner reduction dimension.)")
      .def_rw(
          "unroll_factor_top_of_vectorization",
          &ReductionParams::unroll_factor_top_of_vectorization,
          R"(Extra unroll on top of vectorization.)")
      .def_rw(
          "vectorize_inner_reduction",
          &ReductionParams::vectorize_inner_reduction,
          R"(Vectorize instead of unroll for inner reduction.)")
      .def_rw(
          "split_grid_dim_inner_reduction",
          &ReductionParams::split_grid_dim_inner_reduction,
          R"(Split grid dimension for inner reduction if too large.)")
      .def_rw(
          "pad_inner_reduction_to_warp",
          &ReductionParams::pad_inner_reduction_to_warp,
          R"(Pad inner dimension to nearest warp.)")
      .def_rw(
          "batches_per_block_inner_reduction",
          &ReductionParams::batches_per_block_inner_reduction,
          R"(Register persistent buffer size in inner dimension.)")
      .def_rw(
          "block_dim_inner_reduction",
          &ReductionParams::block_dim_inner_reduction,
          R"(Block parallel dimension for inner reduction.)")
      .def_rw(
          "grid_dim_inner_reduction",
          &ReductionParams::grid_dim_inner_reduction,
          R"(Grid parallel dimension for inner reduction.)")
      .def_rw(
          "multiple_reds_per_blk",
          &ReductionParams::multiple_reds_per_blk,
          R"(Perform multiple reductions per block.)")
      .def_rw(
          "unroll_factor_iter_dom",
          &ReductionParams::unroll_factor_iter_dom,
          R"(Unrolling/vectorization factor for iteration dimension.)")
      .def_rw(
          "vectorize_iter_dom",
          &ReductionParams::vectorize_iter_dom,
          R"(Vectorize instead of unroll for iteration domain.)")
      .def_rw(
          "split_grid_dim_iter_dom_inner",
          &ReductionParams::split_grid_dim_iter_dom_inner,
          R"(Inner split grid dimension for iteration axis.)")
      .def_rw(
          "split_grid_dim_iter_dom_outer",
          &ReductionParams::split_grid_dim_iter_dom_outer,
          R"(Outer split grid dimension for iteration axis.)")
      .def_rw(
          "block_dim_iter_dom",
          &ReductionParams::block_dim_iter_dom,
          R"(Block parallel dimension for iteration domain.)")
      .def_rw(
          "grid_dim_iter_dom",
          &ReductionParams::grid_dim_iter_dom,
          R"(Grid parallel dimension for iteration domain.)")
      .def_rw(
          "cross_block_outer_reduction",
          &ReductionParams::cross_block_outer_reduction,
          R"(Reduce across the block for outer reduction.)")
      .def_rw(
          "cross_grid_outer_reduction",
          &ReductionParams::cross_grid_outer_reduction,
          R"(Reduce across the grid for outer reduction.)")
      .def_rw(
          "batches_per_block_outer_reduction",
          &ReductionParams::batches_per_block_outer_reduction,
          R"(Register persistent buffer size in outer dimension.)")
      .def_rw(
          "unroll_factor_outer_reduction",
          &ReductionParams::unroll_factor_outer_reduction,
          R"(Unrolling/vectorization factor for outer reduction.)")
      .def_rw(
          "block_dim_outer_reduction",
          &ReductionParams::block_dim_outer_reduction,
          R"(Block parallel dimension for outer reduction.)")
      .def_rw(
          "grid_dim_outer_reduction",
          &ReductionParams::grid_dim_outer_reduction,
          R"(Grid parallel dimension for outer reduction.)")
      .def_rw(
          "compute_persistent_buffer_with_first_consumer",
          &ReductionParams::compute_persistent_buffer_with_first_consumer,
          R"(Use computeWith to persistent buffers.)")
      .def_rw(
          "static_bdimx",
          &ReductionParams::static_bdimx,
          R"(Static block dimension X.)")
      .def_rw(
          "static_bdimy",
          &ReductionParams::static_bdimy,
          R"(Static block dimension Y.)")
      .def_rw(
          "combined_inner_outer",
          &ReductionParams::combined_inner_outer,
          R"(Combined inner and outer reduction.)")
      .def_rw(
          "tidx_for_outer_reduction",
          &ReductionParams::tidx_for_outer_reduction,
          R"(Use TIDx for outer reduction axis.)")
      .def_rw(
          "pad_outer_reduction_to_warp",
          &ReductionParams::pad_outer_reduction_to_warp,
          R"(Pad outer reduction to warp.)")
      .def_rw(
          "combined_split_grid_inner_dim",
          &ReductionParams::combined_split_grid_inner_dim,
          R"(Further split inner dimension by grid in combined scheduler.)")
      .def_rw(
          "vectorization_factor_outer",
          &ReductionParams::vectorization_factor_outer,
          R"(Vectorization factor for outer reduction partial result.)")
      .def_rw(
          "vectorization_factor_tmp_gmem_write",
          &ReductionParams::vectorization_factor_tmp_gmem_write,
          R"(Vectorization factor for temporary global memory write.)")
      .def_rw(
          "block_dim_inner_reduction_extra",
          &ReductionParams::block_dim_inner_reduction_extra,
          R"(Additional block parallel dimension for inner reduction.)");

  // Supporting types for MatmulParams
  nb::class_<GemmTile>(nvfuser, "GemmTile")
      .def(nb::init<int64_t, int64_t, int64_t>())
      .def_rw("m", &GemmTile::m, R"(M dimension of the GEMM tile.)")
      .def_rw("n", &GemmTile::n, R"(N dimension of the GEMM tile.)")
      .def_rw("k", &GemmTile::k, R"(K dimension of the GEMM tile.)")
      .def("__repr__", [](const GemmTile& self) {
        return "GemmTile(m=" + std::to_string(self.m) +
            ", n=" + std::to_string(self.n) + ", k=" + std::to_string(self.k) +
            ")";
      });

  nb::class_<MatMulTileOptions>(nvfuser, "MatMulTileOptions")
      .def(nb::init<GemmTile, GemmTile>())
      .def_rw(
          "cta_tile", &MatMulTileOptions::cta_tile, R"(CTA tile dimensions.)")
      .def_rw(
          "warp_tile",
          &MatMulTileOptions::warp_tile,
          R"(Warp tile dimensions.)")
      .def("__repr__", [](const MatMulTileOptions& self) {
        return nvfuser::toString(self);
      });

  nb::class_<MatmulParams::CircularBufferOptions>(
      nvfuser, "CircularBufferOptions")
      .def(nb::init<bool, bool, int, int>())
      .def_rw(
          "circular_buffer_smem_read",
          &MatmulParams::CircularBufferOptions::circular_buffer_smem_read,
          R"(Enable circular buffering for shared memory reads.)")
      .def_rw(
          "circular_buffer_smem_write",
          &MatmulParams::CircularBufferOptions::circular_buffer_smem_write,
          R"(Enable circular buffering for shared memory writes.)")
      .def_rw(
          "smem_circular_buffer_stage",
          &MatmulParams::CircularBufferOptions::smem_circular_buffer_stage,
          R"(Number of circular buffering stages.)")
      .def_rw(
          "smem_circular_buffer_prefetch_gap",
          &MatmulParams::CircularBufferOptions::
              smem_circular_buffer_prefetch_gap,
          R"(Circular buffer prefetch gap.)")
      .def("__repr__", [](const MatmulParams::CircularBufferOptions& self) {
        return self.toString();
      });

  nb::class_<MatmulParams::SupportedVectorization>(
      nvfuser, "SupportedVectorization")
      .def(nb::init<int64_t, int64_t, int64_t>())
      .def_rw("a", &MatmulParams::SupportedVectorization::a, R"(
                Vectorization factor for operand A.
              )")
      .def_rw("b", &MatmulParams::SupportedVectorization::b, R"(
                Vectorization factor for operand B.
              )")
      .def_rw("epilogue", &MatmulParams::SupportedVectorization::epilogue, R"(
                Vectorization factor for epilogue.
              )")
      .def("__repr__", [](const MatmulParams::SupportedVectorization& self) {
        return self.toString();
      });

  nb::enum_<MatmulParams::TileRasterizationOrder>(
      nvfuser, "MatmulTileRasterizationOrder")
      .value("column_major", MatmulParams::TileRasterizationOrder::ColumnMajor)
      .value("row_major", MatmulParams::TileRasterizationOrder::RowMajor);

  nb::class_<MatmulParams::ClusterDims>(nvfuser, "ClusterDims")
      .def(nb::init<int64_t, int64_t>())
      .def_rw("m", &MatmulParams::ClusterDims::m, R"(
                M dimension of the cluster.
              )")
      .def_rw("n", &MatmulParams::ClusterDims::n, R"(
                N dimension of the cluster.
              )")
      .def("__repr__", [](const MatmulParams::ClusterDims& self) {
        return self.toString();
      });

  nb::enum_<MmaMacroEncode::Arch>(nvfuser, "MmaMacroArch")
      .value("no_mma", MmaMacroEncode::Arch::NoMma)
      .value("volta", MmaMacroEncode::Arch::Volta)
      .value("turing", MmaMacroEncode::Arch::Turing)
      .value("ampere", MmaMacroEncode::Arch::Ampere)
      .value("hopper", MmaMacroEncode::Arch::Hopper)
      .value("blackwell_1cta", MmaMacroEncode::Arch::Blackwell1CTA)
      .value("blackwell_2cta", MmaMacroEncode::Arch::Blackwell2CTA);

  nb::class_<MmaMacroEncode>(nvfuser, "MmaMacroEncode")
      .def(nb::init<MmaMacroEncode::Arch, uint16_t, uint16_t, uint16_t>())
      .def("mma_macro", &MmaMacroEncode::operator MmaMacro)
      .def_rw("arch", &MmaMacroEncode::arch, R"(
                GPU architecture for MMA instruction.
              )")
      .def_rw("m", &MmaMacroEncode::m, R"(
                M dimension of MMA instruction.
              )")
      .def_rw("n", &MmaMacroEncode::n, R"(
                N dimension of MMA instruction.
              )")
      .def_rw("k", &MmaMacroEncode::k, R"(
                K dimension of MMA instruction.
              )");

  nb::class_<MmaMacro>(nvfuser, "MmaMacro")
      .def_prop_rw(
          "arch",
          [](const MmaMacro& self) { return MmaMacroEncode(self).arch; },
          [](MmaMacro& self, MmaMacroEncode::Arch x) {
            auto enc = MmaMacroEncode(self);
            enc.arch = x;
            self = enc;
          },
          R"(GPU architecture for MMA instruction.)")
      .def_prop_rw(
          "m",
          [](const MmaMacro& self) { return MmaMacroEncode(self).m; },
          [](MmaMacro& self, uint16_t x) {
            auto enc = MmaMacroEncode(self);
            enc.m = x;
            self = enc;
          },
          R"(M dimension of MMA instruction.)")
      .def_prop_rw(
          "n",
          [](const MmaMacro& self) { return MmaMacroEncode(self).n; },
          [](MmaMacro& self, uint16_t x) {
            auto enc = MmaMacroEncode(self);
            enc.n = x;
            self = enc;
          },
          R"(N dimension of MMA instruction.)")
      .def_prop_rw(
          "k",
          [](const MmaMacro& self) { return MmaMacroEncode(self).k; },
          [](MmaMacro& self, uint16_t x) {
            auto enc = MmaMacroEncode(self);
            enc.k = x;
            self = enc;
          },
          R"(K dimension of MMA instruction.)")
      .def("__repr__", [](const MmaMacro& self) {
        return nvfuser::toString(self);
      });

  nb::class_<MatmulParams, HeuristicParams>(nvfuser, "MatmulParams")
      .def(nb::init())
      .def("__repr__", [](const MatmulParams& self) { return self.toString(); })
      .def_rw("tile_sizes", &MatmulParams::tile_sizes, R"(
                Tiling hierarchy on block and warp levels.
              )")
      .def_rw(
          "circular_buffer_options", &MatmulParams::circular_buffer_options, R"(
                Circular buffering configuration.
              )")
      .def_rw("supported_vec_size", &MatmulParams::supported_vec_size, R"(
                Maximum vectorization supported by inputs and outputs.
              )")
      .def_rw(
          "async_gmem_load_operands",
          &MatmulParams::async_gmem_load_operands,
          R"(
                Use cp.async to load operands (Ampere+).
              )")
      .def_rw("grid_traversal_factor", &MatmulParams::grid_traversal_factor, R"(
                Grid swizzle factor to increase L2 hit rate.
              )")
      .def_rw("use_smem_epilogue", &MatmulParams::use_smem_epilogue, R"(
                Unswizzle MMA results in shared memory for coalesced writes.
              )")
      .def_rw("use_ldst_matrix", &MatmulParams::use_ldst_matrix, R"(
                Use stmatrix/ldmatrix instructions in epilogue.
              )")
      .def_rw(
          "promote_prologue_smem_reuse",
          &MatmulParams::promote_prologue_smem_reuse,
          R"(
                Promote reuse of prologue shared memory.
              )")
      .def_rw("splitk_factor", &MatmulParams::splitk_factor, R"(
                Single-kernel split-K factor for parallelizing K dimension.
              )")
      .def_rw("tiling_strategy", &MatmulParams::tiling_strategy, R"(
                Strategy for mapping output tiles to CTAs.
              )")
      .def_rw("buffering_loop_level", &MatmulParams::buffering_loop_level, R"(
                Loop level for circular buffering (CTA tiles vs warp tiles).
              )")
      .def_rw(
          "circular_buffering_strategy",
          &MatmulParams::circular_buffering_strategy,
          R"(Circular buffering strategy (pipelined vs warp specialized).)")
      .def_rw("cta_order", &MatmulParams::cta_order, R"(
                CTA rasterization order (row major vs column major).
              )")
      .def_rw("cluster_dims", &MatmulParams::cluster_dims, R"(
                CGA dimensions for Hopper+ devices.
              )")
      .def_rw("mma_macro", &MatmulParams::mma_macro, R"(
                Type of MMA instruction to use in generated kernel.
              )");
}

} // namespace nvfuser::python
