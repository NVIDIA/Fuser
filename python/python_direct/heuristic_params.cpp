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
  py::class_<LaunchParams> launch_parameters(
      nvfuser, "LaunchParams", py::module_local());
  launch_parameters.def(py::init<>());
  launch_parameters.def(
      py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>());
  launch_parameters.def(
      "__repr__", [](const LaunchParams& self) { return self.toString(); });
  launch_parameters.def_property(
      "bdimx",
      [](LaunchParams& self) { return self.bdimx(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDx);
      },
      R"(
          The number of threads in the x dimension of the block.
      )");
  launch_parameters.def_property(
      "bdimy",
      [](LaunchParams& self) { return self.bdimy(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDy);
      },
      R"(
          The number of threads in the y dimension of the block.
      )");
  launch_parameters.def_property(
      "bdimz",
      [](LaunchParams& self) { return self.bdimz(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDz);
      },
      R"(
          The number of threads in the z dimension of the block.
      )");
  launch_parameters.def_property(
      "gdimx",
      [](LaunchParams& self) { return self.gdimx(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDx);
      },
      R"(
          The number of blocks in the x dimension of the grid.
      )");
  launch_parameters.def_property(
      "gdimy",
      [](LaunchParams& self) { return self.gdimy(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDy);
      },
      R"(
          The number of blocks in the y dimension of the grid.
      )");
  launch_parameters.def_property(
      "gdimz",
      [](LaunchParams& self) { return self.gdimz(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDz);
      },
      R"(
          The number of blocks in the z dimension of the grid.
      )");

  py::class_<CompileParams> compile_parameters(
      nvfuser, "CompileParams", py::module_local());
  compile_parameters.def(
      py::init([](std::optional<PrimDataType> index_type,
                  int64_t maxrregcount,
                  bool enable_magic_zero,
                  bool enable_ptxas_verbose,
                  std::optional<c10::Device> device,
                  std::vector<std::string> include_paths) {
        return CompileParams(
            index_type,
            maxrregcount,
            enable_magic_zero,
            enable_ptxas_verbose,
            device,
            include_paths);
      }),
      py::kw_only(),
      py::arg("index_type") = py::none(),
      py::arg("maxrregcount") = 255,
      py::arg("enable_magic_zero") = true,
      py::arg("enable_ptxas_verbose") = false,
      py::arg("device") = py::none(),
      py::arg("include_paths") = py::list(),
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
            )");
  compile_parameters.def(
      "__repr__", [](const CompileParams& self) { return self.toString(); });
  compile_parameters.def_readwrite("index_type", &CompileParams::index_type, R"(
                The index type to use for the kernel.
              )");
  compile_parameters.def_readwrite(
      "maxrregcount", &CompileParams::maxrregcount, R"(
                The maximum number of registers to use for the kernel.
              )");
  compile_parameters.def_readwrite(
      "enable_magic_zero", &CompileParams::enable_magic_zero, R"(
                Whether to enable magic zero for the kernel.
              )");
  compile_parameters.def_readwrite(
      "enable_ptxas_verbose", &CompileParams::enable_ptxas_verbose, R"(
                Whether to enable verbose output for the kernel.
              )");
  compile_parameters.def_readwrite("device", &CompileParams::device, R"(
                The device to use for the kernel.
              )");
  compile_parameters.def_readwrite(
      "include_paths", &CompileParams::include_paths, R"(
                The additional include paths to use for the kernel.
              )");
  compile_parameters.def_readwrite("device", &CompileParams::device, R"(
                The device to use for the kernel.
              )");
  compile_parameters.def_readwrite(
      "include_paths", &CompileParams::include_paths, R"(
                The additional include paths to use for the kernel.
              )");

  py::class_<HeuristicParams> heuristic_parameters(
      nvfuser, "HeuristicParams", py::module_local());
  heuristic_parameters.def(
      "__repr__", [](const HeuristicParams& self) { return self.toString(); });
  heuristic_parameters.def("__eq__", &HeuristicParams::sameAs, R"(
                Whether the heuristic parameters are the same.
              )");
  heuristic_parameters.def_readwrite("lparams", &HeuristicParams::lparams, R"(
                The launch parameters for the kernel.
              )");
  heuristic_parameters.def_readwrite("cparams", &HeuristicParams::cparams, R"(
                The compile parameters for the kernel.
              )");
  heuristic_parameters.def_readonly(
      "scheduler_type", &HeuristicParams::scheduler_type, R"(
                The type of scheduler that generated these parameters.
              )");
  heuristic_parameters.def("hash", &HeuristicParams::hash, R"(
                The hash of the heuristic parameters.
              )");

  py::class_<PointwiseParams, HeuristicParams> pointwise(
      nvfuser, "PointwiseParams", py::module_local());
  pointwise.def(py::init());
  pointwise.def(
      "__repr__", [](const PointwiseParams& self) { return self.toString(); });
  pointwise.def_readwrite("break_point", &PointwiseParams::break_point, R"(
                Split point from left to right of domain for 2D scheduling.
              )");
  pointwise.def_readwrite("split_block", &PointwiseParams::split_block, R"(
                Split block across left and right dimension.
              )");
  pointwise.def_readwrite(
      "split_grid_y_dim", &PointwiseParams::split_grid_y_dim, R"(
                Split grid y dimension if too large.
              )");
  pointwise.def_readwrite(
      "flip_grid_binding", &PointwiseParams::flip_grid_binding, R"(
                Bind BIDy on innermost dimension for broadcast performance.
              )");
  pointwise.def_readwrite(
      "vectorization_factor", &PointwiseParams::vectorization_factor, R"(
                Vectorization factor.
              )");
  pointwise.def_readwrite(
      "unroll_factor_inner", &PointwiseParams::unroll_factor_inner, R"(
                Unroll factor for inner dimension.
              )");
  pointwise.def_readwrite(
      "unroll_factor_outer", &PointwiseParams::unroll_factor_outer, R"(
                Unroll factor for outer dimension to reuse loaded data.
              )");

  py::class_<ReductionParams, HeuristicParams> reduction(
      nvfuser, "ReductionParams", py::module_local());
  reduction.def(py::init());
  reduction.def(
      "__repr__", [](const ReductionParams& self) { return self.toString(); });
  reduction.def_readwrite("fastest_dim", &ReductionParams::fastest_dim, R"(
                Reduce on innermost dimension.
              )");
  reduction.def_readwrite(
      "persistent_kernel", &ReductionParams::persistent_kernel, R"(
                Store input in shared memory or registers to reduce global memory reads.
              )");
  reduction.def_readwrite(
      "project_persistent_buffers",
      &ReductionParams::project_persistent_buffers,
      R"(Project persistent buffers back to inputs.
              )");
  reduction.def_readwrite("schedule_3D", &ReductionParams::schedule_3D, R"(
                Use 3D scheduling for patterns like [reduction, iteration, reduction].
              )");
  reduction.def_readwrite("flip_grid", &ReductionParams::flip_grid, R"(
                Swap gdimx and gdimy bindings for outer reductions.)");
  reduction.def_readwrite(
      "cross_block_inner_reduction",
      &ReductionParams::cross_block_inner_reduction,
      R"(Reduce across the block for inner reduction.)");
  reduction.def_readwrite(
      "cross_grid_inner_reduction",
      &ReductionParams::cross_grid_inner_reduction,
      R"(Reduce across the grid for inner reduction.)");
  reduction.def_readwrite(
      "unroll_factor_inner_reduction",
      &ReductionParams::unroll_factor_inner_reduction,
      R"(Unrolling/vectorization factor for inner reduction dimension.)");
  reduction.def_readwrite(
      "unroll_factor_top_of_vectorization",
      &ReductionParams::unroll_factor_top_of_vectorization,
      R"(Extra unroll on top of vectorization.)");
  reduction.def_readwrite(
      "vectorize_inner_reduction",
      &ReductionParams::vectorize_inner_reduction,
      R"(Vectorize instead of unroll for inner reduction.)");
  reduction.def_readwrite(
      "split_grid_dim_inner_reduction",
      &ReductionParams::split_grid_dim_inner_reduction,
      R"(Split grid dimension for inner reduction if too large.)");
  reduction.def_readwrite(
      "pad_inner_reduction_to_warp",
      &ReductionParams::pad_inner_reduction_to_warp,
      R"(Pad inner dimension to nearest warp.)");
  reduction.def_readwrite(
      "batches_per_block_inner_reduction",
      &ReductionParams::batches_per_block_inner_reduction,
      R"(Register persistent buffer size in inner dimension.)");
  reduction.def_readwrite(
      "block_dim_inner_reduction",
      &ReductionParams::block_dim_inner_reduction,
      R"(Block parallel dimension for inner reduction.)");
  reduction.def_readwrite(
      "grid_dim_inner_reduction",
      &ReductionParams::grid_dim_inner_reduction,
      R"(Grid parallel dimension for inner reduction.)");
  reduction.def_readwrite(
      "multiple_reds_per_blk",
      &ReductionParams::multiple_reds_per_blk,
      R"(Perform multiple reductions per block.)");
  reduction.def_readwrite(
      "unroll_factor_iter_dom",
      &ReductionParams::unroll_factor_iter_dom,
      R"(Unrolling/vectorization factor for iteration dimension.)");
  reduction.def_readwrite(
      "vectorize_iter_dom",
      &ReductionParams::vectorize_iter_dom,
      R"(Vectorize instead of unroll for iteration domain.)");
  reduction.def_readwrite(
      "split_grid_dim_iter_dom_inner",
      &ReductionParams::split_grid_dim_iter_dom_inner,
      R"(Inner split grid dimension for iteration axis.)");
  reduction.def_readwrite(
      "split_grid_dim_iter_dom_outer",
      &ReductionParams::split_grid_dim_iter_dom_outer,
      R"(Outer split grid dimension for iteration axis.)");
  reduction.def_readwrite(
      "block_dim_iter_dom",
      &ReductionParams::block_dim_iter_dom,
      R"(Block parallel dimension for iteration domain.)");
  reduction.def_readwrite(
      "grid_dim_iter_dom",
      &ReductionParams::grid_dim_iter_dom,
      R"(Grid parallel dimension for iteration domain.)");
  reduction.def_readwrite(
      "cross_block_outer_reduction",
      &ReductionParams::cross_block_outer_reduction,
      R"(Reduce across the block for outer reduction.)");
  reduction.def_readwrite(
      "cross_grid_outer_reduction",
      &ReductionParams::cross_grid_outer_reduction,
      R"(Reduce across the grid for outer reduction.)");
  reduction.def_readwrite(
      "batches_per_block_outer_reduction",
      &ReductionParams::batches_per_block_outer_reduction,
      R"(Register persistent buffer size in outer dimension.)");
  reduction.def_readwrite(
      "unroll_factor_outer_reduction",
      &ReductionParams::unroll_factor_outer_reduction,
      R"(Unrolling/vectorization factor for outer reduction.)");
  reduction.def_readwrite(
      "block_dim_outer_reduction",
      &ReductionParams::block_dim_outer_reduction,
      R"(Block parallel dimension for outer reduction.)");
  reduction.def_readwrite(
      "grid_dim_outer_reduction",
      &ReductionParams::grid_dim_outer_reduction,
      R"(Grid parallel dimension for outer reduction.)");
  reduction.def_readwrite(
      "compute_persistent_buffer_with_first_consumer",
      &ReductionParams::compute_persistent_buffer_with_first_consumer,
      R"(Use computeWith to persistent buffers.)");
  reduction.def_readwrite(
      "static_bdimx",
      &ReductionParams::static_bdimx,
      R"(Static block dimension X.)");
  reduction.def_readwrite(
      "static_bdimy",
      &ReductionParams::static_bdimy,
      R"(Static block dimension Y.)");
  reduction.def_readwrite(
      "combined_inner_outer",
      &ReductionParams::combined_inner_outer,
      R"(Combined inner and outer reduction.)");
  reduction.def_readwrite(
      "tidx_for_outer_reduction",
      &ReductionParams::tidx_for_outer_reduction,
      R"(Use TIDx for outer reduction axis.)");
  reduction.def_readwrite(
      "pad_outer_reduction_to_warp",
      &ReductionParams::pad_outer_reduction_to_warp,
      R"(Pad outer reduction to warp.)");
  reduction.def_readwrite(
      "combined_split_grid_inner_dim",
      &ReductionParams::combined_split_grid_inner_dim,
      R"(Further split inner dimension by grid in combined scheduler.)");
  reduction.def_readwrite(
      "vectorization_factor_outer",
      &ReductionParams::vectorization_factor_outer,
      R"(Vectorization factor for outer reduction partial result.)");
  reduction.def_readwrite(
      "vectorization_factor_tmp_gmem_write",
      &ReductionParams::vectorization_factor_tmp_gmem_write,
      R"(Vectorization factor for temporary global memory write.)");
  reduction.def_readwrite(
      "block_dim_inner_reduction_extra",
      &ReductionParams::block_dim_inner_reduction_extra,
      R"(Additional block parallel dimension for inner reduction.)");
}

} // namespace nvfuser::python
