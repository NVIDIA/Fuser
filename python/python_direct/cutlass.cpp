// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_ENABLE_CUTLASS

#include <bindings.h>
#include <nvf_cutlass.h>

namespace nvfuser::python {

namespace {

void bindGemm(py::module_& cutlass) {
  const char* nvfp4_gemm_docstring =
      R"(nvfp4_scaled_mm(Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor alpha, DataType out_dtype) -> Tensor output)";
  cutlass.def(
      "nvfp4_scaled_mm",
      &cutlass_kernels::nvfp4_scaled_mm,
      nvfp4_gemm_docstring);
}

void bindGroupedGemm(py::module_& cutlass) {
  const char* nvfp4_scaled_grouped_mm_docstring =
      R"(nvfp4_scaled_grouped_mm(Tensor! output, Tensor a, Tensor b, Tensor a_blockscale, "
         "Tensor b_blockscale, Tensor alphas, Tensor ab_strides, Tensor c_strides, Tensor problem_sizes, "
         "Tensor expert_offsets, Tensor sf_offsets) -> ())";
  cutlass.def(
      "nvfp4_scaled_grouped_mm",
      &cutlass_kernels::nvfp4_scaled_grouped_mm,
      nvfp4_scaled_grouped_mm_docstring);
  const char* grouped_mm_docstring =
      R"(grouped_mm(Tensor! output, Tensor a, Tensor b, Tensor alphas, Tensor "
         "ab_strides, Tensor c_strides, Tensor problem_sizes, Tensor expert_offsets) -> ())";
  cutlass.def("grouped_mm", &cutlass_kernels::grouped_mm, grouped_mm_docstring);
}

} // namespace

void bindCutlass(py::module& nvfuser) {
  py::module_ nvf_cutlass = nvfuser.def_submodule(
      "nvf_cutlass", "This submodule contains all cutlass gemms for NvFuser.");
  bindGemm(nvf_cutlass);
}

} // namespace nvfuser::python

#endif
