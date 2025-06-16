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

void bindGroupedGemm(py::module_& cutlass) {
  const char* docstring =
      R"(fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, "
         "Tensor a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, "
         "Tensor stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, "
         "Tensor problem_sizes, Tensor expert_offsets, Tensor workspace))";
  cutlass.def(
      "fp8_blockwise_scaled_grouped_mm",
      &cutlass_kernels::fp8_blockwise_scaled_grouped_mm,
      docstring);
}

} // namespace

void bindCutlass(py::module& nvfuser) {
  py::module_ nvf_cutlass = nvfuser.def_submodule(
      "nvf_cutlass", "This submodule contains all cutlass gemms for NvFuser.");
  bindGroupedGemm(nvf_cutlass);
}

} // namespace nvfuser::python

#endif
