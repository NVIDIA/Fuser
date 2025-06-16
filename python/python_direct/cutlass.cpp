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
      R"(nvfp4_scaled_mm(Tensor output, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor alpha))";
  cutlass.def(
      "nvfp4_scaled_mm",
      &cutlass_kernels::nvfp4_scaled_mm,
      nvfp4_gemm_docstring);
}

} // namespace

void bindCutlass(py::module& nvfuser) {
  py::module_ nvf_cutlass = nvfuser.def_submodule(
      "nvf_cutlass", "This submodule contains all cutlass gemms for NvFuser.");
  bindGemm(nvf_cutlass);
}

} // namespace nvfuser::python

#endif
