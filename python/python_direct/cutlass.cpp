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

torch::Tensor scaled_mm_wrapper(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    at::ScalarType out_dtype) {
  return cutlass_kernels::nvfp4_scaled_mm(
      a, b, scales_a, scales_b, alpha, out_dtype);
}

void bindGemm(py::module_& cutlass) {
  const char* nvfp4_gemm_docstring =
      R"(nvfp4_scaled_mm(Tensor a,
                         Tensor b,
                         Tensor scales_a,
                         Tensor scales_b,
                         Tensor alpha,
                         DataType out_dtype)
                         -> Tensor output)";
  cutlass.def("nvfp4_scaled_mm", &scaled_mm_wrapper, nvfp4_gemm_docstring);

  cutlass.def(
      "nvfp4_scaled_mm_epilogue",
      [](const torch::Tensor& a,
         const torch::Tensor& b,
         const torch::Tensor& scales_a,
         const torch::Tensor& scales_b,
         const torch::Tensor& alpha,
         const torch::Tensor& global_normconst) -> py::tuple {
        std::pair<torch::Tensor, torch::Tensor> output =
            cutlass_kernels::nvfp4_scaled_mm_epilogue(
                a, b, scales_a, scales_b, alpha, global_normconst);
        return py::make_tuple(output.first, output.second);
      },
      R"(nvfp4_scaled_mm_epilouge(Tensor a,
                                  Tensor b,
                                  Tensor scales_a,
                                  Tensor scales_b,
                                  Tensor alpha,
                                  Tensor global_normconst)
                                  -> tuple(Tensor output, Tensor blockscale))");
}

} // namespace

void bindCutlass(py::module& nvfuser) {
  py::module_ nvf_cutlass = nvfuser.def_submodule(
      "nvf_cutlass", "This submodule contains all cutlass gemms for NvFuser.");
  bindGemm(nvf_cutlass);
}

} // namespace nvfuser::python

#endif
