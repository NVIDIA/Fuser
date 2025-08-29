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
  cutlass.def(
      "nvfp4_scaled_mm",
      [](const torch::Tensor& a,
         const torch::Tensor& b,
         const torch::Tensor& scales_a,
         const torch::Tensor& scales_b,
         const torch::Tensor& alpha,
         at::ScalarType out_dtype) -> torch::Tensor {
        return cutlass_kernels::nvfp4_scaled_mm(
            a, b, scales_a, scales_b, alpha, out_dtype);
      },
      R"(Computes nvfp4 matmul and returns bf16, fp16, or fp32 output tensor.
         nvfp4_scaled_mm(Tensor a,
                         Tensor b,
                         Tensor scales_a,
                         Tensor scales_b,
                         Tensor alpha,
                         DataType out_dtype)
                         -> Tensor output)");

  cutlass.def(
      "nvfp4_scaled_mm_blockscale",
      [](const torch::Tensor& a_nvfp4,
         const torch::Tensor& b_nvfp4,
         const torch::Tensor& scales_a,
         const torch::Tensor& scales_b,
         const torch::Tensor& alpha,
         const torch::Tensor& global_normconst) -> py::tuple {
        std::pair<torch::Tensor, torch::Tensor> output =
            cutlass_kernels::nvfp4_scaled_mm_blockscale(
                a_nvfp4, b_nvfp4, scales_a, scales_b, alpha, global_normconst);
        return py::make_tuple(output.first, output.second);
      },
      R"(Computes nvfp4 matmul and blockscale quantization. It returns nvfp4
         output tensor and its blockscale factor.
         nvfp4_scaled_mm_blockscale(Tensor a_nvfp4,
                                    Tensor b_nvfp4,
                                    Tensor scales_a,
                                    Tensor scales_b,
                                    Tensor alpha,
                                    Tensor global_normconst)
                                    -> tuple(Tensor out_nvfp4, Tensor blockscale))");
}

void bindGroupedGemm(py::module_& cutlass) {
  cutlass.def(
      "nvfp4_scaled_grouped_mm",
      &cutlass_kernels::nvfp4_scaled_grouped_mm,
      R"(Computes nvfp4 grouped matmul and returns bf16 or fp16 output tensor.
         nvfp4_scaled_grouped_mm(Tensor output,
                                 Tensor a,
                                 Tensor b,
                                 Tensor a_blockscale,
                                 Tensor b_blockscale,
                                 Tensor alphas,
                                 Tensor ab_strides,
                                 Tensor c_strides,
                                 Tensor problem_sizes,
                                 Tensor expert_offsets,
                                 Tensor sf_offsets) -> ())");
}

} // namespace

void bindCutlass(py::module& nvfuser) {
  py::module_ nvf_cutlass = nvfuser.def_submodule(
      "nvf_cutlass", "This submodule contains all cutlass gemms for NvFuser.");
  bindGemm(nvf_cutlass);
  bindGroupedGemm(nvf_cutlass);
}

} // namespace nvfuser::python

#endif
