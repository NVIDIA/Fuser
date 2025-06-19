// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <nvf_cutlass.h>

namespace nvfuser::python {

namespace {

void bindGroupedGemm(py::module_& cutlass) {
  const char* mxfp8_grouped_gemm_docstring =
      R"(fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, "
         "Tensor a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, "
         "Tensor stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, "
         "Tensor problem_sizes, Tensor expert_offsets, Tensor workspace))";
  cutlass.def(
      "fp8_blockwise_scaled_grouped_mm",
      &cutlass_kernels::fp8_blockwise_scaled_grouped_mm,
      mxfp8_grouped_gemm_docstring);

  const char* prepare_moe_input_docstring =
      R"(prepare_moe_input(Tensor topk_ids, Tensor expert_offsets, Tensor problem_sizes1, "
        "Tensor problem_sizes2, Tensor input_permutation, Tensor output_permutation, "
        "int num_experts, int n, int k) -> ())";
  cutlass.def(
      "prepare_moe_input",
      &cutlass_kernels::prepare_moe_input,
      prepare_moe_input_docstring);
}

} // namespace

void bindCutlass(py::module& nvfuser) {
  py::module_ nvf_cutlass = nvfuser.def_submodule(
      "nvf_cutlass", "This submodule contains all cutlass gemms for NvFuser.");
  bindGroupedGemm(nvf_cutlass);
}

} // namespace nvfuser::python
