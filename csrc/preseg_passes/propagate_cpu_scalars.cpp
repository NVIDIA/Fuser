// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <fusion.h>
#include <ir/utils.h>
#include <preseg_passes/propagate_cpu_scalars.h>

namespace nvfuser::preseg_passes {

void PropagateCpuScalarsPass::runPass(Fusion* fusion) {
  // CPU scalars can either be fusion inputs, or variables
  // philox_seed/philox_offset in SDPA operators. This pass propagates the CPU
  // scalar property to the outputs of expressions using the following rule:
  // Exprs can have the following combination of inputs:
  // 1. expr->inputs() = {scalars} -> generates CUDA tensors
  // These are expressions like `full`, `uniform` that generate CUDA tensors
  // but do not accept any fusion inputs.
  // 2. expr->inputs() = {scalars (optional), CPU scalar tensor} -> generates
  // CPU scalars
  // 3. expr->inputs() = {scalars (optional), CPU scalar tensor (optional), CUDA
  // tensor} -> generates CUDA tensors
  // This information is used in KernelExecutor::supported to reject fusions with
  // any CPU scalar outputs since it is not supported by nvFuser codegen.
  // Note: Alternatively, they can be expression evaluated. 
  // See `test_evalutor.cpp/ExprEvalTest.CpuScalarOutputs`

  for (Expr* expr : StmtSort::getExprs(fusion)) {
    bool has_cpu_scalar_input = false;
    bool has_cuda_input = false;

    for (Val* inp : expr->inputs()) {
      if (auto* inp_tv = dynamic_cast<TensorView*>(inp)) {
        if (inp_tv->isCpuScalar()) {
          has_cpu_scalar_input = true;
        } else {
          has_cuda_input = true;
          // Return early -- found atleast one CUDA input
          break;
        }
      }
    }
    if (!has_cuda_input && has_cpu_scalar_input) {
      // Expr is of the second category, and has all CPU scalar outputs
      for (Val* out : expr->outputs()) {
        if (auto* out_tv = dynamic_cast<TensorView*>(out)) {
          out_tv->setCpuScalar(true);
        }
      }
    }
  }
}
} // namespace nvfuser::preseg_passes
