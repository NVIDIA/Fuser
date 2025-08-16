// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/padded_parallel_dimensions.h>
#include <device_lower/utils.h>
#include <fusion.h>
#include <ir/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

PaddedParallelDimensions collectPaddedParallelDims(Fusion* fusion) {
  bool can_be_single_warp = true;

  const auto warp_size = at::cuda::warp_size();

  PaddedParallelDimensions warp_pad_info;

  auto used_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->getLoopDomain()) {
      if (tv->definition()) {
        // TODO: Support GroupedReductionOp
        if (auto reduction = dynamic_cast<ReductionOp*>(tv->definition())) {
          if (ir_utils::getMaybeWarpReductionDim(
                  reduction->out(), reduction->in())
                  .has_value()) {
            warp_pad_info.has_warp_reduction = true;
          }
        }
      }

      // Check ifi TIDx is padded in this kernel
      if (id->hasPaddingToMultipleOfWarp()) {
        NVF_ERROR(
            id->getParallelType() == ParallelType::TIDx,
            "Padded types supported only on TIDx");
        warp_pad_info.is_tidx_padded = true;
      }

      // Check all possible bindings of TIDx to see
      //  if TIDx will eventually be bound to a single warp.
      if (id->getParallelType() == ParallelType::TIDx) {
        auto size_after_padding = id->getMaybeSizeAfterPadding();
        bool padding_to_single_warp = size_after_padding.has_value() &&
            size_after_padding.value() == warp_size;

        if (id->extent()->isConstInt() &&
            id->extent()->evaluate().as<int64_t>() > warp_size &&
            !padding_to_single_warp) {
          // If we see any other TIDx binding that's larger than
          //  a warp or unknown, we shouldn't lower warp reduce
          //  to a single warp type.
          can_be_single_warp = false;
          warp_pad_info.is_tidx_single_warp = false;
        } else if (can_be_single_warp) {
          if (padding_to_single_warp ||
              (id->extent()->isConstInt() &&
               id->extent()->evaluate().as<int64_t>() == warp_size)) {
            warp_pad_info.is_tidx_single_warp = true;
          }
        }
      }
    }
  }

  return warp_pad_info;
}

} // namespace nvfuser
