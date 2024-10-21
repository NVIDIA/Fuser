// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <scheduler/ampere_multi_matmul.h>
#include <scheduler/hopper_multi_matmul.h>

namespace nvfuser {

void scheduleMultipleMatmuls(Fusion* fusion, const MatmulParams* params) {
  FusionGuard fg(fusion);

  // NOTE: In the future we should be able to simply check the generation of
  // the macro instead of looking at the device properties here. However,
  // until we have Hopper mma ready, we will be using Ampere macros on Hopper
  // machines for testing. This means in order to trigger Hopper code, we need
  // to look at the device instead of the macro for now. See commented
  // conditions below.
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const int cc = device_prop->major * 10 + device_prop->minor;
  if (cc >= 75 && cc < 90) {
    AmpereMultipleMatmulScheduler(fusion, params).run();
  } else if (cc >= 90 && cc < 100) {
    HopperMultipleMatmulScheduler(fusion, params).run();
  } else {
    NVF_THROW(
        "The matrix multiplication scheduler is unavailable for this device: ",
        device_prop->major,
        ".",
        device_prop->minor);
  }
}

} // namespace nvfuser
