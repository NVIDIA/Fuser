// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <preseg_passes/optimization_pass.h>

namespace nvfuser::preseg_passes {

// A pass used in HostIrLower that takes a HostIrContainer as input, reads the
// TensorView's ParallelType::Stream, and modify the the HostIrContainer's top
// level expressions with the corresponding Host For Loops, which bodies contain
// stream assignement, selecting on tensor's axis, and the exprs on those sliced
// tensors. After this pass, the ParallelType::Stream is removed from the
// TensorView's axis.
//
// An illustration of the pass can be found in the tests
// `test_host_ir_stream_lowering.cpp`
// with the option `NVFUSER_DUMP=host_ir`.
class StreamParallelType : public OptimizationPass<StreamParallelType> {
  friend class OptimizationPass<StreamParallelType>;

 protected:
  static void runPass(Fusion* fusion);
  static constexpr std::string_view name() {
    return "StreamParallelType";
  }
};

} // namespace nvfuser::preseg_passes
