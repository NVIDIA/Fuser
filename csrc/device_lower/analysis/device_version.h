// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <compute_at_map.h>
#include <dispatch.h>
#include <ir/all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

class LoopIndexing;
class ComputeAtMap;

//! Traverses a Fusion to find the minimum supported CUDA compute capability
//! that will be able to run the generated kernel.
class MinimumDeviceVersion : private IterVisitor {
 public:
  static std::pair<std::pair<int, int>, std::string> compute(Fusion* fusion) {
    MinimumDeviceVersion mdv;
    mdv.traverse(fusion);
    return {mdv.min_version_, mdv.reason_};
  }

 private:
  using IterVisitor::dispatch;
  using IterVisitor::handle;
  using IterVisitor::traverse;

  //! Check dtypes of all Vals. BFloat16 requires Ampere (8.0+), Float8_xxx
  //! requires Hopper (9.0+)
  void dispatch(Val* v) final;

  //! MmaOp currently supports Turing and newer (7.5+) depending on macro
  void handle(MmaOp* mma_op) final;

  //! LoadStoreOpType::CpAsync requires Ampere (8.0+)
  //! https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
  void handle(LoadStoreOp* ls_op) final;

  //! If TensorView has warp specialized circular buffering, it will use the
  //! setmaxnreg ptx instruction that requires Hopper (9.0+).
  //! https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg
  void handle(TensorView* tv) final;

  //! bump min_version_ to at least this value
  void ensureVersion(std::pair<int, int> version, std::string reason);

 private:
  std::pair<int, int> min_version_ = {7, 0};
  std::string reason_ =
      "nvFuser supports Volta and above (compute capability 7.0+)";
};

} // namespace nvfuser
