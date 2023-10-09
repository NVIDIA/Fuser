#pragma once
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <array>
#include <memory>
#include <vector>
#include <cuda.h>

namespace nvfuser {

class KernelArgumentHolder;
class LaunchParams;
struct CompileParams;
namespace kir {
  class Kernel;
}

struct vm_t final {
  /// Creates a VM object from a preconfigured buffer.
  /// The given buffer should not die while this object is alive.
  vm_t(const void*);
  ~vm_t();

  void initialize();
#if 0
  void initialize(KernelArgumentHolder&,
                  const LaunchParams&,
                  const CompileParams&,
                  // drop outputs arg, force it as part of exec?
                  std::vector<at::Tensor> outputs,
                  const kir::Kernel*);

  void setKernelParams(CUfunction, const std::array<unsigned,3>& grid,
                       const std::array<unsigned,3>& block,
                       unsigned shmem, CUstream,
                       const std::vector<void*>& args);
#endif

  void exec() const;
  void coop_exec() const;

private:
  ///@{
  /// State that is unowned by this object.
  CUstream strm_;
  CUfunction function_;
  ///@}
  std::array<unsigned,3> gridDim_;
  std::array<unsigned,3> blockDim_;
  unsigned shmem_;

  /// Kernel arguments, used for cuLaunchKernel. The void*s that these point to
  /// are unowned by this object.
  std::vector<void*> args_;
};

}
