// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h>
#include <memory>
namespace nvfuser {

constexpr int64_t kHeuristicJitCompileThreads = 4;
struct HeuristicJitImpl;

class HeuristicJit {
 public:
  HeuristicJit(
      Fusion* fusion,
      SchedulerType scheduler_type,
      int num_threads = kHeuristicJitCompileThreads);

  bool canReuse(
      const HeuristicParams* heuristic_params);
      
  ~HeuristicJit();

 private:
  std::unique_ptr<HeuristicJitImpl> pimpl_;
};
} // namespace nvfuser
