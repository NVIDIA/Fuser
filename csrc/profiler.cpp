// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <profiler.h>

namespace nvfuser {

void FusionProfile::reset() {
  total_time = 0.0;
  host_time = 0.0;
  kernel_time = 0.0;

  input_bytes = 0;
  output_bytes = 0;
  total_bytes = 0;

  device_name.clear();
  device_peak_bandwidth = 0.0;

  effective_bandwidth = 0.0;
  perentage_peak_bandwidth = 0.0;
}

std::mutex Profiler::singleton_lock_;
Profiler* Profiler::singleton_ = nullptr;

void Profiler::start() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new Profiler();
  } else {
    singleton_->reset();
  }

  singleton_->timer_.init();
  singleton_->timer_.start();
}

void Profiler::stop() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr, "Profiler singleton is unexpectedly null!");
 
  singleton_->profile_.total_time = singleton_->timer_.elapsed();
}

void Profiler::reset() {
  profile_.reset();
}

} // namespace nvfuser
