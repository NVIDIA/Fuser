// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <profiler.h>

namespace nvfuser {

// CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

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

std::mutex FusionProfiler::singleton_lock_;
FusionProfiler* FusionProfiler::singleton_ = nullptr;

void FusionProfiler::start() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new FusionProfiler();
  } else {
    singleton_->reset();
  }

  singleton_->timer_.init();
  singleton_->timer_.start();
}

void FusionProfiler::stop() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");
 
  singleton_->profile_.total_time = singleton_->timer_.elapsed();
  singleton_->print();
}

void FusionProfiler::reset() {
  profile_.reset();
}

void FusionProfiler::print() const {
  std::cout << "\nFusion Total Time: " << profile_.total_time << std::endl;
}

} // namespace nvfuser
