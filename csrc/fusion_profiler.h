// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <cupti.h>
#include <options.h>
#include <utils.h>

namespace nvfuser {

class CudaEventTimer {
 public:
  CudaEventTimer(cudaStream_t s) : 
    stream_(s),
    start_event_(),
    stop_event_(),
    time_ms_(0.0), 
    time_calculated_(false),
    start_recorded_(false),
    stop_recorded_(false) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop_event_));
  }

  ~CudaEventTimer() {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop_event_));
  }

  void reset() {
    time_ms_ = 0.0;
    time_calculated_ = false;
    start_recorded_ = false;
    stop_recorded_ = false;
  }

  void start() {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event_, stream_));
    start_recorded_ = true;
    time_calculated_ = false;
  }
  void stop() {
    NVF_CHECK(start_recorded_, "Cude Start Event was not recorded!");
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(stop_event_, stream_));
    stop_recorded_ = true;
  }

  float time() {
    if (!time_calculated_) {
      NVF_CHECK(start_recorded_ && stop_recorded_, "Cuda Start and Stop events not recorded!");
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event_));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop_event_));
      NVFUSER_CUDA_RT_SAFE_CALL(
          cudaEventElapsedTime(&time_ms_, start_event_, stop_event_));
      time_calculated_ = true;
      start_recorded_ = false;
      stop_recorded_ = false;
    }
    return time_ms_;
  }

 private:
  cudaStream_t stream_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  float time_ms_;
  bool time_calculated_;
  bool start_recorded_;
  bool stop_recorded_;
};

/*struct KernelProfile {
  float kernel_time;
};*/

struct FusionProfile {
  void reset();

  float total_time;
  float host_time;
  float compile_time;
  float kernel_time;

  int64_t input_bytes;
  int64_t output_bytes;
  int64_t total_bytes;

  std::string device_name;
  float device_peak_bandwidth;

  float effective_bandwidth;
  float perentage_peak_bandwidth;

  //std::vector<SegmentProfile> segment_profiles;
};

// Singleton
class FusionProfiler : public NonCopyable {
 public:
  static void start();
  static void stop();

  static void start_kernel_compile();
  static void stop_kernel_compile();
  
  static void start_kernel();
  static void stop_kernel();

 private:
  FusionProfiler();

  void reset();
  void print() const;

 private:
  static FusionProfiler* singleton_;
  static std::mutex singleton_lock_;

  CudaEventTimer fusion_timer_;
  CudaEventTimer compile_timer_;
  FusionProfile profile_;
  bool fusion_profile_started_;
  bool kernel_compile_recorded_;
  bool kernel_profile_recorded_;
};

#define FUSION_PROFILER_START_PROFILE \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::start(); \
  }
#define FUSION_PROFILER_STOP_PROFILE \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::stop(); \
  }
#define FUSION_PROFILER_START_KERNEL \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::start_kernel(); \
  }
#define FUSION_PROFILER_STOP_KERNEL \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::stop_kernel(); \
  }
#define FUSION_PROFILER_START_KERNEL_COMPILE \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::start_kernel_compile(); \
  }
#define FUSION_PROFILER_STOP_KERNEL_COMPILE \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::stop_kernel_compile(); \
  }

} // namespace nvfuser
