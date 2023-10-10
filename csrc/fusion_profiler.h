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

enum class ProfilerState {
  Ready,
  Running,
  Finished,
  Processed,
};

std::ostream& operator<<(std::ostream&, const ProfilerState&);

class CudaEventTimer {
 public:
  CudaEventTimer(cudaStream_t s) : 
    stream_(s),
    start_event_(),
    stop_event_(),
    time_ms_(0.0), 
    state_(ProfilerState::Ready) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop_event_));
  }

  ~CudaEventTimer() {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop_event_));
  }

  void reset() {
    time_ms_ = 0.0;
    state_ = ProfilerState::Ready;
  }

  void start() {
    NVF_CHECK(state_ == ProfilerState::Ready, "ProfilerState is not Ready! ", state_);
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event_, stream_));
    state_ = ProfilerState::Running;
  }

  void stop() {
    NVF_CHECK(state_ == ProfilerState::Running, "ProfilerState is not Running! ", state_);
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(stop_event_, stream_));
    state_ = ProfilerState::Finished;
  }

  float time() {
    if (state_ == ProfilerState::Finished) {
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event_));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop_event_));
      NVFUSER_CUDA_RT_SAFE_CALL(
          cudaEventElapsedTime(&time_ms_, start_event_, stop_event_));
      state_ = ProfilerState::Processed;
    } else {
      NVF_CHECK(state_ == ProfilerState::Processed, "ProfilerState is not Processed! ", state_);
    }
    return time_ms_;
  }

  ProfilerState state() const {
    return state_;
  }

 private:
  cudaStream_t stream_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  float time_ms_;
  ProfilerState state_;
};

/*struct KernelProfile {
  float kernel_time;
};*/

class SegmentProfiler {
 public:
  SegmentProfiler();

  void start_gpu_profile();
  void stop_gpu_profile();
  void start_gpu_compile();
  void stop_gpu_compile();

 private:
  int64_t segment_id_;

  CudaEventTimer compile_timer_;
};

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
  static FusionProfiler* get();

  void start();
  void stop();

  void addSegment();
  SegmentProfiler* lastSegment();
  SegmentProfiler* segment(int64_t idx);

  void bytesAccessed(int64_t input_bytes, int64_t output_bytes);

 private:
  FusionProfiler();

  void reset();
  void print() const;

 private:
  static FusionProfiler* singleton_;
  static std::mutex singleton_lock_;

  FusionProfile profile_;
  CudaEventTimer fusion_timer_;
  std::vector<unique_ptr<SegmentProfiler>> segments_;
};

#define FUSION_PROFILER_START_PROFILE \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::get->start(); \
  }
#define FUSION_PROFILER_STOP_PROFILE \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    FusionProfiler::get->stop(); \
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
