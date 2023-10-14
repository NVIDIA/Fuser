// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <unordered_map>

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
  CudaEventTimer(cudaStream_t s);
  ~CudaEventTimer();

  void reset();
  void start();
  void stop();
  float time();
  ProfilerState state() const;

 private:
  cudaStream_t stream_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  float time_ms_;
  ProfilerState state_;
};

struct DeviceDescriptor{
  void generate(size_t device);

  int device{0};
  std::string name{"NVIDIA Unknown GPU"};
  int bus_width{0};
  int memory_clock{0}; 

  double peak_bandwidth{0.0};
};

struct KernelProfile {
  std::string name;
  uint32_t device{0};
  uint32_t stream{0};
  uint32_t correlation_id{0};

  double compile_time_ms{0.0};
  double time_ms{0.0};

  std::array<int32_t, 3> grid{0, 0, 0};
  std::array<int32_t, 3> block{0, 0, 0};
  std::array<uint32_t, 3> cluster{0, 0, 0};

  int32_t dynamic_shared_mem{0};
  int32_t static_shared_mem{0};
  uint32_t registers{0};
  
  size_t input_bytes{0};
  size_t output_bytes{0};
  size_t total_bytes{0};
  
  float effective_bandwidth{0.0};
  float perentage_peak_bandwidth{0.0};
};

struct FusionProfile {
  void reset();

  double time_ms{0.0};
  double host_time_ms{0.0};
  double compile_time_ms{0.0};
  double kernel_time_ms{0.0};

  size_t input_bytes{0};
  size_t output_bytes{0};
  size_t total_bytes{0};

  float effective_bandwidth{0.0};
  float perentage_peak_bandwidth{0.0};

  //std::vector<SegmentProfile> segment_profiles;
};

class SegmentProfiler {
 public:
  SegmentProfiler();

  void setSegmentId(int64_t id);

  void startCompile();
  void stopCompile();

  void startKernel();
  void stopKernel();

  void bytesAccessed(size_t input_bytes, size_t output_bytes);

 private:
  int64_t segment_id_;

  CudaEventTimer compile_timer_;
  ProfilerState kernel_profile_state_;
};

class FusionProfiler {

 public: // Static Methods
  static FusionProfiler* get();
 
  // Static Methods to capture Asynchronous CUPTI activity
  // Collects CUPTI activity to map Profiler Id -> CUPTI Correlation Id
  // Segment ID -> Correlation ID
  static void recordAsyncCorrIdActivity(uint32_t seg_id, uint32_t corr_id);
  // Collects CUPTI Kernel Activity
  // Segment ID -> KernelProfile
  static void recordAsyncKernelActivity(
      size_t device, uint32_t corr_id, KernelProfile profile);

 public:
  FusionProfiler(size_t device);
  
  void createSegments(size_t num);
  SegmentProfiler& segment(size_t idx);

  void start();
  void stop();

  void bytesAccessed(size_t input_bytes, size_t output_bytes);

 private:
  FusionProfiler();
  void reset();
  void print() const;

 private:
  static FusionProfiler* singleton_;
  static std::mutex singleton_lock_;

  DeviceDescriptor device_descriptor_;
  FusionProfile profile_;
  CudaEventTimer fusion_timer_;
  std::vector<SegmentProfiler> segments_;
  std::unordered_map<uint32_t, KernelProfile> corrid_2_kprof_;
};

#define _FP_ENABLE(code) \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    code; \
  }
// Fusion Profiling Macros
#define FUSION_PROFILER_START_PROFILE \
  _FP_ENABLE(FusionProfiler::get()->start())
#define FUSION_PROFILER_STOP_PROFILE \
  _FP_ENABLE(FusionProfiler::get()->stop())
#define FUSION_PROFILER_CREATE_SEGMENTS(segments) \
  _FP_ENABLE(FusionProfiler::get()->createSegments(segments))
#define FUSION_PROFILER_BYTES_ACCESSED(inputs, outputs) \
  _FP_ENABLE(FusionProfiler::get()->bytesAcccessed(inputs, outputs))

// Segment Profiling Macros
#define SEGMENT_PROFILER_START_COMPILE(device, idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).startCompile())
#define SEGMENT_PROFILER_STOP_COMPILE(device, idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).stopCompile())
#define SEGMENT_PROFILER_START_KERNEL(device, idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).startKernel())
#define SEGMENT_PROFILER_STOP_KERNEL(device, idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).stopKernel())
#define SEGMENT_PROFILER_BYTES_ACCESSED(device, idx, inputs, outputs) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).bytesAccessed(inputs, outputs))

} // namespace nvfuser
