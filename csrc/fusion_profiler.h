// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <chrono>
#include <unordered_map>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <cupti.h>
#include <debug.h>
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
  double time();
  ProfilerState state() const;

 private:
  cudaStream_t stream_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  double time_ms_;
  ProfilerState state_;
};

class HostTimer {
 public:
  using Clock = std::chrono::steady_clock;

  HostTimer();

  void reset();
  void start();
  void stop();
  double time();
  ProfilerState state() const;

 private:
  Clock::time_point start_event_;
  Clock::time_point stop_event_;
  double time_ms_;
  ProfilerState state_;
};

struct DeviceDescriptor{
  void generate(int device);

  int device{-1};
  std::string name{"NVIDIA Unknown GPU"};
  int bus_width{0};
  int memory_clock{0}; 

  double peak_bandwidth_gbs{0.0};
};

struct KernelProfile {
  std::string name;
  int device{-1};
  uint32_t stream{0};
  uint32_t correlation_id{0};

  double compile_time_ms{0.0};
  double time_ms{0.0};
  double effective_bandwidth_gbs{0.0};
  double percentage_peak_bandwidth{0.0};

  std::array<int32_t, 3> grid{0, 0, 0};
  std::array<int32_t, 3> block{0, 0, 0};
  std::array<uint32_t, 3> cluster{0, 0, 0};

  int32_t dynamic_shared_mem{0};
  int32_t static_shared_mem{0};
  uint32_t registers{0};
  
  int64_t input_bytes{0};
  int64_t output_bytes{0};
  
  std::string device_name;
  double peak_bandwidth_gbs{0.0};
};

struct FusionProfile {
  static std::array<const char*, 25> column_strs;

  void reset();

  bool verbose{isDebugDumpEnabled(DebugDumpOption::FusionProfilerVerbose)};
  int64_t fusion_id{-1};

  double time_ms{0.0};
  double host_time_ms{0.0};
  double compile_time_ms{0.0};
  double kernel_time_ms{0.0};

  int64_t input_bytes{0};
  int64_t output_bytes{0};

  double effective_bandwidth_gbs{0.0};
  double percentage_peak_bandwidth{0.0};

  std::vector<KernelProfile> kernel_profiles;
};

std::ostream& operator<<(std::ostream&, const FusionProfile&);

class SegmentProfiler {
 public:
  SegmentProfiler(uint32_t id, bool disable_cupti);

  void startCompile(int device);
  void stopCompile();

  void startKernel(int device);
  void stopKernel();

  void inputBytesAccessed(int64_t bytes);
  void outputBytesAccessed(int64_t bytes);

  uint32_t segmentId() const;
  int device() const { return device_; }

  int64_t inputBytes() const { return input_bytes_; }
  int64_t outputBytes() const { return output_bytes_; }
  double compileTime() { return compile_timer_.time(); }
  ProfilerState state() const { return kernel_profile_state_; }

 private:
  // The disable_cupti option is for testing
  bool disable_cupti_;

  int device_;
  uint32_t segment_id_;

  HostTimer compile_timer_;
  int64_t input_bytes_;
  int64_t output_bytes_;
  ProfilerState kernel_profile_state_;
};

class FusionProfiler {
  // The disable_cupti option is for testing
  FusionProfiler(bool disable_cupti);
  void reset();

 public: 
  // The disable_cupti option is for testing
  static FusionProfiler* get(bool disable_cupti = false);

  ProfilerState state() const;
 
  void createSegments(size_t num);
  SegmentProfiler& segment(size_t idx);

  void start();
  void stop();
  void startParallelCompile();
  void stopParallelCompile();
  void inputBytesAccessed(int64_t bytes);
  void outputBytesAccessed(int64_t bytes);
  const FusionProfile& profile() const;
  
  // Methods to capture Asynchronous CUPTI activity that get called from
  // functions registered with CUPTI.
  // Correlation ID -> Segment ID
  void recordAsyncCorrIdActivity(uint32_t seg_id, uint32_t corr_id);
  // Collects CUPTI Kernel Activity
  void recordAsyncKernelActivity(KernelProfile prof);

 private:
  static FusionProfiler* singleton_;
  static std::mutex singleton_lock_;

  // This is a debug option for testing.
  bool disable_cupti_;
  // The state is used to check for errors in usage
  ProfilerState state_;

  // Data members with information that is aggregated into a FusionProfile
  int64_t fusion_id_;
  bool parallel_compile_;
  FusionProfile profile_;
  CudaEventTimer fusion_timer_;
  HostTimer host_timer_;
  HostTimer parallel_compile_timer_;
  std::vector<SegmentProfiler> segments_;
  // The FusionProfiler collects a cache of device descriptors so each segment
  // does not need to spend time re-generating the information.
  std::vector<DeviceDescriptor> device_descriptors_;

  // These 3 data members are used to collect and connect asynchronous records,
  // generated by CUPTI, to the segments responsible for the activity
  std::vector<KernelProfile> kernel_profiles_;
  std::unordered_map<uint32_t, uint32_t> corrid_2_segid_; 
  std::unordered_map<uint32_t, size_t> segid_2_idx_;
};

#define _FP_ENABLE(code) \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isDebugDumpEnabled(DebugDumpOption::FusionProfilerVerbose) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    code; \
  }
// Fusion Profiling Macros
#define FUSION_PROFILER_START \
  _FP_ENABLE(FusionProfiler::get()->start())
#define FUSION_PROFILER_STOP \
  _FP_ENABLE(FusionProfiler::get()->stop())
#define FUSION_PROFILER_CREATE_SEGMENTS(segments) \
  _FP_ENABLE(FusionProfiler::get()->createSegments(segments))
#define FUSION_PROFILER_START_PARALLEL_COMPILE(segments) \
  _FP_ENABLE( \
    if ((segments > 1) && !isOptionDisabled(DisableOption::ParallelCompile)) { \
      FusionProfiler::get()->startParallelCompile(); \
    })
#define FUSION_PROFILER_STOP_PARALLEL_COMPILE(segments) \
  _FP_ENABLE( \
    if ((segments > 1) && !isOptionDisabled(DisableOption::ParallelCompile)) { \
      FusionProfiler::get()->stopParallelCompile(); \
    })
#define FUSION_PROFILER_INPUT_BYTES_ACCESSED(input_fn) \
  _FP_ENABLE(FusionProfiler::get()->inputBytesAccessed(input_fn()))
#define FUSION_PROFILER_OUTPUT_BYTES_ACCESSED(output_fn) \
  _FP_ENABLE(FusionProfiler::get()->outputBytesAccessed(output_fn()))
#define FUSION_PROFILER_PRINT \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) || \
      isDebugDumpEnabled(DebugDumpOption::FusionProfilerVerbose)) { \
    debug() << FusionProfiler::get()->profile(); \
  }

// Segment Profiling Macros
#define SEGMENT_PROFILER_START_COMPILE(device, idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).startCompile(device))
#define SEGMENT_PROFILER_STOP_COMPILE(idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).stopCompile())
#define SEGMENT_PROFILER_START_KERNEL(device, idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).startKernel(device))
#define SEGMENT_PROFILER_STOP_KERNEL(idx) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).stopKernel())
#define SEGMENT_PROFILER_INPUT_BYTES_ACCESSED(idx, input_fn) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).inputBytesAccessed(input_fn()));
#define SEGMENT_PROFILER_OUTPUT_BYTES_ACCESSED(idx, output_fn) \
  _FP_ENABLE(FusionProfiler::get()->segment(idx).outputBytesAccessed(output_fn()));

} // namespace nvfuser
