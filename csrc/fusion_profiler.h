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

/*struct KernelProfile {
  float kernel_time;
};*/

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

struct DeviceDescriptor{
  int device;
  std::string name;
  int bus_width;
  int memory_clock; 
};

class FusionProfiler {
 public:
  FusionProfiler(size_t device);
  
  void createSegments(size_t num);
  SegmentProfiler& segment(size_t idx);

  void start();
  void stop();

  void bytesAccessed(size_t input_bytes, size_t output_bytes);

 private:
  void reset();
  void print() const;

 private:
  DeviceDescriptor device_descriptor_;
  FusionProfile profile_;
  CudaEventTimer fusion_timer_;
  std::vector<SegmentProfiler> segments_;
};

// Singleton
class Profiler : public NonCopyable {
 public:
  static FusionProfiler& get(size_t device);
  static FusionProfiler& get(std::optional<int8_t> device);

 private:
  Profiler(size_t devices);
 
 private:
  static Profiler* singleton_;
  static std::mutex singleton_lock_;

  std::vector<FusionProfiler> fusion_profilers_;
};

#define _FP_ENABLE(code) \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    code; \
  }
// Fusion Level Profiling Macros
#define FUSION_PROFILER_START_PROFILE(device) \
  _FP_ENABLE(Profiler::get(device).start())
#define FUSION_PROFILER_STOP_PROFILE(device) \
  _FP_ENABLE(Profiler::get(device).stop())
#define FUSION_PROFILER_CREATE_SEGMENTS(device, segments) \
  _FP_ENABLE(Profiler::get(device).createSegments(segments))
#define FUSION_PROFILER_BYTES_ACCESSED(device, inputs, outputs) \
  _FP_ENABLE(Profiler::get(device).bytesAcccessed(inputs, outputs))
// Fusion Segment Profiling Macros

#define FUSION_PROFILER_SEGMENT_START_COMPILE(device, idx) \
  _FP_ENABLE(Profiler::get(device).segment(idx).startCompile())
#define FUSION_PROFILER_SEGMENT_STOP_COMPILE(device, idx) \
  _FP_ENABLE(Profiler::get(device).segment(idx).stopCompile())
#define FUSION_PROFILER_SEGMENT_START_KERNEL(device, idx) \
  _FP_ENABLE(Profiler::get(device).segment(idx).startKernel())
#define FUSION_PROFILER_SEGMENT_STOP_KERNEL(device, idx) \
  _FP_ENABLE(Profiler::get(device).segment(idx).stopKernel())
#define FUSION_PROFILER_SEGMENT_BYTES_ACCESSED(device, idx, inputs, outputs) \
  _FP_ENABLE(Profiler::get(device).segment(idx).bytesAccessed(inputs, outputs))

} // namespace nvfuser
