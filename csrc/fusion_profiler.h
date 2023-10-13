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

struct KernelProfile {
  std::string name;
  uint32_t device = 0;
  uint32_t stream = 0;
  uint32_t correlation_id = 0;

  double time_ms;

  int32_t grid_x = 1;
  int32_t grid_y = 1;
  int32_t grid_z = 1;

  int32_t block_x = 1;
  int32_t block_y = 1;
  int32_t block_z = 1;

  uint32_t cluster_x = 1;
  uint32_t cluster_y = 1;
  uint32_t cluster_z = 1;

  int32_t dynamic_shared_mem = 0;
  int32_t static_shared_mem = 0;
  uint32_t registers = 0;
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
  DeviceDescriptor(size_t device);

  int device;
  std::string name;
  int bus_width;
  int memory_clock; 

  double peak_bandwidth;
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
  std::unordered_map<uint32_t, KernelProfile> corrid_2_kernel_profile_;
};

// Singleton
class Profiler : public NonCopyable {
 public:
  static FusionProfiler& getProfiler(size_t device);
  static FusionProfiler& getProfiler(std::optional<int8_t> device);
 
  // Static Methods to capture Asynchronous CUPTI activity
  // Collects CUPTI activity to map Profiler Id -> CUPTI Correlation Id
  // Segment ID -> Correlation ID
  static void recordAsyncCorrIdActivity(uint32_t seg_id, uint32_t corr_id);
  // Collects CUPTI Kernel Activity
  // Segment ID -> KernelProfile
  static void recordAsyncKernelActivity(
      size_t device, uint32_t corr_id, KernelProfile profile);

 private:
  Profiler(size_t devices);
 
 private:
  static Profiler* singleton_;
  static std::mutex singleton_lock_;

  std::vector<FusionProfiler> fusion_profilers_;
  std::unordered_map<uint32_t, uint32_t> segid_2_corrid_;
};

#define _FP_ENABLE(code) \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    code; \
  }
// Fusion Profiling Macros
#define FUSION_PROFILER_START_PROFILE(device) \
  _FP_ENABLE(Profiler::getProfiler(device).start())
#define FUSION_PROFILER_STOP_PROFILE(device) \
  _FP_ENABLE(Profiler::getProfiler(device).stop())
#define FUSION_PROFILER_CREATE_SEGMENTS(device, segments) \
  _FP_ENABLE(Profiler::getProfiler(device).createSegments(segments))
#define FUSION_PROFILER_BYTES_ACCESSED(device, inputs, outputs) \
  _FP_ENABLE(Profiler::getProfiler(device).bytesAcccessed(inputs, outputs))

// Segment Profiling Macros
#define SEGMENT_PROFILER_START_COMPILE(device, idx) \
  _FP_ENABLE(Profiler::getProfiler(device).segment(idx).startCompile())
#define SEGMENT_PROFILER_STOP_COMPILE(device, idx) \
  _FP_ENABLE(Profiler::getProfiler(device).segment(idx).stopCompile())
#define SEGMENT_PROFILER_START_KERNEL(device, idx) \
  _FP_ENABLE(Profiler::getProfiler(device).segment(idx).startKernel())
#define SEGMENT_PROFILER_STOP_KERNEL(device, idx) \
  _FP_ENABLE(Profiler::getProfiler(device).segment(idx).stopKernel())
#define SEGMENT_PROFILER_BYTES_ACCESSED(device, idx, inputs, outputs) \
  _FP_ENABLE(Profiler::getProfiler(device).segment(idx).bytesAccessed(inputs, outputs))

} // namespace nvfuser
