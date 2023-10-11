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

  void startCompile();
  void stopCompile();

  void startKernel();
  void stopKernel();

  void bytesAccessed(size_t input_bytes, size_t output_bytes);

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

  void createSegments(size_t num);
  SegmentProfiler& segment(size_t idx);

  void bytesAccessed(size_t input_bytes, size_t output_bytes);

 private:
  FusionProfiler();

  void reset();
  void print() const;

 private:
  static FusionProfiler* singleton_;
  static std::mutex singleton_lock_;

  FusionProfile profile_;
  CudaEventTimer fusion_timer_;
  std::vector<SegmentProfiler> segments_;
};

#define FP_ENABLE(code) \
  if (isDebugDumpEnabled(DebugDumpOption::FusionProfiler) \
      || isOptionEnabled(EnableOption::FusionProfiler)) { \
    code \
  }
// Fusion Level Profiling Macros
#define FUSION_PROFILER_START_PROFILE FP_ENABLE(FusionProfiler::get->start())
#define FUSION_PROFILER_STOP_PROFILE FP_ENABLE(FusionProfiler::get->stop())
#define FUSION_PROFILER_CREATE_SEGMENTS(n) \
  FP_ENABLE(FusionProfiler::get->createSegments(n))
#define FUSION_PROFILER_BYTES_ACCESSED(inputs, outputs) \
  FP_ENABLE(FusionProfiler::get->bytesAcccessed(inputs, outputs))
// Fusion Segment Profiling Macros
#define FUSION_PROFILER_SEGMENT_START_COMPILE(idx) \
  FP_ENABLE(FusionProfiler::get->segment(idx).startCompile())
#define FUSION_PROFILER_SEGMENT_STOP_COMPILE(idx) \
  FP_ENABLE(FusionProfiler::get->segment(idx).stopCompile())
#define FUSION_PROFILER_SEGMENT_START_KERNEL(idx) \
  FP_ENABLE(FusionProfiler::get->segment(idx).startKernel())
#define FUSION_PROFILER_SEGMENT_STOP_KERNEL(idx) \
  FP_ENABLE(FusionProfiler::get->segment(idx).stopKernel())
#define FUSION_PROFILER_SEGMENT_BYTES_ACCESSED(idx, inputs, outputs) \
  FP_ENABLE(FusionProfiler::get->segment(idx).bytesAccessed(inputs, outputs))

} // namespace nvfuser
