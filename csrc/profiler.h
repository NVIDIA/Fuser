// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <c10/cuda/CUDAStream.h>
#include <executor_utils.h>

namespace nvfuser {

//struct SegmentProfile {
//
//};

struct FusionProfile {
  void reset();

  float total_time;
  float host_time;
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
class Profiler : public NonCopyable {
 public:
  static void start();
  static void stop();

 private:
  Profiler() :
    timer_(at::cuda::getCurrentCUDAStream()),
    profile_() {}

  void reset();
  void print() const;

 private:
  static Profiler* singleton_;
  static std::mutex singleton_lock_;

  executor_utils::CudaKernelTimer timer_;
  FusionProfile profile_;
};

} // namespace nvfuser
