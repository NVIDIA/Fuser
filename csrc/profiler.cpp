// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <profiler.h>

namespace nvfuser {

namespace {

// Copying some code from the CUPTI samples/common code
// CUPTI buffer size 8 MB
#define BUF_SIZE (8 * 1024 * 1024)
// 8-byte alignment for the buffers
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                 \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

void buffer_requested(
    uint8_t **ppBuffer,
    size_t *pSize,
    size_t *pMaxNumRecords)
{
    uint8_t *pBuffer = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
    NVF_ERROR(pBuffer, "CUPTI Malloced buffer Pointer is null!");

    *pSize = BUF_SIZE;
    *ppBuffer = ALIGN_BUFFER(pBuffer, ALIGN_SIZE);
    *pMaxNumRecords = 0;
     std::cout << "\nBuffer requested!" << std::endl;
}

void buffer_completed(
    CUcontext context,
    uint32_t streamId,
    uint8_t *pBuffer,
    size_t size,
    size_t validSize)
{
    std::cout << "\nBuffer completed!" << std::endl;
    if (validSize > 0) {
      FusionProfiler::kernel_profiler()->
          recordKernelActivity(pBuffer, validSize);
    }

    free(pBuffer);
}

} // annonymous

// CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
KernelProfiler::KernelProfiler() :
    kernel_activity_recorded_(false)
{
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
}

KernelProfiler::~KernelProfiler() {
  // TODO: What does the cuptiActivityFlushAll 1 parameter mean?
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityFlushAll(1));
}

void KernelProfiler::start() {
  kernel_activity_recorded_ = false;
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
}

KernelProfile KernelProfiler::stop() {
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  //NVF_CHECK(kernel_activity_recorded_,
  //    "CUPTI Kernel Activity was not recorded!");
  kernel_activity_recorded_ = false;
  return KernelProfile();
}

void KernelProfiler::recordKernelActivity(uint8_t* activity_buffer, size_t activity_size) {
  NVF_ERROR(!kernel_activity_recorded_, "Kernel Activity is already recorded!");
  kernel_activity_recorded_ = true;

  std::cout << "\nRecord Kernel Activity!" << std::endl;
}

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

  singleton_->profile_started_ = true;
}

void FusionProfiler::stop() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");
  NVF_ERROR(singleton_->profile_started_,
            "FusionProfiler cannot stop a profile that is not started!");

  singleton_->profile_started_ = false;
  singleton_->profile_.total_time = singleton_->timer_.elapsed();
  singleton_->print();
}

KernelProfiler* FusionProfiler::kernel_profiler() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");
  NVF_ERROR(singleton_->profile_started_,
            "The kernel profiler is not valid when a profile is not started!");
 
  return singleton_->kernel_profiler_.get();
} 

void FusionProfiler::start_kernel() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");
  NVF_ERROR(singleton_->profile_started_,
            "FusionProfiler profile is not in progress!");

  singleton_->kernel_profiler_.get()->start();
}
void FusionProfiler::stop_kernel() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");
  NVF_ERROR(singleton_->profile_started_,

            "FusionProfiler profile is not in progress!");
  singleton_->kernel_profiler_.get()->stop();
}

void FusionProfiler::reset() {
  profile_.reset();
  profile_started_ = false;
}

void FusionProfiler::print() const {
  std::cout << "\nFusion Total Time: " << profile_.total_time << std::endl;
}

} // namespace nvfuser
