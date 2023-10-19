// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cxxabi.h>
#include <fusion_profiler.h>

namespace nvfuser {

namespace {

// Copying some code from the CUPTI samples/common code
// CUPTI buffer size 8 MB
#define BUF_SIZE (8 * 1024 * 1024)
// 8-byte alignment for the buffers
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                 \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

const char* GetName(const char *pName) {
  if (pName == nullptr) {
    return "<null>";
  }
  int status = 0;
  return abi::__cxa_demangle(pName, nullptr, nullptr, &status);
}

void PrintActivity(CUpti_Activity *pRecord, FILE *pFileHandle) {
  CUpti_ActivityKind activityKind = pRecord->kind;

  switch (activityKind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      CUpti_ActivityKernel8 *pKARecord = (CUpti_ActivityKernel8 *)pRecord;

      KernelProfile prof;
      prof.name.assign(GetName(pKARecord->name));
      prof.device = (int)pKARecord->deviceId;
      prof.stream = pKARecord->streamId;
      prof.correlation_id = pKARecord->correlationId;
      constexpr double ms_convert = 1.0 / 1000000.0;
      prof.time_ms = static_cast<double>(pKARecord->end - pKARecord->start) * ms_convert;
      prof.grid = {pKARecord->gridX, pKARecord->gridY, pKARecord->gridZ};
      prof.block = {pKARecord->blockX, pKARecord->blockY, pKARecord->blockZ};
      prof.cluster = {pKARecord->clusterX, pKARecord->clusterY, pKARecord->clusterZ};
      prof.dynamic_shared_mem = pKARecord->dynamicSharedMemory;
      prof.static_shared_mem = pKARecord->staticSharedMemory;
      prof.registers = pKARecord->registersPerThread;

      FusionProfiler::get()->recordAsyncKernelActivity(std::move(prof));

      break;
    }
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
    {
      CUpti_ActivityExternalCorrelation *pExternalCorrelationRecord = (CUpti_ActivityExternalCorrelation *)pRecord;
      FusionProfiler::get()->recordAsyncCorrIdActivity(pExternalCorrelationRecord->externalId, pExternalCorrelationRecord->correlationId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_DRIVER:
      break;
    default:
      fprintf(pFileHandle, "  <unknown>\n");
      break;
  }
}

void PrintActivityBuffer(
    uint8_t *pBuffer,
    size_t validBytes,
    FILE *pFileHandle,
    void *pUserData) {
  CUpti_Activity *pRecord = nullptr;
  CUptiResult status = CUPTI_SUCCESS;

  do {
    status = cuptiActivityGetNextRecord(pBuffer, validBytes, &pRecord);
    if (status == CUPTI_SUCCESS) {
      PrintActivity(pRecord, stdout);
    }
    else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
       break;
    }
    else {
      NVFUSER_CUPTI_SAFE_CALL(status);
    }
  } while (true);
}

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
}

void buffer_completed(
    CUcontext context,
    uint32_t streamId,
    uint8_t *pBuffer,
    size_t size,
    size_t validSize)
{
    if (validSize > 0) {
      PrintActivityBuffer(pBuffer, validSize, stdout, nullptr); 
      //FusionProfiler::kernel_profiler()->
      //    recordKernelActivity(pBuffer, validSize);
    }

    free(pBuffer);
}

const char* profiler_state2string(const ProfilerState& pstate) {
  switch (pstate) {
    case ProfilerState::Ready:
      return "Ready";
    case ProfilerState::Running:
      return "Running";
    case ProfilerState::Finished:
      return "Finished";
    case ProfilerState::Processed:
      return "Processed";
    default:
      NVF_ERROR(false, "Unexpected ProfilerState enum value!");
  }
}
 
} // annonymous

std::ostream& operator<<(std::ostream& out, const ProfilerState& pstate) {
  return out << profiler_state2string(pstate); 
}
  
CudaEventTimer::CudaEventTimer(cudaStream_t s) : 
  stream_(s),
  start_event_(),
  stop_event_(),
  time_ms_(0.0), 
  state_(ProfilerState::Ready) {
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event_));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&stop_event_));
}

CudaEventTimer::~CudaEventTimer() {
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event_));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(stop_event_));
}

void CudaEventTimer::reset() {
  time_ms_ = 0.0;
  state_ = ProfilerState::Ready;
}

void CudaEventTimer::start() {
  NVF_CHECK(state_ == ProfilerState::Ready, "ProfilerState is not Ready! ", state_);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event_, stream_));
  state_ = ProfilerState::Running;
}

void CudaEventTimer::stop() {
  NVF_CHECK(state_ == ProfilerState::Running, "ProfilerState is not Running! ", state_);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(stop_event_, stream_));
  state_ = ProfilerState::Finished;
}

double CudaEventTimer::time() {
  if (state_ == ProfilerState::Finished) {
    float tmp{0.0};
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(stop_event_));
    NVFUSER_CUDA_RT_SAFE_CALL(
        cudaEventElapsedTime(&tmp, start_event_, stop_event_));
    time_ms_ = static_cast<double>(tmp);
    state_ = ProfilerState::Processed;
  } else {
    NVF_CHECK((state_ == ProfilerState::Processed) || (state_ == ProfilerState::Ready), "ProfilerState is not Processed or Ready! ", state_);
  }
  return time_ms_;
}

ProfilerState CudaEventTimer::state() const {
  return state_;
}

void DeviceDescriptor::generate(int _device) {
  device = static_cast<int>(_device);
  name.reserve(100);
  NVFUSER_CUDA_SAFE_CALL(
      cuDeviceGetName(name.data(), 100, device));
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &bus_width,
      CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
      device));
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &memory_clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));

    // Peak bandwidth calculation:
    // Bus width is given in bits, so dividing by 8 converts to bytes.
    // Clock is given in kHz. 1 GB = 1e9 bytes (don't report GiB = 1024^3 bytes)
    // A factor of 2 is multiplied to account for double data rate (DDR):
    // (clock in kHz * width in bits) * (1000 Hz / kHz) * (1 GB / 8e9 bits) * 2
    // factor = 2.5e-7
  peak_bandwidth_gbs = 2.5e-7 * static_cast<double>(memory_clock) * static_cast<double>(bus_width);
}

SegmentProfiler::SegmentProfiler(uint32_t id) :
  device_(-1),
  segment_id_(id),
  compile_timer_(at::cuda::getCurrentCUDAStream()),
  input_bytes_(0),
  output_bytes_(0),
  kernel_profile_state_(ProfilerState::Ready) {}

void SegmentProfiler::startCompile(int device) {
  device_ = device;
  compile_timer_.start();
}

void SegmentProfiler::stopCompile() {
  compile_timer_.stop();
}

void SegmentProfiler::startKernel(int device) {
  device_ = device;
  //NVF_CHECK(segment_id_ > -1, "Segment Id is not valid! ", segment_id_);
  NVF_CHECK(kernel_profile_state_ == ProfilerState::Ready, "ProfilerState is not Ready!", kernel_profile_state_);
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(segment_id_)));
  kernel_profile_state_ = ProfilerState::Running;
}

void SegmentProfiler::stopKernel() {
  NVF_CHECK(kernel_profile_state_ == ProfilerState::Running, "ProfilerState is not Running!", kernel_profile_state_);
  uint64_t corr_id = 0;
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &corr_id));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  NVF_CHECK(corr_id == static_cast<uint64_t>(segment_id_), "Correlation Id does not match segment id! Corr Id: ", corr_id, " Segment Id: ", segment_id_);
  kernel_profile_state_ = ProfilerState::Finished;
}

void SegmentProfiler::inputBytesAccessed(int64_t bytes) {
  input_bytes_ = bytes;
}

void SegmentProfiler::outputBytesAccessed(int64_t bytes) {
  output_bytes_ = bytes;
}

uint32_t SegmentProfiler::segmentId() const {
  return segment_id_;
}

void FusionProfile::reset() {
  time_ms = 0.0;
  host_time_ms = 0.0;
  compile_time_ms = 0.0;
  kernel_time_ms = 0.0;
  
  input_bytes = 0;
  output_bytes = 0;

  effective_bandwidth_gbs = 0.0;
  percentage_peak_bandwidth = 0.0;

  kernel_profiles.clear();
}

std::ostream& operator<<(std::ostream&, const FusionProfile&) {
  constexpr std::array<const char*,  23> column_strs{"Fus#", "NumSegs",
      "Time(ms)", "HstTime(ms)", "CmpTime(ms)", "KerTime(ms)", "EffBw(GB/s)",
      "%PeakBw", "Seg#", "S-KerName", "S-Dev", "S-Str", "S-KerTime(ms)",
      "S-CmpTime(ms)", "S-EffBw(GB/s)", "S-%PeakBw", "S-Grid",
      "S-Block", "S-Cluster", "S-Smem[Dyn,Stat]", "S-Regs", "S-In(MB)",
      "S-Out(MB)"};

  

  

}

std::mutex FusionProfiler::singleton_lock_;
FusionProfiler* FusionProfiler::singleton_ = nullptr;

FusionProfiler::FusionProfiler() :
  fusion_id_(0),
  profile_(),
  fusion_timer_(at::cuda::getCurrentCUDAStream()),
  segments_(),
  device_descriptors_(),
  kernel_profiles_(),
  corrid_2_segid_() {
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
}

void FusionProfiler::reset() {
  ++fusion_id_; 

  profile_.reset();
  fusion_timer_.reset();
  segments_.clear();
  kernel_profiles_.clear();
  corrid_2_segid_.clear();
  segid_2_idx_.clear();
}

FusionProfiler* FusionProfiler::get() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new FusionProfiler();
  } 
  return singleton_;
}

void FusionProfiler::createSegments(size_t num) {
  segments_.reserve(num);
  uint32_t hash_preamble = (0xffff & fusion_id_) << 15;
  for (size_t i = 0; i < num; ++i) {
    uint32_t id = hash_preamble | (0xffff & static_cast<uint32_t>(i));
    segid_2_idx_[id] = segments_.size();
    segments_.emplace_back(id);
  }
}
SegmentProfiler& FusionProfiler::segment(size_t idx) {
  return segments_.at(idx);
}

void FusionProfiler::start() {
  reset();
  fusion_timer_.start();
}

void FusionProfiler::stop() {
  fusion_timer_.stop();
  profile_.time_ms = fusion_timer_.time();
  kernel_profiles_.reserve(segments_.size());
  profile_.kernel_profiles.resize(segments_.size());
  
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityFlushAll(0));

  NVF_CHECK(kernel_profiles_.size() == segments_.size(), "All of the kernel profiles have not been recorded!");
 
  double compile_time_ms = 0.0;
  double kernel_time_ms = 0.0;
  constexpr double mb_divider = 1.0 / 1.0e6;
  for(auto &kp : kernel_profiles_) {
    auto corr_id = kp.correlation_id;
    NVF_CHECK(kp.device >= 0, "Device Descriptor index is not valid! ", kp.device);
    if ((size_t)kp.device >= device_descriptors_.size()) {
      device_descriptors_.resize(kp.device + 1);
    }
    NVF_CHECK((size_t)kp.device < device_descriptors_.size(), "Device idx is beyond size of Device Descriptors! ", kp.device);
    if (device_descriptors_[kp.device].device != kp.device) {
      device_descriptors_[kp.device].generate(kp.device);
    }
    kp.device_name = device_descriptors_[kp.device].name;
    kp.peak_bandwidth_gbs = device_descriptors_[kp.device].peak_bandwidth_gbs;
    NVF_CHECK(corrid_2_segid_.count(corr_id) > 0, "Correlation Id is not found in corrid -> segid hashmap! ", corr_id);
    auto seg_id = corrid_2_segid_[corr_id];
    NVF_CHECK(segid_2_idx_.count(seg_id) > 0, "Seg id is not found in seg id -> idx hashmap! ", seg_id);
    auto kp_idx = segid_2_idx_[seg_id];
    NVF_CHECK(kp_idx < profile_.kernel_profiles.size(), "Index is out of range of Kernel Profiles size! ", kp_idx, " ", profile_.kernel_profiles.size());
    NVF_CHECK(segments_[kp_idx].state() == ProfilerState::Finished, "SegmentProfiler ProfilerState is not Finished!", segments_[kp_idx].state());
    kp.input_bytes = segments_.at(kp_idx).inputBytes();
    kp.output_bytes = segments_.at(kp_idx).outputBytes();
    kp.effective_bandwidth_gbs = (double)(kp.input_bytes + kp.output_bytes) / kp.time_ms * mb_divider;
    kp.percentage_peak_bandwidth = kp.effective_bandwidth_gbs / kp.peak_bandwidth_gbs * 100.0;
    kp.compile_time_ms = segments_.at(kp_idx).compileTime();

    compile_time_ms += kp.compile_time_ms;
    kernel_time_ms += kp.time_ms;
    profile_.kernel_profiles[kp_idx] = std::move(kp);
  }

  int device = segments_[0].device();
  for (auto &seg : segments_) {
    NVF_CHECK(seg.device() == device, "All Segment profiles must be on the same device!");
  }
  profile_.fusion_id = fusion_id_;
  profile_.host_time_ms = profile_.time_ms - compile_time_ms - kernel_time_ms;
  profile_.compile_time_ms = compile_time_ms;
  profile_.kernel_time_ms = compile_time_ms;
  profile_.effective_bandwidth_gbs = (double)(profile_.input_bytes + profile_.output_bytes) / profile_.time_ms * mb_divider;
  profile_.percentage_peak_bandwidth = profile_.effective_bandwidth_gbs / device_descriptors_[segments_[0].device()].peak_bandwidth_gbs * 100.0;
}
  
void FusionProfiler::inputBytesAccessed(int64_t bytes) {
  profile_.input_bytes = bytes;
}

void FusionProfiler::outputBytesAccessed(int64_t bytes) {
  profile_.output_bytes = bytes;
}

const FusionProfile& FusionProfiler::profile() const {
  return profile_;
}

void FusionProfiler::recordAsyncCorrIdActivity(uint32_t seg_id, uint32_t corr_id) {
  NVF_CHECK(corrid_2_segid_.count(corr_id) == 0, "Segment Correlation Activity asociated with this correlation id already exists! ", corr_id);
  corrid_2_segid_[corr_id] = seg_id;
}

void FusionProfiler::recordAsyncKernelActivity(KernelProfile prof) {
  kernel_profiles_.emplace_back(std::move(prof));
}

} // namespace nvfuser
