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

const char *
GetActivityKindString(
    CUpti_ActivityKind activityKind)
{
    switch (activityKind)
    {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            return "CONCURRENT_KERNEL";
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
            return "EXTERNAL_CORRELATION";
        case CUPTI_ACTIVITY_KIND_DRIVER:
            return "DRIVER";
        default:
            return "<unknown>";
    }
}

const char *
GetName(
    const char *pName)
{
    if (pName == NULL)
    {
        return "<null>";
    }
    int status = 0;
    return abi::__cxa_demangle(pName, 0, 0, &status);
    //return cppDemange(pName);
}

const char *
GetChannelType(
    CUpti_ChannelType channelType)
{
    switch (channelType)
    {
        case CUPTI_CHANNEL_TYPE_INVALID:
            return "INVALID";
        case CUPTI_CHANNEL_TYPE_COMPUTE:
            return "COMPUTE";
        case CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY:
            return "ASYNC_MEMCPY";
        default:
            return "<unknown>";
    }
}

const char *
GetExternalCorrelationKindString(
    CUpti_ExternalCorrelationKind externalCorrelationKind)
{
    switch (externalCorrelationKind)
    {
        case CUPTI_EXTERNAL_CORRELATION_KIND_INVALID:
            return "INVALID";
        case CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN:
            return "UNKNOWN";
        case CUPTI_EXTERNAL_CORRELATION_KIND_OPENACC:
            return "OPENACC";
        case CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0:
            return "CUSTOM0";
        case CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1:
            return "CUSTOM1";
        case CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2:
            return "CUSTOM2";
        default:
            return "<unknown>";
    }
}

void
PrintActivity(
    CUpti_Activity *pRecord,
    FILE *pFileHandle)
{
  CUpti_ActivityKind activityKind = pRecord->kind;

    switch (activityKind)
    {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            CUpti_ActivityKernel8 *pKernelRecord = (CUpti_ActivityKernel8 *)pRecord;

            fprintf(pFileHandle, "%s [ %llu, %llu ] duration %f ms, \"%s\", correlationId %u\n"
                    "\tgrid [ %u, %u, %u ], block [ %u, %u, %u ], cluster [ %u, %u, %u ], sharedMemory (static %u, dynamic %u)\n"
                    "\tdeviceId %u, contextId %u, streamId %u, graphId %u, graphNodeId %llu, channelId %u, channelType %s\n",
                    GetActivityKindString(pKernelRecord->kind),
                    (unsigned long long)pKernelRecord->start,
                    (unsigned long long)pKernelRecord->end,
                    //(unsigned long long)(pKernelRecord->end - pKernelRecord->start) / 1000000.0,
                    (pKernelRecord->end - pKernelRecord->start) / 1000000.0,
                    GetName(pKernelRecord->name),
                    pKernelRecord->correlationId,
                    pKernelRecord->gridX,
                    pKernelRecord->gridY,
                    pKernelRecord->gridZ,
                    pKernelRecord->blockX,
                    pKernelRecord->blockY,
                    pKernelRecord->blockZ,
                    pKernelRecord->clusterX,
                    pKernelRecord->clusterY,
                    pKernelRecord->clusterZ,
                    pKernelRecord->staticSharedMemory,
                    pKernelRecord->dynamicSharedMemory,
                    pKernelRecord->deviceId,
                    pKernelRecord->contextId,
                    pKernelRecord->streamId,
                    pKernelRecord->graphId,
                    (unsigned long long)pKernelRecord->graphNodeId,
                    pKernelRecord->channelID,
                    GetChannelType(pKernelRecord->channelType));

            break;
        }
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
        {
            CUpti_ActivityExternalCorrelation *pExternalCorrelationRecord = (CUpti_ActivityExternalCorrelation *)pRecord;

            fprintf(pFileHandle, "%s externalKind %s, correlationId %llu, externalId %llu\n",
                    GetActivityKindString(pExternalCorrelationRecord->kind),
                    GetExternalCorrelationKindString(pExternalCorrelationRecord->externalKind),
                    (long long unsigned)pExternalCorrelationRecord->correlationId,
                    (long long unsigned)pExternalCorrelationRecord->externalId);

            break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        {
            CUpti_ActivityAPI *pApiRecord = (CUpti_ActivityAPI *)pRecord;
            const char* pName = NULL;

            cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, pApiRecord->cbid, &pName);

            fprintf(pFileHandle, "%s [ %llu, %llu ] duration %llu, \"%s\", cbid %u, processId %u, threadId %u, correlationId %u\n",
                    GetActivityKindString(pApiRecord->kind),
                    (unsigned long long)pApiRecord->start,
                    (unsigned long long)pApiRecord->end,
                    (unsigned long long)(pApiRecord->end - pApiRecord->start),
                    GetName(pName),
                    pApiRecord->cbid,
                    pApiRecord->processId,
                    pApiRecord->threadId,
                    pApiRecord->correlationId);

            break;
        }
        default:
            fprintf(pFileHandle, "  <unknown>\n");
            break;
    }
}

void
PrintActivityBuffer(
    uint8_t *pBuffer,
    size_t validBytes,
    FILE *pFileHandle,
    void *pUserData)
{
    CUpti_Activity *pRecord = NULL;
    CUptiResult status = CUPTI_SUCCESS;

    do {
      status = cuptiActivityGetNextRecord(pBuffer, validBytes, &pRecord);
      if (status == CUPTI_SUCCESS) {
        std::cout << "\nKernel Profile Success!" << std::endl;
        PrintActivity(pRecord, stdout);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        std::cout << "\nKernel Profile Max Limit Reached!" << std::endl;
         break;
      }
      else {
        std::cout << "\nKernel Profile Error?" << std::endl;
        NVFUSER_CUPTI_SAFE_CALL(status);
      }
    } while (1);
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

float CudaEventTimer::time() {
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

ProfilerState CudaEventTimer::state() const {
  return state_;
}

SegmentProfiler::SegmentProfiler() :
  segment_id_(-1),
  compile_timer_(at::cuda::getCurrentCUDAStream()),
  kernel_profile_state_(ProfilerState::Ready) {}

void SegmentProfiler::setSegmentId(int64_t id) {
  segment_id_ = id;
}

void SegmentProfiler::startCompile() {
  compile_timer_.start();
}

void SegmentProfiler::stopCompile() {
  compile_timer_.stop();
}

void SegmentProfiler::startKernel() {
  NVF_CHECK(segment_id_ > -1, "Segment Id is not valid! ", segment_id_);
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

void FusionProfile::reset() {
  total_time = 0.0;
  host_time = 0.0;
  compile_time = 0.0;
  kernel_time = 0.0;

  input_bytes = 0;
  output_bytes = 0;
  total_bytes = 0;

  device_name.clear();
  device_peak_bandwidth = 0.0;

  effective_bandwidth = 0.0;
  perentage_peak_bandwidth = 0.0;
}

std::mutex Profiler::singleton_lock_;
Profiler* Profiler::singleton_ = nullptr;

Profiler::Profiler(size_t devices) :
  fusion_profilers_() {}

FusionProfiler& Profiler::get(size_t device) {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new Profiler(device + 1);
  } 
  if (device >= singleton_->fusion_profilers_.size()) {
    for (size_t i = singleton_->fusion_profilers_.size(); i < device + 1; ++i) {
      singleton_->fusion_profilers_.emplace_back(i);
    }
  }
  return singleton_->fusion_profilers_[device];
}

FusionProfiler& Profiler::get(std::optional<int8_t> device) {
  int selected_device = device.has_value() ? static_cast<int>(device.value()) : 0;
  return get(selected_device);
}

void FusionProfiler::start() {
  reset();
  fusion_timer_.start();
}

void FusionProfiler::stop() {
  fusion_timer_.stop();
  profile_.total_time = fusion_timer_.time();
  print();
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityFlushAll(0));
}

void FusionProfiler::createSegments(size_t num) {
  segments_.resize(num);
}

FusionProfiler::FusionProfiler(size_t device) :
  device_descriptor_(),
  profile_(),
  fusion_timer_(at::cuda::getCurrentCUDAStream()),
  segments_() {
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
}

void FusionProfiler::reset() {
  profile_.reset();
  fusion_timer_.reset();
  segments_.clear();
}

void FusionProfiler::print() const {
  std::cout << "\nFusion Total Time: " << profile_.total_time << " ms" << std::endl;
  //std::cout << "\nCompile Time: " << profile_.compile_time << " ms" << std::endl;
}

} // namespace nvfuser
