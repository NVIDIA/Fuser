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

} // annonymous

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

std::mutex FusionProfiler::singleton_lock_;
FusionProfiler* FusionProfiler::singleton_ = nullptr;

void FusionProfiler::start() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new FusionProfiler();
  } else {
    singleton_->reset();
  }
  singleton_->fusion_timer_.reset();
  singleton_->fusion_timer_.start();
  singleton_->compile_timer_.reset();
  singleton_->fusion_profile_recorded_ = true;
}

void FusionProfiler::stop() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");

  singleton_->fusion_profile_started_ = false;
  singleton_->fusion_timer_.stop();
  singleton_->profile_.total_time = singleton_->fusion_timer_.time();
  singleton_->profile_.compile_time = singleton_->compile_timer_.time();
  singleton_->print();
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityFlushAll(0));
}

void FusionProfiler::start_kernel_compile() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  std::cout << "Start Kernel Compile!" << std::endl;
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");

  singleton_->compile_timer_.start();
  singleton_->kernel_compile_recorded_ = true;
}

void FusionProfiler::stop_kernel_compile() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  std::cout << "Stop Kernel Compile!" << std::endl;
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");

  singleton_->compile_timer_.stop();
  singleton_->kernel_compile_started_ = false;
}

void FusionProfiler::start_kernel() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, static_cast<uint64_t>(9999)));
  singleton_->kernel_profile_started_ = true;
}
void FusionProfiler::stop_kernel() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  NVF_ERROR(singleton_ != nullptr,
            "FusionProfiler singleton is unexpectedly null!");
  uint64_t id = 0;
  NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  std::cout << "\nPopped External Correlation Id? " << id << std::endl;
  singleton_->kernel_profile_started_ = false;
}

FusionProfiler::FusionProfiler() :
  fusion_timer_(at::cuda::getCurrentCUDAStream()),
  compile_timer_(at::cuda::getCurrentCUDAStream()),
  profile_(),
  fusion_profile_started_(false),
  kernel_compile_started_(false),
  kernel_profile_started_(false) {
  NVFUSER_CUPTI_SAFE_CALL(
      cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
}

void FusionProfiler::reset() {
  profile_.reset();
  fusion_profile_started_ = false;
  kernel_profile_started_ = false;
}

void FusionProfiler::print() const {
  std::cout << "\nFusion Total Time: " << profile_.total_time << " ms" << std::endl;
  std::cout << "\nCompile Time: " << profile_.compile_time << " ms" << std::endl;
}

} // namespace nvfuser
