// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cupti.h>
#include <fusion_profiler.h>
#include <iomanip>

namespace nvfuser {

namespace {

//! The following CUPTI code is adapted from the CUTPI samples/common and
//! sample/activity_trace_async examples shipped with CUPTI.
//! \note The CUPTI usage should be isolated to this file!

#define NVFUSER_CUPTI_SAFE_CALL(x)                     \
  do {                                                 \
    CUptiResult _status = x;                           \
    if (_status != CUPTI_SUCCESS) {                    \
      const char* errorString;                         \
      cuptiGetResultString(_status, &errorString);     \
      fprintf(                                         \
          stderr,                                      \
          "%s:%d: Error: %s failed with error: %s.\n", \
          __FILE__,                                    \
          __LINE__,                                    \
          #x,                                          \
          errorString);                                \
      exit(EXIT_FAILURE);                              \
    }                                                  \
  } while (0)

void record_cupti_activity(CUpti_Activity* pRecord, FILE* pFileHandle) {
  CUpti_ActivityKind activityKind = pRecord->kind;

  switch (activityKind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      CUpti_ActivityKernel8* pKARecord = (CUpti_ActivityKernel8*)pRecord;

      KernelProfile prof;
      prof.name.assign(demangle(pKARecord->name));

      size_t start = prof.name.find("nvfuser");
      if (start != std::string::npos) {
        size_t end = prof.name.find('(', start);
        prof.name = prof.name.substr(start, end - start);
      }
      prof.device = (int)pKARecord->deviceId;
      prof.stream = pKARecord->streamId;
      prof.correlation_id = pKARecord->correlationId;
      constexpr double ms_convert = 1.0 / 1000000.0;
      prof.time_ms =
          static_cast<double>(pKARecord->end - pKARecord->start) * ms_convert;
      prof.grid = {pKARecord->gridX, pKARecord->gridY, pKARecord->gridZ};
      prof.block = {pKARecord->blockX, pKARecord->blockY, pKARecord->blockZ};
      prof.cluster = {
          pKARecord->clusterX, pKARecord->clusterY, pKARecord->clusterZ};
      prof.dynamic_shared_mem = pKARecord->dynamicSharedMemory;
      prof.static_shared_mem = pKARecord->staticSharedMemory;
      prof.registers = pKARecord->registersPerThread;

      FusionProfiler::recordAsyncKernelActivity(std::move(prof));

      break;
    }
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
      CUpti_ActivityExternalCorrelation* pExternalCorrelationRecord =
          (CUpti_ActivityExternalCorrelation*)pRecord;
      FusionProfiler::recordAsyncCorrIdActivity(
          pExternalCorrelationRecord->externalId,
          pExternalCorrelationRecord->correlationId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_DRIVER:
      // NOTE: Driver activity is enabled in order to capture ext correlation
      // records but the driver activity records are not of interest to record.
      // Therefore, we read and skip over them from the buffer.
      break;
    default:
      fprintf(pFileHandle, "  <unknown>\n");
      break;
  }
}

// This is the function that reads and processes each CUPTI Activity Record
// recorded in a buffer.  The specific types of records are uniquely processed
// in a separate function called below: record_cupti_activity.
void record_cupti_activity_buffer(
    uint8_t* pBuffer,
    size_t validBytes,
    FILE* pFileHandle,
    void* pUserData) {
  CUpti_Activity* pRecord = nullptr;
  CUptiResult status = CUPTI_SUCCESS;

  // This is an arbitrary record limit to make sure we do not get into an
  // infinite loop;
  const size_t max_records = 100;
  bool found_max_limit = false;

  for (size_t i = 0; i < max_records; ++i) {
    status = cuptiActivityGetNextRecord(pBuffer, validBytes, &pRecord);
    if (status == CUPTI_SUCCESS) {
      // Processes a valid CUPTI Activty record and records it with the
      // fusion profiling infrastructure if the record is of interest.
      record_cupti_activity(pRecord, stdout);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // This case is hit every time an activity buffer is read and you reach
      // the end of the buffer activity to consume. Therefore, it is the
      // intended way to break out of the loop!  The name is miss leading as
      // you aren't reaching a Max Limit, necessarily!
      found_max_limit = true;
      break;
    } else {
      NVFUSER_CUPTI_SAFE_CALL(status);
    }
  }
  NVF_ERROR(
      found_max_limit,
      "The CUPTI buffer has more than ",
      max_records,
      " record! Is that expected?");
}

// The functions cupti_buffer_requested and cupti_buffer_completed are
// registered with the CUPTI Activiy Record Callback API:
// cuptiActivityRegisterCallbacks.  Each of the functions APIs is prescribed
// by CUPTI and you can find their signatured definitions in the CUPT docs.

void cupti_buffer_requested(
    uint8_t** ppBuffer,
    size_t* pSize,
    size_t* pMaxNumRecords) {
  uint8_t* pBuffer = FusionProfiler::cuptiBufferPtr();
  NVF_ERROR(pBuffer, "CUPTI Activity Record buffer pointer is null!");
  const size_t align_size = 8;
  NVF_ERROR(
      ((uintptr_t)pBuffer & (align_size - 1)) == 0,
      "The CUPTI Activity Record buffer needs to be 8 byte aligned!");

  *ppBuffer = pBuffer;
  *pSize = FusionProfiler::cupti_activity_buffer_size;
  // NOTE: The Max Number of records limits the number of records that can be
  // recorded in the activity buffer.  When set to 0, it puts as many records
  // as it can which effectively disables a max limit.
  *pMaxNumRecords = 0;
}

void cupti_buffer_completed(
    CUcontext context,
    uint32_t streamId,
    uint8_t* pBuffer,
    size_t size,
    size_t validSize) {
  if (validSize > 0) {
    record_cupti_activity_buffer(pBuffer, validSize, stdout, nullptr);
  }
}

//! A local utility function to give ProfilerState enum state strings
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

} // namespace

std::ostream& operator<<(std::ostream& out, const ProfilerState& pstate) {
  return out << profiler_state2string(pstate);
}

CudaEventTimer::CudaEventTimer(cudaStream_t s)
    : stream_(s),
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
  NVF_CHECK(
      state_ == ProfilerState::Ready, "ProfilerState is not Ready! ", state_);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event_, stream_));
  state_ = ProfilerState::Running;
}

void CudaEventTimer::stop() {
  NVF_CHECK(
      state_ == ProfilerState::Running,
      "ProfilerState is not Running! ",
      state_);
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
    NVF_CHECK(
        (state_ == ProfilerState::Processed) ||
            (state_ == ProfilerState::Ready),
        "ProfilerState is not Processed or Ready! ",
        state_);
  }
  return time_ms_;
}

ProfilerState CudaEventTimer::state() const {
  return state_;
}

HostTimer::HostTimer()
    : start_event_(),
      stop_event_(),
      time_ms_(0.0),
      state_(ProfilerState::Ready) {}

void HostTimer::reset() {
  time_ms_ = 0.0;
  state_ = ProfilerState::Ready;
}

void HostTimer::start() {
  NVF_CHECK(
      state_ == ProfilerState::Ready, "ProfilerState is not Ready! ", state_);
  start_event_ = Clock::now();
  state_ = ProfilerState::Running;
}

void HostTimer::stop() {
  NVF_CHECK(
      state_ == ProfilerState::Running,
      "ProfilerState is not Running! ",
      state_);
  stop_event_ = Clock::now();
  state_ = ProfilerState::Finished;
}

double HostTimer::time() {
  if (state_ == ProfilerState::Finished) {
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            stop_event_ - start_event_)
            .count();
    time_ms_ = elapsed_seconds * 1000.0;
    state_ = ProfilerState::Processed;
  } else {
    NVF_CHECK(
        (state_ == ProfilerState::Processed) ||
            (state_ == ProfilerState::Ready),
        "ProfilerState is not Processed or Ready! ",
        state_);
  }
  return time_ms_;
}

ProfilerState HostTimer::state() const {
  return state_;
}

void DeviceDescriptor::generate(DeviceDescriptor& desc, int device) {
  desc.device = device;
  desc.name.reserve(100);
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetName(desc.name.data(), 100, device));
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &desc.bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &desc.memory_clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));

  // Peak bandwidth calculation:
  // Bus width is given in bits, so dividing by 8 converts to bytes.
  // Clock is given in kHz. 1 GB = 1e9 bytes (don't report GiB = 1024^3 bytes)
  // A factor of 2 is multiplied to account for double data rate (DDR):
  // (clock in kHz * width in bits) * (1000 Hz / kHz) * (1 GB / 8e9 bits) * 2
  // factor = 2.5e-7
  constexpr double static_comp = 2.0 * /*2x for DDR*/
      1000.0 * /*kHz->Hz*/
      (1.0 / 8.0) * /*bits->bytes*/
      (1.0 / 1.0e9); /*Bytes->Gigabytes*/
  desc.peak_bandwidth_gbs = static_comp *
      static_cast<double>(desc.memory_clock) *
      static_cast<double>(desc.bus_width);
}

SegmentProfiler::SegmentProfiler(uint32_t id, bool cupti_disabled)
    : cupti_disabled_(cupti_disabled),
      device_(-1),
      segment_id_(id),
      compile_timer_(),
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
  NVF_CHECK(
      kernel_profile_state_ == ProfilerState::Ready,
      "ProfilerState is not Ready!",
      kernel_profile_state_);
  if (!cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN,
        static_cast<uint64_t>(segment_id_)));
  }
  kernel_profile_state_ = ProfilerState::Running;
}

void SegmentProfiler::stopKernel() {
  NVF_CHECK(
      kernel_profile_state_ == ProfilerState::Running,
      "ProfilerState is not Running!",
      kernel_profile_state_);
  uint64_t corr_id = 0;
  if (!cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &corr_id));
    NVF_CHECK(
        corr_id == static_cast<uint64_t>(segment_id_),
        "Correlation Id does not match segment id! Corr Id: ",
        corr_id,
        " Segment Id: ",
        segment_id_);
  }
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
  fusion_id = -1;
  segments = 0;

  cuda_evt_time_ms = 0.0;
  host_time_ms = 0.0;
  compile_time_ms = 0.0;
  kernel_time_ms = 0.0;

  input_bytes = 0;
  output_bytes = 0;

  effective_bandwidth_gbs = 0.0;
  percentage_peak_bandwidth = 0.0;

  kernel_profiles.clear();
}

std::array<const char*, 25> column_strs{
    "Fus#",      "NSegs",       "CuEvtTm(ms)",  "HstTm(ms)",
    "CmpTm(ms)", "KerTm(ms)",   "EffBw(GB/s)",  "%PeakBw",
    "S-Seg#",    "S-KerTm(ms)", "S-CmpTm(ms)",  "S-EffBw(GB/s)",
    "S-%PeakBw", "S-In(MB)",    "S-Out(MB)",    "S-Smem[Dyn,Stat]",
    "S-Regs",    "S-Grid",      "S-Block",      "S-Cluster",
    "S-Dev",     "S-Stm",       "S-PkBw(GB/s)", "S-DeviceName",
    "S-KerName"};

std::ostream& operator<<(std::ostream& os, const FusionProfile& fp) {
  if (fp.fusion_id == 0) {
    os << std::left << std::setw(5) << std::get<0>(column_strs) << " "
       << std::setw(5) << std::get<1>(column_strs) << " " << std::setw(11)
       << std::get<2>(column_strs) << " " << std::setw(9)
       << std::get<3>(column_strs) << " " << std::setw(9)
       << std::get<4>(column_strs);

    if (!fp.kernel_profiles.empty()) {
      os << " " << std::setw(9) << std::get<5>(column_strs) << " "
         << std::setw(11) << std::get<6>(column_strs) << " " << std::setw(9)
         << std::get<7>(column_strs);

      os << " " << std::setw(6) << std::get<8>(column_strs) << " "
         << std::setw(9) << std::get<9>(column_strs);

      if (fp.verbose) {
        os << " " << std::setw(11) << std::get<10>(column_strs);
      }

      os << " " << std::setw(13) << std::get<11>(column_strs) << " "
         << std::setw(9) << std::get<12>(column_strs) << " " << std::setw(9)
         << std::get<13>(column_strs) << " " << std::setw(9)
         << std::get<14>(column_strs) << " " << std::setw(16)
         << std::get<15>(column_strs) << " " << std::setw(6)
         << std::get<16>(column_strs) << " " << std::setw(16)
         << std::get<17>(column_strs) << " " << std::setw(16)
         << std::get<18>(column_strs);

      if (fp.verbose) {
        os << " " << std::setw(16) << std::get<19>(column_strs) << " "
           << std::setw(5) << std::get<20>(column_strs) << " " << std::setw(5)
           << std::get<21>(column_strs) << " " << std::setw(12)
           << std::get<22>(column_strs) << " " << std::setw(20)
           << std::get<23>(column_strs);
      }

      os << " " << std::setw(20) << std::get<24>(column_strs);
    }

    os << std::endl;
  }

  if (fp.kernel_profiles.empty()) {
    os << std::setfill(' ') << std::right << std::fixed << std::setw(5)
       << fp.fusion_id << " " << std::setw(5) << fp.segments << " "
       << std::setw(11) << std::setprecision(3) << fp.cuda_evt_time_ms << " "
       << std::setw(9) << std::setprecision(3) << fp.host_time_ms << " "
       << std::setw(9) << std::setprecision(3) << fp.compile_time_ms
       << std::endl;
  } else {
    bool first_prof = true;
    int idx = 0;
    for (auto& kp : fp.kernel_profiles) {
      if (first_prof) {
        os << std::setfill(' ') << std::right << std::fixed << std::setw(5)
           << fp.fusion_id << " " << std::setw(5) << fp.segments << " "
           << std::setw(11) << std::setprecision(3) << fp.cuda_evt_time_ms
           << " " << std::setw(9) << std::setprecision(3) << fp.host_time_ms
           << " " << std::setw(9) << std::setprecision(3) << fp.compile_time_ms
           << " " << std::setw(9) << std::setprecision(3) << fp.kernel_time_ms
           << " " << std::setw(11) << std::setprecision(2)
           << fp.effective_bandwidth_gbs << " " << std::setw(9)
           << std::setprecision(2) << fp.percentage_peak_bandwidth;
        first_prof = false;
      } else {
        os << std::setfill(' ') << std::right << std::fixed << std::setw(5)
           << "-"
           << " " << std::setw(5) << "-"
           << " " << std::setw(11) << "-"
           << " " << std::setw(9) << "-"
           << " " << std::setw(9) << "-"
           << " " << std::setw(9) << "-"
           << " " << std::setw(11) << "-"
           << " " << std::setw(9) << "-";
      }
      std::stringstream grid;
      grid << "[" << std::get<0>(kp.grid) << ", " << std::get<1>(kp.grid)
           << ", " << std::get<2>(kp.grid) << "]";
      std::stringstream block;
      block << "[" << std::get<0>(kp.block) << ", " << std::get<1>(kp.block)
            << ", " << std::get<2>(kp.block) << "]";
      std::stringstream cluster;
      cluster << "[" << std::get<0>(kp.cluster) << ", "
              << std::get<1>(kp.cluster) << ", " << std::get<2>(kp.cluster)
              << "]";
      std::stringstream smem;
      smem << "[" << kp.dynamic_shared_mem << ", " << kp.static_shared_mem
           << "]";
      os << std::setfill(' ') << std::right << std::fixed << " " << std::setw(6)
         << idx << " " << std::setw(11) << std::setprecision(3) << kp.time_ms;

      if (fp.verbose) {
        os << " " << std::setw(11) << std::setprecision(3)
           << kp.compile_time_ms;
      }

      os << " " << std::setw(13) << std::setprecision(2)
         << kp.effective_bandwidth_gbs << " " << std::setw(9)
         << std::setprecision(2) << kp.percentage_peak_bandwidth << " "
         << std::setw(9) << std::setprecision(3)
         << ((double)kp.input_bytes / 1000000.0) << " " << std::setw(9)
         << std::setprecision(3) << ((double)kp.output_bytes / 1000000.0) << " "
         << std::setw(16) << smem.str() << " " << std::setw(6) << kp.registers
         << " " << std::setw(16) << grid.str() << " " << std::setw(16)
         << block.str();
      if (fp.verbose) {
        os << " " << std::setw(16) << cluster.str() << " " << std::setw(5)
           << kp.device << " " << std::setw(5) << kp.stream << " "
           << std::setw(12) << std::setprecision(2) << kp.peak_bandwidth_gbs
           << " " << std::setw(20) << kp.device_name;
      }
      os << " " << std::setw(20) << kp.name;
      os << std::endl;
      ++idx;
    }
  }
  return os;
}

FusionProfiler::FusionProfiler()
    : cupti_disabled_(false),
      cupti_buffer_(FusionProfiler::cupti_activity_buffer_size),
      state_(ProfilerState::Ready),
      fusion_id_(-1),
      profile_(),
      fusion_timer_(at::cuda::getCurrentCUDAStream()),
      host_timer_(),
      compile_timer_(),
      segments_(),
      device_descriptors_(),
      kernel_profiles_(),
      corrid_2_segid_() {
  if (!cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityRegisterCallbacks(
        cupti_buffer_requested, cupti_buffer_completed));
  }
}

FusionProfiler* FusionProfiler::get() {
  static std::mutex singleton_lock;
  static FusionProfiler* singleton = nullptr;

  std::lock_guard<std::mutex> guard(singleton_lock);
  if (singleton == nullptr) {
    singleton = new FusionProfiler();
  }
  return singleton;
}

void FusionProfiler::reset() {
  FusionProfiler* fp = get();
  fp->state_ = ProfilerState::Ready;
  ++(fp->fusion_id_);

  fp->profile_.reset();
  fp->fusion_timer_.reset();
  fp->host_timer_.reset();
  fp->compile_timer_.reset();
  fp->segments_.clear();
  fp->kernel_profiles_.clear();
  fp->corrid_2_segid_.clear();
}

ProfilerState FusionProfiler::state() {
  return get()->state_;
}

void FusionProfiler::createSegments(size_t num) {
  FusionProfiler* fp = get();
  NVF_CHECK(
      state() == ProfilerState::Running,
      "FusionProfiler state is not Running!",
      state());
  fp->segments_.reserve(num);
  for (uint32_t i = 0; i < num; ++i) {
    fp->segments_.emplace_back(i, fp->cupti_disabled_);
  }
}
SegmentProfiler& FusionProfiler::segment(size_t idx) {
  return get()->segments_.at(idx);
}

void FusionProfiler::start(bool cupti_disable) {
  FusionProfiler* fp = get();
  fp->cupti_disabled_ = cupti_disable;
  reset();
  if (!fp->cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    NVFUSER_CUPTI_SAFE_CALL(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  }
  cudaDeviceSynchronize();
  fp->fusion_timer_.start();
  fp->host_timer_.start();
  fp->state_ = ProfilerState::Running;
}

void FusionProfiler::stop() {
  FusionProfiler* fp = get();
  NVF_CHECK(
      state() == ProfilerState::Running,
      "FusionProfiler state is not Running!",
      state());
  fp->host_timer_.stop();
  fp->fusion_timer_.stop();
  fp->state_ = ProfilerState::Finished;
  auto& fprof = fp->profile_;
  fprof.cuda_evt_time_ms = fp->fusion_timer_.time();
  fprof.host_time_ms = fp->host_timer_.time();
  fprof.fusion_id = fp->fusion_id_;
  fprof.segments = (int64_t)fp->segments_.size();

  double kernel_time_ms = 0.0;
  constexpr double mb_divider = 1.0 / 1.0e6;
  if (!fp->cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
    NVFUSER_CUPTI_SAFE_CALL(
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    fp->kernel_profiles_.reserve(fp->segments_.size());
    fprof.kernel_profiles.resize(fp->segments_.size());

    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityFlushAll(0));

    NVF_CHECK(
        fp->kernel_profiles_.size() >= fp->segments_.size(),
        "All of the kernel profiles have not been recorded!");

    for (auto& kprof : fp->kernel_profiles_) {
      auto corr_id = kprof.correlation_id;
      if (fp->corrid_2_segid_.count(corr_id) == 0) {
        continue;
      }
      NVF_CHECK(
          kprof.device >= 0,
          "Device Descriptor index is not valid! ",
          kprof.device);
      if ((size_t)kprof.device >= fp->device_descriptors_.size()) {
        fp->device_descriptors_.resize(kprof.device + 1);
      }
      NVF_CHECK(
          (size_t)kprof.device < fp->device_descriptors_.size(),
          "Device idx is beyond size of Device Descriptors! ",
          kprof.device);
      if (fp->device_descriptors_[kprof.device].device != kprof.device) {
        DeviceDescriptor::generate(
            fp->device_descriptors_[kprof.device], kprof.device);
      }
      kprof.device_name = fp->device_descriptors_[kprof.device].name;
      kprof.peak_bandwidth_gbs =
          fp->device_descriptors_[kprof.device].peak_bandwidth_gbs;
      NVF_CHECK(
          fp->corrid_2_segid_.count(corr_id) > 0,
          "Correlation Id is not found in corrid -> segid hashmap! ",
          corr_id);
      auto kp_idx = fp->corrid_2_segid_[corr_id];
      NVF_CHECK(
          kp_idx < fprof.kernel_profiles.size(),
          "Index is out of range of Kernel Profiles size! ",
          kp_idx,
          " ",
          fprof.kernel_profiles.size());
      NVF_CHECK(
          fp->segments_[kp_idx].state() == ProfilerState::Finished,
          "SegmentProfiler ProfilerState is not Finished!",
          fp->segments_[kp_idx].state());
      kprof.input_bytes = segment(kp_idx).inputBytes();
      kprof.output_bytes = segment(kp_idx).outputBytes();
      kprof.effective_bandwidth_gbs =
          (double)(kprof.input_bytes + kprof.output_bytes) / kprof.time_ms *
          mb_divider;
      kprof.percentage_peak_bandwidth =
          kprof.effective_bandwidth_gbs / kprof.peak_bandwidth_gbs * 100.0;
      kprof.compile_time_ms = segment(kp_idx).compileTime();

      kernel_time_ms += kprof.time_ms;
      fprof.kernel_profiles[kp_idx] = std::move(kprof);
    }

    for (auto& seg : fp->segments_) {
      NVF_CHECK(
          seg.device() == segment(0).device(),
          "All Segment profiles must be on the same device!");
    }
    fprof.kernel_time_ms = kernel_time_ms;
    fprof.effective_bandwidth_gbs =
        (double)(fprof.input_bytes + fprof.output_bytes) / kernel_time_ms *
        mb_divider;
    fprof.percentage_peak_bandwidth = fprof.effective_bandwidth_gbs /
        fp->device_descriptors_[segment(0).device()].peak_bandwidth_gbs * 100.0;
  }
  fprof.compile_time_ms = fp->compile_timer_.time();

  fp->state_ = ProfilerState::Processed;
}

void FusionProfiler::startCompile() {
  NVF_CHECK(
      state() == ProfilerState::Running,
      "FusionProfiler state is not Running!",
      state());
  get()->compile_timer_.start();
}

void FusionProfiler::stopCompile() {
  NVF_CHECK(
      state() == ProfilerState::Running,
      "FusionProfiler state is not Running!",
      state());
  get()->compile_timer_.stop();
}

void FusionProfiler::inputBytesAccessed(int64_t bytes) {
  NVF_CHECK(
      state() == ProfilerState::Running,
      "FusionProfiler state is not Running!",
      state());
  get()->profile_.input_bytes = bytes;
}

void FusionProfiler::outputBytesAccessed(int64_t bytes) {
  NVF_CHECK(
      state() == ProfilerState::Running,
      "FusionProfiler state is not Running!",
      state());
  get()->profile_.output_bytes = bytes;
}

const FusionProfile& FusionProfiler::profile() {
  NVF_CHECK(
      state() == ProfilerState::Processed,
      "The FusionProfile struct data is not valid because it has not been processed! ",
      state());
  return get()->profile_;
}

void FusionProfiler::recordAsyncCorrIdActivity(
    uint32_t seg_id,
    uint32_t corr_id) {
  FusionProfiler* fp = get();
  NVF_CHECK(
      fp->corrid_2_segid_.count(corr_id) == 0,
      "Segment Correlation Activity asociated with this correlation id already exists! ",
      corr_id);
  fp->corrid_2_segid_[corr_id] = seg_id;
}

void FusionProfiler::recordAsyncKernelActivity(KernelProfile prof) {
  get()->kernel_profiles_.emplace_back(std::move(prof));
}

uint8_t* FusionProfiler::cuptiBufferPtr() {
  return get()->cupti_buffer_.data();
}

} // namespace nvfuser
