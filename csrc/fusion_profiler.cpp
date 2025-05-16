// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <iomanip>

#include <cupti.h>

#include <exceptions.h>
#include <fusion_profiler.h>

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
      prof.shared_mem = {
          pKARecord->dynamicSharedMemory, pKARecord->staticSharedMemory};
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
    case CUPTI_ACTIVITY_KIND_RUNTIME:
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
  const size_t max_records = 10000;
  bool found_max_limit = false;

  for (size_t i = 0; i < max_records; ++i) {
    status = cuptiActivityGetNextRecord(pBuffer, validBytes, &pRecord);
    if (status == CUPTI_SUCCESS) {
      // Processes a valid CUPTI Activty record and records it with the
      // fusion profiling infrastructure if the record is of interest.
      record_cupti_activity(pRecord, pFileHandle);
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
      NVF_THROW("Unexpected ProfilerState enum value!");
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
  NVF_CHECK_EQ(state_, ProfilerState::Ready);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event_, stream_));
  state_ = ProfilerState::Running;
}

void CudaEventTimer::stop() {
  NVF_CHECK_EQ(state_, ProfilerState::Running);
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
        "ProfilerState (",
        state_,
        ") is not Processed or Ready!");
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
  NVF_CHECK_EQ(state_, ProfilerState::Ready);
  start_event_ = Clock::now();
  state_ = ProfilerState::Running;
}

void HostTimer::stop() {
  NVF_CHECK_EQ(state_, ProfilerState::Running);
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
        "ProfilerState (",
        state_,
        ") is not Processed or Ready!");
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

void SegmentProfiler::startCompile() {
  compile_timer_.start();
}

void SegmentProfiler::stopCompile() {
  compile_timer_.stop();
}

void SegmentProfiler::startKernel() {
  NVF_CHECK_EQ(kernel_profile_state_, ProfilerState::Ready);
  if (!cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN,
        static_cast<uint64_t>(segment_id_)));
  }
  kernel_profile_state_ = ProfilerState::Running;
}

void SegmentProfiler::stopKernel() {
  NVF_CHECK_EQ(kernel_profile_state_, ProfilerState::Running);
  uint64_t corr_id = 0;
  if (!cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &corr_id));
    NVF_CHECK_EQ(
        corr_id,
        static_cast<uint64_t>(segment_id_),
        "Correlation Id does not match segment id!");
  }
  kernel_profile_state_ = ProfilerState::Finished;
}

void SegmentProfiler::inputBytesAccessed(int64_t bytes) {
  input_bytes_ = bytes;
}

void SegmentProfiler::outputBytesAccessed(int64_t bytes) {
  output_bytes_ = bytes;
}

void SegmentProfiler::scheduler(const std::string& name) {
  scheduler_ = name;
}
const std::string& SegmentProfiler::scheduler() const {
  return scheduler_;
}

uint32_t SegmentProfiler::segmentId() const {
  return segment_id_;
}

auto FusionProfile::toTuple(const FusionProfile& prof, size_t seg_id) {
  NVF_CHECK(
      !prof.kernel_profiles.empty(),
      "Cannot convert FusionProfile to a tuple containing CUPTI gathered stats!");
  NVF_CHECK(
      seg_id < prof.kernel_profiles.size(),
      "Invalid seg_id for FusionProfile. Segments: ",
      prof.kernel_profiles.size(),
      " seg_id: ",
      seg_id);
  const auto& kp = prof.kernel_profiles[seg_id];

  return std::tie(
      prof.fusion_id,
      prof.segments,
      prof.cuda_evt_time_ms,
      prof.host_time_ms,
      prof.compile_time_ms,
      prof.kernel_time_ms,
      prof.effective_bandwidth_gbs,
      prof.percentage_peak_bandwidth,
      prof.input_bytes,
      prof.output_bytes,
      kp.segment_id,
      kp.time_ms,
      kp.compile_time_ms,
      kp.effective_bandwidth_gbs,
      kp.percentage_peak_bandwidth,
      kp.input_bytes,
      kp.output_bytes,
      kp.shared_mem_str,
      kp.registers,
      kp.grid_str,
      kp.block_str,
      kp.cluster_str,
      kp.scheduler,
      kp.device,
      kp.stream,
      kp.peak_bandwidth_gbs,
      kp.device_name,
      kp.name);
}

auto FusionProfile::toNocuptiTuple(const FusionProfile& prof) {
  NVF_CHECK(
      prof.kernel_profiles.empty(),
      "Cannot convert FusionProfile to a tuple without CUPTI gathered stats!");

  return std::tie(
      prof.fusion_id,
      prof.segments,
      prof.cuda_evt_time_ms,
      prof.host_time_ms,
      prof.compile_time_ms);
}

void FusionProfile::reset() {
  fusion_id = -1;
  segments = 0;

  cuda_evt_time_ms = 0.0;
  host_time_ms = 0.0;
  compile_time_ms = 0.0;
  kernel_time_ms = 0.0;

  effective_bandwidth_gbs = 0.0;
  percentage_peak_bandwidth = 0.0;

  input_bytes = 0;
  output_bytes = 0;

  kernel_profiles.clear();
}

const std::vector<ProfileAttrDescriptor> FusionProfile::profile_attr_descs{
    // column_header, verbose, segment, list, column_width, number,
    // mantissa_width, unit_multiplier
    {"Fus#", false, false, false, 5, true, 0, std::nullopt},
    {"NSegs", false, false, false, 5, true, 0, std::nullopt},
    {"CuEvtTm(ms)", false, false, false, 11, true, 3, std::nullopt},
    {"HstTm(ms)", false, false, false, 9, true, 3, std::nullopt},
    {"CmpTm(ms)", false, false, false, 9, true, 3, std::nullopt},
    {"KerTm(ms)", false, false, false, 9, true, 3, std::nullopt},
    {"EffBw(GB/s)", false, false, false, 11, true, 3, std::nullopt},
    {"%PkBw", false, false, false, 7, true, 2, std::nullopt},
    {"In(MB)", true, false, false, 8, true, 3, 1.0e-6},
    {"Out(MB)", true, false, false, 9, true, 3, 1.0e-6},
    {"S-Seg#", false, true, false, 6, true, 0, std::nullopt},
    {"S-KerTm(ms)", false, true, false, 11, true, 3, std::nullopt},
    {"S-CmpTm(ms)", true, true, false, 11, true, 3},
    {"S-EffBw(GB/s)", false, true, false, 13, true, 3, std::nullopt},
    {"S-%PkBw", false, true, false, 7, true, 2, std::nullopt},
    {"S-In(MB)", false, true, false, 8, true, 3, 1.0e-6},
    {"S-Out(MB)", false, true, false, 9, true, 3, 1.0e-6},
    {"S-Smem[Dyn,Stat]", false, true, true, 16, false, 0, std::nullopt},
    {"S-Regs", false, true, false, 6, true, 0, std::nullopt},
    {"S-Grid", false, true, true, 16, true, 0, std::nullopt},
    {"S-Block", false, true, true, 16, false, 0, std::nullopt},
    {"S-Cluster", true, true, true, 16, false, 0, std::nullopt},
    {"S-Sched", true, true, false, 15, false, 0, std::nullopt},
    {"S-Dev", true, true, false, 5, true, 0, std::nullopt},
    {"S-Stm", true, true, false, 5, true, 0, std::nullopt},
    {"S-PkBw(GB/s)", true, true, false, 12, true, 3, std::nullopt},
    {"S-DeviceName", true, true, false, 20, false, 0, std::nullopt},
    {"S-KerName", false, true, false, 20, false, 0, std::nullopt}};

namespace {
// The operator* overloads are to satisfy the compiler and should not be called!
double operator*(const std::basic_string<char>& a, double b) {
  NVF_THROW("This types operator* overload should not be called!");
  return 0.0;
}

template <typename T, size_t I>
std::string toString(const std::array<T, I>& cont) {
  std::string out{"["};
  bool first_elem = true;
  for (const auto& elem : cont) {
    if (first_elem) {
      first_elem = false;
    } else {
      out += ", ";
    }
    out += std::to_string(elem);
  }
  out += "]";
  return out;
}

template <bool NOCUPTI = false, size_t I = 0, typename... Ts>
constexpr std::ostream& printTuple(
    std::ostream& os,
    std::tuple<Ts...> tup,
    size_t seg_id,
    bool verbose) {
  if constexpr (
      (I == sizeof...(Ts)) ||
      (NOCUPTI && (I == FusionProfile::first_cupti_idx))) {
    return os;
  } else {
    os << std::setfill(' ') << std::fixed << std::right;
    const auto& desc = FusionProfile::profile_attr_descs.at(I);
    // Print the tuple and go to next element
    if ((verbose && desc.verbose) || !desc.verbose) {
      if constexpr (I > 0) {
        os << " ";
      }
      os << std::setw(desc.column_width);
      if (seg_id > 0 && !desc.segment) {
        os << "-";
      } else {
        if (desc.number) {
          os << std::setprecision(desc.mantissa_width);
        }
        if (desc.unit_multiplier.has_value()) {
          // NOTE: The "operator*(const std::basic_string<char>& a, double b)"
          // that is defined in this anonymous namespace is used to prevent a
          // compiler error in the following line as some tuple values are
          // strings and trigger this overload even though it does not make
          // sense to execute.
          os << static_cast<double>(
              std::get<I>(tup) * desc.unit_multiplier.value());
        } else {
          os << std::get<I>(tup);
        }
      }
    }
    // Going for next element.
    return printTuple<NOCUPTI, I + 1>(os, tup, seg_id, verbose);
  }
}
} // namespace

std::ostream& operator<<(std::ostream& os, const FusionProfile& fp) {
  // Print headers only for first fusion
  if (fp.fusion_id == 0) {
    // `os` may have leftover characters in the line
    // before the header is printed. So we start with a newline.
    os << std::endl;
    // Print headers starting on the left
    os << std::setfill(' ') << std::left;

    // Print no-cupti headers
    for (size_t i = 0; i < FusionProfile::first_cupti_idx; ++i) {
      const auto& desc = FusionProfile::profile_attr_descs.at(i);
      if (i > 0) {
        os << " ";
      }
      os << std::setw(desc.column_width) << desc.column_header;
    }

    // Print cupti collected column headers
    if (!fp.kernel_profiles.empty()) {
      for (size_t i = FusionProfile::first_cupti_idx;
           i < FusionProfile::profile_attr_descs.size();
           ++i) {
        const auto& desc = FusionProfile::profile_attr_descs.at(i);
        if ((fp.verbose && desc.verbose) || !desc.verbose) {
          os << " " << std::setw(desc.column_width) << desc.column_header;
        }
      }
    }
    os << std::endl;
  }

  // Print no-cupti data per fusion
  if (fp.kernel_profiles.empty()) {
    printTuple<true>(os, FusionProfile::toNocuptiTuple(fp), 0, fp.verbose);
    os << std::endl;
    // Print segment data per segment
  } else {
    for (size_t i = 0; i < fp.kernel_profiles.size(); ++i) {
      printTuple(os, FusionProfile::toTuple(fp, i), i, fp.verbose);
      os << std::endl;
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
  NVF_CHECK_EQ(state(), ProfilerState::Running);
  fp->segments_.reserve(num);
  for (uint32_t i = 0; i < num; ++i) {
    fp->segments_.emplace_back(i, fp->cupti_disabled_);
  }
}
SegmentProfiler& FusionProfiler::segment(size_t idx) {
  NVF_CHECK(
      get()->segments_.size() > idx,
      "FusionProfiler: You are attempting to access non-existent segments! Segments: ",
      get()->segments_.size(),
      " Idx: ",
      idx);
  return get()->segments_.at(idx);
}

/*static*/ void FusionProfiler::start(bool cupti_disable) {
  FusionProfiler* fp = get();
  fp->cupti_disabled_ = cupti_disable;
  reset();
  if (!fp->cupti_disabled_) {
    NVFUSER_CUPTI_SAFE_CALL(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    NVFUSER_CUPTI_SAFE_CALL(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  }
  cudaDeviceSynchronize();
  fp->fusion_timer_.start();
  fp->host_timer_.start();
  fp->state_ = ProfilerState::Running;
}

const DeviceDescriptor& FusionProfiler::deviceDescriptor(const int device_id) {
  NVF_CHECK(device_id >= 0, "Invalid device index: ", device_id);
  if ((size_t)device_id >= device_descriptors_.size()) {
    device_descriptors_.resize(device_id + 1);
  }
  DeviceDescriptor& desc = device_descriptors_[device_id];

  if (desc.device != device_id) {
    // This happens when device_descriptors_[device_id] is initialized (and
    // thus device==-1) but not populated.
    DeviceDescriptor::generate(desc, device_id);
  }
  return desc;
}

/*static*/ void FusionProfiler::stop() {
  FusionProfiler* fp = get();
  NVF_CHECK_EQ(state(), ProfilerState::Running);
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
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    NVFUSER_CUPTI_SAFE_CALL(
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    // This will be populated by the following `cuptiActivityFlushAll` call.
    fp->kernel_profiles_.reserve(fp->segments_.size());
    NVFUSER_CUPTI_SAFE_CALL(cuptiActivityFlushAll(0));

    fprof.kernel_profiles.resize(fp->segments_.size());
    for (auto& kprof : fp->kernel_profiles_) {
      auto corr_id = kprof.correlation_id;
      if (fp->corrid_2_segid_.count(corr_id) == 0) {
        continue;
      }
      const DeviceDescriptor& device_desc = fp->deviceDescriptor(kprof.device);
      kprof.device_name = device_desc.name;
      kprof.peak_bandwidth_gbs = device_desc.peak_bandwidth_gbs;
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
      NVF_CHECK_EQ(
          fp->segments_[kp_idx].state(),
          ProfilerState::Finished,
          "SegmentProfiler ProfilerState is not Finished!");
      kprof.segment_id = static_cast<size_t>(kp_idx);
      kprof.input_bytes = segment(kp_idx).inputBytes();
      kprof.output_bytes = segment(kp_idx).outputBytes();
      kprof.effective_bandwidth_gbs =
          (double)(kprof.input_bytes + kprof.output_bytes) / kprof.time_ms *
          mb_divider;
      kprof.percentage_peak_bandwidth =
          kprof.effective_bandwidth_gbs / kprof.peak_bandwidth_gbs * 100.0;
      kprof.compile_time_ms = segment(kp_idx).compileTime();

      kprof.grid_str = toString(kprof.grid);
      kprof.block_str = toString(kprof.block);
      kprof.cluster_str = toString(kprof.cluster);
      kprof.shared_mem_str = toString(kprof.shared_mem);

      kprof.scheduler = segment(kp_idx).scheduler();

      kernel_time_ms += kprof.time_ms;
      fprof.kernel_profiles[kp_idx] = std::move(kprof);
    }

    for (auto& seg : fp->segments_) {
      NVF_CHECK(
          seg.device() == segment(0).device(),
          "All Segment profiles must be on the same device!");
    }
    fprof.kernel_time_ms = kernel_time_ms;
    if (!fp->kernel_profiles_.empty()) {
      fprof.effective_bandwidth_gbs =
          (double)(fprof.input_bytes + fprof.output_bytes) / kernel_time_ms *
          mb_divider;
    }
    if (!fp->segments_.empty()) {
      fprof.percentage_peak_bandwidth = fprof.effective_bandwidth_gbs /
          fp->deviceDescriptor(segment(0).device()).peak_bandwidth_gbs * 100.0;
    }
  }
  fprof.compile_time_ms = fp->compile_timer_.time();

  fp->state_ = ProfilerState::Processed;
}

void FusionProfiler::startCompile() {
  NVF_CHECK_EQ(state(), ProfilerState::Running);
  get()->compile_timer_.start();
}

void FusionProfiler::stopCompile() {
  NVF_CHECK_EQ(state(), ProfilerState::Running);
  get()->compile_timer_.stop();
}

void FusionProfiler::inputBytesAccessed(int64_t bytes) {
  NVF_CHECK_EQ(state(), ProfilerState::Running);
  get()->profile_.input_bytes = bytes;
}

void FusionProfiler::outputBytesAccessed(int64_t bytes) {
  NVF_CHECK_EQ(state(), ProfilerState::Running);
  get()->profile_.output_bytes = bytes;
}

const FusionProfile& FusionProfiler::profile() {
  NVF_CHECK_EQ(
      state(),
      ProfilerState::Processed,
      "The FusionProfile struct data is not valid because it has not been processed!");
  return get()->profile_;
}

double FusionProfiler::lastKernelTime() {
  const auto& fprof = profile();
  NVF_CHECK(
      !fprof.kernel_profiles.empty(), "There are no kernel profiles to query!");
  return fprof.kernel_profiles.back().time_ms;
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
