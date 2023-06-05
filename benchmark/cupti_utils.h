#pragma once

#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#include <iostream>
#include <iomanip>


#define RETURN_IF_NVPW_ERROR(retval, actual)                            \
  do {                                                                  \
    NVPA_Status status = actual;                                        \
    if (NVPA_STATUS_SUCCESS != status) {                                \
      fprintf(stderr, "FAILED: %s with error %s\n", #actual, NV::Metric::Utils::GetNVPWResultString(status)); \
      return retval;                                                    \
    }                                                                   \
  } while (0)


template <typename T>

class ScopeExit
{
public:
    ScopeExit(T t) : t(t) {}
    ~ScopeExit() { t(); }
    T t;
};

template <typename T>
ScopeExit<T> MoveScopeExit(T t) {
    return ScopeExit<T>(t);
};

#define NV_ANONYMOUS_VARIABLE_DIRECT(name, line) name##line
#define NV_ANONYMOUS_VARIABLE_INDIRECT(name, line) NV_ANONYMOUS_VARIABLE_DIRECT(name, line)

#define SCOPE_EXIT(func) const auto NV_ANONYMOUS_VARIABLE_INDIRECT(EXIT, __LINE__) = MoveScopeExit([=](){(void)func;})

namespace NV {
namespace Metric {
namespace Utils {

inline static const char* GetNVPWResultString(NVPA_Status status) {
  const char* errorMsg = NULL;
  switch (status)
  {
    case NVPA_STATUS_ERROR:
      errorMsg = "NVPA_STATUS_ERROR";
      break;
    case NVPA_STATUS_INTERNAL_ERROR:
      errorMsg = "NVPA_STATUS_INTERNAL_ERROR";
      break;
    case NVPA_STATUS_NOT_INITIALIZED:
      errorMsg = "NVPA_STATUS_NOT_INITIALIZED";
      break;
    case NVPA_STATUS_NOT_LOADED:
      errorMsg = "NVPA_STATUS_NOT_LOADED";
      break;
    case NVPA_STATUS_FUNCTION_NOT_FOUND:
      errorMsg = "NVPA_STATUS_FUNCTION_NOT_FOUND";
      break;
    case NVPA_STATUS_NOT_SUPPORTED:
      errorMsg = "NVPA_STATUS_NOT_SUPPORTED";
      break;
    case NVPA_STATUS_NOT_IMPLEMENTED:
      errorMsg = "NVPA_STATUS_NOT_IMPLEMENTED";
      break;
    case NVPA_STATUS_INVALID_ARGUMENT:
      errorMsg = "NVPA_STATUS_INVALID_ARGUMENT";
      break;
    case NVPA_STATUS_INVALID_METRIC_ID:
      errorMsg = "NVPA_STATUS_INVALID_METRIC_ID";
      break;
    case NVPA_STATUS_DRIVER_NOT_LOADED:
      errorMsg = "NVPA_STATUS_DRIVER_NOT_LOADED";
      break;
    case NVPA_STATUS_OUT_OF_MEMORY:
      errorMsg = "NVPA_STATUS_OUT_OF_MEMORY";
      break;
    case NVPA_STATUS_INVALID_THREAD_STATE:
      errorMsg = "NVPA_STATUS_INVALID_THREAD_STATE";
      break;
    case NVPA_STATUS_FAILED_CONTEXT_ALLOC:
      errorMsg = "NVPA_STATUS_FAILED_CONTEXT_ALLOC";
      break;
    case NVPA_STATUS_UNSUPPORTED_GPU:
      errorMsg = "NVPA_STATUS_UNSUPPORTED_GPU";
      break;
    case NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION:
      errorMsg = "NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION";
      break;
    case NVPA_STATUS_OBJECT_NOT_REGISTERED:
      errorMsg = "NVPA_STATUS_OBJECT_NOT_REGISTERED";
      break;
    case NVPA_STATUS_INSUFFICIENT_PRIVILEGE:
      errorMsg = "NVPA_STATUS_INSUFFICIENT_PRIVILEGE";
      break;
    case NVPA_STATUS_INVALID_CONTEXT_STATE:
      errorMsg = "NVPA_STATUS_INVALID_CONTEXT_STATE";
      break;
    case NVPA_STATUS_INVALID_OBJECT_STATE:
      errorMsg = "NVPA_STATUS_INVALID_OBJECT_STATE";
      break;
    case NVPA_STATUS_RESOURCE_UNAVAILABLE:
      errorMsg = "NVPA_STATUS_RESOURCE_UNAVAILABLE";
      break;
    case NVPA_STATUS_DRIVER_LOADED_TOO_LATE:
      errorMsg = "NVPA_STATUS_DRIVER_LOADED_TOO_LATE";
      break;
    case NVPA_STATUS_INSUFFICIENT_SPACE:
      errorMsg = "NVPA_STATUS_INSUFFICIENT_SPACE";
      break;
    case NVPA_STATUS_OBJECT_MISMATCH:
      errorMsg = "NVPA_STATUS_OBJECT_MISMATCH";
      break;
    case NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED:
      errorMsg = "NVPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED";
      break;
    default:
      break;
  }

  return errorMsg;
}
}
}
}

namespace NV {
namespace Metric {
namespace Parser {
inline bool ParseMetricNameString(const std::string& metricName, std::string* reqName, bool* isolated, bool* keepInstances)
{
  std::string& name = *reqName;
  name = metricName;
  if (name.empty())
  {
    return false;
  }

  // boost program_options sometimes inserts a \n between the metric name and a '&' at the end
  size_t pos = name.find('\n');
  if (pos != std::string::npos)
  {
    name.erase(pos, 1);
  }

  // trim whitespace
  while (name.back() == ' ')
  {
    name.pop_back();
    if (name.empty())
    {
      return false;
    }
  }

  *keepInstances = false;
  if (name.back() == '+')
  {
    *keepInstances = true;
    name.pop_back();
    if (name.empty())
    {
      return false;
    }
  }

  *isolated = true;
  if (name.back() == '$')
  {
    name.pop_back();
    if (name.empty())
    {
      return false;
    }
  }
  else if (name.back() == '&')
  {
    *isolated = false;
    name.pop_back();
    if (name.empty())
    {
      return false;
    }
  }

  return true;
}
}
}
}

namespace NV {
namespace Metric {
namespace Config {

inline bool GetRawMetricRequests(std::string chipName,
                                 const std::vector<std::string>& metricNames,
                                 std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
                                 const uint8_t* pCounterAvailabilityImage = NULL)
{
  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculateScratchBufferSizeParam.pChipName = chipName.c_str();
  calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

  std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
  metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
  metricEvaluatorInitializeParams.pChipName = chipName.c_str();
  metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
  NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

  bool isolated = true;
  bool keepInstances = true;
  std::vector<const char*> rawMetricNames;
  for (auto& metricName : metricNames)
  {
    std::string reqName;
    NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
    keepInstances = true;
    NVPW_MetricEvalRequest metricEvalRequest;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
    convertMetricToEvalRequest.pMetricName = reqName.c_str();
    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));

    std::vector<const char*> rawDependencies;
    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params getMetricRawDependenciesParms = {NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
    getMetricRawDependenciesParms.pMetricsEvaluator = metricEvaluator;
    getMetricRawDependenciesParms.pMetricEvalRequests = &metricEvalRequest;
    getMetricRawDependenciesParms.numMetricEvalRequests = 1;
    getMetricRawDependenciesParms.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    getMetricRawDependenciesParms.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDependenciesParms));
    rawDependencies.resize(getMetricRawDependenciesParms.numRawDependencies);
    getMetricRawDependenciesParms.ppRawDependencies = rawDependencies.data();
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_GetMetricRawDependencies(&getMetricRawDependenciesParms));

    for (size_t i = 0; i < rawDependencies.size(); ++i)
    {
      rawMetricNames.push_back(rawDependencies[i]);
    }
  }

  for (auto& rawMetricName : rawMetricNames)
  {
    NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
    metricRequest.pMetricName = rawMetricName;
    metricRequest.isolated = isolated;
    metricRequest.keepInstances = keepInstances;
    rawMetricRequests.push_back(metricRequest);
  }

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
  RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
  return true;
}

inline bool GetConfigImage(std::string chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& configImage, const uint8_t* pCounterAvailabilityImage = NULL)
{
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  GetRawMetricRequests(chipName, metricNames, rawMetricRequests, pCounterAvailabilityImage);

  NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = { NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE };
  rawMetricsConfigCreateParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
  rawMetricsConfigCreateParams.pChipName = chipName.c_str();
  rawMetricsConfigCreateParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_RawMetricsConfig_Create_V2(&rawMetricsConfigCreateParams));
  NVPA_RawMetricsConfig* pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;

  if(pCounterAvailabilityImage)
  {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
    setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
    setCounterAvailabilityParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_SetCounterAvailability(&setCounterAvailabilityParams));
  }

  NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
  rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
  SCOPE_EXIT([&]() -> void {
    auto x = NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams);
    (void)x;
  });

  NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
  beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

  NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
  addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

  NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
  endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

  NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
  generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

  NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
  getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
  getConfigImageParams.bytesAllocated = 0;
  getConfigImageParams.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

  configImage.resize(getConfigImageParams.bytesCopied);
  getConfigImageParams.bytesAllocated = configImage.size();
  getConfigImageParams.pBuffer = configImage.data();
  RETURN_IF_NVPW_ERROR(false, NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

  return true;
}

bool GetCounterDataPrefixImage(std::string chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& counterDataImagePrefix, const uint8_t* pCounterAvailabilityImage = NULL)
{
  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  GetRawMetricRequests(chipName, metricNames, rawMetricRequests, pCounterAvailabilityImage);

  NVPW_CUDA_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE };
  counterDataBuilderCreateParams.pChipName = chipName.c_str();
  counterDataBuilderCreateParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

  NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
  counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
  SCOPE_EXIT([&]() {
    auto x = NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams);
    (void)x;
  });

  NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
  addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
  addMetricsParams.pRawMetricRequests = rawMetricRequests.data();
  addMetricsParams.numMetricRequests = rawMetricRequests.size();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

  //size_t counterDataPrefixSize = 0;
  NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
  getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
  getCounterDataPrefixParams.bytesAllocated = 0;
  getCounterDataPrefixParams.pBuffer = NULL;
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

  counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);
  getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
  getCounterDataPrefixParams.pBuffer = counterDataImagePrefix.data();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

  return true;
}

}
}
}


inline bool WriteBinaryFile(const char* pFileName, const std::vector<uint8_t>& data)
{
    FILE* fp = fopen(pFileName, "wb");
    if (fp)
    {
        if (data.size())
        {
            fwrite(&data[0], 1, data.size(), fp);
        }
        fclose(fp);
    }
    else
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has write access\n";
        return false;
    }
    return true;
}

inline bool ReadBinaryFile(const char* pFileName, std::vector<uint8_t>& image)
{
    FILE* fp = fopen(pFileName, "rb");
    if (!fp)
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has read access\n";
        return false;
    }

    fseek(fp, 0, SEEK_END);
    const long fileLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (!fileLength)
    {
        std::cout << pFileName << " has zero length\n";
        fclose(fp);
        return false;
    }

    image.resize((size_t)fileLength);
    fread(&image[0], 1, image.size(), fp);
    fclose(fp);
    return true;
}


namespace NV {
namespace Metric {
namespace Eval {

struct MetricNameValue
{
  std::string metricName;
  int numRanges;
  // <rangeName , metricValue> pair
  std::vector < std::pair<std::string, double> > rangeNameMetricValueMap;
};

inline std::string GetHwUnit(const std::string& metricName)
{
  return metricName.substr(0, metricName.find("__", 0));
}

inline bool GetMetricGpuValue( std::string chipName,
                               const std::vector<uint8_t>& counterDataImage,
                               const std::vector<std::string>& metricNames,
                               std::vector<MetricNameValue>& metricNameValueMap,
                               const uint8_t* pCounterAvailabilityImage)
{
  if (!counterDataImage.size())
  {
    std::cout << "Counter Data Image is empty!\n";
    return false;
  }

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculateScratchBufferSizeParam.pChipName = chipName.c_str();
  calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

  std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
  metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
  metricEvaluatorInitializeParams.pChipName = chipName.c_str();
  metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
  NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
  getNumRangesParams.pCounterDataImage = counterDataImage.data();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

  bool isolated = true;
  bool keepInstances = true;
  for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
  {
    std::string reqName;
    NV::Metric::Parser::ParseMetricNameString(metricNames[metricIndex], &reqName, &isolated, &keepInstances);
    NVPW_MetricEvalRequest metricEvalRequest;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
    convertMetricToEvalRequest.pMetricName = reqName.c_str();
    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));

    MetricNameValue metricNameValue;
    metricNameValue.numRanges = getNumRangesParams.numRanges;
    metricNameValue.metricName = metricNames[metricIndex];
    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
    {
      NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
      getRangeDescParams.pCounterDataImage = counterDataImage.data();
      getRangeDescParams.rangeIndex = rangeIndex;
      RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
      std::vector<const char*> descriptionPtrs(getRangeDescParams.numDescriptions);
      getRangeDescParams.ppDescriptions = descriptionPtrs.data();
      RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

      std::string rangeName;
      for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
      {
        if (descriptionIndex)
        {
          rangeName += "/";
        }
        rangeName += descriptionPtrs[descriptionIndex];
      }

      NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = { NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE };
      setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
      setDeviceAttribParams.pCounterDataImage = counterDataImage.data();
      setDeviceAttribParams.counterDataImageSize = counterDataImage.size();
      RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));

      double metricValue = 0.0;
      NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateToGpuValuesParams = { NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE };
      evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
      evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
      evaluateToGpuValuesParams.numMetricEvalRequests = 1;
      evaluateToGpuValuesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
      evaluateToGpuValuesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
      evaluateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
      evaluateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
      evaluateToGpuValuesParams.rangeIndex = rangeIndex;
      evaluateToGpuValuesParams.isolated = true;
      evaluateToGpuValuesParams.pMetricValues = &metricValue;
      RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateToGpuValuesParams));
      metricNameValue.rangeNameMetricValueMap.push_back(std::make_pair(rangeName, metricValue));
    }
    metricNameValueMap.push_back(metricNameValue);
  }

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
  RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
  return true;
}

inline bool PrintMetricValues( std::string chipName,
                               const std::vector<uint8_t>& counterDataImage,
                               const std::vector<std::string>& metricNames,
                               const uint8_t* pCounterAvailabilityImage = NULL)
{
  if (!counterDataImage.size())
  {
    std::cout << "Counter Data Image is empty!\n";
    return false;
  }

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculateScratchBufferSizeParam.pChipName = chipName.c_str();
  calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

  std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
  metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
  metricEvaluatorInitializeParams.pChipName = chipName.c_str();
  metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
  metricEvaluatorInitializeParams.pCounterDataImage = counterDataImage.data();
  metricEvaluatorInitializeParams.counterDataImageSize = counterDataImage.size();
  RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
  NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
  getNumRangesParams.pCounterDataImage = counterDataImage.data();
  RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

  std::cout << "\n" << std::setw(40) << std::left << "Range Name"
            << std::setw(100) << std::left        << "Metric Name"
            << "Metric Value" << std::endl;
  std::cout << std::setfill('-') << std::setw(160) << "" << std::setfill(' ') << std::endl;

  std::string reqName;
  bool isolated = true;
  bool keepInstances = true;
  for (std::string metricName : metricNames)
  {
    NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
    NVPW_MetricEvalRequest metricEvalRequest;
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
    convertMetricToEvalRequest.pMetricName = reqName.c_str();
    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
    convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));

    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
    {
      NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
      getRangeDescParams.pCounterDataImage = counterDataImage.data();
      getRangeDescParams.rangeIndex = rangeIndex;
      RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
      std::vector<const char*> descriptionPtrs(getRangeDescParams.numDescriptions);
      getRangeDescParams.ppDescriptions = descriptionPtrs.data();
      RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

      std::string rangeName;
      for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
      {
        if (descriptionIndex)
        {
          rangeName += "/";
        }
        rangeName += descriptionPtrs[descriptionIndex];
      }

      NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = { NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE };
      setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
      setDeviceAttribParams.pCounterDataImage = counterDataImage.data();
      setDeviceAttribParams.counterDataImageSize = counterDataImage.size();
      RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));

      double metricValue;
      NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateToGpuValuesParams = { NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE };
      evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
      evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
      evaluateToGpuValuesParams.numMetricEvalRequests = 1;
      evaluateToGpuValuesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
      evaluateToGpuValuesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
      evaluateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
      evaluateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
      evaluateToGpuValuesParams.rangeIndex = rangeIndex;
      evaluateToGpuValuesParams.isolated = true;
      evaluateToGpuValuesParams.pMetricValues = &metricValue;
      RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateToGpuValuesParams));

      std::cout << std::setw(40) << std::left << rangeName << std::setw(100)
                << std::left << metricName << metricValue << std::endl;
    }
  }

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
  RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
  return true;
}
}
}
}
