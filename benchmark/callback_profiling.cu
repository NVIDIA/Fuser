//
// Copyright 2020 NVIDIA Corporation. All rights reserved
//

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <nvperf_host.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <Metric.h>
#include <Eval.h>
#include <FileOp.h>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, errstr);                     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                     \
    }                                                                          \
} while (0)

#define METRIC_NAME "sm__ctas_launched.sum"

struct ProfilingData_t
{
    int numRanges = 2;
    bool bProfiling = false;
    std::string chipName;
    std::vector<std::string> metricNames;
    std::string CounterDataFileName = "SimpleCupti.counterdata";
    std::string CounterDataSBFileName = "SimpleCupti.counterdataSB";
    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_UserReplay;
    bool allPassesSubmitted = true;
    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;
};

__global__
void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void enableProfiling(ProfilingData_t* pProfilingData)
{
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    if (pProfilingData->profilerReplayMode == CUPTI_KernelReplay)
    {
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    }
    else if (pProfilingData->profilerReplayMode == CUPTI_UserReplay)
    {
        CUpti_Profiler_BeginPass_Params beginPassParams = { CUpti_Profiler_BeginPass_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
        CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    }
}

void disableProfiling(ProfilingData_t* pProfilingData)
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    if (pProfilingData->profilerReplayMode == CUPTI_UserReplay)
    {
        CUpti_Profiler_EndPass_Params endPassParams = { CUpti_Profiler_EndPass_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
        pProfilingData->allPassesSubmitted = (endPassParams.allPassesSubmitted == 1) ? true : false;
    }
    else if (pProfilingData->profilerReplayMode == CUPTI_KernelReplay)
    {
        pProfilingData->allPassesSubmitted = true;
    }

    if (pProfilingData->allPassesSubmitted)
    {
        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = { CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    }
}

void beginSession(ProfilingData_t* pProfilingData)
{
    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = pProfilingData->counterDataImage.size();
    beginSessionParams.pCounterDataImage = &pProfilingData->counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = pProfilingData->counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &pProfilingData->counterDataScratchBuffer[0];
    beginSessionParams.range = pProfilingData->profilerRange;
    beginSessionParams.replayMode = pProfilingData->profilerReplayMode;
    beginSessionParams.maxRangesPerPass = pProfilingData->numRanges;
    beginSessionParams.maxLaunchesPerPass = pProfilingData->numRanges;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
}

void setConfig(ProfilingData_t* pProfilingData)
{
    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = &pProfilingData->configImage[0];
    setConfigParams.configSize = pProfilingData->configImage.size();
    setConfigParams.passIndex = 0;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
}

void createCounterDataImage(int numRanges,
    std::vector<uint8_t>& counterDataImagePrefix,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImage
)
{
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    counterDataImageOptions.maxNumRanges = numRanges;
    counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));
}

void setupProfiling(ProfilingData_t* pProfilingData)
{
    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

    if (pProfilingData->metricNames.size())
    {
        if (!NV::Metric::Config::GetConfigImage(pProfilingData->chipName, pProfilingData->metricNames, pProfilingData->configImage))
        {
            std::cout << "Failed to create configImage" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!NV::Metric::Config::GetCounterDataPrefixImage(pProfilingData->chipName, pProfilingData->metricNames, pProfilingData->counterDataImagePrefix))
        {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cout << "No metrics provided to profile" << std::endl;
        exit(EXIT_FAILURE);
    }

    createCounterDataImage(pProfilingData->numRanges, pProfilingData->counterDataImagePrefix,
                           pProfilingData->counterDataScratchBuffer, pProfilingData->counterDataImage);

    beginSession(pProfilingData);
    setConfig(pProfilingData);
}

void stopProfiling(ProfilingData_t* pProfilingData)
{
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};

    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
    CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    // Dump counterDataImage and counterDataScratchBuffer in file.
    WriteBinaryFile(pProfilingData->CounterDataFileName.c_str(), pProfilingData->counterDataImage);
    WriteBinaryFile(pProfilingData->CounterDataSBFileName.c_str(), pProfilingData->counterDataScratchBuffer);
}

void callbackHandler(void* userdata, CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid, void* cbdata)
{
    ProfilingData_t* profilingData = (ProfilingData_t*)(userdata);
    const CUpti_CallbackData* cbInfo = (CUpti_CallbackData*)cbdata;
    switch (domain)
    {
    case CUPTI_CB_DOMAIN_DRIVER_API:
        switch (cbid)
        {
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
        {
            if (cbInfo->callbackSite == CUPTI_API_ENTER)
            {
                enableProfiling(profilingData);
            }
            else
            {
              disableProfiling(profilingData);
            }
        }
        break;
        default:
            break;
        }
        break;
    case CUPTI_CB_DOMAIN_RESOURCE:
        switch (cbid)
        {
        case CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
        {
            setupProfiling(profilingData);
            profilingData->bProfiling = true;
        }
        break;
        default:
            break;
        }
        break;
    default:
        break;
    }

}

void initVec(int* vec, int n)
{
    for (int i = 0; i < n; i++)
        vec[i] = i;
}

void cleanUp(int* h_A, int* h_B, int* h_C, int* d_A, int* d_B, int* d_C)
{
    if (d_A)
        RUNTIME_API_CALL(cudaFree(d_A));
    if (d_B)
        RUNTIME_API_CALL(cudaFree(d_B));
    if (d_C)
        RUNTIME_API_CALL(cudaFree(d_C));

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
}

void VectorAdd()
{
    int N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int* h_A, * h_B, * h_C;
    int* d_A, * d_B, * d_C;
    int i, sum;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);

    // Allocate vectors in device memory
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

    // Copy vectors from host memory to device memory
    RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n",
        blocksPerGrid, threadsPerBlock);

    VecAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (i = 0; i < N; ++i)
    {
        sum = h_A[i] + h_B[i];
        if (h_C[i] != sum)
        {
            fprintf(stderr, "error: result verification failed\n");
            exit(EXIT_FAILURE);
        }
    }

    cleanUp(h_A, h_B, h_C, d_A, d_B, d_C);
}

int main(int argc, char* argv[])
{
    CUdevice cuDevice = 0;
    int deviceCount, deviceNum = 0;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;

    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        exit(EXIT_WAIVED);
    }

    DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));

    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

    // Initialize profiler API support and test device compatibility
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = deviceNum;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << deviceNum << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }
        exit(EXIT_WAIVED);
    }

    ProfilingData_t* profilingData = new ProfilingData_t();
    for (int i = 1; i < argc; ++i)
    {
        char* arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
        {
            printf("Usage: %s -d [device_num] -m [metric_names comma separated] -n [num of ranges] -r [kernel or user] -o [counterdata filename]\n", argv[0]);
            exit(EXIT_SUCCESS);
        }

        if (strcmp(arg, "--device") == 0 || strcmp(arg, "-d") == 0)
        {
            deviceNum = atoi(argv[i + 1]);
            printf("CUDA Device Number: %d\n", deviceNum);
            i++;
        }
        else if (strcmp(arg, "--metrics") == 0 || strcmp(arg, "-m") == 0)
        {
            char* metricName = strtok(argv[i + 1], ",");
            while (metricName != NULL)
            {
                profilingData->metricNames.push_back(metricName);
                metricName = strtok(NULL, ",");
            }
            i++;
        }
        else if (strcmp(arg, "--numRanges") == 0 || strcmp(arg, "-n") == 0)
        {
            int numRanges = atoi(argv[i + 1]);
            profilingData->numRanges = numRanges;
            i++;
        }
        else if (strcmp(arg, "--replayMode") == 0 || strcmp(arg, "-r") == 0)
        {
            std::string replayMode(argv[i + 1]);
            if (replayMode == "kernel")
                profilingData->profilerReplayMode = CUPTI_KernelReplay;
            else if (replayMode == "user")
                profilingData->profilerReplayMode = CUPTI_UserReplay;
            else {
                printf("Invalid --replayMode argument supported replayMode type 'kernel' or 'user'\n");
                exit(EXIT_FAILURE);
            }
            i++;
        }
        else if (strcmp(arg, "--outputCounterData") == 0 || strcmp(arg, "-o") == 0)
        {
            std::string outputCounterData(argv[i + 1]);
            profilingData->CounterDataFileName = outputCounterData;
            profilingData->CounterDataSBFileName = outputCounterData + "SB";
            i++;
        }
        else {
            printf("Error!! Invalid Arguments\n");
            printf("Usage: %s -d [device_num] -m [metric_names comma separated] -n [num of ranges] -r [kernel or user] -o [counterdata filename]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (profilingData->metricNames.empty())
        profilingData->metricNames.push_back(METRIC_NAME);

    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = deviceNum;
    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    profilingData->chipName = getChipNameParams.pChipName;

    CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callbackHandler, profilingData));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));

    do
    {
        VectorAdd();
    }
    while (!profilingData->allPassesSubmitted);

    if (profilingData->bProfiling)
    {
        stopProfiling(profilingData);
        profilingData->bProfiling = false;

        /* Evaluation of metrics collected in counterDataImage, this can also be done offline*/
        NV::Metric::Eval::PrintMetricValues(profilingData->chipName, profilingData->counterDataImage, profilingData->metricNames);
    }

    delete profilingData;
    exit(EXIT_SUCCESS);
}
