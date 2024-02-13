#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                  \
  do {                                 \
    CUresult _result = x;              \
    if (_result != CUDA_SUCCESS) {     \
      const char* msg;                 \
      const char* name;                \
      cuGetErrorName(_result, &name);  \
      cuGetErrorString(_result, &msg); \
      std::cout << "CUDA error: "      \
          << name                      \
          << " failed with error "     \
          << msg                       \
          << std::endl;                \
      exit(_result);                   \
    }                                  \
  } while (0)

#define CUDA_RUNTIME_CHECK(x)                         \
  do {                                                \
    cudaError_t _result = x;                          \
    if (_result != cudaSuccess) {                     \
      const char* name = cudaGetErrorName(_result);   \
      const char* msg = cudaGetErrorString(_result);  \
      std::cout << "CUDA error: "                     \
          << name                                     \
          << " failed with error "                    \
          << msg                                      \
          << std::endl;                               \
      exit(_result);                                  \
    }                                                 \
  } while (0)
