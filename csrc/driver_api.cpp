// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_utils.h>
#include <driver_api.h>
#include <exceptions.h>
#include <sys_utils.h>
#include <utils.h>

// How does the magic work?
//
// Let's take driver API cuGetErrorName as an example. Because all nvFuser's
// code are in the nvfuser namespace, when we define nvfuser::cuGetErrorName,
// this name will shadow the driver API cuGetErrorName. So when we write code
// cuGetErrorName(...), we will be using nvfuser::cuGetErrorName, instead of
// the driver API cuGetErrorName, due to C++'s name lookup rule. So the goal is
// to make nvfuser::cuGetErrorName behave just like the driver API
// cuGetErrorName, except that it is lazily loaded.
//
// We define nvfuser::cuGetErrorName as a pointer which is initialized to a
// function lazilyLoadAndInvoke. When nvfuser::cuGetErrorName is invoked for the
// first time, it will invoke its lazilyLoadAndInvoke function. This function
// lazily loads the driver API cuGetErrorName, replaces nvfuser::cuGetErrorName
// with the loaded driver API function pointer, and call the newly loaded
// cuGetErrorName driver API. The next time when nvfuser::cuGetErrorName is
// invoked, it will be calling driver API directly.
//
// For each driver API, we need to define its own lazilyLoadAndInvoke function.
// The function signature of each lazilyLoadAndInvoke must be exactly the same
// as its corresponding driver API, because otherwise, we can not assign it to
// our function pointers like nvfuser::cuGetErrorName.
//
// We could of course define these lazilyLoadAndInvoke functions manually for
// each driver API, but it would be very tedious and error-prone. We want to
// automate this process so that adding a new driver API is as trivial as:
//   DEFINE_DRIVER_API_WRAPPER(cuDriverAPIName)
//
// C++'s syntax only allows us to create a function like
//   ReturnType lazilyLoadAndInvoke(Arg1 arg1, Arg2 arg2, ...) {
//     ...
//   }
// Because the number of parameters of the driver API can vary, the only way to
// do it generally that I can think of is to put lazilyLoadAndInvoke into a
// struct template so that we can use parameter pack `typename ... Args`, which
// will be deducted from decltype(cuDriverAPIName), as struct template
// parameters. To make the decltype(cuDriverAPIName) -> ReturnType(Args...)
// deduction work, we can define a ctor for the struct template and add a CTAD
// rule to tell the compiler how to deduce the template parameters.
//
// Doc for CTAD:
// https://en.cppreference.com/w/cpp/language/class_template_argument_deduction
//
// Driver APIs are loaded using cudaGetDriverEntryPoint as recommended by
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-the-runtime-api
namespace {
void getDriverEntryPoint(
    const char* symbol,
    unsigned int version,
    void** entry_point,
    cudaDriverEntryPointQueryResult* query_result) {
#if (CUDA_VERSION >= 12050)
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDriverEntryPointByVersion(
      symbol, entry_point, version, cudaEnableDefault, query_result));
#else
  (void)version;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDriverEntryPoint(
      symbol, entry_point, cudaEnableDefault, query_result));
#endif
}
} // namespace

#define DEFINE_DRIVER_API_WRAPPER(funcName, version)            \
  namespace {                                                   \
  template <typename ReturnType, typename... Args>              \
  struct funcName##Loader {                                     \
    static ReturnType lazilyLoadAndInvoke(Args... args) {       \
      static auto* entry_point = [&]() {                        \
        decltype(::funcName)* entry_point;                      \
        cudaDriverEntryPointQueryResult query_result;           \
        getDriverEntryPoint(                                    \
            #funcName,                                          \
            version,                                            \
            reinterpret_cast<void**>(&entry_point),             \
            &query_result);                                     \
        NVF_CHECK(                                              \
            query_result == cudaDriverEntryPointSuccess,        \
            "Failed to get the entry point for ",               \
            #funcName,                                          \
            ": ",                                               \
            query_result);                                      \
        return entry_point;                                     \
      }();                                                      \
      return entry_point(args...);                              \
    }                                                           \
    /* This ctor is just a CTAD helper, it is only used in a */ \
    /* non-evaluated environment*/                              \
    funcName##Loader(ReturnType(Args...)){};                    \
  };                                                            \
                                                                \
  /* Use CTAD rule to deduct return and argument types */       \
  template <typename ReturnType, typename... Args>              \
  funcName##Loader(ReturnType(Args...))                         \
      ->funcName##Loader<ReturnType, Args...>;                  \
  } /* namespace */                                             \
                                                                \
  decltype(::funcName)* funcName =                              \
      decltype(funcName##Loader(::funcName))::lazilyLoadAndInvoke

namespace nvfuser {

ALL_DRIVER_API_WRAPPER(DEFINE_DRIVER_API_WRAPPER);

} // namespace nvfuser

#undef DEFINE_DRIVER_API_WRAPPER
