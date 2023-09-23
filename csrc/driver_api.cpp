// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <driver_api.h>

#include <cuda.h>
#include <dlfcn.h>

#include <iostream>

#include <exceptions.h>

namespace {

class CUDADriverAPIDynamicLoader {
  void* handle_ = nullptr;

 public:
  constexpr static const char* filename = "libcuda.so";

  ~CUDADriverAPIDynamicLoader() {
    if (handle_) {
      dlclose(handle_);
    }
  }

  void* sym(const char* symbolName) {
    if (!handle_) {
      handle_ = dlopen(filename, RTLD_LAZY);
    }
    NVF_CHECK(
        handle_, "Dynamic library not loaded. Please check CUDA installation");
    void* symbol = dlsym(handle_, symbolName);
    NVF_CHECK(
        symbol,
        "Failed to load symbol: ",
        symbolName,
        " ",
        dlerror(),
        "Please check CUDA installation");
    return symbol;
  }
} loader;

} // namespace

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
// Because C++'s syntax only allows us to create a function like:
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
#define DEFINE_DRIVER_API_WRAPPER(funcName)                     \
  namespace {                                                   \
  template <typename ReturnType, typename... Args>              \
  struct funcName##Loader {                                     \
    static ReturnType lazilyLoadAndInvoke(Args... args) {       \
      funcName = (decltype(funcName))loader.sym(#funcName);     \
      return funcName(args...);                                 \
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
  }                                                             \
                                                                \
  decltype(::funcName)* funcName =                              \
      decltype(funcName##Loader(::funcName))::lazilyLoadAndInvoke

namespace nvfuser {

DEFINE_DRIVER_API_WRAPPER(cuGetErrorName);
DEFINE_DRIVER_API_WRAPPER(cuGetErrorString);
DEFINE_DRIVER_API_WRAPPER(cuTensorMapEncodeTiled);

} // namespace nvfuser

#undef DEFINE_DRIVER_API_WRAPPER
