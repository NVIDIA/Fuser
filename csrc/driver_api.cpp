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

class CUDADriverDynamicLoader {
  void* handle_ = nullptr;

 public:
  constexpr static const char* filename = "libcuda.so";

  ~CUDADriverDynamicLoader() {
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

#define DEFINE_DRIVER_API_WRAPPER(funcName)                 \
  namespace {                                               \
  template <typename ReturnType, typename... Args>          \
  struct funcName##Loader {                                 \
    static ReturnType lazilyLoadAndInvoke(Args... args) {   \
      std::cout << "lazy load" << std::endl;                \
      funcName = (decltype(funcName))loader.sym(#funcName); \
      return funcName(args...);                             \
    }                                                       \
                                                            \
    funcName##Loader(ReturnType(Args...)){};                \
  };                                                        \
                                                            \
  template <typename ReturnType, typename... Args>          \
  funcName##Loader(ReturnType(Args...))                     \
      ->funcName##Loader<ReturnType, Args...>;              \
  }                                                         \
                                                            \
  decltype(::funcName)* funcName =                          \
      decltype(funcName##Loader(::funcName))::lazilyLoadAndInvoke

namespace nvfuser {

DEFINE_DRIVER_API_WRAPPER(cuTensorMapEncodeTiled);

} // namespace nvfuser

#undef DEFINE_DRIVER_API_WRAPPER
