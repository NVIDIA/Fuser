// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#if defined(__linux__) && !defined(_GNU_SOURCE)
// dl_iterate_phdr is only defined when _GNU_SOURCE is defined. The
// macro needs to be defined before any header file is included. See
// the man page for more info.
#define _GNU_SOURCE
#endif

#include <exceptions.h>
#include <runtime/executor_utils.h>
#include <sys_utils.h>

#if defined(__linux__)

#include <filesystem>
#include <sstream>
#include <string>

#include <dlfcn.h>
#include <link.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

namespace nvfuser {
void* LibraryLoader::getSymbol(const char* symbol_name) {
  if (handle_ == nullptr) {
    handle_ = dlopen(filename_.c_str(), RTLD_LAZY);
    NVF_CHECK(
        handle_ != nullptr,
        "Dynamic library ",
        filename_,
        " could not be loaded. ",
        dlerror());
  }
  void* symbol = dlsym(handle_, symbol_name);
  NVF_CHECK(
      symbol != nullptr,
      "Failed to load symbol: ",
      symbol_name,
      " ",
      dlerror());
  return symbol;
}

LibraryLoader::~LibraryLoader() {
  if (handle_ != nullptr) {
    dlclose(handle_);
    handle_ = nullptr;
  }
}

namespace {

// Callback should return 0 to continue iterationg. A non-zero return
// value would stop the iteration.
int detectComputeSanitizerCallback(
    struct dl_phdr_info* info,
    size_t size,
    void* data) {
  std::string lib_name = info->dlpi_name;
  return lib_name.find("compute-sanitizer") != std::string::npos;
}

} // namespace

bool detectComputeSanitizer() {
  return dl_iterate_phdr(detectComputeSanitizerCallback, nullptr) != 0;
}

} // namespace nvfuser

#else // #if defined(__linux__)

namespace nvfuser {

void* LibraryLoader::getSymbol(const char* symbol_name) {
  NVF_THROW("LibraryLoader::getSymbol is only supported on Linux");
  return nullptr;
}

LibraryLoader::~LibraryLoader {
  // TODO: implement non-linux versions of LibraryLoader
}

bool detectComputeSanitizer() {
  // Not implemented. Just return false for now.
  return false;
}

} // namespace nvfuser

#endif // #if defined(__linux__)
