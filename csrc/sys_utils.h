// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <utils.h>

#include <string>

namespace nvfuser {

class LibraryLoader : public NonCopyable {
 public:
  LibraryLoader() = default;

  ~LibraryLoader();

  void* getSymbol(const char* symbol_name);

  std::string filename() const {
    return filename_;
  }

  void setFilename(std::string filename) {
    filename_ = filename;
  }

 private:
  std::string filename_ = "";
  void* handle_ = nullptr;
};

// Return true if compute-sanitizer is attached
bool detectComputeSanitizer();

} // namespace nvfuser
