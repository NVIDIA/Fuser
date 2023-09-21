// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <string>
#include <vector>

#include <c10/macros/Export.h>

namespace nvfuser {

//! Helper methods to faciliate moving data between data buffers and files based
//! on what type of data is being copied.

bool append_to_text_file(const std::string& file_path, const std::string& src);

bool copy_from_binary_file(
    const std::string& file_path,
    std::vector<char>& dst);
bool copy_from_text_file(const std::string& file_path, std::string& dst);

bool copy_to_binary_file(
    const std::string& file_path,
    const std::vector<char>& dst);
bool copy_to_text_file(const std::string& file_path, const std::string& src);

} // namespace nvfuser
