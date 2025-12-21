// SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Precompiled header for external dependencies
//
// These headers are included by many compilation units but never change during
// nvFuser development. Precompiling them eliminates redundant parsing overhead.
//
// Note: This file should ONLY contain external headers (stdlib, PyTorch, CUDA).
// Do NOT add nvFuser internal headers here.

#pragma once

// ============================================================================
// C++ Standard Library (most frequently included)
// ============================================================================

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

// C++20 ranges (used by 42 files)
#if __cplusplus >= 202002L
#include <ranges>
#endif

// ============================================================================
// PyTorch / ATen
// ============================================================================

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ArrayRef.h>

// ============================================================================
// CUDA Runtime
// ============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
