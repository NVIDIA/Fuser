// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>
#include <string_view>

namespace nvfuser {

inline constexpr std::string_view kMainFuncName = "main";
inline constexpr std::string_view kTensorSizeFuncName = "tensor_size";
inline constexpr std::string_view kTensorStrideFuncName = "tensor_stride";
inline constexpr std::string_view kTensorDataPtrFuncName = "tensor_data_ptr";
inline constexpr std::string_view kLaunchKernelDirectFuncName =
    "launch_kernel_direct";
inline constexpr std::string_view kNewTensorFuncName = "new_tensor";
inline constexpr std::string_view kDeleteTensorFuncName = "delete_tensor";
inline constexpr std::string_view kSetTensorFuncName = "set_tensor";
inline constexpr std::string_view kAtEmptyStridedCudaWrapper =
    "at_empty_strided_cuda";
inline constexpr std::string_view kAtTensorType = "at.Tensor";
inline constexpr std::string_view kNvtxRangePushFuncName = "nvtx_range_push";
inline constexpr std::string_view kNvtxRangePopFuncName = "nvtx_range_pop";
inline constexpr std::string_view kMatmulOutFuncName = "matmul_out";
inline constexpr std::string_view kLinearOutFuncName = "linear_out";
inline constexpr std::string_view kPermuteFuncName = "permute";
inline constexpr std::string_view kReshapeFuncName = "reshape";
inline constexpr std::string_view kMainFuncOutputTensorName =
    "output_aten_tensor_addr";
inline constexpr int64_t kMaxTensorDim = 8;

} // namespace nvfuser
