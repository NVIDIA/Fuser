// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/fusion_info.h>

namespace nvfuser {

/*static*/ thread_local FusionInfo* FusionInfoGuard::active_fusion_info_ = nullptr;

FusionInfoGuard::FusionInfoGuard(FusionInfo* fusion_info) : prev_fusion_info_(active_fusion_info_) {
  active_fusion_info_ = fusion_info;
}

FusionInfoGuard::~FusionInfoGuard() {
  active_fusion_info_ = prev_fusion_info_;
}

FusionInfo* FusionInfoGuard::current() {
  return active_fusion_info_;
}

bool FusionInfoGuard::hasCurrent() {
  return active_fusion_info_ != nullptr;
}

} // namespace nvfuser
