// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion_guard.h>

namespace nvfuser {

/*static*/ thread_local Fusion* FusionGuard::active_fusion_ = nullptr;

FusionGuard::FusionGuard(Fusion* fusion) : prev_fusion_(active_fusion_) {
  active_fusion_ = fusion;
}

FusionGuard::~FusionGuard() {
  active_fusion_ = prev_fusion_;
}

// Cast to non-cast because many users need it.
/*static*/ Fusion* FusionGuard::getCurFusion() {
  return active_fusion_;
}

/*static*/ void FusionGuard::setCurFusion(Fusion* fusion) {
  active_fusion_ = fusion;
}

} // namespace nvfuser
