// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <visibility.h>

namespace nvfuser {

class Fusion;

//! Fusion Guard is our "context manager". It holds the active fusion and
//! allows it to be accessed anywhere through
//! FusionGuard::getCurFusion().
//!
//! See also the comments in fusion.h
class FusionGuard {
 public:
  //! Set the active fusion so it can be manipulated.
  NVF_API explicit FusionGuard(Fusion* fusion);

  NVF_API ~FusionGuard();

  NVF_API static Fusion* getCurFusion();
  static void setCurFusion(Fusion* fusion);

 private:
  Fusion* prev_fusion_;

  static thread_local Fusion* active_fusion_;
};

} // namespace nvfuser
