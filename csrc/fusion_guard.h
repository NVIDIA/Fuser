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

//! Usage: FusionGuard and Fusion are required user interfaces for any operation
//! underlying the code generator. In order to create values, expressions, and
//! generate code a Fusion instance must be active. It is the responsibility of
//! the user to create a Fusion instance and register it with the fusion guard.
//! The simplest example of this is:
//!
//!     Fusion fusion;
//!     FusionGuard fg(&fusion);
//!
//! Once a fusion is active all values and operations will be registered with
//! it.
//!
//! FusionGuard and Fusion are critical to the lifetime model of the IR system.
//! FusionGuard is a convenient way to set what base container instance holds
//! the defined IR. Statements that are defined are registered through the
//! FusionGuard with a particular Fusion. FusionGuard provides convenient
//! methods to access the active fusion so it doesn't need to be passed around
//! constantly. Any IR node derived classes from Statement must register with
//! Fusion to avoid memory leaks.
//!
//! Fusion is generally thought of as a translated fusion group from the JIT. It
//! is likely a single kernel, although, we don't have to stick to this in the
//! future and could in theory generate multiple kernels with an executor to run
//! them.
//!
//! Fusion also allows users to set input/output values that will allow us to
//! figure out how to hook up runtime data to and from the JIT as well as
//! provide us mechanisms for dependency analysis and DCE including safety
//! checks.

class Fusion;

//! Fusion Guard is our "context manager". It holds the active fusion and
//! allows it to be accessed anywhere through FusionGuard::getCurFusion()
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
