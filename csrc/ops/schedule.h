// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

#include <ir/interface_nodes.h>
#include <ops/utils.h>
#include <type.h>

namespace nvfuser {

// NvFuser's paradigm is to define math operations first and then apply
// scheduling. The main limitations are that the inputs and outputs for an
// expression must be an actual TensorView and new expressions generally cannot
// be added to Fusion IR. These schedule expressions allow experimental support
// for expressing new hardware features like TMA, Programmatic Dependent Launch
// (PDL, or Cluster Launch Control (CLC) in Fusion IR.
//
// Scheduling PDL in NvFuser
//
// Choosing when to launch dependent grids and wait for primary grids to finish
// is a scheduling decision. Instead of manually adding PDL expressions during
// lowering, insert them into directly fusion IR and leverage IterDomain
// scheduling.
//
// launch_dependent_grid and wait_for_prior_grid accept a list of input Vals.
// Without loss of generality, the first TensorView is taken as a reference.
// An output TensorView is created with a **broadcast-only** logical domain that
// is derived from a reference TensorView. Val::addDependency inserts the output
// into the Fusion IR by adding to the consumer's inputs. During lowering,
// expression sorting correctly places these operations in Kernel IR. Register
// allocation is skipped for this TensorView.
//
// NOTE: All the inputs must be inlinable together to avoid issues with
// ComputeAtMap and IdModel. For incompatible inputs, issue multiple expressions
// for each group of inlinable inputs.
//
// NOTE: Adding schedule operations after a terminating output is not supported.

// The launch dependent grid instruction allows the driver to start the
// dependent grid in programmatic dependent launch. This is commonly used after
// computation is finished but before storing results to global memory.
NVF_API TensorView* launch_dependent_grid(std::vector<Val*> inputs);

// The wait for prior grid instruction prevents the kernel from running before
// the prior grid is finished. This is used before any operations access global
// input variables modified by the prior kernel.
NVF_API TensorView* wait_for_prior_grid(std::vector<Val*> inputs);

} // namespace nvfuser
