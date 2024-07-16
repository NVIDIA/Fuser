// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/reorder_sharded_axis.h>

#include <device_lower/utils.h>
#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/alias.h>

namespace nvfuser::preseg_passes {

void ReorderShardedAxisPass::runPass(Fusion* fusion) {}

} // namespace nvfuser::preseg_passes
