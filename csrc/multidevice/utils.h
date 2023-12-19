// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/interface_nodes.h>

namespace nvfuser {

// Returns whether a TensorView has its first non-reduction axis parallelized
// on Didx
// Checks that the other non-reduction axis are not parallelized on Didx
bool isSharded(TensorView*);

// returns the number of device indices present accross all
// device meshes in the Fusion
int64_t requestedNumberOfDevices(Fusion*);

} // namespace nvfuser
