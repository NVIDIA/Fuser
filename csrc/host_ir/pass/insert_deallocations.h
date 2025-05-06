// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <host_ir/container.h>

namespace nvfuser::hir {

/* For each input in every expression in the container, find the index of its
 * last use and insert a deallocate directly after */
void insertDeallocations(HostIrContainer* hic);

} // namespace nvfuser::hir
