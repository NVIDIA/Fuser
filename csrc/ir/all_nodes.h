// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

// All nodes defined in fusion IR.
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/internal_nodes.h>
#include <kernel_ir.h>

// Don't include host_ir/ir.h here because the nodes there are not part of
// fusion IR.
//
// #include <host_ir/ir.h>
