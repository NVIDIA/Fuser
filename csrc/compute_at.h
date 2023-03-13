// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <inlining.h>
#include <root_domain_map.h>
#include <transform_replay.h>

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class TensorDomain;
class TensorView;

struct ComputeAt {
 public:
  // Runs the compute at pass making producer look like consumer, computing
  // producer relative to consumer
  static void runAt(
      TensorView* producer,
      TensorView* consumer,
      int64_t consumer_position,
      ComputeAtMode mode = ComputeAtMode::Standard);
};

} // namespace nvfuser
