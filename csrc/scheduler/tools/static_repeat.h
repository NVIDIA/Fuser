// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>
#include <optional>

namespace nvfuser {

class IterDomain;
class TensorView;

namespace scheduler_tools {

struct StaticRepeatInfo {
  IterDomain* ref_repeating_id = nullptr;
  std::vector<TensorView*> repeated_tvs;
};

std::optional<StaticRepeatInfo> getMaybeStaticRepeatId(TensorView* ref_tv);

} // namespace scheduler_tools
} // namespace nvfuser
