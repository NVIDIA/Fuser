// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion_executor/executor_params.h>
#include <scheduler/heuristic_types.h>
#include <utils.h>

#include <string>

namespace nvfuser {

class HeuristicParams : public PolymorphicBase {
 public:
  std::string tag = "";

  LaunchParams lparams;
  CompileParams cparams;
  const ScheduleHeuristic heuristic_type;

  virtual std::string toString() const {
    return "Undefined Heuristic Params";
  }

  virtual size_t hash() const {
    return 0;
  };

  virtual bool sameAs(const std::shared_ptr<HeuristicParams>& other) const {
    if (!other->isStrictlyA<HeuristicParams>()) {
      return false;
    }
    if (other->heuristic_type != heuristic_type) {
      return false;
    }
    return other->cparams == cparams;
  }

  HeuristicParams() = delete;
  explicit HeuristicParams(ScheduleHeuristic type) : heuristic_type(type) {};
};

} // namespace nvfuser
