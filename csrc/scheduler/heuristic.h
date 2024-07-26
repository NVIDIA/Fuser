// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <executor_params.h>
#include <utils.h>

#include <string>

namespace nvfuser {

class HeuristicParams : public PolymorphicBase {
 public:
  std::string tag = "";

  LaunchParams lparams;
  CompileParams cparams;

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
    return other->cparams == cparams;
  }

  virtual std::shared_ptr<HeuristicParams> clone() const {
    return std::make_shared<HeuristicParams>();
  }

  HeuristicParams() = default;
  HeuristicParams(std::string tag, PrimDataType index_type)
      : tag(std::move(tag)), cparams({.index_type = index_type}) {};
  HeuristicParams(std::string tag, KernelIndexMode index_mode)
      : tag(std::move(tag)),
        cparams({.index_type = indexModeToDtype(index_mode)}) {};
};

} // namespace nvfuser
