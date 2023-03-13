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

  virtual size_t hash() const = 0;

  virtual ~HeuristicParams() = default;

  virtual bool sameAs(const std::shared_ptr<HeuristicParams>& other) const = 0;

  virtual std::shared_ptr<HeuristicParams> clone() const = 0;

  HeuristicParams() = default;
  HeuristicParams(const std::string& tag, PrimDataType index_type)
      : tag(tag), cparams({.index_type = index_type}){};
  HeuristicParams(const std::string& tag, KernelIndexMode index_mode)
      : tag(tag), cparams({.index_type = indexModeToDtype(index_mode)}){};
};

} // namespace nvfuser
