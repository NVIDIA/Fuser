// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <fusion_executor/executor_params.h>
#include <scheduler/heuristic_types.h>
#include <scheduler/runtime_info.h>
#include <utils.h>

#include <string>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicSummary;

class HeuristicParams : public PolymorphicBase {
 public:
  std::string tag = "";

  LaunchParams lparams;
  CompileParams cparams;
  const HeuristicType heuristic_type;

  virtual std::string toString() const {
    return "Undefined Heuristic Params";
  }

  virtual size_t hash() const {
    return 0;
  };

  virtual bool sameAs(const HeuristicParams* other) const {
    if (!other->isStrictlyA<HeuristicParams>()) {
      return false;
    }
    if (other->heuristic_type != heuristic_type) {
      return false;
    }
    return other->cparams == cparams;
  }

  HeuristicParams() = delete;
  explicit HeuristicParams(HeuristicType type) : heuristic_type(type) {};

  virtual std::unique_ptr<HeuristicParams> clone() const {
    return std::make_unique<HeuristicParams>(heuristic_type);
  }
};

//! Auxiliary class for storing heuristics. The managed data is either
//!  a single heursitic for complete fusion, or a vector of heuristics used for
//!  a segmented fusion.
class HeuristicParamsList {
 public:
  //! Constructor for segmented fusion case. Created with empty list and
  //!  uses emplaceBack for inserting heuristics in order
  explicit HeuristicParamsList() = default;

  //! Constructor fills heuristics_ with nullptr, which allows us to create
  //! SchedulerEntries out of order.
  explicit HeuristicParamsList(size_t num_heuristics) {
    heuristics_.reserve(num_heuristics);
    std::fill_n(std::back_inserter(heuristics_), num_heuristics, nullptr);
  }

  //! Constructor for complete fusion case, generates the scheduler entry
  //!  for the fusion owning the given expression
  explicit HeuristicParamsList(
      HeuristicType schedule_heuristic,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicSummary* data_cache = nullptr);

  HeuristicParamsList(const HeuristicParamsList&) = delete;
  HeuristicParamsList& operator=(const HeuristicParamsList&) = delete;

  std::unique_ptr<HeuristicParams>& at(int index) {
    return heuristics_.at(index);
  }

  //! Place a scheduler entry on the list. Applies to segmented fusion only.
  void emplaceBack(std::unique_ptr<HeuristicParams>&& pt) {
    NVF_ERROR(is_segmented_);
    heuristics_.emplace_back(std::move(pt));
  }

  //! Returns list of schedulers for a segmneted fusion.
  const std::vector<std::unique_ptr<HeuristicParams>>& heuristicsList() const {
    return heuristics_;
  }

  //! Returns the single scheduler for a complete fusion.
  HeuristicParams* singleKernelHeuristics() {
    NVF_ERROR(!is_segmented_);
    return heuristics_.begin()->get();
  }

 private:
  std::vector<std::unique_ptr<HeuristicParams>> heuristics_;
  bool is_segmented_ = true;
};

} // namespace nvfuser
