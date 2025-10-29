// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <runtime/executor_params.h>
#include <scheduler/scheduler_types.h>
#include <utils.h>

#include <string>

namespace nvfuser {

class SchedulerRuntimeInfo;
class HeuristicDataCache;

// Top-level class representing heuristic parameters. Most schedulers
// have their own subclasses to have their specific parameters, except
// for ExprEval schedulers.
class NVF_API HeuristicParams : public PolymorphicBase {
 public:
  std::string tag = "";

  LaunchParams lparams;
  CompileParams cparams;
  const SchedulerType scheduler_type;

  virtual std::string toString() const {
    std::stringstream ss;
    ss << "Heuristic Params (" << scheduler_type << ")";
    return ss.str();
  }

  virtual size_t hash() const {
    return 0;
  };

  virtual bool sameAs(const HeuristicParams* other) const {
    if (!other->isStrictlyA<HeuristicParams>()) {
      return false;
    }
    if (other->scheduler_type != scheduler_type) {
      return false;
    }
    return other->cparams == cparams;
  }

  HeuristicParams() = delete;
  explicit HeuristicParams(SchedulerType _scheduler_type)
      : scheduler_type(_scheduler_type) {};

  virtual std::unique_ptr<HeuristicParams> clone() const {
    return std::make_unique<HeuristicParams>(*this);
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
      SchedulerType scheduler_type,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr);

  HeuristicParamsList(const HeuristicParamsList&) = delete;
  HeuristicParamsList& operator=(const HeuristicParamsList&) = delete;

  std::unique_ptr<HeuristicParams>& at(int index) {
    return heuristics_.at(index);
  }

  //! Place a heuristics on the list. Applies to segmented fusion only.
  void emplaceBack(std::unique_ptr<HeuristicParams>&& pt) {
    NVF_ERROR(is_segmented_);
    heuristics_.emplace_back(std::move(pt));
  }

  //! Returns list of heuristics for a segmented fusion.
  const std::vector<std::unique_ptr<HeuristicParams>>& heuristicsList() const {
    return heuristics_;
  }

  //! Returns the single heuristics for a complete fusion.
  HeuristicParams* singleKernelHeuristics() const {
    NVF_ERROR(!is_segmented_);
    return heuristics_.begin()->get();
  }

 private:
  std::vector<std::unique_ptr<HeuristicParams>> heuristics_;
  bool is_segmented_ = true;
};

} // namespace nvfuser
