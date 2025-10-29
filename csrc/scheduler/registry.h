// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <exceptions.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <scheduler/compile_time_info.h>
#include <scheduler/utils.h>

namespace nvfuser {

class HeuristicDataCache;
class HeuristicParams;
class SchedulerRuntimeInfo;

//! Virtual base class for schedule heuristics
//!   heuristic implementations derive from this
//!   class and implement a schedule(Fusion*)
//!   and a bool canSchedule(Fusion*) interface
class SchedulerEntry {
 public:
  NVF_API virtual ~SchedulerEntry() = default;

  //! Fusion runtime facing API,
  //!   schedule the given fusion with heuristics owned
  //!   by this entry, for actual heuristics to override
  NVF_API virtual void schedule(
      Fusion* fusion,
      const HeuristicParams* params) = 0;

  virtual std::unique_ptr<HeuristicParams> computeHeuristics(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) = 0;

  // Compile check that the scheduler maybe able to schedule the fusion
  virtual bool canScheduleCompileTime(Fusion* fusion) = 0;

  // Runtime check that the scheduler can take the fusion. Scheduler must be
  // able to schedule the fusion if canScheduleCompileTime && this returns True.
  virtual bool canScheduleRunTime(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info,
      HeuristicDataCache* data_cache = nullptr) = 0;

  // Dispatch heuristic type to the right derived class of scheduler entry.
  // Scheduler entries are stateless so it's a lightweight class to dispatch to
  // the virtual functions in this abstract class.
  NVF_API static std::unique_ptr<SchedulerEntry> makeSchedulerInstance(
      SchedulerType scheduler_type);

  // Checks the provided scheduler type can schedule the fusion with the
  // provided inputs. Schedules the fusion according to the heuristics provided
  // by the scheduler. Returns the heuristics. This is simply a convenience
  // function for a common testing pattern. If validate_scheduler is set to
  // false canSchedule will not be checked.
  NVF_API static std::unique_ptr<HeuristicParams> scheduleWith(
      Fusion* fusion,
      SchedulerType scheduler_type,
      const KernelArgumentHolder& runtime_inputs,
      bool validate_scheduler = true);

  //! Heuristic comparison
  NVF_API bool sameAs(const SchedulerEntry* other);

  NVF_API const HeuristicParams* params() const {
    return params_.get();
  }

  //! Set scheduler hyperparameters for schedulers that use them
  void setSchedulerHyperParameters(
      const scheduler_utils::SchedulerHyperParameters* hyperparams) {
    scheduler_hyperparams_ = hyperparams;
  }

  //! Get scheduler hyperparameters (may be nullptr)
  const scheduler_utils::SchedulerHyperParameters* getSchedulerHyperParameters()
      const {
    return scheduler_hyperparams_;
  }

  std::unique_ptr<HeuristicParams> params_ = nullptr;

 private:
  //! Optional scheduler hyperparameters for schedulers that use them
  const scheduler_utils::SchedulerHyperParameters* scheduler_hyperparams_ =
      nullptr;
};

namespace Schedule {

//! External access for canSchedule utilities through SchedulerEntry
//!  to avoid exposing a single function to the namespace
NVF_API bool canSchedule(
    SchedulerType sh,
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache = nullptr,
    bool skip_compile_time_checks = false);

//! Fusion segmenter facing API,
//!   returns a schedule that applies in the given fusion, returns
//!   SchedulerType::None if no schedule in the registry can handle.
SchedulerType proposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info);
} // namespace Schedule

} // namespace nvfuser
