// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace nvfuser {

class TaskGraph {
 public:
  using TaskId = int16_t;
  using DataId = int16_t;
  using Size = int64_t;

  //! A Task consumes some input Data and produces some output Data. To do so,
  //! it might use some intermediate space.
  struct Task {
    std::vector<DataId> inputs;
    std::vector<DataId> outputs;
    //! This amount of temporary space is required only while executing the Task
    //! and is immediately freed afterward
    Size temp_space = 0;

    std::string toString() const;
  };

  struct Data {
    std::optional<TaskId> definition;
    std::vector<TaskId> uses;
    // If set, this means we do not allocate a new output when executing this
    // Data's definition, instead we re-use the space from the specified input.
    // Note that this implies an ordering constraint which we will check, since
    // the definition must be the last use of the aliased input.
    std::optional<DataId> input_alias;
    Size size;

    //! This indicates whether we are able to free this data after its last use.
    //! For a segmented fusion, unsegmented fusion inputs and outputs cannot be
    //! freed (with the exception of an aliased input), while any intermediate
    //! tensors should be freed as soon as possible.
    bool can_free = true;

    std::string toString() const;
  };

  TaskGraph(const std::vector<Task>& tasks, const std::vector<Data>& data);

  //! This represents the execution of a single Task in a given ordering. It
  //! tracks some cumulative state representing the amount of space required up
  //! to this point.
  struct Step {
    TaskId task;

    //! This is the sum of all Data that is active _after_ execution of this
    //! task and after any inputs with no more uses are freed.
    Size allocated;

    //! This is the maximum active space used until this step is completed.
    Size high_water_mark;

    std::string toString() const;
  };

  TaskId numTasks() const {
    return (TaskId)tasks_.size();
  }

  const Task& getTask(TaskId id) const {
    return tasks_.at((size_t)id);
  }

  TaskId numData() const {
    return (DataId)data_.size();
  }

  const Data& getData(DataId id) const {
    return data_.at((size_t)id);
  }

  Size getInitialAllocation() const {
    return initial_allocation_;
  }

  //! Given a list of steps, recompute the active space and high water mark.
  //! This is useful for validating that our backtracking algorithm does not
  //! corrupt this data. Raises an exception if corruption is detected.
  void validateSteps(const std::vector<Step>& steps) const;

  struct SortResult {
    std::vector<Step> steps;

    //! Number of iterations computed
    int64_t iterations;

    //! Whether the search was exhaustive. If not, then it was likely cut off
    //! early because of an iteration limit.
    bool exhaustive;

    std::string toString() const;
  };

  //! This does an exhaustive search of all possible orderings using a modified
  //! Kahn's algorithm to efficiently traverse the set of possible topological
  //! orderings.
  SortResult findOptimalOrder() const;

  std::string toString() const;

 private:
  std::vector<Task> tasks_;
  std::vector<Data> data_;

  //! How much data is allocated by data that has no definition, i.e. input data
  Size initial_allocation_ = 0;

  std::vector<TaskId> num_uses_;
  std::vector<DataId> num_dependencies_;
};

inline std::ostream& operator<<(std::ostream& os, const TaskGraph::Task& task) {
  os << task.toString();
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const TaskGraph::Data& data) {
  os << data.toString();
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const TaskGraph& graph) {
  os << graph.toString();
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const TaskGraph::Step& step) {
  os << step.toString();
  return os;
}
inline std::ostream& operator<<(
    std::ostream& os,
    const TaskGraph::SortResult& result) {
  os << result.toString();
  return os;
}

} // namespace nvfuser
