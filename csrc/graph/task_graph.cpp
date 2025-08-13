// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <exceptions.h>
#include <graph/task_graph.h>
#include <utils.h>

#include <set>

namespace nvfuser {

void TaskGraph::validateSteps(const std::vector<Step>& steps) const {
  // First find any Data in the graph that has no definition. This must be
  // preallocated before running the program, so we initialize allocated and
  // high_water_mark to the sum of their sizes.
  TaskGraph::Size allocated = getInitialAllocation();
  TaskGraph::Size high_water_mark = allocated;

  std::vector<TaskId> future_uses = num_uses_;
  std::vector<DataId> outstanding_dependencies = num_dependencies_;

  // Now we are ready to process steps
  for (const Step& step : steps) {
    const Task& task = getTask(step.task);

    // Allocate outputs
    for (const DataId output_id : task.outputs) {
      const Data& data = getData(output_id);
      if (!data.input_alias.has_value()) {
        // Don't allocate outputs if they are reusing input memory
        allocated += data.size;
      }
    }

    // Add temporary space
    allocated += task.temp_space;

    // This is the most space we will use, so update high water mark here
    high_water_mark = std::max(high_water_mark, allocated);
    NVF_ERROR(step.high_water_mark == high_water_mark);

    // reduce use count for inputs and free them if possible
    for (const DataId input_id : task.inputs) {
      if (--future_uses.at((size_t)input_id)) {
        // There are no more uses for this Data, so free it if we're allowed to
        const Data& data = getData(input_id);
        if (data.can_free) {
          allocated -= data.size;
        }
      }
    }

    // step.allocated indicates how much space is allocated _upon completion_ of
    // this step
    NVF_ERROR(step.allocated == allocated);
  }
}

namespace {

//! [Backtracking algorithm to find optimal topological ordering]
//!
//! If validate==true, then we will validate the steps vector after every
//! backtracking step.
//!
//! c.f. https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
class TaskSorter {
 public:
  TaskSorter(const TaskGraph& graph, bool validate, int64_t max_iters)
      : graph_(graph), validate_(validate), max_iters_(max_iters) {
    sort();
  }

  const std::vector<TaskGraph::Step>& steps() const {
    return steps_;
  }

 private:
  inline void validate() const {
    if (validate_) {
      graph_.validateSteps(steps_);
    }
  }

  //! This pushes a step indicating that we should execute the given task next.
  void advance(TaskGraph::TaskId task_id) {
    TaskGraph::Size allocated = 0;
    TaskGraph::Size high_water_mark = 0;
    if (steps_.empty()) {
      // (Re-)Initialize allocated and high_water_mark to starting values
      allocated = graph_.getInitialAllocation();
      high_water_mark = allocated;
    } else {
      allocated = steps_.back().allocated;
      high_water_mark = steps_.back().high_water_mark;
    }

    // Compute the new allocated amount and high water mark for this step
    const TaskGraph::Task& task = graph_.getTask(task_id);

    for (const TaskGraph::DataId output_id : task.outputs) {
      const TaskGraph::Data& output = graph_.getData(output_id);
      // Allocate outputs if not aliased
      if (!output.input_alias.has_value()) {
        allocated += output.size;
      }

      // Update outstanding_dependencies_ and ready_tasks_ for each use
      for (const TaskGraph::TaskId use_id : output.uses) {
        if (--outstanding_dependencies_.at((size_t)use_id) == 0) {
          ready_tasks_.insert(use_id);
        }
      }
    }

    // Add temp space
    allocated += task.temp_space;

    // Update high water mark
    high_water_mark = std::max(high_water_mark, allocated);

    // Decrement future_uses_ and deallocate dead inputs
    for (const TaskGraph::DataId input_id : task.inputs) {
      const TaskGraph::Data& input = graph_.getData(input_id);
      if (--future_uses_.at((size_t)input_id) == 0) {
        if (input.can_free) {
          allocated -= input.size;
        }
      }
    }

    steps_.emplace_back(task_id, allocated, high_water_mark);
  }

  //! Backtrack a single step. This returns the TaskId of the step that was
  //! popped.
  TaskGraph::TaskId backtrack() {
    validate();
    TaskGraph::TaskId last_task_id = steps_.back().task;
    const TaskGraph::Task& last_task = graph_.getTask(last_task_id);
    steps_.pop_back();

    ready_tasks_.erase(last_task_id);

    // Update outstanding_dependencies to reflect that the outputs of last_task
    // are no longer available
    for (const TaskGraph::DataId& output_id : last_task.outputs) {
      const TaskGraph::Data& output = graph_.getData(output_id);
      for (const TaskGraph::TaskId use_id : output.uses) {
        outstanding_dependencies_.at((size_t)use_id)++;
      }
    }

    // Update future_uses to reflect that the inputs to last_task will need to
    // compute last_task later
    for (const TaskGraph::DataId& input_id : last_task.inputs) {
      future_uses_.at((size_t)input_id)++;
    }

    return last_task_id;
  }

  void sort() {
    // Set up outstanding_dependencies_, future_uses_, and ready_tasks_
    outstanding_dependencies_.reserve(graph_.numTasks());
    for (const TaskGraph::TaskId task_id : arange(graph_.numTasks())) {
      const TaskGraph::Task& task = graph_.getTask(task_id);
      TaskGraph::DataId inputs_to_compute = 0;
      for (const TaskGraph::DataId data_id : task.inputs) {
        const TaskGraph::Data& data = graph_.getData(data_id);
        if (data.definition.has_value()) {
          // Skip counting input data since these are available before we start
          inputs_to_compute++;
        }
      }
      outstanding_dependencies_.push_back(inputs_to_compute);
      if (inputs_to_compute == 0) {
        ready_tasks_.insert(task_id);
      }
    }

    future_uses_.reserve(graph_.numData());
    for (const TaskGraph::DataId data_id : arange(graph_.numData())) {
      const TaskGraph::Data& data = graph_.getData(data_id);
      future_uses_.push_back(data.uses.size());
    }

    // Initialize best_usage
    TaskGraph::Size best_usage = std::numeric_limits<TaskGraph::Size>::max();
    std::vector<TaskGraph::Step> best_steps;

    // This is the main optimization loop
    TaskGraph::TaskId backtracked_task_id = -1;
    for (int64_t _ : arange(max_iters_)) {
      NVF_ERROR(
          !ready_tasks_.empty() || steps_.size() == (size_t)graph_.numTasks(),
          "Ran out of ready tasks before completing ordering");

      TaskGraph::TaskId next_task_id = -1;
      for (const TaskGraph::TaskId ready_id : ready_tasks_) {
        if (ready_id > backtracked_task_id) {
          next_task_id = ready_id;
          break;
        }
      }

      if (next_task_id == -1) {
        // There are no ready tasks with ID above the backtracked_task_id. This
        // means it is time to backtrack
        backtracked_task_id = backtrack();
        continue;
      }

      advance(next_task_id);

      // If our high water mark is above best_usage, terminate early and
      // backtrack
      if (steps_.back().high_water_mark > best_usage) {
        backtracked_task_id = backtrack();
        continue;
      }

      // Our usage is at or below best_usage. Have we completed an ordering? If
      // so, update best_steps
      if (steps_.size() == (size_t)graph_.numTasks()) {
        best_steps = steps_;
      }
    }

    // Record our best found steps
    steps_ = best_steps;

    // Validate final result
    NVF_ERROR(steps_.size() == graph_.numTasks());
    validate();
  }

 private:
  const TaskGraph& graph_;
  bool validate_;
  int64_t max_iters_;
  std::vector<TaskGraph::Step> steps_;

  //! There is one entry here for each task and indicating how many
  //! dependencies are currently unmet. When this reaches zero the task becomes
  //! ready.
  std::vector<TaskGraph::DataId> outstanding_dependencies_;

  //! There is one entry here for each Data and indicating how many uses there
  //! are remaining. When it reaches zero, the Data can be freed if allowed.
  std::vector<TaskGraph::TaskId> future_uses_;

  //! This holds all candidates for the next step, sorted by ID
  std::set<TaskGraph::TaskId> ready_tasks_;
};

} // namespace

std::vector<TaskGraph::Step> TaskGraph::findOptimalOrder() const {
  // TODO: Find a reasonable default number of iterations. Note that one
  // iteration equals one task, not one ordering
  TaskSorter sorter(*this, /*validate=*/true, /*max_iters=*/2000);
  return sorter.steps();
}

} // namespace nvfuser
