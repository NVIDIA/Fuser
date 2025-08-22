// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <exceptions.h>
#include <graph/task_graph.h>
#include <options.h>
#include <utils.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include "options.h"

namespace nvfuser {

TaskGraph::TaskGraph(
    const std::vector<Task>& tasks,
    const std::vector<Data>& data)
    : tasks_(tasks), data_(data) {
  NVF_ERROR(
      tasks.size() <= std::numeric_limits<TaskGraph::TaskId>::max(),
      "There are too many tasks to represent with TaskGraph::TaskId");
  NVF_ERROR(
      data.size() <= std::numeric_limits<TaskGraph::DataId>::max(),
      "There are too many data objects to represent with TaskGraph::DataId");

  // Initialize the counts of future uses of data and unmet dependencies of
  // tasks. These are the out-degrees of Data and in-degrees of Tasks,
  // respectively.
  num_dependencies_.reserve(tasks_.size());
  for (const Task& task : tasks_) {
    // Only count task inputs that are not already available (i.e. they have no
    // definition)
    num_dependencies_.push_back((DataId)std::count_if(
        task.inputs.begin(), task.inputs.end(), [&](DataId data_id) {
          return getData(data_id).definition.has_value();
        }));
    // Validate input
    for (DataId input_id : task.inputs) {
      NVF_ERROR(input_id >= 0 && (size_t)input_id < data_.size());
    }
    for (DataId output_id : task.outputs) {
      NVF_ERROR(output_id >= 0 && (size_t)output_id < data_.size());
    }
  }
  num_uses_.reserve(data_.size());
  for (const Data& data : data_) {
    num_uses_.push_back((TaskId)data.uses.size());
    if (!data.definition.has_value()) {
      initial_allocation_ += (Size)data.size;
    }
    // Validate input
    if (data.definition.has_value()) {
      DataId d = data.definition.value();
      NVF_ERROR(d >= 0 && (size_t)d < tasks_.size());
    }
    if (data.aliases_input.has_value()) {
      DataId a = data.aliases_input.value();
      NVF_ERROR(a >= 0 && (size_t)a < tasks_.size());
    }
    for (TaskId use : data.uses) {
      NVF_ERROR(use >= 0 && (size_t)use < tasks_.size());
    }
  }
}

std::string TaskGraph::toMermaid() const {
  std::stringstream ss;

  ss << "flowchart TD\n";

  bool print_data_size = false;
  if (numData() > 0) {
    Size sz = -1;
    for (const Data& data : data_) {
      if (data.size == 0) {
        continue;
      }
      if (sz == -1) {
        sz = data.size;
        continue;
      }
      if (data.size != 0 && data.size != sz) {
        print_data_size = true;
        break;
      }
    }
  }

  std::vector<bool> is_aliased_input(numData(), false);

  // Declare nodes with shapes and labels
  for (const auto& [data_id, data] : enumerate(data_)) {
    if (data.aliases_input.has_value()) {
      is_aliased_input.at(data.aliases_input.value()) = true;
    }
    ss << "    d" << data_id << "([\"d" << data_id;
    if (print_data_size || data.size == 0) {
      // Print data size if there are different sized data elements. Always
      // print [0] for empty data (these will be shown in gray)
      ss << " [" << data.size << "]";
    }
    ss << "\"]);\n";
  }
  for (const auto& [task_id, task] : enumerate(tasks_)) {
    if (task.temp_space != 0) {
      ss << "    t" << task_id << "[\"t" << task_id << " [" << task.temp_space
         << "]\"];\n";
    }
  }

  for (const auto& [task_id, task] : enumerate(tasks_)) {
    for (const DataId& input_id : task.inputs) {
      ss << "    d" << input_id << " --> t" << task_id << "\n";
    }
    for (const DataId& output_id : task.outputs) {
      ss << "    t" << task_id << " --> d" << output_id << "\n";
    }
  }

  ss << "\n";
  ss << "    classDef task fill:orange;\n";
  ss << "    classDef data fill:lightblue;\n";
  ss << "    classDef dataInput fill:lightgreen;\n";
  ss << "    classDef dataOutput fill:pink;\n";
  ss << "    classDef dataEmpty fill:#EEE,stroke:#DDD,color:#999;\n";
  ss << "    classDef aliasedInput fill:yellow;\n";
  ss << "    classDef aliasEdge stroke-dasharray:3,stroke:blue;\n";

  ss << "\n";
  for (const TaskId task_id : arange(numTasks())) {
    ss << "    class t" << task_id << " task;\n";
  }
  ss << "\n";
  for (const auto& [data_id, data] : enumerate(data_)) {
    // Create edges for aliases
    if (data.aliases_input.has_value()) {
      ss << "    d" << data_id << " alias" << data_id << "@--> d"
         << data.aliases_input.value() << ";\n";
      ss << "    class alias" << data_id << " aliasEdge;\n";
    }

    std::string class_name = "data";
    if (!data.definition.has_value()) {
      if (is_aliased_input.at(data_id)) {
        class_name = "aliasedInput";
      } else {
        class_name = "dataInput";
      }
    } else if (!data.can_free) {
      class_name = "dataOutput";
    } else if (data.size == 0) {
      class_name = "dataEmpty";
    }
    ss << "    class d" << data_id << " " << class_name << ";\n";
  }

  return ss.str();
}

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
    NVF_ERROR(
        outstanding_dependencies.at((size_t)step.task) == 0,
        "Invalid ordering found: task id ",
        step.task,
        " is executed before all its dependencies are available");

    const Task& task = getTask(step.task);

    // Allocate outputs
    for (const DataId output_id : task.outputs) {
      const Data& output = getData(output_id);
      if (output.aliases_input.has_value()) {
        // Check that the aliased input has no further uses
        // Note that we will decrement this use count later in this function
        NVF_ERROR(
            future_uses.at((size_t)output.aliases_input.value()) == 1,
            "Tried to execute segment that would overwrite input alias before "
            "some of its uses");
      } else {
        // Don't allocate outputs if they are reusing input memory
        allocated += output.size;
      }
    }

    // Add temporary space
    allocated += task.temp_space;

    // This is the most space we will use, so update high water mark here
    high_water_mark = std::max(high_water_mark, allocated);

    NVF_ERROR(
        step.high_water_mark == high_water_mark,
        "Mismatch in high water mark during validation");

    // reduce use count for inputs and free them if possible
    for (const DataId input_id : task.inputs) {
      if (--future_uses.at((size_t)input_id) == 0) {
        // There are no more uses for this Data, so free it if we're allowed to
        const Data& data = getData(input_id);
        if (data.can_free) {
          allocated -= data.size;
        }
      }
    }

    for (const DataId output_id : task.outputs) {
      const Data& data = getData(output_id);
      for (const TaskId use_id : data.uses) {
        --outstanding_dependencies.at((size_t)use_id);
      }
    }

    // step.allocated indicates how much space is allocated _upon completion_ of
    // this step
    NVF_ERROR(
        step.allocated == allocated, "Mismatch in allocated during validation");
  }
}

TaskGraph TaskGraph::convertAliasesToDependencies() const {
  // Begin with a copy of the tasks and data
  std::vector<Task> tasks{tasks_};
  std::vector<Data> data{data_};

  // This is used to ensure we don't have multiple aliases of the same input
  std::unordered_set<DataId> aliased_inputs;

  // If we modify data while traversing it, then we run the risk

  for (TaskId task_id : arange((TaskId)tasks.size())) {
    Task& task = tasks.at(task_id);
    for (DataId output_id : task.outputs) {
      Data& output = data.at((size_t)output_id);
      if (output.aliases_input.has_value()) {
        DataId& alias_id = output.aliases_input.value();
        // Reset the aliases_input flag before modifying the data vector
        output.aliases_input = std::nullopt;
        Data& alias = data.at((size_t)alias_id);
        NVF_ERROR_EQ(
            output.size,
            alias.size,
            "Expected alias to have same size as alias");
        // Reset to unaliased and set size to zero
        output.size = 0;

        NVF_ERROR(
            !aliased_inputs.contains(alias_id),
            "Found multiple outputs aliasing the same input");
        aliased_inputs.insert(alias_id);

        // For each use of the aliased input, add a new output to it and make
        // that output a new input to the current task
        for (TaskId use_id : alias.uses) {
          if (use_id == task_id) {
            continue;
          }
          Task& use = tasks.at((size_t)use_id);

          auto dummy_data_id = (DataId)data.size();
          data.emplace_back(
              /*definition=*/std::optional<TaskId>{use_id},
              /*uses=*/std::vector<TaskId>{task_id},
              /*aliases_input=*/std::nullopt,
              /*size=*/0,
              /*can_free=*/true);

          use.outputs.push_back(dummy_data_id);
          task.inputs.push_back(dummy_data_id);
        }
      }
    }
  }

  return {tasks, data};
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
  TaskSorter(
      const TaskGraph& graph,
      bool validate,
      int64_t max_time_us,
      bool print_debug)
      : orig_graph_(graph),
        graph_(graph.convertAliasesToDependencies()),
        debug_(print_debug),
        validate_(validate),
        max_time_us_(max_time_us) {
    if (debug_) {
      has_aliasing_ = std::ranges::any_of(
          arange(orig_graph_.numData()), [&](TaskGraph::DataId data_id) {
            return orig_graph_.getData(data_id).aliases_input.has_value();
          });
      if (has_aliasing_) {
        debug() << "Aliasing detected in task graph. Original graph:\n";
        debug() << orig_graph_.toString() << "\n\n";
        if (hasDebugDumpArgument(DebugDumpOption::TaskGraph, "mermaid")) {
          debug() << "Original graph (mermaid):\n"
                  << graph_.toMermaid() << std::endl;
        }
        debug() << "Modified graph without aliasing:\n";
        debug() << graph_.toString() << "\n\n";
        if (hasDebugDumpArgument(DebugDumpOption::TaskGraph, "mermaid")) {
          debug() << "Modified graph (mermaid):\n"
                  << graph_.toMermaid() << std::endl;
        }
      } else {
        debug() << graph_.toString() << "\n\n";
        if (hasDebugDumpArgument(DebugDumpOption::TaskGraph, "mermaid")) {
          debug() << "Mermaid graph:\n" << graph_.toMermaid() << std::endl;
        }
      }
    }
    sort();
  }

  const TaskGraph::SortResult& result() const {
    return result_;
  }

 private:
  inline void validate() const {
    if (validate_) {
      orig_graph_.validateSteps(steps_);
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

    NVF_ERROR(
        ready_tasks_.erase(task_id) == 1,
        "Attempted to advance to task that was not marked ready");

    // Compute the new allocated amount and high water mark for this step
    const TaskGraph::Task& task = graph_.getTask(task_id);

    for (const TaskGraph::DataId output_id : task.outputs) {
      const TaskGraph::Data& output = graph_.getData(output_id);
      // Allocate outputs if not aliased
      if (!output.aliases_input.has_value()) {
        allocated += output.size;
      }

      // Update outstanding_dependencies_ and ready_tasks_ for each use
      for (const TaskGraph::TaskId use_id : output.uses) {
        --outstanding_dependencies_.at((size_t)use_id);
        if (taskIsReady(use_id)) {
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

    ready_tasks_.insert(last_task_id);

    // Update outstanding_dependencies to reflect that the outputs of last_task
    // are no longer available
    for (const TaskGraph::DataId& output_id : last_task.outputs) {
      const TaskGraph::Data& output = graph_.getData(output_id);
      for (const TaskGraph::TaskId use_id : output.uses) {
        if (outstanding_dependencies_.at((size_t)use_id)++ == 0) {
          // This task _was_ ready but not it is not
          ready_tasks_.erase((size_t)use_id);
        }
      }
    }

    // Update future_uses to reflect that the inputs to last_task will need to
    // compute last_task later
    for (const TaskGraph::DataId& input_id : last_task.inputs) {
      future_uses_.at((size_t)input_id)++;
    }

    return last_task_id;
  }

  //! A task is ready if it has no outstanding_dependencies _and_ it is the last
  //! use for all of its aliased inputs.
  bool taskIsReady(TaskGraph::TaskId task_id) const {
    if (outstanding_dependencies_.at((size_t)task_id) != 0) {
      return false;
    }
    if (!has_aliasing_ || !task_has_aliased_input_.at((size_t)task_id)) {
      return true;
    }
    // The rest of this function is the aliasing dependency check
    for (const TaskGraph::DataId output_id : arange(graph_.numData())) {
      const TaskGraph::Data& output_data = graph_.getData(output_id);
      if (output_data.aliases_input.has_value()) {
        TaskGraph::DataId input_id = output_data.aliases_input.value();
        // Check for future uses (beyond the current one)
        if (future_uses_.at((size_t)input_id) > 1) {
          return false;
        }
      }
    }
    return true;
  }

  void sort() {
    if (has_aliasing_) {
      task_has_aliased_input_.resize(graph_.numTasks(), false);
      for (const TaskGraph::DataId data_id : arange(graph_.numData())) {
        const TaskGraph::Data& data = graph_.getData(data_id);
        if (data.aliases_input.has_value()) {
          NVF_ERROR(
              data.definition.has_value(),
              "Data that aliases input must have a definition");
          task_has_aliased_input_.at(data.definition.value()) = true;
          continue;
        }
      }
    }

    // Set up outstanding_dependencies_, future_uses_, and ready_tasks_
    future_uses_.resize(graph_.numData(), 0);
    for (const TaskGraph::DataId data_id : arange(graph_.numData())) {
      const TaskGraph::Data& data = graph_.getData(data_id);
      future_uses_.at((size_t)data_id) = data.uses.size();
    }

    outstanding_dependencies_.resize(graph_.numTasks(), 0);
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
      outstanding_dependencies_.at((size_t)task_id) = inputs_to_compute;
      if (taskIsReady(task_id)) {
        ready_tasks_.insert(task_id);
      }
    }

    // Initialize best steps found so far
    std::vector<TaskGraph::Step> best_steps;

    // This is the main optimization loop
    TaskGraph::TaskId backtracked_task_id = -1;

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start = Clock::now();

    for (int64_t iter : arange(10000000)) {
      if (iter % 64 == 0) {
        Clock::time_point end = Clock::now();
        if (std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() > max_time_us_) {
          result_.iterations = iter;
          break;
        }
      }

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

      // Reset backtracked_task_id
      backtracked_task_id = -1;

      if (next_task_id == -1) {
        // There are no ready tasks with ID above the backtracked_task_id. This
        // means it is time to backtrack

        if (steps_.empty()) {
          // If there is nowhere to backtrack it means we are done with the
          // search
          result_.exhaustive = true;
          break;
        }
        backtracked_task_id = backtrack();
        continue;
      }

      advance(next_task_id);

      // If our high water mark is above best_usage, terminate early and
      // backtrack
      if (!best_steps.empty() &&
          steps_.back().high_water_mark > best_steps.back().high_water_mark) {
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
    result_.steps = best_steps;

    // Validate final result
    NVF_ERROR(result_.steps.size() == (size_t)graph_.numTasks());
    validate();
  }

 private:
  const TaskGraph& orig_graph_;
  const TaskGraph graph_;
  const bool debug_;
  const bool validate_;
  const int64_t max_time_us_;

  //! This allows us to skip aliasing checks in the common case where no inputs
  //! are aliased by outputs
  bool has_aliasing_ = false;
  //! This tells us which tasks overwrite one of their inputs. For these, we
  //! will need to check that the aliased input has no future uses before
  //! advancing to it.
  std::vector<bool> task_has_aliased_input_;

  TaskGraph::SortResult result_;
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

std::string TaskGraph::Task::toString() const {
  std::stringstream ss;
  ss << "Task{";
  ss << "input ids={" << inputs << "}";
  ss << ", output ids={" << outputs << "}";
  ss << ", temp space=" << temp_space;
  ss << "}";
  return ss.str();
}

std::string TaskGraph::Data::toString() const {
  std::stringstream ss;
  ss << "Data{";
  ss << "definition="
     << (definition.has_value() ? std::to_string(definition.value()) : "none");
  ss << ", uses={" << uses << "}";
  ss << ", size=" << size;
  ss << ", aliases_input="
     << (aliases_input.has_value() ? std::to_string(aliases_input.value())
                                   : "none");
  ss << ", can_free=" << (can_free ? "yes" : "no");
  ss << "}";
  return ss.str();
}

std::string TaskGraph::Step::toString() const {
  std::stringstream ss;
  ss << "Step{";
  ss << "task id=" << task;
  ss << ", allocated=" << allocated;
  ss << ", high water mark=" << high_water_mark;
  ss << "}";
  return ss.str();
}

std::string TaskGraph::SortResult::toString() const {
  std::stringstream ss;
  ss << "SortResult{";
  ss << "steps={" << steps << "}";
  ss << ", iterations=" << iterations;
  ss << ", exhaustive=" << (exhaustive ? "yes" : "no");
  ss << "}";
  return ss.str();
}

std::string TaskGraph::toString() const {
  std::stringstream ss;
  ss << "TaskGraph{\n";
  ss << "  data:\n";
  for (DataId i : arange(numData())) {
    ss << "    " << i << " = " << getData(i) << "\n";
  }
  ss << "  tasks:\n";
  for (TaskId j : arange(numTasks())) {
    ss << "    " << j << " = " << getTask(j) << "\n";
  }
  ss << "}";
  return ss.str();
}

TaskGraph::SortResult TaskGraph::findOptimalOrder() const {
  TaskSorter sorter(
      *this,
      /*validate=*/true,
      /*max_time_us=*/100000,
      /*debug=*/isDebugDumpEnabled(DebugDumpOption::TaskGraph));
  return sorter.result();
}

} // namespace nvfuser
