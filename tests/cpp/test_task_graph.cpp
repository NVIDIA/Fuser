/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <macros.h>

#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <graph/task_graph.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using Tasks = std::vector<TaskGraph::Task>;
using TaskGraphTest = NVFuserTest;

struct SimpleAlias {
  TaskGraph::DataId output;
  TaskGraph::DataId input;
};

std::vector<TaskGraph::Data> inferData(const Tasks& tasks) {
  // Find number of data items so we can resize
  TaskGraph::DataId max_data_id = 0;
  for (const TaskGraph::Task& task : tasks) {
    for (TaskGraph::DataId input_id : task.inputs) {
      max_data_id = std::max(max_data_id, input_id);
    }
    for (TaskGraph::DataId output_id : task.outputs) {
      max_data_id = std::max(max_data_id, output_id);
    }
  }
  std::vector<TaskGraph::Data> all_data((size_t)max_data_id + 1);

  for (const auto& [task_id, task] : enumerate(tasks)) {
    for (TaskGraph::DataId input_id : task.inputs) {
      all_data.at(input_id).uses.push_back(task_id);
    }
    for (TaskGraph::DataId output_id : task.outputs) {
      all_data.at(output_id).definition = task_id;
    }
  }

  // Detect inputs and outputs and ensure they are not freed
  for (TaskGraph::Data& data : all_data) {
    data.can_free = data.definition.has_value() && !data.uses.empty();
  }

  return all_data;
}

std::vector<TaskGraph::TaskId> getTasks(const TaskGraph::SortResult& result) {
  const std::vector<TaskGraph::Step>& steps = result.steps;
  std::vector<TaskGraph::TaskId> tasks;
  tasks.reserve(steps.size());
  for (const TaskGraph::Step& step : steps) {
    tasks.push_back(step.task);
  }
  return tasks;
}

TEST_F(TaskGraphTest, Basic) {
  Tasks tasks{{{0, 1}, {2}}, {{0, 2}, {3}}};
  auto data = inferData(tasks);
  auto graph = TaskGraph(tasks, data);

  std::vector<TaskGraph::TaskId> expected{0, 1};
  EXPECT_EQ(getTasks(graph.findOptimalOrder()), expected);
}

// This example includes two segments, each of which aliases the other
TEST_F(TaskGraphTest, ImpossibleAlias) {
  // Two tasks, each takes the same two inputs
  Tasks tasks{{{0, 1}, {2}}, {{0, 1}, {3}}};
  auto data = inferData(tasks);
  // Each of the segment outputs aliases a different input
  data[2].aliases_input = 0;
  data[3].aliases_input = 1;
  // This graph can't be ordered without breaking the aliasing constraint
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { getTasks(graph.findOptimalOrder()); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

TEST_F(TaskGraphTest, SelfEdge) {
  Tasks tasks{{{0}, {0}}};
  auto data = inferData(tasks);
  // This graph can't be ordered because it contains an edge from a Data node
  // back to itself. A task can't be both producer and consumer to a Data.
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { getTasks(graph.findOptimalOrder()); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

TEST_F(TaskGraphTest, TwoCycle) {
  Tasks tasks{{{0}, {1}}, {{1}, {0}}};
  auto data = inferData(tasks);
  // This graph can't be ordered because it contains a cycle
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { getTasks(graph.findOptimalOrder()); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

TEST_F(TaskGraphTest, ThreeCycle) {
  Tasks tasks{{{0}, {1}}, {{1}, {2}}, {{2}, {0}}};
  auto data = inferData(tasks);
  // This graph can't be ordered because it contains a cycle
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { getTasks(graph.findOptimalOrder()); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

} // namespace nvfuser
